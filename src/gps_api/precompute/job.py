"""Precompute orchestration + the ``gps-api-precompute`` console script.

Two run shapes share one chain: :func:`run_precompute` handles a single
region; :func:`run_fleet` (Phase-2 rollout) iterates **every** configured
region into one coherent store — per-region ``velocities/*.geojson`` and
``models/*_breaks.json``, one combined ``stations.geojson`` catalog
spanning all regions, and a single ``meta/run.json`` summarizing the whole
fleet run. Fault tolerance holds at both levels: one bad station is
recorded and skipped inside its region; one bad region is recorded and
skipped by the fleet.

Per region, for every configured station, the job calls the
``gps_analysis`` public API — no math is derived here (MATH_STANDARDS
rule):

1. :func:`gps_analysis.fit_components` — ``lineperiodic`` (or configured)
   trajectory fit per component, and :func:`gps_analysis.remove_trend` —
   the detrended series.
2. :func:`gps_analysis.estimate_velocity` — fixed-window WLS secular
   velocity with formal σ (``method="wls"``, the fleet baseline) — or,
   where the region configures ``velocity_method: mle`` (contract
   Amendment A5), :func:`gps_analysis.estimate_velocity_mle` — the
   colored-noise MLE with honest σ and per-component noise-model
   provenance on the feature. The GBIS velocity upgrade stays a later
   slice (PLAN-analysis-lane §1).
3. :func:`gps_analysis.detect_breakpoints` — GBIS4TS velocity break points
   + colored-noise parameters (``method="gbis"``), per component. The
   displacement series is passed straight in — the estimator
   zero-references internally (input-contract decision, PLAN-analysis-lane
   §7), so the job must NOT pre-reference it. **Cost gate:** GBIS4TS runs
   only for the regions in ``breakpoints.enabled_regions`` (selective by
   design — WLS is the fleet-wide baseline; the 1e6-iteration chains are
   never run across all stations). The chains themselves are fanned out
   over a process pool with an optional triage -> confirm screening stage
   (:mod:`gps_api.precompute.breaks` — perf-audit #1/#6, plan §10.7):
   the cheap per-station stages (fit/detrend/velocity/Parquet) stay
   inline; only the MCMC work items are parallelized.

4. :func:`gps_api.precompute.deformation.compute_mogi_series` — the Mogi
   ΔV(t) deformation-source stage (Amendment A6), gated by
   ``deformation.enabled_regions`` exactly like break detection. Its
   product feeds ``GET /v1/deformation/{region}``; a stage failure is
   recorded on the run summary without sinking the region.

5. :func:`gps_api.precompute.outliers.detect_station_outliers` — the
   outlier-detection stage (design §5.1, contract Amendment A8), gated by
   ``outliers.enabled`` (CLI ``--no-outliers``). Runs FIRST per station:
   flags ride as additive Parquet columns (raw columns byte-identical),
   the detrended columns become the residuals of the outlier-robust
   step-augmented fit (``steps.csv`` known steps — the SENG lesson), all
   downstream estimates (velocity, breaks, deformation) fit on the
   INLIERS, and the protected ``SuspectedEvent`` clusters are written to
   ``meta/suspected_steps.csv`` for operator review. Aborts/failures are
   recorded in ``meta/run.json`` — loud, never silent.

Products land in the file store (:mod:`gps_api.precompute.products`);
``GET /v1/velocities`` serves the velocity GeoJSON directly.

The job always runs in the foreground — scheduling (cron/systemd timer) is
the deployment's business, detaching is not this module's.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from gps_analysis import (
    VelocityEstimate,
    VelocityEstimateMLE,
    estimate_velocity,
    estimate_velocity_mle,
    fit_components,
    linear,
    lineperiodic,
    remove_trend,
)
from numpy.typing import NDArray

from gps_api import settings
from gps_api.precompute import products
from gps_api.precompute.breaks import (
    BreakSummary,
    ComponentTask,
    TriageStats,
    detect_station_breaks,
)
from gps_api.precompute.config import (
    AnalysisConfig,
    DeformationConfig,
    StationMeta,
    load_analysis_config,
    load_station_meta,
    load_step_catalog,
)
from gps_api.precompute.deformation import compute_mogi_series
from gps_api.precompute.outliers import (
    StationOutliers,
    config_hash,
    detect_station_outliers,
    mask_station_series,
)
from gps_api.precompute.products import Provenance
from gps_api.precompute.slip import compute_slip_distribution
from gps_api.precompute.sources import (
    COMPONENTS,
    FloatArray,
    StationSeries,
    load_neu,
    synthetic_station,
    yearf_to_datetime,
)
from gps_api.schemas import (
    ComponentNoise,
    PointGeometry,
    StationCollection,
    StationFeature,
    StationProperties,
    VelocityCollection,
    VelocityFeature,
    VelocityProperties,
)

#: Trajectory models the config may name (gps_analysis callables).
_MODEL_FUNCS: dict[str, Any] = {
    "lineperiodic": lineperiodic,
    "linear": linear,
}

SeriesLoader = Callable[[str], StationSeries]


@dataclasses.dataclass(frozen=True)
class RunSummary:
    """What one per-region precompute run produced.

    Written to ``meta/run.json`` for a single-region run; embedded per
    region in the :class:`FleetSummary` for a fleet run.
    """

    region: str
    store: Path
    fitted_at: datetime.datetime
    source: str
    stations_ok: tuple[str, ...]
    stations_failed: dict[str, str]
    products: tuple[str, ...]
    api_max_points: int | None = None
    #: Region-level note when the (config-gated) deformation stage (Mogi or
    #: Okada) failed — the region's other products survive (fault-tolerance
    #: rule), but the failure is recorded, never silent.
    deformation_failed: str | None = None
    #: Whether the outlier-detection stage ran for this region — controls
    #: whether the ``outliers_*`` keys appear in ``meta/run.json``.
    outliers_enabled: bool = False
    #: Stations whose detection hit the §3.5 excess-candidate abort: their
    #: series proceeded UNMASKED (flags all-False) — loud, never silent
    #: (design §5.1 ``outliers_aborted``).
    outliers_aborted: tuple[str, ...] = ()
    #: Stations whose detection *raised*: they proceed unmasked with their
    #: products intact (fault tolerance, like ``deformation_failed``) and
    #: the reason is recorded here.
    outliers_failed: dict[str, str] = dataclasses.field(default_factory=dict)
    #: Suspected-event rows destined for ``meta/suspected_steps.csv``
    #: (kept on the summary so fleet runs can aggregate across regions).
    suspected_steps: tuple[dict[str, Any], ...] = ()

    def outliers_dict(self) -> dict[str, Any]:
        """The ``outliers_*`` members of the run.json payloads."""
        return {
            "outliers_aborted": list(self.outliers_aborted),
            "outliers_failed": dict(self.outliers_failed),
            "suspected_steps": len(self.suspected_steps),
        }

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready mapping for ``meta/run.json``."""
        payload: dict[str, Any] = {
            "region": self.region,
            "store": str(self.store),
            "fitted_at": self.fitted_at.isoformat().replace("+00:00", "Z"),
            "source": self.source,
            "stations_ok": list(self.stations_ok),
            "stations_failed": dict(self.stations_failed),
            "products": list(self.products),
        }
        if self.deformation_failed is not None:
            payload["deformation_failed"] = self.deformation_failed
        if self.outliers_enabled:
            payload.update(self.outliers_dict())
        if self.api_max_points is not None:
            payload["api"] = {"max_points": self.api_max_points}
        return payload


@dataclasses.dataclass(frozen=True)
class FleetSummary:
    """What one fleet run produced (written to ``meta/run.json``).

    Aggregates the per-region :class:`RunSummary` results plus the regions
    that failed outright; totals give the per-station success/failure
    counts across the whole fleet.
    """

    store: Path
    fitted_at: datetime.datetime
    source: str
    regions: dict[str, RunSummary]
    regions_failed: dict[str, str]
    products: tuple[str, ...]
    api_max_points: int | None = None

    @property
    def stations_ok_total(self) -> int:
        """Stations that produced products, summed over successful regions."""
        return sum(len(s.stations_ok) for s in self.regions.values())

    @property
    def stations_failed_total(self) -> int:
        """Stations recorded as failed, summed over successful regions."""
        return sum(len(s.stations_failed) for s in self.regions.values())

    @property
    def deformation_failed_total(self) -> int:
        """Regions whose gated deformation stage (Mogi or Okada) failed."""
        return sum(1 for s in self.regions.values() if s.deformation_failed is not None)

    @property
    def outliers_failed_total(self) -> int:
        """Stations whose outlier detection raised, summed over regions."""
        return sum(len(s.outliers_failed) for s in self.regions.values())

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready mapping for ``meta/run.json`` (fleet shape)."""
        payload: dict[str, Any] = {
            "fleet": True,
            "store": str(self.store),
            "fitted_at": self.fitted_at.isoformat().replace("+00:00", "Z"),
            "source": self.source,
            "regions": {
                name: {
                    "stations_ok": list(summary.stations_ok),
                    "stations_failed": dict(summary.stations_failed),
                    "products": list(summary.products),
                    **(
                        {"deformation_failed": summary.deformation_failed}
                        if summary.deformation_failed is not None
                        else {}
                    ),
                    **(summary.outliers_dict() if summary.outliers_enabled else {}),
                }
                for name, summary in self.regions.items()
            },
            "regions_failed": dict(self.regions_failed),
            "totals": {
                "regions_ok": len(self.regions),
                "regions_failed": len(self.regions_failed),
                "stations_ok": self.stations_ok_total,
                "stations_failed": self.stations_failed_total,
                # Outlier-stage totals only when the stage ran anywhere —
                # keeps pre-A8 fleet payload shapes byte-stable.
                **(
                    {
                        "outliers_failed": self.outliers_failed_total,
                        "outliers_aborted": sum(
                            len(s.outliers_aborted) for s in self.regions.values()
                        ),
                    }
                    if any(s.outliers_enabled for s in self.regions.values())
                    else {}
                ),
            },
            "products": list(self.products),
        }
        if self.api_max_points is not None:
            payload["api"] = {"max_points": self.api_max_points}
        return payload


def _point(meta: StationMeta) -> PointGeometry:
    """GeoJSON point ([lon, lat] or [lon, lat, height_m]) for one station."""
    coords = [meta.lon, meta.lat]
    if meta.height is not None:
        coords.append(meta.height)
    return PointGeometry(coordinates=coords)


def _velocity_feature(
    series: StationSeries,
    meta: StationMeta,
    cfg: AnalysisConfig,
    model_name: str,
    method: str,
) -> VelocityFeature:
    """Velocity vector for one station as a contract GeoJSON feature.

    ``method`` is the region's resolved estimator (Amendment A5): ``"wls"``
    calls :func:`gps_analysis.estimate_velocity` (formal σ, the fleet
    baseline); ``"mle"`` calls :func:`gps_analysis.estimate_velocity_mle`
    (colored-noise GLS σ — honest for flicker-dominated series) and stamps
    the per-component noise models onto the feature as provenance.
    """
    t_last = float(series.t[-1])
    window = (t_last - cfg.velocity_window_years, None)
    noise: dict[str, ComponentNoise] | None = None
    estimate: VelocityEstimate
    if method == "mle":
        mle_kwargs: dict[str, Any] = {}
        if cfg.velocity_kappa_bounds is not None:
            mle_kwargs["kappa_bounds"] = cfg.velocity_kappa_bounds
        mle: VelocityEstimateMLE = estimate_velocity_mle(
            series.t,
            series.y,
            model=model_name,
            window=window,
            names=COMPONENTS,
            **mle_kwargs,
        )
        estimate = mle
        noise = {
            component: ComponentNoise(
                sigma_white_mm=model.sigma_white,
                amplitude_mm=model.amplitude_powerlaw,
                spectral_index=model.spectral_index,
            )
            for component, model in zip(COMPONENTS, mle.noise, strict=True)
        }
    else:
        estimate = estimate_velocity(
            series.t,
            series.y,
            series.sigma,
            model=model_name,
            window=window,
            names=COMPONENTS,
        )
    if estimate.magnitude is None or estimate.azimuth is None:
        raise RuntimeError(
            f"{series.marker}: no horizontal magnitude/azimuth on the "
            "velocity estimate (north/east components missing?)"
        )
    i_n, i_e, i_u = (series.component_index(c) for c in ("north", "east", "up"))
    return VelocityFeature(
        geometry=_point(meta),
        properties=VelocityProperties(
            marker=series.marker,
            east=float(estimate.rates[i_e]),
            north=float(estimate.rates[i_n]),
            up=float(estimate.rates[i_u]),
            sigma_east=float(estimate.sigmas[i_e]),
            sigma_north=float(estimate.sigmas[i_n]),
            sigma_up=float(estimate.sigmas[i_u]),
            magnitude=float(estimate.magnitude),
            azimuth=float(estimate.azimuth),
            method="mle" if method == "mle" else "wls",
            window_start=yearf_to_datetime(estimate.span[0]),
            window_end=yearf_to_datetime(estimate.span[1]),
            noise=noise,
        ),
    )


def _break_parameters(model: str, optimal: Sequence[float]) -> dict[str, float]:
    """Flatten an optimal parameter vector (MATLAB order) to named floats.

    BPD1: ``[a, v, g, t_b, κ, β]``; BPD2: ``[a, v, g1, t_b1, g2, t_b2, κ, β]``
    — intercept mm, secular rate mm/yr, rate change(s) mm/yr, break
    epoch(s) yr, colored-noise spectral index κ and amplitude β.
    """
    opt = [float(v) for v in optimal]
    parameters = {
        "intercept_mm": opt[0],
        "trend_mm_yr": opt[1],
        "trend_change_mm_yr": opt[2],
        "breakpoint_yearf": opt[3],
    }
    if model == "BPD2":
        parameters["trend_change2_mm_yr"] = opt[4]
        parameters["breakpoint2_yearf"] = opt[5]
    parameters["kappa"] = opt[-2]
    parameters["amp_mm"] = opt[-1]
    return parameters


def _station_break_tasks(
    series: StationSeries,
    detrended: FloatArray,
    *,
    n_breaks: int,
    n_runs: int,
    t_runs: int,
    seed: int | None,
    outlier_flags: NDArray[np.bool_] | None = None,
) -> list[ComponentTask]:
    """Per-component GBIS4TS work items for one station (confirm settings).

    The displacement series goes straight into the task (the estimator
    zero-references internally — do not pre-reference here). The fixed
    white-noise amplitude follows the dev-viz heuristic: median
    observation σ, or the residual std when the source carries no σ.

    ``outlier_flags`` (shape ``(3, N)``, True = outlier) applies the
    per-component INLIER mask before the chain is queued — GBIS4TS
    consumes the cleaned series (BGÓ Q8: its likelihood assumes no
    blunders); the store keeps every epoch regardless.
    """
    tasks: list[ComponentTask] = []
    for i, component in enumerate(COMPONENTS):
        keep = (
            np.ones(series.t.size, dtype=np.bool_)
            if outlier_flags is None
            else ~outlier_flags[i]
        )
        if series.sigma is not None:
            wn_amp = float(np.median(series.sigma[i][keep]))
        else:
            wn_amp = float(np.std(detrended[i][keep]))
        tasks.append(
            ComponentTask(
                marker=series.marker,
                component=component,
                t=series.t[keep],
                y=series.y[i][keep],
                wn_amp=wn_amp,
                n_breaks=n_breaks,
                n_runs=n_runs,
                t_runs=t_runs,
                seed=seed,
            )
        )
    return tasks


def _entry_from_summary(
    summary: BreakSummary, fitted_at: datetime.datetime
) -> dict[str, Any]:
    """One break-catalog entry (contract ``BreakEntry`` shape) per chain."""
    parameters = _break_parameters(summary.model, summary.optimal)
    return {
        "marker": summary.marker,
        "component": summary.component,
        "model": summary.model,
        "method": "gbis",
        "fitted_at": fitted_at.isoformat().replace("+00:00", "Z"),
        "breakpoint_time": yearf_to_datetime(parameters["breakpoint_yearf"])
        .isoformat()
        .replace("+00:00", "Z"),
        "parameters": parameters,
        "wn_amp_mm": summary.wn_amp,
        "y_ref_mm": summary.y_ref,
        "n_runs": summary.n_runs,
    }


def _write_mogi_deformation(
    store: Path,
    region_name: str,
    station_series: dict[str, StationSeries],
    detrended: dict[str, FloatArray],
    meta: dict[str, StationMeta],
    dcfg: DeformationConfig,
    fitted_at: datetime.datetime,
    seed: int | None,
    provenance: Callable[..., Provenance],
) -> str:
    """Mogi ΔV(t) stage (A6): fit + write ``models/<region>_deformation.json``."""
    outcome = compute_mogi_series(
        region_name, station_series, detrended, meta, dcfg, fitted_at, seed=seed
    )
    extra: dict[str, Any] = {
        "series": dcfg.series,
        "window_years": dcfg.window_years,
        "step_days": dcfg.step_days,
        "epoch_mean_days": dcfg.epoch_mean_days,
        "min_stations": dcfg.min_stations,
        "nu": dcfg.nu,
        "depth_bounds_km": list(dcfg.depth_bounds_km),
        "dv_bounds_m3": (
            list(dcfg.dv_bounds_m3) if dcfg.dv_bounds_m3 is not None else None
        ),
        "epochs_skipped": outcome.epochs_skipped,
        "stations_excluded": list(outcome.stations_excluded),
        # Independent GNSS-only product — compared against, never derived
        # from, Vincent's InSAR-side inv_volume_mogi.dat.
        "cross_check": (
            "independent GNSS-only analog of the operational Mogi "
            "dV(t) at insar.vedur.is:/mnt/scratch/vincent/model/"
            "svartsengi/inflation*/inv_volume_mogi.dat"
        ),
    }
    if dcfg.bayes_n_runs > 0:
        extra["bayes"] = {
            "n_runs": dcfg.bayes_n_runs,
            "t_runs": dcfg.bayes_t_runs,
            "seed": seed,
        }
    path = products.write_deformation_json(
        store, region_name, outcome.result, provenance("mogi", **extra)
    )
    print(
        f"[region {region_name}] mogi deformation: "
        f"{len(outcome.result.fits)} epoch(s) fitted, "
        f"{outcome.epochs_skipped} skipped",
        flush=True,
    )
    return str(path)


def _write_okada_slip(
    store: Path,
    region_name: str,
    station_series: dict[str, StationSeries],
    detrended: dict[str, FloatArray],
    meta: dict[str, StationMeta],
    dcfg: DeformationConfig,
    fitted_at: datetime.datetime,
    provenance: Callable[..., Provenance],
) -> str:
    """Okada distributed-slip stage (A7): invert + write ``<region>_slip.json``."""
    outcome = compute_slip_distribution(
        region_name, station_series, detrended, meta, dcfg, fitted_at
    )
    okada = dcfg.okada
    assert okada is not None  # DeformationConfig guarantees it for source=okada
    result = outcome.result
    extra: dict[str, Any] = {
        "series": dcfg.series,
        "window_years": dcfg.window_years,
        "epoch_mean_days": dcfg.epoch_mean_days,
        "min_stations": dcfg.min_stations,
        "nu": dcfg.nu,
        "plane": {
            "origin": {"lon": okada.origin_lon, "lat": okada.origin_lat},
            "strike": okada.strike,
            "dip": okada.dip,
            "length_km": okada.length_km,
            "width_km": okada.width_km,
            "top_depth_km": okada.top_depth_km,
            "n_strike": okada.n_strike,
            "n_dip": okada.n_dip,
        },
        "components": list(okada.components),
        "nonneg": okada.nonneg,
        "edge": okada.edge,
        "smoothing": result.smoothing,
        "smoothing_selected_by": result.smoothing_selected_by,
        "stations_excluded": list(outcome.stations_excluded),
        # Formal σ caveat (see slip.compute_slip_distribution): the per-patch
        # covariance is the unconstrained linear-Gaussian one; under NNLS it
        # is not exact for patches pinned at the slip >= 0 bound.
        "sigma_note": (
            "per-patch sigma is the unconstrained linear-Gaussian formal "
            "covariance propagated through okada_greens/patch_laplacian; not "
            "exact for patches pinned by the non-negativity constraint"
        ),
    }
    path = products.write_slip_json(
        store, region_name, result, provenance("okada", **extra)
    )
    print(
        f"[region {region_name}] okada slip: {len(result.patches)} patch(es), "
        f"lambda={result.smoothing:.3g} ({result.smoothing_selected_by}), "
        f"{result.n_stations} station(s)",
        flush=True,
    )
    return str(path)


def run_precompute(
    cfg: AnalysisConfig,
    region_name: str,
    series_loader: SeriesLoader,
    store: Path,
    source: str,
    *,
    detect_breaks: bool | None = None,
    compute_deformation: bool | None = None,
    compute_outliers: bool | None = None,
    n_runs: int | None = None,
    t_runs: int | None = None,
    triage_n_runs: int | None = None,
    triage_t_runs: int | None = None,
    max_workers: int | None = None,
    seed: int | None = 0,
    write_catalog: bool = True,
    write_meta: bool = True,
) -> RunSummary:
    """Run the full precompute for one region and write the products.

    Args:
        cfg: Loaded analysis-lane configuration.
        region_name: Region key in ``cfg.regions``.
        series_loader: ``marker -> StationSeries`` (``.NEU`` or synthetic —
            the data source is a parameter).
        store: Store root directory (created as needed).
        source: Provenance tag describing the data source for the run.
        detect_breaks: Force break detection on/off; ``None`` follows
            ``breakpoints.enabled_regions`` in the config.
        compute_deformation: Force the Mogi deformation stage on/off;
            ``None`` follows ``deformation.enabled_regions`` in the config
            (Amendment A6 — gated exactly like break detection). A stage
            failure is recorded on the summary
            (:attr:`RunSummary.deformation_failed`) without sinking the
            region's other products.
        compute_outliers: Force the outlier-detection stage on/off;
            ``None`` follows ``outliers.enabled`` in the config (design
            §5.4; CLI ``--no-outliers``). When on, every station's series
            gets flagged non-destructively (raw Parquet columns unchanged,
            additive flag columns per design §5.2), the downstream
            estimates (velocity, GBIS4TS breaks, deformation) fit on the
            **inliers**, and the protected suspected-event clusters land in
            ``meta/suspected_steps.csv``. A per-station detection failure
            is recorded (:attr:`RunSummary.outliers_failed`) and the
            station proceeds unmasked — fault tolerance, never silent.
        n_runs / t_runs: Override the configured GBIS4TS confirm chain
            lengths (dev runs; production uses the configured 1e6).
        triage_n_runs / triage_t_runs: Override the configured triage
            screen chain lengths; ``None`` follows
            ``breakpoints.triage_n_runs`` / ``triage_t_runs`` (a resolved
            value of 0 keeps triage off — every gated station confirmed).
        max_workers: Override ``breakpoints.max_workers`` for the break
            chain process pool (``None`` follows the config; the config's
            own default is one worker per core, capped at the number of
            chains; 0 runs inline).
        seed: MCMC RNG seed (reproducibility — pooled chains use exactly
            the per-call seeds the serial path used, so results match).
        write_catalog / write_meta: Whether to write ``stations.geojson``
            and ``meta/run.json`` (defaults on). :func:`run_fleet` turns
            both off per region and writes the combined catalog + fleet
            summary itself, so a fleet store stays coherent.

    Per-station failures are recorded and skipped — one bad station must
    not sink the batch (fault-tolerance rule of the ops packages).
    """
    region = cfg.region(region_name)
    # Resolved per-region estimator (Amendment A5) — raises on 'gbis'/typos.
    velocity_method = cfg.velocity_method_for(region.name)
    meta = load_station_meta(region.stations)
    fitted_at = datetime.datetime.now(datetime.UTC)
    breaks_on = (
        cfg.breakpoints.enabled_for(region.name)
        if detect_breaks is None
        else detect_breaks
    )
    deformation_on = (
        cfg.deformation.enabled_for(region.name)
        if compute_deformation is None
        else compute_deformation
    )
    outliers_on = cfg.outliers.enabled if compute_outliers is None else compute_outliers
    step_catalog = load_step_catalog(cfg.config_dir) if outliers_on else {}
    if outliers_on and not step_catalog:
        # The real-data finding (SENG): a stepless trajectory model
        # over-flags real signal on active stations — say so up front.
        print(
            f"[region {region.name}] outliers: no steps.csv under "
            f"{cfg.config_dir} — detection runs without known-step terms "
            "(over-flagging risk on stations with equipment/coseismic steps)",
            file=sys.stderr,
            flush=True,
        )
    runs = cfg.breakpoints.n_runs if n_runs is None else n_runs
    truns = cfg.breakpoints.t_runs if t_runs is None else t_runs
    triage_runs = (
        cfg.breakpoints.triage_n_runs if triage_n_runs is None else triage_n_runs
    )
    triage_truns = (
        cfg.breakpoints.triage_t_runs if triage_t_runs is None else triage_t_runs
    )
    workers = cfg.breakpoints.max_workers if max_workers is None else max_workers
    if breaks_on and 16 * truns >= runs:
        raise ValueError(
            f"breakpoints: n_runs ({runs}) must exceed the annealing span "
            f"16*t_runs ({16 * truns}) — adjust analysis.yaml or --runs/--t-runs"
        )
    if breaks_on and triage_runs > 0 and 16 * triage_truns >= triage_runs:
        raise ValueError(
            f"breakpoints: triage_n_runs ({triage_runs}) must exceed the "
            f"annealing span 16*triage_t_runs ({16 * triage_truns}) — adjust "
            "analysis.yaml or --triage-runs/--triage-t-runs"
        )

    def _provenance(method: str, **extra: Any) -> Provenance:
        return Provenance(
            method=method,
            frame=region.reference_frame,
            fitted_at=fitted_at,
            source=source,
            extra=extra,
        )

    velocity_features: list[VelocityFeature] = []
    break_entries: list[dict[str, Any]] = []
    break_tasks: dict[str, list[ComponentTask]] = {}
    triage_stats: TriageStats | None = None
    # Kept only for deformation-gated regions (small, config-gated sets) —
    # the Mogi stage needs the whole region's fields at once.
    kept_series: dict[str, StationSeries] = {}
    kept_detrended: dict[str, FloatArray] = {}
    written: list[str] = []
    ok: list[str] = []
    failed: dict[str, str] = {}
    deformation_failed: str | None = None
    outliers_aborted: list[str] = []
    outliers_failed: dict[str, str] = {}
    suspected_rows: list[dict[str, Any]] = []

    for marker in region.stations:
        model_name = cfg.detrend_model_for(marker)
        if model_name not in _MODEL_FUNCS:
            raise ValueError(
                f"unknown trajectory model {model_name!r} for {marker} — "
                f"configure one of {sorted(_MODEL_FUNCS)}"
            )
        try:
            series = series_loader(marker)
            print(f"[{marker}] {series.t.size} epochs ({series.source})", flush=True)

            # Outlier stage (design §5.1): flags + step-augmented robust
            # fits. NON-destructive by construction — the raw series is
            # written unchanged; detection failure is recorded and the
            # station proceeds unmasked (fault tolerance, never silent).
            station_outliers: StationOutliers | None = None
            if outliers_on:
                try:
                    station_outliers = detect_station_outliers(
                        series,
                        _MODEL_FUNCS[model_name],
                        cfg.outliers,
                        step_catalog.get(marker, ()),
                    )
                except Exception as exc:  # noqa: BLE001 — station proceeds unmasked
                    outliers_failed[marker] = f"{type(exc).__name__}: {exc}"
                    print(
                        f"[{marker}] outlier detection FAILED — "
                        f"{outliers_failed[marker]} (station proceeds unmasked)",
                        file=sys.stderr,
                        flush=True,
                    )
            if station_outliers is not None:
                if station_outliers.aborted:
                    # §3.5 excess-candidate abort: flags are all-False —
                    # the station proceeds unmasked, loudly recorded.
                    outliers_aborted.append(marker)
                    print(
                        f"[{marker}] outlier detection ABORTED "
                        "(candidate fraction > max_flag_fraction — likely "
                        "unmodeled signal; station proceeds unmasked, "
                        "suspected events recorded)",
                        file=sys.stderr,
                        flush=True,
                    )
                suspected_rows.extend(
                    event.as_row(region.name, station_aborted=station_outliers.aborted)
                    for event in station_outliers.events
                )
                # Detrended columns = residuals of the outlier-robust
                # step-augmented fit, evaluated at ALL epochs (design §5.1).
                detrended = station_outliers.detrended
                n_flagged = int(np.count_nonzero(station_outliers.union_flags))
                print(
                    f"[{marker}] outliers: {n_flagged} epoch(s) flagged, "
                    f"{len(station_outliers.events)} suspected event(s)",
                    flush=True,
                )
            else:
                fits = tuple(
                    fit_components(
                        _MODEL_FUNCS[model_name],
                        series.t,
                        series.y,
                        sigma=series.sigma,
                        names=COMPONENTS,
                    )
                )
                detrended = np.asarray(
                    remove_trend(_MODEL_FUNCS[model_name], series.t, series.y, fits),
                    dtype=np.float64,
                )
            times = [yearf_to_datetime(float(v)) for v in series.t]
            series_extra: dict[str, Any] = {"units": "mm", "marker": marker}
            if station_outliers is not None:
                series_extra["outliers"] = station_outliers.provenance()
            written.append(
                str(
                    products.write_series_parquet(
                        store,
                        series,
                        detrended,
                        times,
                        _provenance(model_name, **series_extra),
                        outliers=station_outliers,
                    )
                )
            )
            # HARD RULE (BGÓ): downstream parameter estimates fit on the
            # INLIERS; the store keeps every epoch. Velocity/deformation
            # take the union-inlier view (shared time axis across the
            # three components); the per-component GBIS chains mask per
            # component below.
            if station_outliers is not None:
                keep_union = ~station_outliers.union_flags
                estimate_series = mask_station_series(series, keep_union)
                estimate_detrended = detrended[:, keep_union]
            else:
                estimate_series = series
                estimate_detrended = detrended
            velocity_features.append(
                _velocity_feature(
                    estimate_series, meta[marker], cfg, model_name, velocity_method
                )
            )
            print(
                f"[{marker}] fit + {velocity_method.upper()} velocity done",
                flush=True,
            )
            if deformation_on:
                kept_series[marker] = estimate_series
                kept_detrended[marker] = estimate_detrended
            if breaks_on:
                # Chains are the fleet's cost — queue them for the pool
                # instead of running the MCMC inline (perf-audit #1).
                # GBIS4TS consumes the CLEANED (inlier) series (BGÓ Q8) —
                # per-component masks, since each chain is per component.
                break_tasks[marker] = _station_break_tasks(
                    series,
                    detrended,
                    n_breaks=cfg.breakpoints.n_breaks,
                    n_runs=runs,
                    t_runs=truns,
                    seed=seed,
                    outlier_flags=(
                        None if station_outliers is None else station_outliers.flags
                    ),
                )
            ok.append(marker)
        except Exception as exc:  # noqa: BLE001 — batch survives one bad station
            failed[marker] = f"{type(exc).__name__}: {exc}"
            print(f"[{marker}] FAILED — {failed[marker]}", file=sys.stderr, flush=True)

    if breaks_on and break_tasks:
        outcome = detect_station_breaks(
            break_tasks,
            triage_n_runs=triage_runs,
            triage_t_runs=triage_truns,
            triage_sigma=cfg.breakpoints.triage_sigma,
            max_workers=workers,
        )
        triage_stats = outcome.triage
        for marker, reason in outcome.failures.items():
            # Same semantics as the serial path: a station whose break
            # detection failed is recorded as failed (its series/velocity
            # products, already written, stay in the store).
            failed[marker] = reason
            ok.remove(marker)
        for marker in region.stations:  # station order, component order
            break_entries.extend(
                _entry_from_summary(summary, fitted_at)
                for summary in outcome.summaries.get(marker, ())
            )

    if not ok:
        raise RuntimeError(
            f"precompute produced nothing — every station failed: {failed}"
        )

    if write_catalog:
        catalog = StationCollection(
            features=[
                StationFeature(
                    geometry=_point(meta[m]),
                    properties=StationProperties(
                        marker=m, name=meta[m].name, regions=[region.name]
                    ),
                )
                for m in region.stations
            ]
        )
        written.append(
            str(
                products.write_stations_geojson(
                    store,
                    catalog,
                    _provenance("catalog", config=str(cfg.analysis_yaml)),
                )
            )
        )
    velocity_extra: dict[str, Any] = {
        "model": cfg.detrend_model,
        "window_years": cfg.velocity_window_years,
    }
    if velocity_method == "mle" and cfg.velocity_kappa_bounds is not None:
        velocity_extra["kappa_bounds"] = list(cfg.velocity_kappa_bounds)
    written.append(
        str(
            products.write_velocities_geojson(
                store,
                region.name,
                VelocityCollection(features=velocity_features),
                _provenance(velocity_method, **velocity_extra),
            )
        )
    )
    if breaks_on:
        gbis_extra: dict[str, Any] = {
            "n_breaks": cfg.breakpoints.n_breaks,
            "n_runs": runs,
            "t_runs": truns,
            "seed": seed,
        }
        if outliers_on:
            # BGÓ Q8: the chains consumed the CLEANED (inlier) series —
            # record the outlier-params hash that shaped that cleaning.
            gbis_extra["outliers"] = {
                "cleaned_input": True,
                "params_hash": config_hash(cfg.outliers),
            }
        if triage_stats is not None:
            # Triage is never a silent cap: the screen/flag counts ride in
            # the product provenance (and were logged during the run).
            gbis_extra["triage"] = {
                "n_runs": triage_stats.n_runs,
                "t_runs": triage_stats.t_runs,
                "sigma": triage_stats.sigma,
                "stations_screened": triage_stats.stations_screened,
                "stations_flagged": list(triage_stats.stations_flagged),
            }
        written.append(
            str(
                products.write_breaks_json(
                    store,
                    region.name,
                    break_entries,
                    _provenance("gbis", **gbis_extra),
                )
            )
        )

    if deformation_on:
        # Deformation stage (Amendment A6 Mogi / A7 Okada slip, by
        # deformation.source). Stations that failed anywhere in the chain are
        # excluded; a stage failure is recorded — the region's
        # velocity/series/break products stay in the store.
        dcfg = cfg.deformation
        kept_ok_series = {m: s for m, s in kept_series.items() if m in ok}
        kept_ok_detrended = {m: d for m, d in kept_detrended.items() if m in ok}
        try:
            if dcfg.source == "okada":
                written.append(
                    _write_okada_slip(
                        store,
                        region.name,
                        kept_ok_series,
                        kept_ok_detrended,
                        meta,
                        dcfg,
                        fitted_at,
                        _provenance,
                    )
                )
            else:
                written.append(
                    _write_mogi_deformation(
                        store,
                        region.name,
                        kept_ok_series,
                        kept_ok_detrended,
                        meta,
                        dcfg,
                        fitted_at,
                        seed,
                        _provenance,
                    )
                )
        except Exception as exc:  # noqa: BLE001 — stage must not sink the region
            deformation_failed = f"{type(exc).__name__}: {exc}"
            print(
                f"[region {region.name}] deformation FAILED: {deformation_failed}",
                file=sys.stderr,
                flush=True,
            )

    if outliers_on and write_meta:
        # Operator-review deliverable (design §5.1 / BGÓ Q5): the protected
        # SuspectedEvent clusters as candidate steps.csv entries. Fleet
        # runs suppress this (write_meta=False) and write the aggregate.
        written.append(str(products.write_suspected_steps_csv(store, suspected_rows)))

    summary = RunSummary(
        region=region.name,
        store=store,
        fitted_at=fitted_at,
        source=source,
        stations_ok=tuple(ok),
        stations_failed=failed,
        products=tuple(written),
        api_max_points=cfg.api_max_points,
        deformation_failed=deformation_failed,
        outliers_enabled=outliers_on,
        outliers_aborted=tuple(outliers_aborted),
        outliers_failed=outliers_failed,
        suspected_steps=tuple(suspected_rows),
    )
    if write_meta:
        products.write_run_meta(store, summary.as_dict())
    return summary


def run_fleet(
    cfg: AnalysisConfig,
    series_loader: SeriesLoader,
    store: Path,
    source: str,
    *,
    detect_breaks: bool | None = None,
    compute_deformation: bool | None = None,
    compute_outliers: bool | None = None,
    n_runs: int | None = None,
    t_runs: int | None = None,
    triage_n_runs: int | None = None,
    triage_t_runs: int | None = None,
    max_workers: int | None = None,
    seed: int | None = 0,
) -> FleetSummary:
    """Run the precompute for **every** configured region into one store.

    Reuses :func:`run_precompute` per region (no math is duplicated) with
    the per-region catalog/meta writes suppressed, then aggregates:

    - ``stations.geojson`` — one combined catalog spanning all successful
      regions; a station configured in several regions carries them all in
      ``properties.regions`` (sorted). Stations of a *failed* region are
      not cataloged (their products were not produced this run).
    - ``velocities/<region>.geojson`` / ``models/<region>_breaks.json`` /
      ``models/<region>_deformation.json`` (Mogi) or ``<region>_slip.json``
      (Okada) — written per region by :func:`run_precompute`; break detection
      stays gated by ``breakpoints.enabled_regions`` and the deformation stage
      by ``deformation.enabled_regions`` (both ``None`` overrides).
    - ``meta/run.json`` — a single :class:`FleetSummary` with per-region
      and per-station success/failure counts.

    Fault tolerance mirrors the station rule one level up: a region that
    raises (bad config entry, missing station metadata, every station
    failing) is recorded in ``regions_failed`` and skipped — one bad
    region must not sink the fleet. Only when *every* region fails does
    the run raise.

    Args:
        cfg: Loaded analysis-lane configuration (all its ``regions`` run).
        series_loader: ``marker -> StationSeries`` shared by all regions.
        store: Store root directory (created as needed).
        source: Provenance tag describing the data source for the run.
        detect_breaks / compute_deformation / compute_outliers / n_runs /
            t_runs / triage_n_runs / triage_t_runs / max_workers / seed:
            Passed through to :func:`run_precompute` (same semantics).

    Raises:
        RuntimeError: When every configured region failed.
    """
    fitted_at = datetime.datetime.now(datetime.UTC)
    summaries: dict[str, RunSummary] = {}
    regions_failed: dict[str, str] = {}
    written: list[str] = []

    for name in cfg.regions:
        print(f"=== region {name} ===", flush=True)
        try:
            summary = run_precompute(
                cfg,
                name,
                series_loader,
                store,
                source,
                detect_breaks=detect_breaks,
                compute_deformation=compute_deformation,
                compute_outliers=compute_outliers,
                n_runs=n_runs,
                t_runs=t_runs,
                triage_n_runs=triage_n_runs,
                triage_t_runs=triage_t_runs,
                max_workers=max_workers,
                seed=seed,
                write_catalog=False,
                write_meta=False,
            )
        except Exception as exc:  # noqa: BLE001 — fleet survives one bad region
            regions_failed[name] = f"{type(exc).__name__}: {exc}"
            print(
                f"[region {name}] FAILED — {regions_failed[name]}",
                file=sys.stderr,
                flush=True,
            )
            continue
        summaries[name] = summary
        written.extend(summary.products)

    if not summaries:
        message = "fleet precompute produced nothing — every region failed"
        raise RuntimeError(f"{message}: {regions_failed}")

    # Combined catalog: every station of every successful region, with the
    # full (sorted) list of regions that contain it.
    memberships: dict[str, list[str]] = {}
    for name in summaries:
        for marker in cfg.region(name).stations:
            memberships.setdefault(marker, []).append(name)
    meta = load_station_meta(sorted(memberships))
    frames = sorted({cfg.region(name).reference_frame for name in summaries})
    catalog = StationCollection(
        features=[
            StationFeature(
                geometry=_point(meta[marker]),
                properties=StationProperties(
                    marker=marker,
                    name=meta[marker].name,
                    regions=sorted(region_names),
                ),
            )
            for marker, region_names in sorted(memberships.items())
        ]
    )
    written.append(
        str(
            products.write_stations_geojson(
                store,
                catalog,
                Provenance(
                    method="catalog",
                    frame=",".join(frames),
                    fitted_at=fitted_at,
                    source=source,
                    extra={
                        "config": str(cfg.analysis_yaml),
                        "regions": sorted(summaries),
                    },
                ),
            )
        )
    )

    if any(summary.outliers_enabled for summary in summaries.values()):
        # One aggregated operator-review file across all regions (each row
        # carries its region; region runs wrote nothing — write_meta=False).
        fleet_rows = [
            row for summary in summaries.values() for row in summary.suspected_steps
        ]
        written.append(str(products.write_suspected_steps_csv(store, fleet_rows)))

    fleet = FleetSummary(
        store=store,
        fitted_at=fitted_at,
        source=source,
        regions=summaries,
        regions_failed=regions_failed,
        products=tuple(written),
        api_max_points=cfg.api_max_points,
    )
    products.write_run_meta(store, fleet.as_dict())
    return fleet


def _neu_loader(neu_dir: Path) -> SeriesLoader:
    """Loader for ``<MARKER>.NEU`` files under ``neu_dir``."""

    def load(marker: str) -> StationSeries:
        for candidate in (neu_dir / f"{marker}.NEU", neu_dir / f"{marker}.neu"):
            if candidate.is_file():
                return load_neu(candidate, marker=marker)
        raise FileNotFoundError(f"no {marker}.NEU under {neu_dir}")

    return load


def _synthetic_loader(seed: int, n_days: int) -> SeriesLoader:
    """Loader generating the deterministic synthetic fixture per marker."""

    def load(marker: str) -> StationSeries:
        return synthetic_station(marker, seed=seed, n_days=n_days)

    return load


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gps-api-precompute",
        description=(
            "Scheduled precompute for the GNSS analysis lane: runs the "
            "gps_analysis chain (trajectory fit + detrend, WLS or "
            "colored-noise MLE velocity, GBIS4TS break points and Mogi/Okada "
            "deformation sources for gated regions) for one configured "
            "region — or, with --fleet, every configured region — and "
            "writes Parquet/GeoJSON/JSON products to the store gps_api "
            "serves."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="analysis.yaml path (default: <gpsconfig dir>/analysis.yaml "
        "via gps_parser / $GPS_CONFIG_PATH)",
    )
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        "--region",
        help="region to precompute (default: the first configured region)",
    )
    scope.add_argument(
        "--fleet",
        "--all-regions",
        action="store_true",
        help="run every configured region into one coherent store "
        "(combined station catalog, per-region products, one fleet "
        "meta/run.json); break detection stays gated by "
        "breakpoints.enabled_regions",
    )
    parser.add_argument(
        "--store",
        type=Path,
        help="store root (default: analysis.yaml store.path, else "
        "$GPS_API_STORE / ~/.cache/gps_analysis)",
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--neu-dir",
        type=Path,
        help=".NEU directory (default: analysis.yaml data.neu_dir)",
    )
    src.add_argument(
        "--synthetic",
        action="store_true",
        help="run on the built-in synthetic fixture (no station data needed)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="synthetic series length in daily epochs (default 365)",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument(
        "--runs",
        type=int,
        help="override breakpoints.n_runs (kept MCMC iterations; "
        "production config uses 1e6 — use small values for dev runs)",
    )
    parser.add_argument(
        "--t-runs",
        type=int,
        help="override breakpoints.t_runs (annealing iterations per "
        "temperature; 16*t_runs must stay below the run count)",
    )
    parser.add_argument(
        "--triage-runs",
        type=int,
        help="override breakpoints.triage_n_runs (short screening chains "
        "of the triage->confirm stage; 0 disables triage so every gated "
        "station gets the full confirm chain)",
    )
    parser.add_argument(
        "--triage-t-runs",
        type=int,
        help="override breakpoints.triage_t_runs (annealing iterations of "
        "the screening chains; 16*triage_t_runs must stay below "
        "triage_n_runs)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="override breakpoints.max_workers (break-detection process "
        "pool size; default: one worker per CPU core, capped at the "
        "number of station x component chains; 0 runs the chains inline)",
    )
    parser.add_argument(
        "--no-breakpoints",
        action="store_true",
        help="skip GBIS4TS break detection (velocities + series only)",
    )
    parser.add_argument(
        "--no-deformation",
        action="store_true",
        help="skip the deformation stage — Mogi ΔV(t) or Okada slip, by "
        "deformation.source (otherwise gated by deformation.enabled_regions "
        "in analysis.yaml)",
    )
    parser.add_argument(
        "--no-outliers",
        action="store_true",
        help="skip outlier detection (otherwise gated by outliers.enabled "
        "in analysis.yaml): no flag columns, no suspected_steps.csv, and "
        "downstream estimates fit on the unmasked series",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point (``gps-api-precompute``) — foreground only."""
    args = _build_parser().parse_args(argv)
    cfg = load_analysis_config(args.config)
    store = args.store or cfg.store_path or settings.store_path()

    loader: SeriesLoader
    if args.synthetic:
        source = f"synthetic:seed={args.seed},n_days={args.days}"
        loader = _synthetic_loader(args.seed, args.days)
    else:
        neu_dir = args.neu_dir or cfg.neu_dir
        if neu_dir is None:
            print(
                "no data source: pass --neu-dir/--synthetic or set "
                f"data.neu_dir in {cfg.analysis_yaml}",
                file=sys.stderr,
            )
            return 2
        source = f"neu:{neu_dir}"
        loader = _neu_loader(neu_dir)

    if args.fleet:
        fleet = run_fleet(
            cfg,
            loader,
            store,
            source,
            detect_breaks=False if args.no_breakpoints else None,
            compute_deformation=False if args.no_deformation else None,
            compute_outliers=False if args.no_outliers else None,
            n_runs=args.runs,
            t_runs=args.t_runs,
            triage_n_runs=args.triage_runs,
            triage_t_runs=args.triage_t_runs,
            max_workers=args.workers,
            seed=args.seed,
        )
        print(
            f"fleet: {len(fleet.regions)} region(s) ok, "
            f"{len(fleet.regions_failed)} failed; "
            f"{fleet.stations_ok_total} station(s) ok, "
            f"{fleet.stations_failed_total} failed; "
            f"{fleet.deformation_failed_total} deformation stage(s) failed; "
            f"{fleet.outliers_failed_total} outlier detection(s) failed; "
            f"{len(fleet.products)} product file(s) under {fleet.store}"
        )
        for name, reason in fleet.regions_failed.items():
            print(f"  region {name} FAILED — {reason}", file=sys.stderr)
        clean = (
            not fleet.regions_failed
            and fleet.stations_failed_total == 0
            and fleet.deformation_failed_total == 0
            and fleet.outliers_failed_total == 0
        )
        return 0 if clean else 1

    region_name = args.region or next(iter(cfg.regions))
    summary = run_precompute(
        cfg,
        region_name,
        loader,
        store,
        source,
        detect_breaks=False if args.no_breakpoints else None,
        compute_deformation=False if args.no_deformation else None,
        compute_outliers=False if args.no_outliers else None,
        n_runs=args.runs,
        t_runs=args.t_runs,
        triage_n_runs=args.triage_runs,
        triage_t_runs=args.triage_t_runs,
        max_workers=args.workers,
        seed=args.seed,
    )
    print(
        f"region {summary.region}: {len(summary.stations_ok)} station(s) ok, "
        f"{len(summary.stations_failed)} failed; "
        f"{len(summary.products)} product file(s) under {summary.store}"
    )
    for path in summary.products:
        print(f"  {path}")
    clean = (
        not summary.stations_failed
        and summary.deformation_failed is None
        and not summary.outliers_failed
    )
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
