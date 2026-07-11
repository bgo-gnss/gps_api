"""Precompute orchestration + the ``gps-api-precompute`` console script.

Per region (Phase-1 slice: one region per run), for every configured
station, the job calls the ``gps_analysis`` public API — no math is derived
here (MATH_STANDARDS rule):

1. :func:`gps_analysis.fit_components` — ``lineperiodic`` (or configured)
   trajectory fit per component, and :func:`gps_analysis.remove_trend` —
   the detrended series.
2. :func:`gps_analysis.estimate_velocity` — fixed-window WLS secular
   velocity with formal σ (``method="wls"``; the GBIS honest-σ upgrade is a
   later slice, PLAN-analysis-lane §1).
3. :func:`gps_analysis.detect_breakpoints` — GBIS4TS velocity break points
   + colored-noise parameters (``method="gbis"``), per component. The
   displacement series is passed straight in — the estimator
   zero-references internally (input-contract decision, PLAN-analysis-lane
   §7), so the job must NOT pre-reference it.

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
    InversionResult,
    detect_breakpoints,
    estimate_velocity,
    fit_components,
    linear,
    lineperiodic,
    remove_trend,
)

from gps_api import settings
from gps_api.precompute import products
from gps_api.precompute.config import (
    AnalysisConfig,
    StationMeta,
    load_analysis_config,
    load_station_meta,
)
from gps_api.precompute.products import Provenance
from gps_api.precompute.sources import (
    COMPONENTS,
    FloatArray,
    StationSeries,
    load_neu,
    synthetic_station,
    yearf_to_datetime,
)
from gps_api.schemas import (
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
    """What one precompute run produced (also written to ``meta/run.json``)."""

    region: str
    store: Path
    fitted_at: datetime.datetime
    source: str
    stations_ok: tuple[str, ...]
    stations_failed: dict[str, str]
    products: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready mapping for ``meta/run.json``."""
        return {
            "region": self.region,
            "store": str(self.store),
            "fitted_at": self.fitted_at.isoformat().replace("+00:00", "Z"),
            "source": self.source,
            "stations_ok": list(self.stations_ok),
            "stations_failed": dict(self.stations_failed),
            "products": list(self.products),
        }


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
) -> VelocityFeature:
    """WLS velocity vector for one station as a contract GeoJSON feature."""
    t_last = float(series.t[-1])
    estimate = estimate_velocity(
        series.t,
        series.y,
        series.sigma,
        model=model_name,
        window=(t_last - cfg.velocity_window_years, None),
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
            method="wls",
            window_start=yearf_to_datetime(estimate.span[0]),
            window_end=yearf_to_datetime(estimate.span[1]),
        ),
    )


def _break_parameters(result: InversionResult) -> dict[str, float]:
    """Flatten ``InversionResult.optimal`` (MATLAB order) to named floats.

    BPD1: ``[a, v, g, t_b, κ, β]``; BPD2: ``[a, v, g1, t_b1, g2, t_b2, κ, β]``
    — intercept mm, secular rate mm/yr, rate change(s) mm/yr, break
    epoch(s) yr, colored-noise spectral index κ and amplitude β.
    """
    opt = [float(v) for v in result.optimal]
    parameters = {
        "intercept_mm": opt[0],
        "trend_mm_yr": opt[1],
        "trend_change_mm_yr": opt[2],
        "breakpoint_yearf": opt[3],
    }
    if result.model == "BPD2":
        parameters["trend_change2_mm_yr"] = opt[4]
        parameters["breakpoint2_yearf"] = opt[5]
    parameters["kappa"] = opt[-2]
    parameters["amp_mm"] = opt[-1]
    return parameters


def _station_breaks(
    series: StationSeries,
    detrended: FloatArray,
    fitted_at: datetime.datetime,
    *,
    n_breaks: int,
    n_runs: int,
    t_runs: int,
    seed: int | None,
) -> list[dict[str, Any]]:
    """GBIS4TS break detection per component → break-catalog entries.

    The displacement series goes straight to
    :func:`~gps_analysis.detect_breakpoints` (it zero-references
    internally — do not pre-reference here). The fixed white-noise
    amplitude follows the dev-viz heuristic: median observation σ, or the
    residual std when the source carries no σ.
    """
    entries: list[dict[str, Any]] = []
    for i, component in enumerate(COMPONENTS):
        if series.sigma is not None:
            wn_amp = float(np.median(series.sigma[i]))
        else:
            wn_amp = float(np.std(detrended[i]))
        result = detect_breakpoints(
            series.t,
            series.y[i],
            wn_amp,
            n_breaks=n_breaks,
            n_runs=n_runs,
            t_runs=t_runs,
            seed=seed,
        )
        parameters = _break_parameters(result)
        entries.append(
            {
                "marker": series.marker,
                "component": component,
                "model": result.model,
                "method": "gbis",
                "fitted_at": fitted_at.isoformat().replace("+00:00", "Z"),
                "breakpoint_time": yearf_to_datetime(parameters["breakpoint_yearf"])
                .isoformat()
                .replace("+00:00", "Z"),
                "parameters": parameters,
                "wn_amp_mm": wn_amp,
                "y_ref_mm": float(result.y_ref),
                "n_runs": n_runs,
            }
        )
    return entries


def run_precompute(
    cfg: AnalysisConfig,
    region_name: str,
    series_loader: SeriesLoader,
    store: Path,
    source: str,
    *,
    detect_breaks: bool | None = None,
    n_runs: int | None = None,
    t_runs: int | None = None,
    seed: int | None = 0,
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
        n_runs / t_runs: Override the configured GBIS4TS chain lengths
            (dev runs; production uses the configured 1e6).
        seed: MCMC RNG seed (reproducibility).

    Per-station failures are recorded and skipped — one bad station must
    not sink the batch (fault-tolerance rule of the ops packages).
    """
    if cfg.velocity_method != "wls":
        raise ValueError(
            f"velocity.default_method={cfg.velocity_method!r} — only 'wls' is "
            "implemented in this slice ('gbis' velocities are a later slice)"
        )
    region = cfg.region(region_name)
    meta = load_station_meta(region.stations)
    fitted_at = datetime.datetime.now(datetime.UTC)
    breaks_on = (
        cfg.breakpoints.enabled_for(region.name)
        if detect_breaks is None
        else detect_breaks
    )
    runs = cfg.breakpoints.n_runs if n_runs is None else n_runs
    truns = cfg.breakpoints.t_runs if t_runs is None else t_runs
    if breaks_on and 16 * truns >= runs:
        raise ValueError(
            f"breakpoints: n_runs ({runs}) must exceed the annealing span "
            f"16*t_runs ({16 * truns}) — adjust analysis.yaml or --runs/--t-runs"
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
    written: list[str] = []
    ok: list[str] = []
    failed: dict[str, str] = {}

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
            written.append(
                str(
                    products.write_series_parquet(
                        store,
                        series,
                        detrended,
                        times,
                        _provenance(model_name, units="mm", marker=marker),
                    )
                )
            )
            velocity_features.append(
                _velocity_feature(series, meta[marker], cfg, model_name)
            )
            print(f"[{marker}] trajectory fit + WLS velocity done", flush=True)
            if breaks_on:
                break_entries.extend(
                    _station_breaks(
                        series,
                        detrended,
                        fitted_at,
                        n_breaks=cfg.breakpoints.n_breaks,
                        n_runs=runs,
                        t_runs=truns,
                        seed=seed,
                    )
                )
                print(f"[{marker}] GBIS4TS break detection done", flush=True)
            ok.append(marker)
        except Exception as exc:  # noqa: BLE001 — batch survives one bad station
            failed[marker] = f"{type(exc).__name__}: {exc}"
            print(f"[{marker}] FAILED — {failed[marker]}", file=sys.stderr, flush=True)

    if not ok:
        raise RuntimeError(
            f"precompute produced nothing — every station failed: {failed}"
        )

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
                store, catalog, _provenance("catalog", config=str(cfg.analysis_yaml))
            )
        )
    )
    written.append(
        str(
            products.write_velocities_geojson(
                store,
                region.name,
                VelocityCollection(features=velocity_features),
                _provenance(
                    "wls",
                    model=cfg.detrend_model,
                    window_years=cfg.velocity_window_years,
                ),
            )
        )
    )
    if breaks_on:
        written.append(
            str(
                products.write_breaks_json(
                    store,
                    region.name,
                    break_entries,
                    _provenance(
                        "gbis",
                        n_breaks=cfg.breakpoints.n_breaks,
                        n_runs=runs,
                        t_runs=truns,
                        seed=seed,
                    ),
                )
            )
        )

    summary = RunSummary(
        region=region.name,
        store=store,
        fitted_at=fitted_at,
        source=source,
        stations_ok=tuple(ok),
        stations_failed=failed,
        products=tuple(written),
    )
    products.write_run_meta(store, summary.as_dict())
    return summary


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
            "gps_analysis chain (trajectory fit + detrend, WLS velocity, "
            "GBIS4TS break points) for one configured region and writes "
            "Parquet/GeoJSON products to the store gps_api serves."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="analysis.yaml path (default: <gpsconfig dir>/analysis.yaml "
        "via gps_parser / $GPS_CONFIG_PATH)",
    )
    parser.add_argument(
        "--region",
        help="region to precompute (default: the first configured region)",
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
        "--no-breakpoints",
        action="store_true",
        help="skip GBIS4TS break detection (velocities + series only)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point (``gps-api-precompute``) — foreground only."""
    args = _build_parser().parse_args(argv)
    cfg = load_analysis_config(args.config)
    region_name = args.region or next(iter(cfg.regions))
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

    summary = run_precompute(
        cfg,
        region_name,
        loader,
        store,
        source,
        detect_breaks=False if args.no_breakpoints else None,
        n_runs=args.runs,
        t_runs=args.t_runs,
        seed=args.seed,
    )
    print(
        f"region {summary.region}: {len(summary.stations_ok)} station(s) ok, "
        f"{len(summary.stations_failed)} failed; "
        f"{len(summary.products)} product file(s) under {summary.store}"
    )
    for path in summary.products:
        print(f"  {path}")
    return 0 if not summary.stations_failed else 1


if __name__ == "__main__":
    sys.exit(main())
