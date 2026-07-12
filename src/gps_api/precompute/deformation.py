"""Mogi deformation-source stage of the precompute job (Amendment A6).

Fits one Mogi point source per grid epoch to a region's GNSS displacement
field — all estimation through the ``gps_analysis`` public API
(:func:`gps_analysis.mogi_invert`, :func:`gps_analysis.mogi_invert_bayes`,
:func:`gps_analysis.local_coordinates`); no math is derived here
(MATH_STANDARDS rule). The product is the ΔV(t)/depth/position time series
served by ``GET /v1/deformation/{region}``.

Pipeline per gated region (config: ``deformation:`` in ``analysis.yaml``,
:class:`gps_api.precompute.config.DeformationConfig`):

1. **Reference + epoch grid.** A trailing ``window_years`` window ends at
   the newest station epoch. Per station, the displacement at each grid
   epoch (spaced ``step_days``) is the mean of the samples within
   ``epoch_mean_days`` around it, minus the same mean over the reference
   window at the window start — so ΔV(reference) = 0 by construction.
2. **Per-epoch inversion.** Stations with data at the epoch (≥
   ``min_stations``) enter a weighted NLLS Mogi inversion in a local
   tangent-plane frame (metres; origin from config or the participating-
   station centroid). Depth/ΔV bounds come from config; horizontal bounds
   follow the ``mogi_invert`` footprint default. Each fit warm-starts from
   the previous epoch's source.
3. **Optional Bayesian tail.** When ``bayes.n_runs > 0``, the newest
   fitted epoch also gets a GBIS posterior (``mogi_invert_bayes``) whose
   percentile summary rides on the product — honest uncertainties next to
   the formal per-epoch σ.

Fault tolerance: an epoch whose inversion fails to converge is skipped and
counted (never silently); the stage raises only when *no* epoch could be
fitted — and the caller (:mod:`gps_api.precompute.job`) records that as a
region-level note without sinking the region's other products.

**Cross-check, not dependency:** this is an independent GNSS-only analog
of Vincent's operational Mogi ΔV(t) (``insar.vedur.is:/mnt/scratch/
vincent/model/svartsengi/inflation*/inv_volume_mogi.dat``). The two are
meant to be compared against each other; this stage never reads his files.
"""

from __future__ import annotations

import dataclasses
import datetime
import sys

import numpy as np
from gps_analysis import (
    InversionConfig,
    MogiSource,
    PriorBounds,
    local_coordinates,
    mogi_invert,
    mogi_invert_bayes,
)

from gps_api.precompute.config import DeformationConfig, StationMeta
from gps_api.precompute.sources import (
    COMPONENTS,
    FloatArray,
    StationSeries,
    yearf_to_datetime,
)
from gps_api.schemas import (
    DeformationResult,
    MogiPosteriorSummary,
    MogiSourceEstimate,
)

#: mm → m (station displacements arrive in mm; Mogi runs in metres/m³).
_MM_TO_M = 1.0e-3

#: Days per Julian year (grid arithmetic only — no estimation).
_DAYS_PER_YEAR = 365.25

#: Posterior percentile summary levels (keys of the served product).
_PERCENTILES: tuple[tuple[str, float], ...] = (
    ("p2_5", 2.5),
    ("p16", 16.0),
    ("p50", 50.0),
    ("p84", 84.0),
    ("p97_5", 97.5),
)

#: Served parameter keys in ``[x, y, depth, dv]`` inversion order.
_PARAM_KEYS: tuple[str, ...] = ("east_m", "north_m", "depth_km", "dv_m3")

#: Initial MCMC step as a fraction of each uniform-prior range (the GBIS
#: sampler adapts steps continuously — this only seeds the adaptation).
_BAYES_STEP_FRACTION = 0.05


@dataclasses.dataclass(frozen=True)
class MogiSeriesOutcome:
    """What the deformation stage produced for one region.

    Attributes:
        result: The contract-shaped product (``provenance`` unset — the
            writer stamps it).
        epochs_skipped: Grid epochs dropped (too few stations, or a
            non-converged inversion) — surfaced in provenance, never a
            silent cap.
        stations_excluded: Markers that carried no usable reference-window
            samples and were left out of every epoch.
    """

    result: DeformationResult
    epochs_skipped: int
    stations_excluded: tuple[str, ...]


def _epoch_mean(
    series: StationSeries,
    detrended: FloatArray,
    use_detrended: bool,
    center: float,
    half_width: float,
) -> tuple[FloatArray, FloatArray] | None:
    """Mean displacement + 1-σ of one station around one grid epoch.

    Returns ``(mean (3,), sigma (3,))`` in mm over the samples with
    ``|t − center| ≤ half_width`` [yr], rows in :data:`COMPONENTS` order,
    or ``None`` when the station has no samples there. The σ of the mean
    is the mean observation σ over √n; a source without observation σ
    falls back to the detrended-residual std (the break-stage heuristic).
    """
    mask = np.abs(series.t - center) <= half_width
    n = int(np.count_nonzero(mask))
    if n == 0:
        return None
    values = detrended if use_detrended else series.y
    mean = values[:, mask].mean(axis=1)
    if series.sigma is not None:
        sigma = series.sigma[:, mask].mean(axis=1) / np.sqrt(n)
    else:
        sigma = detrended.std(axis=1) / np.sqrt(n)
    return mean.astype(np.float64), np.maximum(sigma.astype(np.float64), 1e-12)


def _local_scale(lon0: float, lat0: float) -> tuple[float, float]:
    """Metres per degree of longitude/latitude at the frame origin.

    The tangent-plane mapping of :func:`gps_analysis.local_coordinates` is
    exactly linear in (λ − λ₀) and (φ − φ₀), so these two factors invert
    it exactly — used to report the fitted source position back as
    lon/lat without deriving any projection math here.
    """
    e_per_deg, _ = local_coordinates(lon0 + 1.0, lat0, lon0, lat0)
    _, n_per_deg = local_coordinates(lon0, lat0 + 1.0, lon0, lat0)
    return float(e_per_deg), float(n_per_deg)


def _mogi_bounds(
    e: FloatArray, n: FloatArray, cfg: DeformationConfig
) -> tuple[FloatArray, FloatArray]:
    """LSQ/prior bounds over ``[x, y, depth, dv]`` (metres / m³).

    Horizontal bounds follow the ``mogi_invert`` footprint default
    (network extent ± one span); depth comes from
    ``deformation.depth_bounds_km``; ΔV from ``deformation.dv_bounds_m3``
    (±∞ when unset — LSQ only; the Bayesian stage requires finite bounds,
    enforced by the config).
    """
    span = max(float(np.ptp(e)), float(np.ptp(n)), 1.0)
    depth_lo, depth_hi = (v * 1000.0 for v in cfg.depth_bounds_km)
    dv_lo, dv_hi = (
        cfg.dv_bounds_m3
        if cfg.dv_bounds_m3 is not None
        else (
            -np.inf,
            np.inf,
        )
    )
    lower = np.array(
        [e.min() - span, n.min() - span, depth_lo, dv_lo], dtype=np.float64
    )
    upper = np.array(
        [e.max() + span, n.max() + span, depth_hi, dv_hi], dtype=np.float64
    )
    return lower, upper


def compute_mogi_series(
    region_name: str,
    station_series: dict[str, StationSeries],
    detrended: dict[str, FloatArray],
    meta: dict[str, StationMeta],
    cfg: DeformationConfig,
    fitted_at: datetime.datetime,
    *,
    seed: int | None = 0,
) -> MogiSeriesOutcome:
    """Fit the Mogi ΔV(t) source time series for one region.

    Args:
        region_name: Region key (product tag only).
        station_series: ``marker -> StationSeries`` of the stations that
            survived the per-station chain (mm, fractional-year epochs).
        detrended: ``marker -> (3, N)`` trajectory-model residuals [mm],
            aligned with each station's series.
        meta: ``marker -> StationMeta`` coordinates from ``stations.cfg``.
        cfg: The validated ``deformation:`` block.
        fitted_at: Run timestamp stamped on the product (UTC).
        seed: RNG seed for the optional Bayesian stage.

    Raises:
        RuntimeError: When no station carries reference-window data, or
            no grid epoch could be fitted — the caller records this as a
            region-level deformation failure (other products survive).
    """
    if not station_series:
        raise RuntimeError("deformation: no station series available")
    use_detrended = cfg.series == "detrended"
    half_width = 0.5 * cfg.epoch_mean_days / _DAYS_PER_YEAR
    step = cfg.step_days / _DAYS_PER_YEAR

    t_end = max(float(s.t[-1]) for s in station_series.values())
    t_ref = t_end - cfg.window_years

    # Per-station reference (mean over the window-start epoch); stations
    # without reference coverage are excluded from every epoch.
    references: dict[str, tuple[FloatArray, FloatArray]] = {}
    excluded: list[str] = []
    for marker, series in station_series.items():
        ref = _epoch_mean(series, detrended[marker], use_detrended, t_ref, half_width)
        if ref is None:
            excluded.append(marker)
        else:
            references[marker] = ref
    if len(references) < cfg.min_stations:
        raise RuntimeError(
            f"deformation: only {len(references)} station(s) carry data in "
            f"the reference window around {t_ref:.3f} — need "
            f">= {cfg.min_stations}"
        )

    # Local tangent-plane frame: config origin or participant centroid.
    markers = sorted(references)
    if cfg.origin_lon is not None and cfg.origin_lat is not None:
        lon0, lat0 = cfg.origin_lon, cfg.origin_lat
    else:
        lon0 = float(np.mean([meta[m].lon for m in markers]))
        lat0 = float(np.mean([meta[m].lat for m in markers]))
    east_by_marker: dict[str, float] = {}
    north_by_marker: dict[str, float] = {}
    for marker in markers:
        e_m, n_m = local_coordinates(meta[marker].lon, meta[marker].lat, lon0, lat0)
        east_by_marker[marker] = float(e_m)
        north_by_marker[marker] = float(n_m)
    e_per_deg, n_per_deg = _local_scale(lon0, lat0)

    i_n, i_e, i_u = (COMPONENTS.index(c) for c in ("north", "east", "up"))
    centers = np.arange(t_ref + step, t_end + 0.5 * step, step, dtype=np.float64)

    fits: list[MogiSourceEstimate] = []
    epochs_skipped = 0
    last_fit_inputs: (
        tuple[FloatArray, FloatArray, FloatArray, FloatArray, MogiSource] | None
    ) = None
    warm_start: MogiSource | None = None
    for center in centers:
        e_list: list[float] = []
        n_list: list[float] = []
        obs_cols: list[FloatArray] = []
        sig_cols: list[FloatArray] = []
        for marker in markers:
            epoch = _epoch_mean(
                station_series[marker],
                detrended[marker],
                use_detrended,
                float(center),
                half_width,
            )
            if epoch is None:
                continue
            ref_mean, ref_sigma = references[marker]
            mean, sigma = epoch
            disp = (mean - ref_mean) * _MM_TO_M
            # Displacement σ: epoch-mean and reference-mean uncertainties
            # combined in quadrature (independent windows), in metres.
            sig = np.hypot(sigma, ref_sigma) * _MM_TO_M
            # StationSeries rows are (north, east, up); mogi_invert wants
            # (east, north, up) — reorder here, once.
            obs_cols.append(np.array([disp[i_e], disp[i_n], disp[i_u]]))
            sig_cols.append(np.array([sig[i_e], sig[i_n], sig[i_u]]))
            e_list.append(east_by_marker[marker])
            n_list.append(north_by_marker[marker])
        if len(e_list) < cfg.min_stations:
            epochs_skipped += 1
            continue
        e_arr = np.asarray(e_list, dtype=np.float64)
        n_arr = np.asarray(n_list, dtype=np.float64)
        obs = np.column_stack(obs_cols)
        sig = np.column_stack(sig_cols)
        bounds = _mogi_bounds(e_arr, n_arr, cfg)
        try:
            fit = mogi_invert(
                e_arr,
                n_arr,
                obs,
                sig,
                x0=warm_start,
                bounds=bounds,
                nu=cfg.nu,
            )
        except (RuntimeError, ValueError) as exc:
            epochs_skipped += 1
            print(
                f"[{region_name}] deformation epoch {float(center):.4f} "
                f"skipped — {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            continue
        warm_start = fit.source
        last_fit_inputs = (e_arr, n_arr, obs, sig, fit.source)
        fits.append(
            MogiSourceEstimate(
                time=yearf_to_datetime(float(center)),
                lon=lon0 + fit.source.x / e_per_deg,
                lat=lat0 + fit.source.y / n_per_deg,
                east_m=fit.source.x,
                north_m=fit.source.y,
                depth_km=fit.source.depth / 1000.0,
                dv_m3=fit.source.dv,
                sigma_east_m=float(fit.sigma[0]),
                sigma_north_m=float(fit.sigma[1]),
                sigma_depth_km=float(fit.sigma[2]) / 1000.0,
                sigma_dv_m3=float(fit.sigma[3]),
                chi2_reduced=fit.chi2_reduced,
                rms_mm=fit.rms / _MM_TO_M,
                n_stations=len(e_list),
            )
        )
    if not fits:
        raise RuntimeError(
            f"deformation: no grid epoch could be fitted for {region_name!r} "
            f"({epochs_skipped} skipped of {centers.size})"
        )

    posterior: MogiPosteriorSummary | None = None
    if cfg.bayes_n_runs > 0 and last_fit_inputs is not None:
        posterior = _bayes_summary(fits[-1], last_fit_inputs, cfg, seed)

    result = DeformationResult(
        region=region_name,
        source_type="mogi",
        reference_time=yearf_to_datetime(t_ref),
        origin_lon=lon0,
        origin_lat=lat0,
        series_kind="detrended" if use_detrended else "raw",
        stations=markers,
        fits=fits,
        posterior=posterior,
        fitted_at=fitted_at,
    )
    return MogiSeriesOutcome(
        result=result,
        epochs_skipped=epochs_skipped,
        stations_excluded=tuple(excluded),
    )


def _bayes_summary(
    last_fit: MogiSourceEstimate,
    inputs: tuple[FloatArray, FloatArray, FloatArray, FloatArray, MogiSource],
    cfg: DeformationConfig,
    seed: int | None,
) -> MogiPosteriorSummary:
    """GBIS posterior of the newest fitted epoch, as percentile summaries.

    Uniform priors are the LSQ bounds (finite ΔV bounds enforced by the
    config); the chain starts at the LSQ optimum. The annealed burn-in
    (``16·t_runs`` samples, the GBIS4TS convention) is discarded before
    the percentiles are taken.
    """
    e_arr, n_arr, obs, sig, source = inputs
    lower, upper = _mogi_bounds(e_arr, n_arr, cfg)
    start = np.clip(source.as_array(), lower, upper)
    bounds = PriorBounds(
        start=start,
        lower=lower,
        upper=upper,
        step=(upper - lower) * _BAYES_STEP_FRACTION,
    )
    config = InversionConfig(
        n_runs=cfg.bayes_n_runs, t_runs=cfg.bayes_t_runs, seed=seed
    )
    post = mogi_invert_bayes(e_arr, n_arr, obs, sig, bounds, config, nu=cfg.nu)
    burn_in = min(16 * cfg.bayes_t_runs, cfg.bayes_n_runs - 1)
    kept = post.m_keep[:, burn_in:]
    # Serve depth in km (contract depth convention); positions/ΔV as-is.
    scale = np.array([1.0, 1.0, 1e-3, 1.0], dtype=np.float64)
    optimal = post.optimal.as_array() * scale
    percentiles = {
        key: {
            p_key: float(np.percentile(kept[i], level) * scale[i])
            for p_key, level in _PERCENTILES
        }
        for i, key in enumerate(_PARAM_KEYS)
    }
    return MogiPosteriorSummary(
        time=last_fit.time,
        n_runs=cfg.bayes_n_runs,
        burn_in=burn_in,
        optimal=dict(zip(_PARAM_KEYS, (float(v) for v in optimal), strict=True)),
        percentiles=percentiles,
    )
