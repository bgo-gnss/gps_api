"""Okada distributed-slip stage of the precompute job (Amendment A7).

Inverts a **single-window** distributed-slip distribution on an
operator-supplied fixed fault/dike plane for a region configured
``source: okada`` — all estimation through the ``gps_analysis`` public API
(:func:`gps_analysis.discretize_fault`, :func:`gps_analysis.okada_greens`,
:func:`gps_analysis.patch_laplacian`, :func:`gps_analysis.okada_invert_slip`,
:func:`gps_analysis.slip_lcurve`); no estimator math is derived here
(MATH_STANDARDS rule). The product is the per-patch slip distribution served
by ``GET /v1/deformation/{region}`` as a
:class:`gps_api.schemas.SlipDistributionResult`.

Pipeline per gated region (config: ``deformation.source: okada`` +
``deformation.okada`` block, :class:`gps_api.precompute.config.DeformationConfig`
/ :class:`~gps_api.precompute.config.OkadaPlaneConfig`):

1. **Net displacement.** A trailing ``window_years`` window ends at the newest
   station epoch. Per station the net displacement is the mean over
   ``epoch_mean_days`` around the window **end** minus the same mean at the
   window **start** — the single displacement field of the intrusion (a dike
   is event-specific; there is no time series here, unlike the Mogi stage).
   The Mogi stage's :func:`~gps_api.precompute.deformation._epoch_mean` /
   ``_local_scale`` are reused so the two stages reference identically.
2. **Fixed-plane inversion.** The operator's plane (centroid = the config
   ``origin``; ``top_depth_km`` → centroid depth) is tiled ``n_strike ×
   n_dip`` (:func:`discretize_fault`) and slip is inverted on it
   (:func:`okada_invert_slip`, Laplacian-regularized ± non-negative). The
   regularization weight λ is the configured fixed value or the L-curve
   corner (:func:`slip_lcurve`) over the configured log-spaced scan.
3. **Formal uncertainties.** Per-patch 1-σ is the unconstrained
   linear-Gaussian formal covariance ``C_s = A⁻¹·GᵂᵀGᵂ·A⁻¹`` with
   ``A = GᵂᵀGᵂ + λ²·LᵀL`` propagated through the **public** Green's/Laplacian
   operators (the honest analog of the Mogi ``fit.sigma``; ``gps_analysis``
   exposes no per-patch σ). Under the non-negativity constraint this σ is not
   exact for patches pinned at the ``slip ≥ 0`` bound — the provenance says so.

Fault tolerance: degenerate solutions (non-finite slip, or an
all-zero/all-pinned solution carrying no resolvable signal) are rejected with
a ``RuntimeError``; the caller (:mod:`gps_api.precompute.job`) records that as
a region-level ``deformation_failed`` note without sinking the region's other
products — the robustness lesson of the Mogi stage.
"""

from __future__ import annotations

import dataclasses
import datetime

import numpy as np
from gps_analysis import (
    FaultPatches,
    OkadaSource,
    discretize_fault,
    local_coordinates,
    okada_greens,
    okada_invert_slip,
    patch_laplacian,
    slip_lcurve,
)

from gps_api.precompute.config import DeformationConfig, OkadaPlaneConfig, StationMeta
from gps_api.precompute.deformation import (
    _DAYS_PER_YEAR,
    _MM_TO_M,
    _epoch_mean,
    _local_scale,
)
from gps_api.precompute.sources import (
    COMPONENTS,
    FloatArray,
    StationSeries,
    yearf_to_datetime,
)
from gps_api.schemas import FaultPatch, SlipDistributionResult

#: km → m (plane geometry arrives in km; Okada runs in metres).
_KM_TO_M = 1.0e3


@dataclasses.dataclass(frozen=True)
class SlipOutcome:
    """What the Okada distributed-slip stage produced for one region.

    Attributes:
        result: The contract-shaped product (``provenance`` unset — the writer
            stamps it).
        stations_excluded: Markers that carried no net-displacement coverage
            (missing reference- or target-window samples) and were left out.
    """

    result: SlipDistributionResult
    stations_excluded: tuple[str, ...]


def _net_displacement(
    station_series: dict[str, StationSeries],
    detrended: dict[str, FloatArray],
    meta: dict[str, StationMeta],
    cfg: DeformationConfig,
    okada: OkadaPlaneConfig,
) -> tuple[list[str], FloatArray, FloatArray, FloatArray, FloatArray, list[str]]:
    """Net (window-end − window-start) displacement field in the plane frame.

    Returns ``(markers, e, n, obs, sig, excluded)``: participating markers,
    their local east/north [m] in the plane-centroid frame, the net
    displacement ``obs`` and its 1-σ ``sig`` as ``(3, N)`` arrays with rows
    **(east, north, up)** in metres (Okada order), and the excluded markers.
    """
    use_detrended = cfg.series == "detrended"
    half_width = 0.5 * cfg.epoch_mean_days / _DAYS_PER_YEAR
    t_end = max(float(s.t[-1]) for s in station_series.values())
    t_ref = t_end - cfg.window_years

    lon0, lat0 = okada.origin_lon, okada.origin_lat
    i_n, i_e, i_u = (COMPONENTS.index(c) for c in ("north", "east", "up"))

    markers: list[str] = []
    excluded: list[str] = []
    e_list: list[float] = []
    n_list: list[float] = []
    obs_cols: list[FloatArray] = []
    sig_cols: list[FloatArray] = []
    for marker in sorted(station_series):
        series = station_series[marker]
        ref = _epoch_mean(series, detrended[marker], use_detrended, t_ref, half_width)
        target = _epoch_mean(
            series, detrended[marker], use_detrended, t_end, half_width
        )
        if ref is None or target is None:
            excluded.append(marker)
            continue
        ref_mean, ref_sigma = ref
        tgt_mean, tgt_sigma = target
        disp = (tgt_mean - ref_mean) * _MM_TO_M
        # Net-displacement σ: end- and start-window means in quadrature.
        sig = np.hypot(tgt_sigma, ref_sigma) * _MM_TO_M
        e_m, n_m = local_coordinates(meta[marker].lon, meta[marker].lat, lon0, lat0)
        e_list.append(float(e_m))
        n_list.append(float(n_m))
        # StationSeries rows are (north, east, up); Okada wants (east, north, up).
        obs_cols.append(np.array([disp[i_e], disp[i_n], disp[i_u]]))
        sig_cols.append(np.array([sig[i_e], sig[i_n], sig[i_u]]))
        markers.append(marker)

    if len(markers) < cfg.min_stations:
        raise RuntimeError(
            f"okada slip: only {len(markers)} station(s) carry data across the "
            f"reference/target windows — need >= {cfg.min_stations}"
        )
    e_arr = np.asarray(e_list, dtype=np.float64)
    n_arr = np.asarray(n_list, dtype=np.float64)
    obs = np.column_stack(obs_cols)
    sig = np.column_stack(sig_cols)
    return markers, e_arr, n_arr, obs, sig, excluded


def _slip_formal_cov(
    e: FloatArray,
    n: FloatArray,
    sig: FloatArray,
    patches: FaultPatches,
    components: tuple[str, ...],
    smoothing: float,
    edge: str,
    nu: float,
) -> FloatArray:
    """Linear-Gaussian formal covariance of the regularized slip solution.

    ``C_s = A⁻¹·GᵂᵀGᵂ·A⁻¹`` with ``A = GᵂᵀGᵂ + λ²·LᵀL`` (weighted data ⇒
    identity data covariance), assembled from the **public** ``okada_greens``
    (G) and ``patch_laplacian`` (L) exactly as :func:`okada_invert_slip` does
    — verified against the returned slip in ``tests/test_slip.py``. Solved via
    the symmetric eigen-decomposition of the (positive-definite for
    ``edge="zero"``) normal matrix, never an explicit inverse of an
    ill-conditioned operator.
    """
    g = okada_greens(e, n, patches, components, nu)
    w = 1.0 / sig.ravel()
    g_w = g * w[:, None]
    lap = patch_laplacian(patches, edge)
    n_comp = len(components)
    reg = (
        np.asarray(np.kron(np.eye(n_comp), lap), dtype=np.float64)
        if n_comp > 1
        else lap
    )
    gtg = g_w.T @ g_w
    a = gtg + smoothing * smoothing * (reg.T @ reg)
    evals, evecs = np.linalg.eigh(a)
    tol = float(evals.max()) * a.shape[0] * float(np.finfo(np.float64).eps)
    inv = np.where(evals > tol, 1.0 / np.where(evals > tol, evals, 1.0), 0.0)
    a_inv = (evecs * inv) @ evecs.T
    cov: FloatArray = a_inv @ gtg @ a_inv
    return cov


def compute_slip_distribution(
    region_name: str,
    station_series: dict[str, StationSeries],
    detrended: dict[str, FloatArray],
    meta: dict[str, StationMeta],
    cfg: DeformationConfig,
    fitted_at: datetime.datetime,
) -> SlipOutcome:
    """Invert the single-window Okada slip distribution for one region.

    Args:
        region_name: Region key (product tag only).
        station_series: ``marker -> StationSeries`` of the stations that
            survived the per-station chain (mm, fractional-year epochs).
        detrended: ``marker -> (3, N)`` trajectory-model residuals [mm].
        meta: ``marker -> StationMeta`` coordinates from ``stations.cfg``.
        cfg: The validated ``deformation:`` block (``source == "okada"``).
        fitted_at: Run timestamp stamped on the product (UTC).

    Raises:
        RuntimeError: When too few stations carry net-displacement data, or
            the inversion returns a degenerate (non-finite / all-zero /
            all-pinned) slip solution — the caller records this as a
            region-level deformation failure (other products survive).
    """
    if not station_series:
        raise RuntimeError("okada slip: no station series available")
    okada = cfg.okada
    if okada is None:  # pragma: no cover - guarded by DeformationConfig
        raise RuntimeError("okada slip: source='okada' but no okada plane config")

    markers, e_arr, n_arr, obs, sig, excluded = _net_displacement(
        station_series, detrended, meta, cfg, okada
    )

    plane = OkadaSource(
        x=0.0,
        y=0.0,
        depth=okada.centroid_depth_km() * _KM_TO_M,
        strike=okada.strike,
        dip=okada.dip,
        length=okada.length_km * _KM_TO_M,
        width=okada.width_km * _KM_TO_M,
        strike_slip=0.0,
        dip_slip=0.0,
        opening=0.0,
    )
    patches = discretize_fault(plane, okada.n_strike, okada.n_dip)
    components = okada.components

    if okada.smoothing is None:
        lo, hi, count = okada.smoothing_scan
        lams = np.geomspace(lo, hi, count)
        _, _, corner = slip_lcurve(
            e_arr,
            n_arr,
            obs,
            sig,
            patches=patches,
            smoothings=lams,
            components=components,
            nonnegative=okada.nonneg,
            edge=okada.edge,
            nu=cfg.nu,
        )
        smoothing = float(lams[corner])
        selected_by = "lcurve"
    else:
        smoothing = float(okada.smoothing)
        selected_by = "fixed"

    fit = okada_invert_slip(
        e_arr,
        n_arr,
        obs,
        sig,
        patches=patches,
        components=components,
        smoothing=smoothing,
        nonnegative=okada.nonneg,
        edge=okada.edge,
        nu=cfg.nu,
    )
    # Robustness guard (Mogi-stage lesson): reject degenerate solutions —
    # a non-finite solve, or a solution carrying no resolvable signal (all
    # patches zero / pinned at the NNLS bound) — never silently.
    if not bool(np.all(np.isfinite(fit.slip))):
        raise RuntimeError("okada slip: non-finite slip solution")
    if float(np.max(np.abs(fit.slip))) <= 0.0:
        raise RuntimeError(
            "okada slip: degenerate all-zero slip solution — no resolvable "
            "signal on the configured plane (check the plane geometry/window)"
        )

    n_comp = len(components)
    n_p = patches.n_patches
    cov = _slip_formal_cov(
        e_arr, n_arr, sig, patches, components, smoothing, okada.edge, cfg.nu
    )
    sigma_flat = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    sigma_grid = sigma_flat.reshape(n_comp, patches.n_down, patches.n_along)

    e_per_deg, n_per_deg = _local_scale(okada.origin_lon, okada.origin_lat)
    fault_patches: list[FaultPatch] = []
    for k in range(n_p):
        row, col = divmod(k, patches.n_along)
        cx, cy, cd = (float(v) for v in patches.centers[k])
        fault_patches.append(
            FaultPatch(
                index=k,
                row=row,
                col=col,
                lon=okada.origin_lon + cx / e_per_deg,
                lat=okada.origin_lat + cy / n_per_deg,
                east_m=cx,
                north_m=cy,
                depth_km=cd / _KM_TO_M,
                slip_m={
                    comp: float(fit.slip[c, row, col])
                    for c, comp in enumerate(components)
                },
                sigma_m={
                    comp: float(sigma_grid[c, row, col])
                    for c, comp in enumerate(components)
                },
            )
        )

    potency = fit.potency()
    potency_m3 = {comp: float(potency[c]) for c, comp in enumerate(components)}
    # Potency σ per component: Var(P_c) = A_patch²·1ᵀ·C_cc·1 over the
    # component's patch block (P_c = A_patch·Σ_k s_ck).
    ones = np.ones(n_p, dtype=np.float64)
    sigma_potency_m3: dict[str, float] = {}
    for c, comp in enumerate(components):
        block = cov[c * n_p : (c + 1) * n_p, c * n_p : (c + 1) * n_p]
        var_p = patches.patch_area**2 * float(ones @ block @ ones)
        sigma_potency_m3[comp] = float(np.sqrt(max(var_p, 0.0)))

    use_detrended = cfg.series == "detrended"
    t_end = max(float(s.t[-1]) for s in station_series.values())
    t_ref = t_end - cfg.window_years
    result = SlipDistributionResult(
        region=region_name,
        source_type="okada",
        reference_time=yearf_to_datetime(t_ref),
        target_time=yearf_to_datetime(t_end),
        origin_lon=okada.origin_lon,
        origin_lat=okada.origin_lat,
        series_kind="detrended" if use_detrended else "raw",
        strike=okada.strike,
        dip=okada.dip,
        length_km=okada.length_km,
        width_km=okada.width_km,
        top_depth_km=okada.top_depth_km,
        n_strike=okada.n_strike,
        n_dip=okada.n_dip,
        components=list(components),
        nonnegative=okada.nonneg,
        smoothing=smoothing,
        smoothing_selected_by=selected_by,
        edge=okada.edge,
        stations=markers,
        patches=fault_patches,
        potency_m3=potency_m3,
        sigma_potency_m3=sigma_potency_m3,
        residual_norm=fit.residual_norm,
        roughness_norm=fit.roughness_norm,
        rms_mm=fit.rms / _MM_TO_M,
        n_obs=fit.n_obs,
        n_stations=len(markers),
        fitted_at=fitted_at,
    )
    return SlipOutcome(result=result, stations_excluded=tuple(excluded))
