"""Pydantic schemas — the typed half of the API contract (docs/API_CONTRACT.md).

Conventions (skjalftalisa lessons, plan §10.5):

- Anything mappable is GeoJSON: ``FeatureCollection`` of Point features with
  the payload in ``properties`` — map layers consume it directly.
- Timestamps are timezone-aware UTC; pydantic serializes them ISO-8601 ``Z``.
- Units are explicit fields, never implied: displacements mm, velocities
  mm/yr, azimuth degrees clockwise from north, depths km.
- Normalized shapes only — raw GAMIT/processing formats never leak through.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

ProductKind = Literal["series", "velocities", "models"]


def _default_products() -> list[ProductKind]:
    return ["series"]


class PointGeometry(BaseModel):
    """GeoJSON Point. Coordinates are [lon, lat] or [lon, lat, height_m], WGS84."""

    type: Literal["Point"] = "Point"
    coordinates: list[float] = Field(min_length=2, max_length=3)


class StationProperties(BaseModel):
    """Catalog properties of one GNSS station."""

    marker: str = Field(description="4-char station marker, e.g. 'SENG'")
    name: str | None = None
    regions: list[str] = Field(default_factory=list)


class StationFeature(BaseModel):
    type: Literal["Feature"] = "Feature"
    geometry: PointGeometry
    properties: StationProperties


class StationCollection(BaseModel):
    """GET /stations — small, cacheable station catalog."""

    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: list[StationFeature]


class SeriesResponse(BaseModel):
    """GET /stations/{marker}/series — N/E/U displacement time series."""

    marker: str
    frame: str = Field(description="reference frame / plate fix, e.g. 'ITRF2014'")
    units: Literal["mm"] = "mm"
    detrended: bool
    time: list[datetime]
    north: list[float]
    east: list[float]
    up: list[float]
    sigma_north: list[float] | None = None
    sigma_east: list[float] | None = None
    sigma_up: list[float] | None = None


class ComponentNoise(BaseModel):
    """MLE white + power-law noise model behind an honest σ (one component).

    The (σ̂_w, β̂, κ̂) triple of ``C = σ_w²·I + β²·ΔT^(−κ/2)·(T Tᵀ)`` at the
    likelihood optimum (``gps_analysis.noise.NoiseModel``) — the provenance
    record that makes a ``method="mle"`` velocity σ honest (contract
    Amendment A5).
    """

    sigma_white_mm: float = Field(description="white-noise amplitude σ̂_w, mm")
    amplitude_mm: float = Field(
        description="power-law amplitude β̂, mm·yr^(−κ/4) (Williams 2003)"
    )
    spectral_index: float = Field(
        description="spectral index κ̂ (0 white, −1 flicker, −2 random walk)"
    )


class VelocityProperties(BaseModel):
    """Velocity vector for one station over one estimation window."""

    marker: str
    east: float = Field(description="mm/yr")
    north: float = Field(description="mm/yr")
    up: float = Field(description="mm/yr")
    sigma_east: float
    sigma_north: float
    sigma_up: float
    magnitude: float = Field(description="horizontal speed, mm/yr")
    azimuth: float = Field(description="degrees clockwise from north")
    method: Literal["wls", "gbis", "mle"] = Field(
        description=(
            "estimator tag (PLAN-analysis-lane §1): 'wls' = fixed-window "
            "weighted least squares with formal σ (the fleet baseline); "
            "'mle' = colored-noise MLE with honest σ (per-region "
            "velocity_method in analysis.yaml — Amendment A5); "
            "'gbis' = GBIS4TS joint break/colored-noise estimate "
            "(reserved for a later slice)"
        )
    )
    window_start: datetime
    window_end: datetime
    noise: dict[str, ComponentNoise] | None = Field(
        default=None,
        description=(
            "per-component MLE noise models keyed 'north'/'east'/'up' — "
            "present only for method='mle' (Amendment A5)"
        ),
    )


class VelocityFeature(BaseModel):
    type: Literal["Feature"] = "Feature"
    geometry: PointGeometry
    properties: VelocityProperties


class VelocityCollection(BaseModel):
    """GET /velocities — velocity vectors as GeoJSON features."""

    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: list[VelocityFeature]


class BreakEntry(BaseModel):
    """One GBIS4TS break/rate-change estimate (one station, one component).

    Mirrors losslessly what the precompute writer emits into
    ``models/<region>_breaks.json``
    (:func:`gps_api.precompute.products.write_breaks_json`); the parameter
    names follow the BPD1/BPD2 flattening in
    :func:`gps_api.precompute.job._break_parameters` — intercept [mm],
    secular trend [mm/yr], rate change(s) [mm/yr], break epoch(s)
    [fractional yr], colored-noise spectral index κ and amplitude [mm]
    (Yang, Sigmundsson & Geirsson 2023, 2023GL103432).
    """

    marker: str
    component: Literal["north", "east", "up"]
    model: Literal["BPD1", "BPD2"] = Field(
        description="GBIS4TS forward model: one or two velocity break points"
    )
    method: Literal["gbis"] = "gbis"
    fitted_at: datetime
    breakpoint_time: datetime = Field(
        description="first break epoch as UTC time (breakpoint_yearf converted)"
    )
    parameters: dict[str, float] = Field(
        description=(
            "posterior-optimal parameters: intercept_mm, trend_mm_yr, "
            "trend_change_mm_yr, breakpoint_yearf (+ trend_change2_mm_yr, "
            "breakpoint2_yearf for BPD2), kappa, amp_mm"
        )
    )
    wn_amp_mm: float = Field(description="fixed white-noise amplitude used, mm")
    y_ref_mm: float = Field(
        description="start baseline subtracted by the zero-reference conditioning, mm"
    )
    n_runs: int = Field(description="kept MCMC iterations behind the estimate")


class ModelResult(BaseModel):
    """GET /models/{region} — latest model products for a region.

    Two kinds share the endpoint: ``'mogi'`` (single-source snapshot —
    kept reserved; the live Mogi ΔV(t) product is served as a
    :class:`DeformationResult` by ``GET /v1/deformation/{region}``,
    Amendment A6) carries ``parameters``; ``'breakpoints'`` (GBIS4TS
    break/rate-change catalog, analysis lane) carries ``entries`` (one per
    station × component).
    """

    region: str
    kind: Literal["mogi", "breakpoints"] = Field(
        description=(
            "'breakpoints' = GBIS4TS break/rate-change catalog (entries); "
            "'mogi' = deformation-source parameters (reserved); 'okada' "
            "and 'joint' arrive with the modeling lane"
        )
    )
    parameters: dict[str, float] = Field(
        default_factory=dict,
        description="named source parameters (lon, lat, depth_km, dV_m3, ...);"
        " empty for kind='breakpoints'",
    )
    entries: list[BreakEntry] | None = Field(
        default=None,
        description="break/rate-change estimates (kind='breakpoints' only)",
    )
    fitted_at: datetime
    provenance: str | dict[str, Any] | None = Field(
        default=None,
        description=(
            "product provenance: method tag, frame, software versions, "
            "fitted_at, input source (structured), or a free-form note"
        ),
    )


class ModelFit(BaseModel):
    """One historical fit in a region's model time-lapse."""

    fitted_at: datetime
    parameters: dict[str, float]
    provenance: str | dict[str, Any] | None = None


class ModelHistory(BaseModel):
    """GET /models/{region}/history — fit time series (e.g. Svartsengi volume).

    Reserved in v0, wired in Phase 2 alongside the Mogi inversion lane and
    the reconciliation against Vincent's ``inv_volume_mogi.dat``.
    """

    region: str
    kind: Literal["mogi", "breakpoints"]
    fits: list[ModelFit]


class MogiSourceEstimate(BaseModel):
    """One epoch of a region's Mogi deformation-source time series.

    A weighted nonlinear least-squares fit (``gps_analysis.mogi_invert``)
    of one Mogi point source to the region's GNSS displacement field at
    ``time``, relative to the product's ``reference_time``. Position is
    served both geographically (lon/lat, WGS84) and in the local
    tangent-plane frame (east/north metres from the product's origin).
    """

    time: datetime = Field(description="epoch of this fit (UTC)")
    lon: float = Field(description="source longitude, degrees East (WGS84)")
    lat: float = Field(description="source latitude, degrees North (WGS84)")
    east_m: float = Field(description="source east offset from origin, m")
    north_m: float = Field(description="source north offset from origin, m")
    depth_km: float = Field(description="source depth below surface, km")
    dv_m3: float = Field(
        description="Mogi volume change ΔV since reference_time, m³ (+ = inflation)"
    )
    sigma_east_m: float = Field(description="formal 1-σ of east_m, m")
    sigma_north_m: float = Field(description="formal 1-σ of north_m, m")
    sigma_depth_km: float = Field(description="formal 1-σ of depth_km, km")
    sigma_dv_m3: float = Field(description="formal 1-σ of dv_m3, m³")
    chi2_reduced: float = Field(description="reduced χ² of the weighted fit")
    rms_mm: float = Field(description="unweighted residual RMS, mm")
    n_stations: int = Field(description="stations entering this epoch's fit")


class MogiPosteriorSummary(BaseModel):
    """Bayesian posterior of the newest epoch (``gps_analysis.mogi_invert_bayes``).

    Percentile summaries of the post-burn-in GBIS chain over the four Mogi
    parameters — the honest-uncertainty companion to the per-epoch formal
    least-squares σ. Parameter keys: ``east_m``, ``north_m``, ``depth_km``,
    ``dv_m3``; percentile keys: ``p2_5``/``p16``/``p50``/``p84``/``p97_5``.
    """

    time: datetime = Field(description="epoch the posterior refers to (UTC)")
    n_runs: int = Field(description="kept MCMC iterations")
    burn_in: int = Field(description="annealed burn-in iterations discarded")
    optimal: dict[str, float] = Field(
        description="maximum-a-posteriori source (same parameter keys)"
    )
    percentiles: dict[str, dict[str, float]] = Field(
        description="parameter → {p2_5, p16, p50, p84, p97_5} posterior summary"
    )


class DeformationResult(BaseModel):
    """GET /v1/deformation/{region} — Mogi source ΔV(t) time-series product.

    Independent GNSS-only deformation product (contract Amendment A6):
    per-epoch Mogi source fits (position, depth, ΔV + formal σ) relative to
    ``reference_time``, plus an optional Bayesian posterior for the newest
    epoch. The analog of Vincent's operational InSAR-side Mogi ΔV(t)
    (``inv_volume_mogi.dat``) — produced independently for cross-checking,
    never derived from his files.
    """

    region: str
    source_type: Literal["mogi"] = Field(
        description="deformation source model ('okada' and 'joint' reserved)"
    )
    reference_time: datetime = Field(
        description="epoch displacements are referenced to (ΔV = 0 by construction)"
    )
    origin_lon: float = Field(
        description="local tangent-plane frame origin longitude, degrees East"
    )
    origin_lat: float = Field(
        description="local tangent-plane frame origin latitude, degrees North"
    )
    series_kind: Literal["raw", "detrended"] = Field(
        description="which station series fed the inversion (deformation.series)"
    )
    stations: list[str] = Field(
        description="markers whose displacements entered the inversion"
    )
    fits: list[MogiSourceEstimate] = Field(
        description="the ΔV(t)/depth/position time series, ascending epochs"
    )
    posterior: MogiPosteriorSummary | None = Field(
        default=None,
        description="Bayesian posterior of the newest epoch (deformation.bayes)",
    )
    fitted_at: datetime
    provenance: str | dict[str, Any] | None = Field(
        default=None,
        description=(
            "product provenance: method tag, frame, software versions, "
            "fitted_at, input source, run settings (structured)"
        ),
    )


class Layer(BaseModel):
    """One entry in the typed, versioned layer catalog (data-driven overlays)."""

    id: str
    title: str
    kind: Literal["geojson", "wms", "tms", "image"]
    url: str
    version: str
    attribution: str | None = None


class LayerCatalog(BaseModel):
    """GET /layers — catalog the map clients iterate to build overlays."""

    version: str
    layers: list[Layer]


class QueryRequest(BaseModel):
    """POST /query — complex selections as a JSON body, not repeated params."""

    markers: list[str] | None = None
    regions: list[str] | None = None
    polygon: list[list[float]] | None = Field(
        default=None, description="GeoJSON-style ring of [lon, lat] positions"
    )
    start: datetime | None = None
    end: datetime | None = None
    products: list[ProductKind] = Field(default_factory=_default_products)


class QueryResponse(BaseModel):
    """POST /query result — only the requested product blocks are populated."""

    series: list[SeriesResponse] | None = None
    velocities: VelocityCollection | None = None
    models: list[ModelResult] | None = None
