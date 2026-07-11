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
            "'gbis' = GBIS4TS joint break/colored-noise estimate with "
            "honest σ (selective, per breakpoints.enabled_regions); "
            "'mle' reserved"
        )
    )
    window_start: datetime
    window_end: datetime


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

    Two kinds share the endpoint: ``'mogi'`` (deformation-source inversion —
    reserved, backburnered lane) carries ``parameters``; ``'breakpoints'``
    (GBIS4TS break/rate-change catalog, analysis lane) carries ``entries``
    (one per station × component).
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
