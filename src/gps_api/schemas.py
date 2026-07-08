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
from typing import Literal

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
    method: Literal["wls", "mle"]
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


class ModelResult(BaseModel):
    """GET /models/{region} — latest deformation-source model for a region."""

    region: str
    kind: Literal["mogi"] = Field(
        description="source type; 'okada' and 'joint' arrive in Phase 2"
    )
    parameters: dict[str, float] = Field(
        description="named source parameters (lon, lat, depth_km, dV_m3, ...)"
    )
    fitted_at: datetime
    provenance: str | None = Field(
        default=None, description="pipeline version / input window used for the fit"
    )


class ModelFit(BaseModel):
    """One historical fit in a region's model time-lapse."""

    fitted_at: datetime
    parameters: dict[str, float]
    provenance: str | None = None


class ModelHistory(BaseModel):
    """GET /models/{region}/history — fit time series (e.g. Svartsengi volume).

    Reserved in v0, wired in Phase 2 alongside the Mogi inversion lane and
    the reconciliation against Vincent's ``inv_volume_mogi.dat``.
    """

    region: str
    kind: Literal["mogi"]
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
