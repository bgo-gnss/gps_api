"""Station catalog and per-station series.

Contract split (skjalftalisa lesson): ``GET /stations`` is the small,
cacheable catalog; ``GET /stations/{marker}/series`` is the on-demand,
potentially large payload with downsampling support.
"""

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Query

from gps_api.routers import not_implemented
from gps_api.schemas import SeriesResponse, StationCollection

router = APIRouter(tags=["stations"])


@router.get("/stations", response_model=StationCollection)
def list_stations() -> StationCollection:
    """Station catalog as a GeoJSON FeatureCollection (small, cacheable)."""
    not_implemented("The station catalog")


@router.get("/stations/{marker}/series", response_model=SeriesResponse)
def station_series(
    marker: str,
    start: datetime | None = None,
    end: datetime | None = None,
    max_points: Annotated[
        int | None,
        Query(
            ge=2,
            description=(
                "target point count; the server downsamples visually "
                "faithfully (LTTB — peaks, offsets and trend shape survive)"
            ),
        ),
    ] = None,
    detrended: bool = True,
) -> SeriesResponse:
    """N/E/U displacement series for one station (on-demand, downsampleable)."""
    not_implemented(f"The series endpoint for station {marker!r}")
