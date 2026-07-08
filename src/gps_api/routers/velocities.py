"""Velocity vectors as GeoJSON features (magnitude + azimuth in properties)."""

from typing import Annotated

from fastapi import APIRouter, Query

from gps_api.routers import not_implemented
from gps_api.schemas import VelocityCollection

router = APIRouter(tags=["velocities"])


@router.get("/velocities", response_model=VelocityCollection)
def list_velocities(
    region: str | None = None,
    window_years: Annotated[
        float | None, Query(gt=0, description="estimation window length in years")
    ] = None,
) -> VelocityCollection:
    """Velocity field, optionally filtered by region and estimation window."""
    not_implemented("The velocity field")
