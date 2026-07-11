"""Velocity vectors as GeoJSON features (magnitude + azimuth in properties).

First store-wired endpoint (Phase-1 slice): serves the
``velocities/<region>.geojson`` files the precompute job writes
(:mod:`gps_api.precompute`). The files are contract-shaped by construction
(written through :class:`~gps_api.schemas.VelocityCollection`); they are
re-validated here on read so a hand-edited or truncated file cannot leak a
malformed payload. The top-level ``provenance`` foreign member in the files
is dropped by the schema on the way out.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from gps_api import settings
from gps_api.schemas import VelocityCollection, VelocityFeature

router = APIRouter(tags=["velocities"])

#: Days per Julian year — window-length filter arithmetic only.
_DAYS_PER_YEAR = 365.25

#: Relative tolerance when matching ``window_years`` against a feature's
#: actual estimation window (data are daily; windows are configured in
#: round years but realized on available epochs).
_WINDOW_RTOL = 0.1


def _window_years(start: datetime, end: datetime) -> float:
    """Length of one feature's estimation window in years."""
    return (end - start).total_seconds() / (86400.0 * _DAYS_PER_YEAR)


def _load_collection(path: Path) -> VelocityCollection:
    """Read + re-validate one region file; 500 on a corrupt product."""
    try:
        return VelocityCollection.model_validate(json.loads(path.read_text()))
    except (ValueError, OSError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"corrupt velocity product {path.name!r}: {exc}",
        ) from exc


@router.get("/velocities", response_model=VelocityCollection)
def list_velocities(
    region: Annotated[
        str | None,
        Query(
            pattern=r"^[A-Za-z0-9_-]+$",
            description="region key, e.g. 'reykjanes' (default: all regions)",
        ),
    ] = None,
    window_years: Annotated[
        float | None, Query(gt=0, description="estimation window length in years")
    ] = None,
) -> VelocityCollection:
    """Velocity field, optionally filtered by region and estimation window."""
    velocities_dir = settings.store_path() / settings.VELOCITIES_DIR
    if region is not None:
        paths = [velocities_dir / f"{region}.geojson"]
        if not paths[0].is_file():
            raise HTTPException(
                status_code=404,
                detail=f"no velocity products for region {region!r}",
            )
    else:
        paths = sorted(velocities_dir.glob("*.geojson"))
        if not paths:
            raise HTTPException(
                status_code=404,
                detail="no velocity products in the store — run gps-api-precompute",
            )

    features: list[VelocityFeature] = []
    for path in paths:
        features.extend(_load_collection(path).features)
    if window_years is not None:
        features = [
            f
            for f in features
            if abs(
                _window_years(f.properties.window_start, f.properties.window_end)
                - window_years
            )
            <= _WINDOW_RTOL * window_years
        ]
    return VelocityCollection(features=features)
