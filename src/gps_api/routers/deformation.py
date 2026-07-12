"""Deformation-source products per region (Mogi ΔV(t) time series).

``GET /v1/deformation/{region}`` serves ``models/<region>_deformation.json``
— the Mogi source time-series product the precompute job writes for regions
gated by ``deformation.enabled_regions``
(:func:`gps_api.precompute.products.write_deformation_json`, contract
Amendment A6) — re-validated through
:class:`~gps_api.schemas.DeformationResult` on the way out, following the
``models`` pattern. A region without a deformation product (not gated, or
its stage failed and was recorded in ``meta/run.json``) is a 404.

The product is an **independent GNSS-only** analog of Vincent's operational
InSAR-side Mogi ΔV(t) (``inv_volume_mogi.dat``): the two are cross-checked
against each other, never derived from one another.
"""

import json
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException
from fastapi import Path as PathParam

from gps_api import settings
from gps_api.routers.models import REGION_PATTERN
from gps_api.schemas import DeformationResult

router = APIRouter(tags=["deformation"])


@router.get("/deformation/{region}", response_model=DeformationResult)
def region_deformation(
    region: Annotated[str, PathParam(pattern=REGION_PATTERN)],
) -> DeformationResult:
    """Latest Mogi deformation-source time series for a region."""
    path = settings.store_path() / settings.MODELS_DIR / f"{region}_deformation.json"
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"no deformation products for region {region!r}",
        )
    try:
        payload: dict[str, Any] = json.loads(path.read_text())
        return DeformationResult.model_validate(payload)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"corrupt deformation product {path.name!r}: {exc}",
        ) from exc
