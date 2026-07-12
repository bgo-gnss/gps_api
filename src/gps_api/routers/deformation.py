"""Deformation-source products per region (Mogi ΔV(t) or Okada slip).

``GET /v1/deformation/{region}`` serves whichever deformation product the
precompute job wrote for a region gated by ``deformation.enabled_regions``,
discriminated by ``source_type``:

- ``models/<region>_deformation.json`` — the Mogi source time-series product
  (:class:`~gps_api.schemas.DeformationResult`, contract Amendment A6,
  :func:`gps_api.precompute.products.write_deformation_json`).
- ``models/<region>_slip.json`` — the Okada distributed-slip product
  (:class:`~gps_api.schemas.SlipDistributionResult`, contract Amendment A7,
  :func:`gps_api.precompute.products.write_slip_json`).

A region configures ``source: mogi`` XOR ``source: okada``, so at most one
product exists — the router dispatches by which file is present (it never
reads config). The payload is re-validated on the way out, following the
``models`` pattern. A region without either product (not gated, or its stage
failed and was recorded in ``meta/run.json``) is a 404.

The Mogi product is an **independent GNSS-only** analog of Vincent's
operational InSAR-side Mogi ΔV(t) (``inv_volume_mogi.dat``): the two are
cross-checked against each other, never derived from one another.
"""

import json
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException
from fastapi import Path as PathParam

from gps_api import settings
from gps_api.routers.models import REGION_PATTERN
from gps_api.schemas import (
    DeformationProduct,
    DeformationResult,
    SlipDistributionResult,
)

router = APIRouter(tags=["deformation"])


@router.get("/deformation/{region}", response_model=DeformationProduct)
def region_deformation(
    region: Annotated[str, PathParam(pattern=REGION_PATTERN)],
) -> DeformationResult | SlipDistributionResult:
    """Latest deformation product for a region (Mogi ΔV(t) or Okada slip)."""
    models_dir = settings.store_path() / settings.MODELS_DIR
    mogi_path = models_dir / f"{region}_deformation.json"
    slip_path = models_dir / f"{region}_slip.json"
    if mogi_path.is_file():
        path = mogi_path
    elif slip_path.is_file():
        path = slip_path
    else:
        raise HTTPException(
            status_code=404,
            detail=f"no deformation products for region {region!r}",
        )
    try:
        payload: dict[str, Any] = json.loads(path.read_text())
        if path is mogi_path:
            return DeformationResult.model_validate(payload)
        return SlipDistributionResult.model_validate(payload)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"corrupt deformation product {path.name!r}: {exc}",
        ) from exc
