"""Model products per region (store-wired for the GBIS4TS break catalogs).

``GET /v1/models/{region}`` serves ``models/<region>_breaks.json`` — the
GBIS4TS break/rate-change catalog the precompute job writes
(``kind="breakpoints"``; :func:`gps_api.precompute.products.write_breaks_json`)
— re-validated through :class:`~gps_api.schemas.ModelResult` on the way out,
following the ``velocities`` pattern. The Mogi deformation-source kind
shares the endpoint but stays a product of the backburnered modeling lane
(plan §9b): a region without model products is a 404 either way.

``GET /v1/models/{region}/history`` stays a **documented 501** (contract
Decisions #5: reserved in v0). Wiring it needs run accumulation — the file
store keeps only the latest run's products — so it lands with the
Postgres/history slice, not this one.
"""

import json
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException
from fastapi import Path as PathParam

from gps_api import settings
from gps_api.routers import not_implemented
from gps_api.schemas import ModelHistory, ModelResult

router = APIRouter(tags=["models"])

#: Region keys are plain identifiers — no path characters reach the store.
REGION_PATTERN = r"^[A-Za-z0-9_-]+$"


@router.get("/models/{region}", response_model=ModelResult)
def region_model(
    region: Annotated[str, PathParam(pattern=REGION_PATTERN)],
) -> ModelResult:
    """Latest model products for a region (GBIS4TS break catalog first)."""
    path = settings.store_path() / settings.MODELS_DIR / f"{region}_breaks.json"
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"no model products for region {region!r}",
        )
    try:
        payload: dict[str, Any] = json.loads(path.read_text())
        # The file's run-level timestamp lives in its structured provenance.
        fitted_at = payload["provenance"]["fitted_at"]
        return ModelResult.model_validate({"fitted_at": fitted_at, **payload})
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"corrupt model product {path.name!r}: {exc}",
        ) from exc


@router.get("/models/{region}/history", response_model=ModelHistory)
def region_model_history(
    region: str,
    start: datetime | None = None,
    end: datetime | None = None,
) -> ModelHistory:
    """Fit time-lapse for a region — reserved in v0 (contract Decisions #5).

    The file store only keeps the latest run; history needs run
    accumulation and lands with the Postgres slice.
    """
    not_implemented(f"The model history endpoint for region {region!r}")
