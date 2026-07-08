"""Deformation-source model results per region (Mogi first, plan §6)."""

from datetime import datetime

from fastapi import APIRouter

from gps_api.routers import not_implemented
from gps_api.schemas import ModelHistory, ModelResult

router = APIRouter(tags=["models"])


@router.get("/models/{region}", response_model=ModelResult)
def region_model(region: str) -> ModelResult:
    """Latest deformation-source model for a region."""
    not_implemented(f"The model endpoint for region {region!r}")


@router.get("/models/{region}/history", response_model=ModelHistory)
def region_model_history(
    region: str,
    start: datetime | None = None,
    end: datetime | None = None,
) -> ModelHistory:
    """Fit time-lapse for a region (reserved in v0; Phase 2 wires it)."""
    not_implemented(f"The model history endpoint for region {region!r}")
