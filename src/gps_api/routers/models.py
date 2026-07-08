"""Deformation-source model results per region (Mogi first, plan §6)."""

from fastapi import APIRouter

from gps_api.routers import not_implemented
from gps_api.schemas import ModelResult

router = APIRouter(tags=["models"])


@router.get("/models/{region}", response_model=ModelResult)
def region_model(region: str) -> ModelResult:
    """Latest deformation-source model for a region."""
    not_implemented(f"The model endpoint for region {region!r}")
