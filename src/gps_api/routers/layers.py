"""Typed, versioned layer catalog for data-driven map overlays."""

from fastapi import APIRouter

from gps_api.routers import not_implemented
from gps_api.schemas import LayerCatalog

router = APIRouter(tags=["layers"])


@router.get("/layers", response_model=LayerCatalog)
def layer_catalog() -> LayerCatalog:
    """Catalog of overlay layers (GeoJSON/WMS/TMS/image) the map clients render."""
    not_implemented("The layer catalog")
