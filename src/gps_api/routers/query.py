"""Complex selections via POST body (regions, polygons, multi-station, windows)."""

from fastapi import APIRouter

from gps_api.routers import not_implemented
from gps_api.schemas import QueryRequest, QueryResponse

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """One JSON body instead of dozens of repeated query params (plan §10.5)."""
    not_implemented("The query endpoint")
