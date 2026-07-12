"""Contract-shape tests for the service surface.

What these tests pin down is the *contract*: every route exists, every
error is ``{"detail": …}``, and the OpenAPI document describes the full
surface in docs/API_CONTRACT.md. The store-wired endpoints
(``/v1/stations``, ``/v1/stations/{marker}/series``, ``/v1/velocities``,
``/v1/models/{region}``) answer contract-shaped 404s on an empty store;
the reserved endpoints (``/v1/models/{region}/history``, ``/v1/layers``,
``/v1/query``) stay deliberate 501 stubs.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gps_api import __version__
from gps_api.main import create_app

client = TestClient(create_app())

STUB_GET_ROUTES = [
    "/v1/models/svartsengi/history",
    "/v1/layers",
]

WIRED_EMPTY_STORE_404_ROUTES = [
    "/v1/stations",
    "/v1/stations/SENG/series",
    "/v1/velocities",
    "/v1/models/reykjanes",
]

CONTRACT_PATHS = [
    "/healthz",
    "/v1/stations",
    "/v1/stations/{marker}/series",
    "/v1/velocities",
    "/v1/models/{region}",
    "/v1/models/{region}/history",
    "/v1/layers",
    "/v1/query",
]


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point every test at an empty temp store — never at the user's cache."""
    monkeypatch.setenv("GPS_API_STORE", str(tmp_path))


def test_healthz() -> None:
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "version": __version__}


@pytest.mark.parametrize("path", STUB_GET_ROUTES)
def test_stub_get_endpoints_return_501_with_detail(path: str) -> None:
    resp = client.get(path)
    assert resp.status_code == 501
    assert isinstance(resp.json()["detail"], str)


@pytest.mark.parametrize("path", WIRED_EMPTY_STORE_404_ROUTES)
def test_wired_endpoints_empty_store_is_404_with_detail(path: str) -> None:
    """Store-wired endpoints answer a contract-shaped 404 on an empty store."""
    resp = client.get(path)
    assert resp.status_code == 404
    assert isinstance(resp.json()["detail"], str)


def test_query_stub_returns_501_with_detail() -> None:
    resp = client.post("/v1/query", json={"regions": ["reykjanes"]})
    assert resp.status_code == 501
    assert isinstance(resp.json()["detail"], str)


def test_query_validation_error_uses_detail_shape() -> None:
    resp = client.post("/v1/query", json={"products": ["nonsense"]})
    assert resp.status_code == 422
    assert "detail" in resp.json()


def test_unknown_route_uses_detail_shape() -> None:
    resp = client.get("/definitely-not-a-route")
    assert resp.status_code == 404
    assert "detail" in resp.json()


def test_velocities_rejects_pathy_region() -> None:
    """Region names are validated (no path characters reach the store)."""
    resp = client.get("/v1/velocities", params={"region": "../etc"})
    assert resp.status_code == 422
    assert "detail" in resp.json()


def test_series_rejects_pathy_marker() -> None:
    """Markers are validated (no path characters reach the store)."""
    resp = client.get("/v1/stations/SENG!/series")
    assert resp.status_code == 422
    assert "detail" in resp.json()


def test_models_rejects_pathy_region() -> None:
    resp = client.get("/v1/models/bad!region")
    assert resp.status_code == 422
    assert "detail" in resp.json()


def test_series_rejects_undersized_max_points() -> None:
    """The LTTB target keeps at least the two endpoint samples."""
    resp = client.get("/v1/stations/SENG/series", params={"max_points": 1})
    assert resp.status_code == 422
    assert "detail" in resp.json()


def test_openapi_documents_the_contract() -> None:
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    paths = resp.json()["paths"]
    for route in CONTRACT_PATHS:
        assert route in paths, f"contract route {route} missing from OpenAPI"
