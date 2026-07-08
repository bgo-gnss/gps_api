"""FastAPI application factory for the GPS products API.

Read-only service over the precomputed analysis store (Postgres for
metadata/velocities/catalogs; Parquet/GeoJSON files for bulk series and
rasters). The scheduled precompute job writes; this API only reads.

Contract: ``docs/API_CONTRACT.md`` (v0). All data endpoints are 501 stubs
until Phase 1 wires the store.
"""

from fastapi import FastAPI

from gps_api import __version__
from gps_api.routers import layers, models, query, stations, velocities


def create_app() -> FastAPI:
    """Build the application; kept as a factory for tests and future settings."""
    app = FastAPI(
        title="gps_api",
        summary="Read-only API over precomputed GNSS analysis products (IMO)",
        version=__version__,
    )
    app.include_router(stations.router)
    app.include_router(velocities.router)
    app.include_router(models.router)
    app.include_router(layers.router)
    app.include_router(query.router)

    @app.get("/healthz", tags=["service"])
    def healthz() -> dict[str, str]:
        """Liveness probe."""
        return {"status": "ok", "version": __version__}

    return app


app = create_app()


def run() -> None:
    """Development entry point: ``uv run gps-api``."""
    import uvicorn

    uvicorn.run("gps_api.main:app", host="127.0.0.1", port=8000, reload=True)
