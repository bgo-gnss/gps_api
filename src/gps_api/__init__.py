"""gps_api — read-only API over precomputed GNSS analysis products.

Tier-2 service of the gpslibrary ecosystem (plan §3, §10.5): a scheduled
precompute job writes analysis products (detrended series, velocities,
deformation-source models) to the store — Postgres for metadata/velocities/
catalogs, Parquet/GeoJSON files for bulk series and rasters. This API only
reads and serves; the web UIs (Dash QC tool, aflogun SPA) only query the API.

The contract lives in ``docs/API_CONTRACT.md`` (v0, drafted Phase 0) and in
the typed schemas in :mod:`gps_api.schemas`.
"""

__version__ = "0.1.0"
