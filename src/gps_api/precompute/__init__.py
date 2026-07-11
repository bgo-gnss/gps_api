"""Scheduled precompute job for the GNSS analysis lane (plan §3, §6).

This package is the *writer* half of gps_api: a batch job that runs the
``gps_analysis`` public API over configured station sets and writes typed
products to the file store the read-only API serves
(:mod:`gps_api.settings` resolves the store root for both sides).

Phase-1 vertical slice — one region, file store only (Parquet/GeoJSON/JSON;
Postgres is the next slice):

- :mod:`gps_api.precompute.config` — all regions, stations, windows and
  paths come from the deployed gpsconfig via ``gps_parser`` (zero
  hardcoding, plan §10.4).
- :mod:`gps_api.precompute.sources` — displacement series input: published
  ``.NEU`` product files or a synthetic fixture (the data source is a
  parameter, as in ``gps_plot.dev_viz``).
- :mod:`gps_api.precompute.products` — product writers; every product file
  is validated against :mod:`gps_api.schemas` where a schema exists and
  carries provenance (method, frame, software versions, ``fitted_at``).
- :mod:`gps_api.precompute.job` — orchestration + the
  ``gps-api-precompute`` console script (foreground only).

The API routers never import this package (or ``gps_analysis``) — they only
read the files it writes.
"""

from gps_api.precompute.job import main, run_precompute

__all__ = ["main", "run_precompute"]
