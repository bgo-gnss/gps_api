"""Real-data validation harnesses for the precompute products.

Precompute-side tooling (never imported by the API routers): drives the
real pipeline on cached real-data fixtures and reconciles the products
against independent operational references. Currently one harness:
:mod:`gps_api.validation.realdata` — Svartsengi Mogi ΔV(t) vs the
operational model on ``insar.vedur.is``.
"""

from gps_api.validation.realdata import (
    Reconciliation,
    default_fixture_dir,
    run_validation,
)

__all__ = ["Reconciliation", "default_fixture_dir", "run_validation"]
