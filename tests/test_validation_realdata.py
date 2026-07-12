"""Real-data reconciliation regression check (env-gated, Svartsengi Mogi).

Reruns the real-data validation harness from the cached fixture: the real
precompute chain over the fixture's published ``.NEU`` series (clipped at
the inflation-cycle end), the store product read back through the contract
schema, the served endpoint, and the quantitative reconciliation against
the cached operational per-cycle ΔV series (Vincent's model — see
``gps_api.validation.realdata`` for the file-format characterization).

**Skipped when the fixture is absent** (it is gitignored — real station
data plus the operational reference are cached locally, never committed).
Rebuild with::

    uv run gps-api-validate-deformation fetch   # needs CDN HTTP + SSH

Thresholds are regression guards derived from the 2026-07-12 baseline run
(inflation08, 39 stations, 40/43 epochs fitted: r=0.9934, scale=0.797,
final ratio 0.848, depth 3.71±1.05 km vs the operational fixed 4.0 km,
source offset 0.32 km, scatter 1.67 km) with generous margins — they catch
a broken pipeline (sign flips, unit slips, frame errors, epoch
misalignment), not model-difference noise between our free-geometry
inversion and the operational fixed-4 km grid search. They were NOT tuned
to make the two models agree.
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gps_api.main import create_app
from gps_api.schemas import DeformationResult
from gps_api.validation.realdata import (
    REGION_NAME,
    Reconciliation,
    default_fixture_dir,
    run_validation,
)

FIXTURE_DIR = default_fixture_dir()

pytestmark = pytest.mark.skipif(
    not (FIXTURE_DIR / "manifest.json").is_file(),
    reason=(
        "real-data fixture not fetched — run "
        "`uv run gps-api-validate-deformation fetch` (needs CDN + SSH access)"
    ),
)


@pytest.fixture(scope="module")
def outcome(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Reconciliation, DeformationResult, Path]:
    """One full harness run shared by every assertion (it is the slow part)."""
    store = tmp_path_factory.mktemp("store")
    reconciliation, result = run_validation(FIXTURE_DIR, store=store)
    return reconciliation, result, store


def test_pipeline_produces_dense_epoch_series(
    outcome: tuple[Reconciliation, DeformationResult, Path],
) -> None:
    reconciliation, result, _ = outcome
    assert reconciliation.n_epochs >= 20
    assert reconciliation.n_aligned >= 20
    # A real multi-station cluster entered the inversion, per epoch.
    assert len(reconciliation.stations_used) >= 10
    assert reconciliation.n_stations_median >= 10


def test_dv_series_tracks_operational_model(
    outcome: tuple[Reconciliation, DeformationResult, Path],
) -> None:
    reconciliation, _, _ = outcome
    # Shape agreement: the two cumulative dV(t) curves are the same signal.
    assert reconciliation.pearson_r > 0.97
    # Amplitude agreement: free-geometry vs fixed-4km differ by a bounded
    # depth/dV trade-off factor, never a unit/sign slip (0.80 observed).
    assert 0.5 < reconciliation.scale < 1.6
    assert reconciliation.final_ours_m3 > 0  # inflation cycle => net inflation
    assert 0.5 < reconciliation.final_ratio < 1.6


def test_source_geometry_is_physical_and_stable(
    outcome: tuple[Reconciliation, DeformationResult, Path],
) -> None:
    reconciliation, _, _ = outcome
    # Depth: shallow crustal source, same order as the operational 4 km.
    assert 2.0 < reconciliation.depth_mean_km < 8.0
    assert reconciliation.depth_std_km < 2.0
    # Position: within a few km of the operational fixed source.
    if reconciliation.source_offset_km is not None:
        assert reconciliation.source_offset_km < 5.0
    assert reconciliation.source_scatter_km < 5.0


def test_product_is_served_by_the_endpoint(
    outcome: tuple[Reconciliation, DeformationResult, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, result, store = outcome
    monkeypatch.setenv("GPS_API_STORE", str(store))
    client = TestClient(create_app())
    response = client.get(f"/v1/deformation/{REGION_NAME}")
    assert response.status_code == 200
    payload = response.json()
    assert payload["region"] == REGION_NAME
    assert payload["source_type"] == "mogi"
    assert len(payload["fits"]) == len(result.fits)


def test_reconciliation_metrics_are_json_ready(
    outcome: tuple[Reconciliation, DeformationResult, Path],
) -> None:
    reconciliation, _, _ = outcome
    payload = json.loads(json.dumps(reconciliation.as_dict()))
    assert payload["cycle"] == reconciliation.cycle
    assert isinstance(payload["stations_used"], list)
