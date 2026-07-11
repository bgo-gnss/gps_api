"""End-to-end test of the Phase-1 precompute slice.

Drives the real chain — config via ``gps_parser`` (temp gpsconfig dir),
synthetic fixture data, the actual ``gps_analysis`` estimators with short
dev MCMC chains — and asserts the store products exist, match the
``gps_api.schemas`` shapes, carry provenance, and are served by the wired
``GET /v1/velocities`` endpoint. Everything runs in temp dirs, foreground,
no network.
"""

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from gps_api.main import create_app
from gps_api.precompute import main as precompute_main
from gps_api.precompute.products import PROVENANCE_METADATA_KEY
from gps_api.schemas import StationCollection, VelocityCollection

STATIONS = ("SENG", "ELDC")
REGION = "reykjanes"

STATIONS_CFG = """\
[SENG]
station_id = SENG
station_name = Svartsengi
latitude = 63.8721
longitude = -22.4353
height = 65.0

[ELDC]
station_id = ELDC
station_name = Eldvorp C
latitude = 63.8412
longitude = -22.5501
height = 41.2
"""

POSTPROCESS_CFG = """\
[PATHS]
data_prepath = /nonexistent/unused-by-this-slice/
"""

# Short dev chains: 16 * t_runs (annealing span) must stay < n_runs; the
# MCMC optimum is indicative only — the test asserts product shape +
# provenance, not estimator accuracy (gps_analysis owns that).
ANALYSIS_YAML = f"""\
version: 0
regions:
  {REGION}:
    description: test region
    stations: [{", ".join(STATIONS)}]
    default_reference_frame: ITRF2014
velocity:
  default_window_years: 2.0
  default_method: wls
detrend:
  default_model: lineperiodic
  overrides: {{}}
breakpoints:
  enabled_regions: [{REGION}]
  n_breaks_default: 1
  n_runs: 420
  t_runs: 20
"""

PROVENANCE_KEYS = {"method", "frame", "fitted_at", "source", "software"}


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the precompute CLI once (foreground) into a temp store."""
    config_dir = tmp_path_factory.mktemp("gpsconfig")
    (config_dir / "stations.cfg").write_text(STATIONS_CFG)
    (config_dir / "postprocess.cfg").write_text(POSTPROCESS_CFG)
    (config_dir / "analysis.yaml").write_text(ANALYSIS_YAML)
    store_dir = tmp_path_factory.mktemp("store")

    import os

    old = os.environ.get("GPS_CONFIG_PATH")
    os.environ["GPS_CONFIG_PATH"] = str(config_dir)
    try:
        rc = precompute_main(
            [
                "--region",
                REGION,
                "--store",
                str(store_dir),
                "--synthetic",
                "--days",
                "240",
                "--seed",
                "1",
            ]
        )
    finally:
        if old is None:
            os.environ.pop("GPS_CONFIG_PATH", None)
        else:
            os.environ["GPS_CONFIG_PATH"] = old
    assert rc == 0
    return store_dir


def test_velocities_product_matches_schema_and_carries_provenance(
    store: Path,
) -> None:
    path = store / "velocities" / f"{REGION}.geojson"
    assert path.is_file()
    payload = json.loads(path.read_text())

    collection = VelocityCollection.model_validate(payload)
    assert {f.properties.marker for f in collection.features} == set(STATIONS)
    for feature in collection.features:
        props = feature.properties
        assert props.method == "wls"
        assert props.window_start < props.window_end
        assert props.sigma_east > 0 and props.sigma_north > 0 and props.sigma_up > 0
        lon, lat = feature.geometry.coordinates[:2]
        assert -25 < lon < -20 and 63 < lat < 65  # [lon, lat] order kept

    provenance = payload["provenance"]
    assert PROVENANCE_KEYS <= provenance.keys()
    assert provenance["method"] == "wls"
    assert provenance["frame"] == "ITRF2014"
    assert provenance["fitted_at"].endswith("Z")
    assert set(provenance["software"]) == {"gps_api", "gps_analysis"}


def test_station_catalog_matches_schema(store: Path) -> None:
    payload = json.loads((store / "stations.geojson").read_text())
    catalog = StationCollection.model_validate(payload)
    assert {f.properties.marker for f in catalog.features} == set(STATIONS)
    assert all(REGION in f.properties.regions for f in catalog.features)
    assert PROVENANCE_KEYS <= payload["provenance"].keys()


def test_series_parquet_has_detrended_columns_and_provenance(store: Path) -> None:
    for marker in STATIONS:
        table = pq.read_table(store / "series" / f"{marker}.parquet")
        expected = {
            "time",
            "north",
            "east",
            "up",
            "sigma_north",
            "sigma_east",
            "sigma_up",
            "north_detrended",
            "east_detrended",
            "up_detrended",
        }
        assert expected <= set(table.column_names)
        assert table.num_rows == 240
        provenance = json.loads(table.schema.metadata[PROVENANCE_METADATA_KEY])
        assert PROVENANCE_KEYS <= provenance.keys()
        assert provenance["method"] == "lineperiodic"
        assert provenance["units"] == "mm"


def test_breaks_catalog_has_gbis_entries(store: Path) -> None:
    payload = json.loads((store / "models" / f"{REGION}_breaks.json").read_text())
    assert payload["region"] == REGION
    assert payload["provenance"]["method"] == "gbis"
    entries = payload["entries"]
    # one BPD1 entry per station and component
    assert len(entries) == len(STATIONS) * 3
    for entry in entries:
        assert entry["method"] == "gbis"
        assert entry["model"] == "BPD1"
        params = entry["parameters"]
        assert {
            "intercept_mm",
            "trend_mm_yr",
            "trend_change_mm_yr",
            "breakpoint_yearf",
            "kappa",
            "amp_mm",
        } <= params.keys()
        assert entry["breakpoint_time"].endswith("Z")


def test_run_meta_written(store: Path) -> None:
    meta = json.loads((store / "meta" / "run.json").read_text())
    assert meta["region"] == REGION
    assert set(meta["stations_ok"]) == set(STATIONS)
    assert meta["stations_failed"] == {}
    assert meta["source"].startswith("synthetic:")


def test_wired_velocities_endpoint_serves_the_store(
    store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contract exercised end-to-end: precompute wrote it, the API serves it."""
    monkeypatch.setenv("GPS_API_STORE", str(store))
    client = TestClient(create_app())

    resp = client.get("/v1/velocities", params={"region": REGION})
    assert resp.status_code == 200
    collection = VelocityCollection.model_validate(resp.json())
    assert {f.properties.marker for f in collection.features} == set(STATIONS)
    assert "provenance" not in resp.json()  # foreign member stays in the store

    # unfiltered = same single region merged
    resp_all = client.get("/v1/velocities")
    assert resp_all.status_code == 200
    assert len(resp_all.json()["features"]) == len(STATIONS)

    # window filter keeps the ~0.65 yr windows only when asked for them
    length_years = 240 / 365.25
    resp_win = client.get("/v1/velocities", params={"window_years": length_years})
    assert resp_win.status_code == 200
    assert len(resp_win.json()["features"]) == len(STATIONS)
    resp_off = client.get("/v1/velocities", params={"window_years": 5.0})
    assert resp_off.status_code == 200
    assert resp_off.json()["features"] == []

    resp_missing = client.get("/v1/velocities", params={"region": "langjokull"})
    assert resp_missing.status_code == 404
    assert isinstance(resp_missing.json()["detail"], str)
