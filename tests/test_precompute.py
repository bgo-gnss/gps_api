"""End-to-end tests of the precompute slices (single region + fleet).

Drives the real chain — config via ``gps_parser`` (temp gpsconfig dir),
synthetic fixture data, the actual ``gps_analysis`` estimators with short
dev MCMC chains (never the production 1e6) — and asserts the store
products exist, match the ``gps_api.schemas`` shapes, carry provenance,
and are served by the wired endpoints (``/v1/stations``,
``/v1/stations/{marker}/series``, ``/v1/velocities``,
``/v1/models/{region}``). Everything runs in temp dirs, foreground, no
network, no hand-listed real stations (plan §10.4 no-hardcoding rule —
stations/regions live only in the test's own gpsconfig fixtures).
"""

import json
import os
from collections.abc import Iterator
from datetime import UTC
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from gps_api.main import create_app
from gps_api.precompute import main as precompute_main
from gps_api.precompute import run_fleet
from gps_api.precompute.config import load_analysis_config
from gps_api.precompute.products import PROVENANCE_METADATA_KEY
from gps_api.precompute.sources import StationSeries, synthetic_station
from gps_api.schemas import (
    ModelResult,
    SeriesResponse,
    StationCollection,
    VelocityCollection,
)

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


# ---------------------------------------------------------------------------
# Fleet rollout (Phase-2 exit item): multi-region run into ONE coherent store.
# ---------------------------------------------------------------------------

FLEET_REGION_A = "reykjanes"  # break detection gated ON
FLEET_REGION_B = "vesturland"  # break detection gated OFF
FLEET_REGION_BAD = "draugasvaedi"  # station missing from stations.cfg → fails
FLEET_SHARED = "ELDC"  # configured in both A and B → merged regions list
FLEET_STATIONS_A = ("SENG", "ELDC")
FLEET_STATIONS_B = ("ELDC", "SKSH")
FLEET_MARKERS = ("SENG", "ELDC", "SKSH")
FLEET_DAYS = 240
FLEET_MAX_POINTS = 200  # api.max_points ceiling < FLEET_DAYS → LTTB kicks in

FLEET_STATIONS_CFG = f"""{STATIONS_CFG}
[SKSH]
station_id = SKSH
station_name = Skipastigshraun
latitude = 63.9105
longitude = -22.4801
height = 33.0
"""

FLEET_ANALYSIS_YAML = f"""\
version: 0
regions:
  {FLEET_REGION_A}:
    description: test region A (breaks gated on)
    stations: [{", ".join(FLEET_STATIONS_A)}]
    default_reference_frame: ITRF2014
  {FLEET_REGION_B}:
    description: test region B (breaks gated off)
    stations: [{", ".join(FLEET_STATIONS_B)}]
    default_reference_frame: ITRF2014
  {FLEET_REGION_BAD}:
    description: misconfigured region — station not in stations.cfg
    stations: [XXXX]
velocity:
  default_window_years: 2.0
  default_method: wls
detrend:
  default_model: lineperiodic
  overrides: {{}}
breakpoints:
  enabled_regions: [{FLEET_REGION_A}]
  n_breaks_default: 1
  n_runs: 420
  t_runs: 20
api:
  max_points: {FLEET_MAX_POINTS}
"""


@pytest.fixture(scope="module")
def fleet_config_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Temp gpsconfig dir with the multi-region analysis sidecar."""
    config_dir = tmp_path_factory.mktemp("gpsconfig-fleet")
    (config_dir / "stations.cfg").write_text(FLEET_STATIONS_CFG)
    (config_dir / "postprocess.cfg").write_text(POSTPROCESS_CFG)
    (config_dir / "analysis.yaml").write_text(FLEET_ANALYSIS_YAML)
    return config_dir


@pytest.fixture()
def fleet_gpsconfig_env(fleet_config_dir: Path) -> Iterator[Path]:
    """Point gps_parser at the fleet gpsconfig for one test."""
    old = os.environ.get("GPS_CONFIG_PATH")
    os.environ["GPS_CONFIG_PATH"] = str(fleet_config_dir)
    try:
        yield fleet_config_dir
    finally:
        if old is None:
            os.environ.pop("GPS_CONFIG_PATH", None)
        else:
            os.environ["GPS_CONFIG_PATH"] = old


@pytest.fixture(scope="module")
def fleet_store(
    tmp_path_factory: pytest.TempPathFactory, fleet_config_dir: Path
) -> Path:
    """Run the fleet CLI once (``--fleet``) into a temp store.

    Exit code is 1 by design: the misconfigured region fails and is
    recorded — one bad region must not sink the fleet.
    """
    store_dir = tmp_path_factory.mktemp("store-fleet")
    old = os.environ.get("GPS_CONFIG_PATH")
    os.environ["GPS_CONFIG_PATH"] = str(fleet_config_dir)
    try:
        rc = precompute_main(
            [
                "--fleet",
                "--store",
                str(store_dir),
                "--synthetic",
                "--days",
                str(FLEET_DAYS),
                "--seed",
                "1",
            ]
        )
    finally:
        if old is None:
            os.environ.pop("GPS_CONFIG_PATH", None)
        else:
            os.environ["GPS_CONFIG_PATH"] = old
    assert rc == 1  # the bad region is recorded, not fatal
    return store_dir


@pytest.fixture()
def fleet_client(fleet_store: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """API client pointed at the fleet store."""
    monkeypatch.setenv("GPS_API_STORE", str(fleet_store))
    return TestClient(create_app())


def test_fleet_catalog_spans_regions_and_merges_memberships(
    fleet_store: Path,
) -> None:
    payload = json.loads((fleet_store / "stations.geojson").read_text())
    catalog = StationCollection.model_validate(payload)
    by_marker = {f.properties.marker: f.properties for f in catalog.features}
    assert set(by_marker) == set(FLEET_MARKERS)
    assert by_marker[FLEET_SHARED].regions == sorted([FLEET_REGION_A, FLEET_REGION_B])
    assert by_marker["SENG"].regions == [FLEET_REGION_A]
    assert by_marker["SKSH"].regions == [FLEET_REGION_B]
    provenance = payload["provenance"]
    assert PROVENANCE_KEYS <= provenance.keys()
    assert provenance["regions"] == sorted([FLEET_REGION_A, FLEET_REGION_B])


def test_fleet_velocities_written_per_region(fleet_store: Path) -> None:
    for region, stations in (
        (FLEET_REGION_A, FLEET_STATIONS_A),
        (FLEET_REGION_B, FLEET_STATIONS_B),
    ):
        payload = json.loads(
            (fleet_store / "velocities" / f"{region}.geojson").read_text()
        )
        collection = VelocityCollection.model_validate(payload)
        assert {f.properties.marker for f in collection.features} == set(stations)
    assert not (fleet_store / "velocities" / f"{FLEET_REGION_BAD}.geojson").exists()


def test_fleet_break_detection_honors_the_region_gate(fleet_store: Path) -> None:
    """GBIS4TS runs ONLY for breakpoints.enabled_regions — never fleet-wide."""
    models = sorted(p.name for p in (fleet_store / "models").glob("*_breaks.json"))
    assert models == [f"{FLEET_REGION_A}_breaks.json"]


def test_fleet_meta_summarizes_regions_and_stations(fleet_store: Path) -> None:
    meta = json.loads((fleet_store / "meta" / "run.json").read_text())
    assert meta["fleet"] is True
    assert meta["fitted_at"].endswith("Z")
    assert set(meta["regions"]) == {FLEET_REGION_A, FLEET_REGION_B}
    for region, stations in (
        (FLEET_REGION_A, FLEET_STATIONS_A),
        (FLEET_REGION_B, FLEET_STATIONS_B),
    ):
        assert set(meta["regions"][region]["stations_ok"]) == set(stations)
        assert meta["regions"][region]["stations_failed"] == {}
    assert set(meta["regions_failed"]) == {FLEET_REGION_BAD}
    assert "KeyError" in meta["regions_failed"][FLEET_REGION_BAD]
    assert meta["totals"] == {
        "regions_ok": 2,
        "regions_failed": 1,
        "stations_ok": 4,
        "stations_failed": 0,
    }
    assert meta["api"] == {"max_points": FLEET_MAX_POINTS}


def test_run_fleet_survives_station_failure(
    fleet_gpsconfig_env: Path, tmp_path: Path
) -> None:
    """One bad *station* must not sink its region, nor the fleet."""
    cfg = load_analysis_config()

    def flaky_loader(marker: str) -> StationSeries:
        if marker == "SKSH":
            raise RuntimeError("simulated station outage")
        return synthetic_station(marker, seed=1, n_days=120)

    summary = run_fleet(
        cfg,
        flaky_loader,
        tmp_path,
        "synthetic:flaky-test",
        detect_breaks=False,  # velocity/series path is what's under test
        seed=1,
    )
    assert set(summary.regions) == {FLEET_REGION_A, FLEET_REGION_B}
    assert summary.regions[FLEET_REGION_B].stations_ok == (FLEET_SHARED,)
    assert set(summary.regions[FLEET_REGION_B].stations_failed) == {"SKSH"}
    assert set(summary.regions_failed) == {FLEET_REGION_BAD}
    assert summary.stations_ok_total == 3  # SENG + ELDC (A) + ELDC (B)
    assert summary.stations_failed_total == 1
    # The failed station produced no series product.
    assert not (tmp_path / "series" / "SKSH.parquet").exists()
    meta = json.loads((tmp_path / "meta" / "run.json").read_text())
    assert meta["regions"][FLEET_REGION_B]["stations_failed"].keys() == {"SKSH"}


# --- The wired endpoints, served from the fleet store ----------------------


def test_stations_endpoint_serves_the_catalog(fleet_client: TestClient) -> None:
    resp = fleet_client.get("/v1/stations")
    assert resp.status_code == 200
    catalog = StationCollection.model_validate(resp.json())
    by_marker = {f.properties.marker: f.properties for f in catalog.features}
    assert set(by_marker) == set(FLEET_MARKERS)
    assert by_marker[FLEET_SHARED].regions == sorted([FLEET_REGION_A, FLEET_REGION_B])
    assert "provenance" not in resp.json()  # foreign member stays in the store


def test_series_endpoint_serves_the_series_shape(
    fleet_client: TestClient,
) -> None:
    resp = fleet_client.get("/v1/stations/SENG/series")
    assert resp.status_code == 200
    series = SeriesResponse.model_validate(resp.json())
    assert series.marker == "SENG"
    assert series.frame == "ITRF2014"
    assert series.units == "mm"
    assert series.detrended is True
    n = len(series.time)
    assert n == FLEET_MAX_POINTS  # 240 epochs, store ceiling 200 → LTTB
    for values in (series.north, series.east, series.up):
        assert len(values) == n
    assert series.sigma_north is not None and len(series.sigma_north) == n
    stamps = [t.timestamp() for t in series.time]
    assert stamps == sorted(stamps)


def test_series_endpoint_lttb_target_and_ceiling(
    fleet_client: TestClient,
) -> None:
    """analysis.yaml api.max_points is a server-side LTTB ceiling.

    It applies by default (previous test), clamps an oversized client
    request, and leaves smaller client targets untouched. LTTB always
    keeps the first and last epoch of the served window, so even the
    clamped responses expose the true series endpoints.
    """
    default = SeriesResponse.model_validate(
        fleet_client.get("/v1/stations/SENG/series").json()
    )
    resp_big = fleet_client.get(
        "/v1/stations/SENG/series", params={"max_points": 100_000}
    )
    assert len(resp_big.json()["time"]) == FLEET_MAX_POINTS  # clamped

    resp_small = fleet_client.get("/v1/stations/SENG/series", params={"max_points": 50})
    series = SeriesResponse.model_validate(resp_small.json())
    assert len(series.time) == 50
    # Endpoint preservation: both selections keep the true first/last epoch.
    assert series.time[0] == default.time[0]
    assert series.time[-1] == default.time[-1]


def test_series_endpoint_window_filter(fleet_client: TestClient) -> None:
    # LTTB selects *real* epochs, so any served time is a valid window edge.
    default = SeriesResponse.model_validate(
        fleet_client.get("/v1/stations/ELDC/series").json()
    )
    start, end = default.time[0], default.time[40]
    resp = fleet_client.get(
        "/v1/stations/ELDC/series",
        params={"start": start.isoformat(), "end": end.isoformat()},
    )
    assert resp.status_code == 200
    series = SeriesResponse.model_validate(resp.json())
    n_window = len(series.time)
    assert 2 < n_window < FLEET_MAX_POINTS  # a real subrange, not clamped
    assert series.time[0] == start and series.time[-1] == end  # inclusive
    assert min(series.time) >= start and max(series.time) <= end
    # Naive datetimes are interpreted as UTC (contract times are UTC).
    naive = start.astimezone(UTC).replace(tzinfo=None)
    resp_naive = fleet_client.get(
        "/v1/stations/ELDC/series",
        params={"start": naive.isoformat(), "end": end.isoformat()},
    )
    assert len(resp_naive.json()["time"]) == n_window


def test_series_endpoint_raw_vs_detrended(fleet_client: TestClient) -> None:
    # Use an un-downsampled window: LTTB indices depend on the values, so
    # raw and detrended selections would otherwise differ by construction.
    default = SeriesResponse.model_validate(
        fleet_client.get("/v1/stations/SENG/series").json()
    )
    window = {
        "start": default.time[0].isoformat(),
        "end": default.time[80].isoformat(),
    }
    raw = SeriesResponse.model_validate(
        fleet_client.get(
            "/v1/stations/SENG/series", params={**window, "detrended": False}
        ).json()
    )
    det = SeriesResponse.model_validate(
        fleet_client.get(
            "/v1/stations/SENG/series", params={**window, "detrended": True}
        ).json()
    )
    assert raw.detrended is False and det.detrended is True
    assert raw.time == det.time  # same epochs — no value-driven selection
    assert raw.north != det.north  # residuals differ from raw displacements
    # Observation σ is the same either way (detrending shifts the model,
    # not the measurement uncertainty).
    assert raw.sigma_north == det.sigma_north


def test_series_endpoint_pre_a8_store_is_nullable(fleet_client: TestClient) -> None:
    """A8 backwards compatibility: no outliers: block → no flag columns.

    A store whose precompute ran without the outlier stage serves
    ``outlier=null`` / ``outlier_provenance=null`` and ``clean=true`` is a
    no-op (there is nothing to drop) — pre-A8 consumers see the exact
    payload they always did.
    """
    default = SeriesResponse.model_validate(
        fleet_client.get("/v1/stations/SENG/series").json()
    )
    assert default.clean is False
    assert default.outlier is None
    assert default.outlier_provenance is None
    cleaned = SeriesResponse.model_validate(
        fleet_client.get("/v1/stations/SENG/series", params={"clean": True}).json()
    )
    assert cleaned.clean is True
    assert cleaned.time == default.time  # nothing flagged, nothing dropped
    assert cleaned.outlier is None


def test_run_meta_has_no_outlier_keys_when_stage_is_off(fleet_store: Path) -> None:
    """The outliers_* run.json keys appear only when the stage ran."""
    meta = json.loads((fleet_store / "meta" / "run.json").read_text())
    for region_meta in meta["regions"].values():
        assert "outliers_aborted" not in region_meta
        assert "outliers_failed" not in region_meta
    assert "outliers_failed" not in meta["totals"]
    assert not (fleet_store / "meta" / "suspected_steps.csv").exists()


def test_series_parquet_has_no_flag_columns_when_stage_is_off(
    fleet_store: Path,
) -> None:
    table = pq.read_table(fleet_store / "series" / "SENG.parquet")
    assert not {c for c in table.column_names if "outlier" in c}


def test_series_endpoint_unknown_marker_is_404(fleet_client: TestClient) -> None:
    resp = fleet_client.get("/v1/stations/QQQQ/series")
    assert resp.status_code == 404
    assert isinstance(resp.json()["detail"], str)


def test_models_endpoint_serves_the_break_catalog(
    fleet_client: TestClient,
) -> None:
    resp = fleet_client.get(f"/v1/models/{FLEET_REGION_A}")
    assert resp.status_code == 200
    result = ModelResult.model_validate(resp.json())
    assert result.region == FLEET_REGION_A
    assert result.kind == "breakpoints"
    assert result.parameters == {}  # source parameters are the mogi kind's
    assert result.entries is not None
    # one BPD1 entry per station and component
    assert len(result.entries) == len(FLEET_STATIONS_A) * 3
    assert {e.marker for e in result.entries} == set(FLEET_STATIONS_A)
    assert {e.component for e in result.entries} == {"north", "east", "up"}
    for entry in result.entries:
        assert entry.method == "gbis"
        assert entry.model == "BPD1"
        assert {
            "intercept_mm",
            "trend_mm_yr",
            "trend_change_mm_yr",
            "breakpoint_yearf",
            "kappa",
            "amp_mm",
        } <= entry.parameters.keys()
        assert entry.n_runs == 420
    assert isinstance(result.provenance, dict)
    assert result.provenance["method"] == "gbis"
    assert result.fitted_at.tzinfo is not None


def test_models_endpoint_ungated_region_is_404(fleet_client: TestClient) -> None:
    """No break products for a region outside breakpoints.enabled_regions."""
    resp = fleet_client.get(f"/v1/models/{FLEET_REGION_B}")
    assert resp.status_code == 404
    assert isinstance(resp.json()["detail"], str)


def test_models_history_stays_reserved_501(fleet_client: TestClient) -> None:
    resp = fleet_client.get(f"/v1/models/{FLEET_REGION_A}/history")
    assert resp.status_code == 501
    assert isinstance(resp.json()["detail"], str)


def test_velocities_endpoint_merges_fleet_regions(
    fleet_client: TestClient,
) -> None:
    resp = fleet_client.get("/v1/velocities")
    assert resp.status_code == 200
    collection = VelocityCollection.model_validate(resp.json())
    # ELDC is in both regions → one feature per (region, station) pair.
    assert len(collection.features) == len(FLEET_STATIONS_A) + len(FLEET_STATIONS_B)
    resp_a = fleet_client.get("/v1/velocities", params={"region": FLEET_REGION_A})
    assert {f["properties"]["marker"] for f in resp_a.json()["features"]} == set(
        FLEET_STATIONS_A
    )
