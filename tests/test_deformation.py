"""End-to-end tests of the Mogi deformation + MLE velocity productization.

Drives the real chain (contract Amendments A5/A6): a synthetic multi-region
fleet where one region carries a known Mogi inflation ramp and configures
``velocity_method: mle`` + the ``deformation:`` stage (with a short Bayesian
tail), and a second region stays on the WLS baseline with no deformation
gate. Asserts the store products exist, match the ``gps_api.schemas``
shapes, recover the planted source (loose physics bounds — estimator
accuracy is ``gps_analysis``'s to pin), carry provenance, honor the config
gates, survive station/stage failures, and are served by
``GET /v1/deformation/{region}``. Everything runs in temp dirs, foreground,
no network, short chains, no hand-listed real stations (plan §10.4 —
stations/regions live only in this test's own gpsconfig fixtures).
"""

import json
import os
import zlib
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from gps_analysis import MogiSource, local_coordinates, mogi_forward

from gps_api.main import create_app
from gps_api.precompute import run_fleet, run_precompute
from gps_api.precompute.config import (
    AnalysisConfig,
    BreakpointConfig,
    DeformationConfig,
    RegionConfig,
    load_analysis_config,
)
from gps_api.precompute.sources import StationSeries, synthetic_station
from gps_api.schemas import DeformationResult, VelocityCollection

MOGI_REGION = "svartsengi"  # mle velocities + deformation gated ON
WLS_REGION = "hengill"  # wls baseline, deformation gated OFF

MOGI_STATIONS: dict[str, tuple[float, float]] = {
    "SENG": (-22.4353, 63.8721),
    "ELDC": (-22.5501, 63.8412),
    "SKSH": (-22.4801, 63.9105),
    "GRIC": (-22.3702, 63.8608),
}
WLS_STATIONS: dict[str, tuple[float, float]] = {
    "HVER": (-21.3000, 64.0200),
    "OLKE": (-21.4000, 64.0700),
}

# Planted truth: a Mogi source inflating linearly under the network.
TRUTH_LON, TRUTH_LAT = -22.4500, 63.8700
TRUTH_DEPTH_M = 4000.0
TRUTH_DV_RATE_M3_YR = 6.0e6
T0, N_DAYS, NOISE_MM = 2024.0, 240, 1.0

DEFORMATION_WINDOW_YEARS = 0.5
BAYES_N_RUNS, BAYES_T_RUNS = 800, 30

STATIONS_CFG = "\n".join(
    f"[{marker}]\n"
    f"station_id = {marker}\n"
    f"station_name = {marker}\n"
    f"latitude = {lat}\n"
    f"longitude = {lon}\n"
    f"height = 50.0\n"
    for marker, (lon, lat) in {**MOGI_STATIONS, **WLS_STATIONS}.items()
)

POSTPROCESS_CFG = """\
[PATHS]
data_prepath = /nonexistent/unused-by-this-slice/
"""

ANALYSIS_YAML = f"""\
version: 0
regions:
  {MOGI_REGION}:
    description: mogi test region (mle velocities, deformation gated on)
    stations: [{", ".join(MOGI_STATIONS)}]
    default_reference_frame: ITRF2014
    velocity_method: mle
  {WLS_REGION}:
    description: baseline region (wls, no deformation)
    stations: [{", ".join(WLS_STATIONS)}]
    default_reference_frame: ITRF2014
velocity:
  default_window_years: 2.0
  default_method: wls
  kappa_bounds: [-2.5, 0.0]
detrend:
  default_model: linear
  overrides: {{}}
breakpoints:
  enabled_regions: []
  n_breaks_default: 1
  n_runs: 420
  t_runs: 20
deformation:
  enabled_regions: [{MOGI_REGION}]
  source: mogi
  series: raw
  window_years: {DEFORMATION_WINDOW_YEARS}
  step_days: 30
  epoch_mean_days: 11
  min_stations: 3
  depth_bounds_km: [0.5, 15.0]
  dv_bounds_m3: [-5.0e7, 5.0e7]
  bayes:
    n_runs: {BAYES_N_RUNS}
    t_runs: {BAYES_T_RUNS}
"""

PROVENANCE_KEYS = {"method", "frame", "fitted_at", "source", "software"}


def mogi_loader(marker: str) -> StationSeries:
    """Synthetic series: known Mogi ramp for the gated region, plain fixture
    elsewhere — all forward modeling from ``gps_analysis`` (no math here)."""
    if marker in WLS_STATIONS:
        return synthetic_station(marker, seed=1, n_days=N_DAYS)
    lon, lat = MOGI_STATIONS[marker]
    e, n = local_coordinates(lon, lat, TRUTH_LON, TRUTH_LAT)
    t = T0 + np.arange(N_DAYS, dtype=np.float64) / 365.25
    rng = np.random.default_rng([7, zlib.crc32(marker.encode())])
    y = np.zeros((3, N_DAYS))
    for k, tk in enumerate(t):
        dv = TRUTH_DV_RATE_M3_YR * (float(tk) - T0)
        u = mogi_forward(  # rows (east, north, up), metres
            np.atleast_1d(e),
            np.atleast_1d(n),
            MogiSource(0.0, 0.0, TRUTH_DEPTH_M, dv),
        )
        y[0, k] = u[1, 0] * 1e3  # north, mm
        y[1, k] = u[0, 0] * 1e3  # east, mm
        y[2, k] = u[2, 0] * 1e3  # up, mm
    y += rng.normal(0.0, NOISE_MM, y.shape)
    return StationSeries(
        marker=marker,
        t=t,
        y=y,
        sigma=np.full_like(y, NOISE_MM),
        source="synthetic:mogi-ramp",
    )


@pytest.fixture(scope="module")
def config_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Temp gpsconfig dir with the deformation/MLE analysis sidecar."""
    config_dir = tmp_path_factory.mktemp("gpsconfig-deformation")
    (config_dir / "stations.cfg").write_text(STATIONS_CFG)
    (config_dir / "postprocess.cfg").write_text(POSTPROCESS_CFG)
    (config_dir / "analysis.yaml").write_text(ANALYSIS_YAML)
    return config_dir


@pytest.fixture()
def gpsconfig_env(config_dir: Path) -> Iterator[Path]:
    """Point gps_parser at this test's gpsconfig for one test."""
    old = os.environ.get("GPS_CONFIG_PATH")
    os.environ["GPS_CONFIG_PATH"] = str(config_dir)
    try:
        yield config_dir
    finally:
        if old is None:
            os.environ.pop("GPS_CONFIG_PATH", None)
        else:
            os.environ["GPS_CONFIG_PATH"] = old


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory, config_dir: Path) -> Path:
    """Run the fleet once (both regions, one coherent store)."""
    store_dir = tmp_path_factory.mktemp("store-deformation")
    old = os.environ.get("GPS_CONFIG_PATH")
    os.environ["GPS_CONFIG_PATH"] = str(config_dir)
    try:
        cfg = load_analysis_config()
        fleet = run_fleet(cfg, mogi_loader, store_dir, "synthetic:mogi-ramp", seed=2)
    finally:
        if old is None:
            os.environ.pop("GPS_CONFIG_PATH", None)
        else:
            os.environ["GPS_CONFIG_PATH"] = old
    assert not fleet.regions_failed
    assert fleet.stations_failed_total == 0
    assert fleet.deformation_failed_total == 0
    return store_dir


@pytest.fixture()
def client(store: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """API client pointed at the deformation store."""
    monkeypatch.setenv("GPS_API_STORE", str(store))
    return TestClient(create_app())


# --- Product: gating, schema, provenance, source recovery -------------------


def test_deformation_product_honors_the_region_gate(store: Path) -> None:
    """The Mogi stage runs ONLY for deformation.enabled_regions."""
    products = sorted(p.name for p in (store / "models").glob("*_deformation.json"))
    assert products == [f"{MOGI_REGION}_deformation.json"]


def test_deformation_product_schema_and_provenance(store: Path) -> None:
    payload = json.loads(
        (store / "models" / f"{MOGI_REGION}_deformation.json").read_text()
    )
    result = DeformationResult.model_validate(payload)
    assert result.region == MOGI_REGION
    assert result.source_type == "mogi"
    assert result.series_kind == "raw"
    assert result.stations == sorted(MOGI_STATIONS)
    assert result.fitted_at.tzinfo is not None

    provenance = payload["provenance"]
    assert PROVENANCE_KEYS <= provenance.keys()
    assert provenance["method"] == "mogi"
    assert provenance["frame"] == "ITRF2014"
    assert provenance["epochs_skipped"] == 0
    assert provenance["stations_excluded"] == []
    assert provenance["bayes"] == {
        "n_runs": BAYES_N_RUNS,
        "t_runs": BAYES_T_RUNS,
        "seed": 2,
    }
    # Independent GNSS-only product: cross-checked against Vincent's
    # operational Mogi dV(t), never derived from his files.
    assert "inv_volume_mogi.dat" in provenance["cross_check"]


def test_deformation_series_recovers_the_planted_ramp(store: Path) -> None:
    """The ΔV(t) product tracks the planted inflation (loose physics bounds
    — estimator accuracy itself is pinned in gps_analysis)."""
    result = DeformationResult.model_validate_json(
        (store / "models" / f"{MOGI_REGION}_deformation.json").read_text()
    )
    fits = result.fits
    assert len(fits) >= 4
    times = [f.time for f in fits]
    assert times == sorted(times)
    assert all(f.time > result.reference_time for f in fits)
    assert all(f.n_stations == len(MOGI_STATIONS) for f in fits)
    assert all(f.sigma_dv_m3 > 0 and f.sigma_depth_km > 0 for f in fits)

    # Inflation ramp: ΔV grows monotonically and matches the planted rate.
    dv = [f.dv_m3 for f in fits]
    assert dv == sorted(dv)
    span_years = (times[-1] - result.reference_time).total_seconds() / (
        86400.0 * 365.25
    )
    truth_final = TRUTH_DV_RATE_M3_YR * span_years
    assert dv[-1] == pytest.approx(truth_final, rel=0.2)
    # Geometry of the newest fit: right place, right depth (order-of-mag).
    final = fits[-1]
    assert final.lon == pytest.approx(TRUTH_LON, abs=0.02)
    assert final.lat == pytest.approx(TRUTH_LAT, abs=0.02)
    assert 2.0 < final.depth_km < 6.0


def test_deformation_posterior_summarizes_the_final_epoch(store: Path) -> None:
    result = DeformationResult.model_validate_json(
        (store / "models" / f"{MOGI_REGION}_deformation.json").read_text()
    )
    posterior = result.posterior
    assert posterior is not None
    assert posterior.time == result.fits[-1].time
    assert posterior.n_runs == BAYES_N_RUNS
    assert posterior.burn_in == 16 * BAYES_T_RUNS
    assert set(posterior.optimal) == {"east_m", "north_m", "depth_km", "dv_m3"}
    for key, levels in posterior.percentiles.items():
        assert set(levels) == {"p2_5", "p16", "p50", "p84", "p97_5"}
        assert levels["p2_5"] <= levels["p16"] <= levels["p50"], key
        assert levels["p50"] <= levels["p84"] <= levels["p97_5"], key
    # The posterior median agrees with the final least-squares ΔV.
    assert posterior.percentiles["dv_m3"]["p50"] == pytest.approx(
        result.fits[-1].dv_m3, rel=0.25
    )


# --- MLE velocities (Amendment A5) ------------------------------------------


def test_mle_region_velocities_carry_method_and_noise(store: Path) -> None:
    payload = json.loads((store / "velocities" / f"{MOGI_REGION}.geojson").read_text())
    collection = VelocityCollection.model_validate(payload)
    assert {f.properties.marker for f in collection.features} == set(MOGI_STATIONS)
    for feature in collection.features:
        props = feature.properties
        assert props.method == "mle"
        assert props.sigma_east > 0 and props.sigma_north > 0 and props.sigma_up > 0
        assert props.noise is not None
        assert set(props.noise) == {"north", "east", "up"}
        for model in props.noise.values():
            assert model.sigma_white_mm >= 0
            assert model.amplitude_mm >= 0
            assert -2.5 <= model.spectral_index <= 0.0
    provenance = payload["provenance"]
    assert provenance["method"] == "mle"
    assert provenance["kappa_bounds"] == [-2.5, 0.0]


def test_wls_region_stays_on_the_baseline(store: Path) -> None:
    payload = json.loads((store / "velocities" / f"{WLS_REGION}.geojson").read_text())
    collection = VelocityCollection.model_validate(payload)
    for feature in collection.features:
        assert feature.properties.method == "wls"
        assert feature.properties.noise is None
    assert payload["provenance"]["method"] == "wls"


def test_fleet_meta_has_no_deformation_failures(store: Path) -> None:
    meta = json.loads((store / "meta" / "run.json").read_text())
    assert meta["fleet"] is True
    assert set(meta["regions"]) == {MOGI_REGION, WLS_REGION}
    for region_summary in meta["regions"].values():
        assert "deformation_failed" not in region_summary
    deformation_products = [
        p for p in meta["regions"][MOGI_REGION]["products"] if "_deformation" in p
    ]
    assert len(deformation_products) == 1


# --- The wired endpoint ------------------------------------------------------


def test_deformation_endpoint_serves_and_validates(client: TestClient) -> None:
    resp = client.get(f"/v1/deformation/{MOGI_REGION}")
    assert resp.status_code == 200
    result = DeformationResult.model_validate(resp.json())
    assert result.region == MOGI_REGION
    assert result.source_type == "mogi"
    assert len(result.fits) >= 4
    assert result.posterior is not None
    # Structured provenance rides on the payload (schema field, A6).
    assert isinstance(result.provenance, dict)
    assert result.provenance["method"] == "mogi"


def test_deformation_endpoint_ungated_region_is_404(client: TestClient) -> None:
    resp = client.get(f"/v1/deformation/{WLS_REGION}")
    assert resp.status_code == 404
    assert isinstance(resp.json()["detail"], str)


# --- Fault tolerance ---------------------------------------------------------


def test_station_failure_does_not_sink_the_deformation_stage(
    gpsconfig_env: Path, tmp_path: Path
) -> None:
    """One bad station is skipped; min_stations still met → product written."""
    cfg = load_analysis_config()

    def flaky(marker: str) -> StationSeries:
        if marker == "GRIC":
            raise RuntimeError("simulated station outage")
        return mogi_loader(marker)

    summary = run_precompute(
        cfg, MOGI_REGION, flaky, tmp_path, "synthetic:flaky", seed=2
    )
    assert set(summary.stations_failed) == {"GRIC"}
    assert summary.deformation_failed is None
    result = DeformationResult.model_validate_json(
        (tmp_path / "models" / f"{MOGI_REGION}_deformation.json").read_text()
    )
    assert result.stations == sorted(set(MOGI_STATIONS) - {"GRIC"})
    assert all(f.n_stations == 3 for f in result.fits)


def test_deformation_stage_failure_is_recorded_not_fatal(
    gpsconfig_env: Path, tmp_path: Path
) -> None:
    """Below min_stations the stage fails — recorded, region products survive."""
    cfg = load_analysis_config()

    def very_flaky(marker: str) -> StationSeries:
        if marker in ("GRIC", "SKSH"):
            raise RuntimeError("simulated station outage")
        return mogi_loader(marker)

    summary = run_precompute(
        cfg, MOGI_REGION, very_flaky, tmp_path, "synthetic:very-flaky", seed=2
    )
    assert set(summary.stations_failed) == {"GRIC", "SKSH"}
    assert summary.deformation_failed is not None
    assert "min_stations" not in summary.deformation_failed  # message is prose
    assert not (tmp_path / "models" / f"{MOGI_REGION}_deformation.json").exists()
    # The region's other products survived the stage failure.
    assert (tmp_path / "velocities" / f"{MOGI_REGION}.geojson").is_file()
    assert (tmp_path / "series" / "SENG.parquet").is_file()
    meta = json.loads((tmp_path / "meta" / "run.json").read_text())
    assert meta["deformation_failed"] == summary.deformation_failed


# --- Config validation -------------------------------------------------------


def test_deformation_config_rejects_unimplemented_source() -> None:
    with pytest.raises(ValueError, match="okada"):
        DeformationConfig(enabled_regions=("x",), source="okada")


def test_deformation_config_rejects_bad_series_and_bounds() -> None:
    with pytest.raises(ValueError, match="series"):
        DeformationConfig(enabled_regions=(), series="filtered")
    with pytest.raises(ValueError, match="depth_bounds_km"):
        DeformationConfig(enabled_regions=(), depth_bounds_km=(5.0, 1.0))
    with pytest.raises(ValueError, match="min_stations"):
        DeformationConfig(enabled_regions=(), min_stations=1)


def test_deformation_config_bayes_needs_finite_priors_and_room() -> None:
    with pytest.raises(ValueError, match="dv_bounds_m3"):
        DeformationConfig(enabled_regions=(), bayes_n_runs=800)
    with pytest.raises(ValueError, match="annealing"):
        DeformationConfig(
            enabled_regions=(),
            dv_bounds_m3=(-1e7, 1e7),
            bayes_n_runs=100,
            bayes_t_runs=100,
        )


def test_velocity_method_gbis_stays_a_later_slice(tmp_path: Path) -> None:
    cfg = AnalysisConfig(
        config_dir=tmp_path,
        analysis_yaml=tmp_path / "analysis.yaml",
        regions={
            "r": RegionConfig(
                name="r",
                stations=("AAAA",),
                reference_frame="ITRF2014",
                velocity_method="gbis",
            )
        },
        velocity_window_years=2.0,
        velocity_method="wls",
        detrend_model="linear",
        detrend_overrides={},
        breakpoints=BreakpointConfig(
            enabled_regions=(), n_breaks=1, n_runs=420, t_runs=20
        ),
    )
    with pytest.raises(ValueError, match="gbis"):
        cfg.velocity_method_for("r")
