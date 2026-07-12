"""End-to-end tests of the Okada distributed-slip productization (A7).

Drives the real chain: a synthetic multi-region fleet where one region
configures ``source: okada`` with an operator-supplied fault/dike plane and a
grid of stations, and a second region stays on the WLS baseline with no
deformation gate. A known opening distribution is imposed on the configured
plane and forward-modelled (``gps_analysis``) into a net-displacement field
(a linear ramp so the window-end − window-start net equals the planted
field); the pipeline must recover the pattern (correlation, potency ratio,
peak patch), write a schema-valid ``models/<region>_slip.json`` product with
provenance, serve it from ``GET /v1/deformation/{region}`` (discriminated from
the Mogi product by ``source_type``), honor the config gate, survive
station/stage failures, and validate the operator-supplied plane at config
time. The per-patch formal covariance is verified to be assembled exactly as
the leaf's estimator does. Everything runs in temp dirs, foreground, no
network, small grid; stations/regions live only in this test's own gpsconfig
fixtures (plan §10.4 — no hand-listed real stations).
"""

import json
import os
import zlib
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from gps_analysis import (
    OkadaSource,
    discretize_fault,
    local_coordinates,
    okada_greens,
    okada_invert_slip,
    patch_laplacian,
)

from gps_api.main import create_app
from gps_api.precompute import run_fleet, run_precompute
from gps_api.precompute.config import (
    DeformationConfig,
    OkadaPlaneConfig,
    load_analysis_config,
)
from gps_api.precompute.products import Provenance
from gps_api.precompute.sources import StationSeries, synthetic_station
from gps_api.schemas import (
    DeformationResult,
    SlipDistributionResult,
    VelocityCollection,
)

OKADA_REGION = "reykjanes_dike"  # distributed-slip gated ON
WLS_REGION = "hengill"  # wls baseline, deformation gated OFF

# Operator-supplied plane (config-driven; the geometry is NOT auto-found).
PLANE_LON, PLANE_LAT = -22.45, 63.87  # centroid
STRIKE, DIP = 40.0, 80.0
LENGTH_KM, WIDTH_KM, TOP_DEPTH_KM = 8.0, 4.0, 1.0
N_STRIKE, N_DIP = 5, 3

# Planted truth: a Gaussian opening bump on the (N_DIP × N_STRIKE) grid,
# peak at (row=1, col=2), 1.0 m opening.
_ii, _jj = np.meshgrid(np.arange(N_STRIKE), np.arange(N_DIP), indexing="xy")
S_TRUE = 1.0 * np.exp(-(((_ii - 2.0) / 1.1) ** 2 + ((_jj - 1.0) / 0.9) ** 2))
S_TRUE_FLAT = S_TRUE.ravel()

T0, N_DAYS, NOISE_MM = 2024.0, 240, 0.5
WINDOW_YEARS = 0.5

# Station grid surrounding the plane (5×5). Markers are fixture-only.
_LONS = np.linspace(PLANE_LON - 0.12, PLANE_LON + 0.12, 5)
_LATS = np.linspace(PLANE_LAT - 0.06, PLANE_LAT + 0.06, 5)
OKADA_STATIONS: dict[str, tuple[float, float]] = {
    f"D{r}{c}": (float(lon), float(lat))
    for r, lat in enumerate(_LATS)
    for c, lon in enumerate(_LONS)
}
WLS_STATIONS: dict[str, tuple[float, float]] = {
    "HVER": (-21.3000, 64.0200),
    "OLKE": (-21.4000, 64.0700),
}


def _planted_plane() -> OkadaSource:
    """The planted plane as an OkadaSource (centroid depth from top depth)."""
    centroid_depth_m = (
        TOP_DEPTH_KM + np.sin(np.radians(DIP)) * WIDTH_KM / 2.0
    ) * 1000.0
    return OkadaSource(
        x=0.0,
        y=0.0,
        depth=centroid_depth_m,
        strike=STRIKE,
        dip=DIP,
        length=LENGTH_KM * 1000.0,
        width=WIDTH_KM * 1000.0,
        strike_slip=0.0,
        dip_slip=0.0,
        opening=0.0,
    )


def _planted_fields() -> dict[str, np.ndarray]:
    """Net (window-end) displacement per station from the planted opening.

    ``G·s_true`` at every station, rows (east, north, up) [m]. All forward
    modelling from ``gps_analysis`` (no math derived in the test)."""
    markers = sorted(OKADA_STATIONS)
    en = [
        local_coordinates(
            OKADA_STATIONS[m][0], OKADA_STATIONS[m][1], PLANE_LON, PLANE_LAT
        )
        for m in markers
    ]
    e = np.array([float(v[0]) for v in en])
    n = np.array([float(v[1]) for v in en])
    patches = discretize_fault(_planted_plane(), N_STRIKE, N_DIP)
    g = okada_greens(e, n, patches, ("opening",))
    d = (g @ S_TRUE_FLAT).reshape(3, len(markers))  # east, north, up [m]
    return {m: d[:, i] for i, m in enumerate(markers)}


PLANTED = _planted_fields()

STATIONS_CFG = "\n".join(
    f"[{marker}]\n"
    f"station_id = {marker}\n"
    f"station_name = {marker}\n"
    f"latitude = {lat}\n"
    f"longitude = {lon}\n"
    f"height = 50.0\n"
    for marker, (lon, lat) in {**OKADA_STATIONS, **WLS_STATIONS}.items()
)

POSTPROCESS_CFG = """\
[PATHS]
data_prepath = /nonexistent/unused-by-this-slice/
"""

ANALYSIS_YAML = f"""\
version: 0
regions:
  {OKADA_REGION}:
    description: okada distributed-slip test region (dike opening)
    stations: [{", ".join(sorted(OKADA_STATIONS))}]
    default_reference_frame: ITRF2014
  {WLS_REGION}:
    description: baseline region (wls, no deformation)
    stations: [{", ".join(WLS_STATIONS)}]
    default_reference_frame: ITRF2014
velocity:
  default_window_years: 2.0
  default_method: wls
detrend:
  default_model: linear
  overrides: {{}}
breakpoints:
  enabled_regions: []
  n_breaks_default: 1
  n_runs: 420
  t_runs: 20
deformation:
  enabled_regions: [{OKADA_REGION}]
  source: okada
  series: raw
  window_years: {WINDOW_YEARS}
  epoch_mean_days: 11
  min_stations: 6
  okada:
    origin: {{lon: {PLANE_LON}, lat: {PLANE_LAT}}}
    strike: {STRIKE}
    dip: {DIP}
    length_km: {LENGTH_KM}
    width_km: {WIDTH_KM}
    top_depth_km: {TOP_DEPTH_KM}
    n_strike: {N_STRIKE}
    n_dip: {N_DIP}
    components: [opening]
    smoothing: 1.0e6
    nonneg: true
"""

PROVENANCE_KEYS = {"method", "frame", "fitted_at", "source", "software"}


def okada_loader(marker: str) -> StationSeries:
    """Synthetic series: planted dike-opening ramp for the gated region,
    plain fixture elsewhere. The ramp crosses zero at the window start and
    reaches the planted field at the window end, so the net displacement the
    stage measures equals the planted forward field."""
    if marker in WLS_STATIONS:
        return synthetic_station(marker, seed=1, n_days=N_DAYS)
    field = PLANTED[marker]  # (east, north, up) [m]
    t = T0 + np.arange(N_DAYS, dtype=np.float64) / 365.25
    t_end = float(t[-1])
    t_ref = t_end - WINDOW_YEARS
    ramp = (t - t_ref) / WINDOW_YEARS  # 0 at window start, 1 at window end
    rng = np.random.default_rng([9, zlib.crc32(marker.encode())])
    y = np.zeros((3, N_DAYS))
    y[0] = field[1] * ramp * 1e3  # north, mm
    y[1] = field[0] * ramp * 1e3  # east, mm
    y[2] = field[2] * ramp * 1e3  # up, mm
    y += rng.normal(0.0, NOISE_MM, y.shape)
    return StationSeries(
        marker=marker,
        t=t,
        y=y,
        sigma=np.full_like(y, NOISE_MM),
        source="synthetic:okada-dike",
    )


@pytest.fixture(scope="module")
def config_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    config_dir = tmp_path_factory.mktemp("gpsconfig-slip")
    (config_dir / "stations.cfg").write_text(STATIONS_CFG)
    (config_dir / "postprocess.cfg").write_text(POSTPROCESS_CFG)
    (config_dir / "analysis.yaml").write_text(ANALYSIS_YAML)
    return config_dir


@pytest.fixture()
def gpsconfig_env(config_dir: Path) -> Iterator[Path]:
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
    store_dir = tmp_path_factory.mktemp("store-slip")
    old = os.environ.get("GPS_CONFIG_PATH")
    os.environ["GPS_CONFIG_PATH"] = str(config_dir)
    try:
        cfg = load_analysis_config()
        fleet = run_fleet(cfg, okada_loader, store_dir, "synthetic:okada-dike", seed=2)
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
    monkeypatch.setenv("GPS_API_STORE", str(store))
    return TestClient(create_app())


# --- Product: gating, schema, provenance ------------------------------------


def test_slip_product_honors_the_region_gate(store: Path) -> None:
    """The Okada stage runs ONLY for deformation.enabled_regions, and writes
    a _slip.json (not a _deformation.json — that is the Mogi product)."""
    slip = sorted(p.name for p in (store / "models").glob("*_slip.json"))
    mogi = sorted(p.name for p in (store / "models").glob("*_deformation.json"))
    assert slip == [f"{OKADA_REGION}_slip.json"]
    assert mogi == []


def test_slip_product_schema_and_provenance(store: Path) -> None:
    payload = json.loads((store / "models" / f"{OKADA_REGION}_slip.json").read_text())
    result = SlipDistributionResult.model_validate(payload)
    assert result.region == OKADA_REGION
    assert result.source_type == "okada"
    assert result.series_kind == "raw"
    assert result.components == ["opening"]
    assert result.nonnegative is True
    assert result.smoothing_selected_by == "fixed"
    assert result.n_strike == N_STRIKE and result.n_dip == N_DIP
    assert len(result.patches) == N_STRIKE * N_DIP
    assert result.stations == sorted(OKADA_STATIONS)
    assert result.n_stations == len(OKADA_STATIONS)
    assert result.n_obs == 3 * len(OKADA_STATIONS)
    assert result.target_time > result.reference_time
    assert result.fitted_at.tzinfo is not None
    # plane geometry echoed faithfully
    assert result.strike == STRIKE and result.dip == DIP
    assert result.top_depth_km == TOP_DEPTH_KM

    provenance = payload["provenance"]
    assert PROVENANCE_KEYS <= provenance.keys()
    assert provenance["method"] == "okada"
    assert provenance["frame"] == "ITRF2014"
    assert provenance["plane"]["origin"] == {"lon": PLANE_LON, "lat": PLANE_LAT}
    assert provenance["plane"]["n_strike"] == N_STRIKE
    assert provenance["stations_excluded"] == []
    assert "non-negativity" in provenance["sigma_note"]


def test_slip_recovers_the_planted_opening(store: Path) -> None:
    """The distributed-slip product recovers the imposed opening pattern."""
    result = SlipDistributionResult.model_validate_json(
        (store / "models" / f"{OKADA_REGION}_slip.json").read_text()
    )
    rec = np.array(
        [p.slip_m["opening"] for p in sorted(result.patches, key=lambda p: p.index)]
    )
    assert rec.shape == (N_STRIKE * N_DIP,)
    assert bool(np.all(rec >= 0.0))  # NNLS
    corr = float(np.corrcoef(rec, S_TRUE_FLAT)[0, 1])
    assert corr > 0.9
    # peak recovered in the true peak cell (row=1, col=2)
    peak = max(result.patches, key=lambda p: p.slip_m["opening"])
    assert (peak.row, peak.col) == (1, 2)
    # potency (opening volume) within tolerance of the planted potency
    patches = discretize_fault(_planted_plane(), N_STRIKE, N_DIP)
    pot_true = float(S_TRUE_FLAT.sum()) * patches.patch_area
    assert result.potency_m3["opening"] == pytest.approx(pot_true, rel=0.2)
    # formal uncertainties present and positive
    assert all(p.sigma_m["opening"] > 0 for p in result.patches)
    assert result.sigma_potency_m3["opening"] > 0
    # patch centroids map back near the plane origin
    mean_lon = float(np.mean([p.lon for p in result.patches]))
    mean_lat = float(np.mean([p.lat for p in result.patches]))
    assert mean_lon == pytest.approx(PLANE_LON, abs=0.01)
    assert mean_lat == pytest.approx(PLANE_LAT, abs=0.01)


# --- The wired endpoint ------------------------------------------------------


def test_slip_endpoint_serves_and_validates(client: TestClient) -> None:
    resp = client.get(f"/v1/deformation/{OKADA_REGION}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["source_type"] == "okada"
    result = SlipDistributionResult.model_validate(body)
    assert result.region == OKADA_REGION
    assert len(result.patches) == N_STRIKE * N_DIP
    assert isinstance(result.provenance, dict)
    assert result.provenance["method"] == "okada"


def test_slip_endpoint_ungated_region_is_404(client: TestClient) -> None:
    resp = client.get(f"/v1/deformation/{WLS_REGION}")
    assert resp.status_code == 404
    assert isinstance(resp.json()["detail"], str)


def test_wls_region_stays_on_the_baseline(store: Path) -> None:
    payload = json.loads((store / "velocities" / f"{WLS_REGION}.geojson").read_text())
    collection = VelocityCollection.model_validate(payload)
    for feature in collection.features:
        assert feature.properties.method == "wls"


# --- Formal-covariance faithfulness -----------------------------------------


def test_formal_cov_operators_match_the_estimator() -> None:
    """The σ operators (okada_greens/patch_laplacian) reproduce the leaf's
    regularized solve: the augmented least-squares over the reconstructed
    G_w/L equals okada_invert_slip's slip and predicted field (nonneg=False).
    This pins that the reported per-patch σ corresponds to the served slip."""
    from gps_api.precompute.slip import _slip_formal_cov

    markers = sorted(OKADA_STATIONS)
    e = np.array(
        [
            local_coordinates(
                OKADA_STATIONS[m][0], OKADA_STATIONS[m][1], PLANE_LON, PLANE_LAT
            )[0]
            for m in markers
        ]
    )
    n = np.array(
        [
            local_coordinates(
                OKADA_STATIONS[m][0], OKADA_STATIONS[m][1], PLANE_LON, PLANE_LAT
            )[1]
            for m in markers
        ]
    )
    patches = discretize_fault(_planted_plane(), N_STRIKE, N_DIP)
    g = okada_greens(e, n, patches, ("opening",))
    obs = (g @ S_TRUE_FLAT).reshape(3, len(markers))
    sigma = np.full((3, len(markers)), 0.001)
    smoothing, edge = 1.0e6, "zero"

    fit = okada_invert_slip(
        e,
        n,
        obs,
        sigma,
        patches=patches,
        components=("opening",),
        smoothing=smoothing,
        nonnegative=False,
        edge=edge,
    )
    # Reconstruct G_w / L exactly as the leaf does and solve the augmented LSQ.
    w = 1.0 / sigma.ravel()
    g_w = g * w[:, None]
    lap = patch_laplacian(patches, edge)
    a = np.vstack((g_w, smoothing * lap))
    b = np.concatenate((obs.ravel() * w, np.zeros(lap.shape[0])))
    sol, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
    np.testing.assert_allclose(sol, fit.slip.ravel(), rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(
        (g @ sol).reshape(3, len(markers)), fit.predicted, rtol=1e-8, atol=1e-12
    )
    # The covariance built from the same operators is finite and PD-diagonal.
    cov = _slip_formal_cov(e, n, sigma, patches, ("opening",), smoothing, edge, 0.25)
    assert cov.shape == (patches.n_patches, patches.n_patches)
    assert bool(np.all(np.isfinite(cov)))
    assert bool(np.all(np.diag(cov) > 0.0))


# --- L-curve smoothing selection --------------------------------------------


def test_lcurve_selects_smoothing_and_recovers(
    gpsconfig_env: Path, tmp_path: Path
) -> None:
    """With smoothing omitted (L-curve), the stage picks λ at the corner and
    still recovers the pattern; the product records the selection mode."""
    from gps_api.precompute.slip import compute_slip_distribution

    cfg = load_analysis_config()
    dcfg = cfg.deformation
    okada = dcfg.okada
    assert okada is not None
    lcurve_okada = OkadaPlaneConfig(
        origin_lon=okada.origin_lon,
        origin_lat=okada.origin_lat,
        strike=okada.strike,
        dip=okada.dip,
        length_km=okada.length_km,
        width_km=okada.width_km,
        top_depth_km=okada.top_depth_km,
        n_strike=okada.n_strike,
        n_dip=okada.n_dip,
        components=okada.components,
        smoothing=None,  # L-curve corner selection
        nonneg=okada.nonneg,
    )
    lcurve_dcfg = DeformationConfig(
        enabled_regions=dcfg.enabled_regions,
        source="okada",
        series=dcfg.series,
        window_years=dcfg.window_years,
        epoch_mean_days=dcfg.epoch_mean_days,
        min_stations=dcfg.min_stations,
        okada=lcurve_okada,
    )
    from gps_api.precompute.config import load_station_meta

    markers = tuple(sorted(OKADA_STATIONS))
    series = {m: okada_loader(m) for m in markers}
    from gps_analysis import fit_components, linear, remove_trend

    detrended = {}
    for m, s in series.items():
        fits = tuple(
            fit_components(
                linear, s.t, s.y, sigma=s.sigma, names=("north", "east", "up")
            )
        )
        detrended[m] = np.asarray(
            remove_trend(linear, s.t, s.y, fits), dtype=np.float64
        )
    meta = load_station_meta(markers)
    import datetime

    outcome = compute_slip_distribution(
        OKADA_REGION,
        series,
        detrended,
        meta,
        lcurve_dcfg,
        datetime.datetime.now(datetime.UTC),
    )
    assert outcome.result.smoothing_selected_by == "lcurve"
    assert outcome.result.smoothing > 0.0
    rec = np.array([p.slip_m["opening"] for p in outcome.result.patches])
    assert float(np.corrcoef(rec, S_TRUE_FLAT)[0, 1]) > 0.8


# --- Fault tolerance ---------------------------------------------------------


def test_station_failure_does_not_sink_the_slip_stage(
    gpsconfig_env: Path, tmp_path: Path
) -> None:
    """A few bad stations are skipped; min_stations still met → product."""
    cfg = load_analysis_config()
    victims = {"D00", "D44"}

    def flaky(marker: str) -> StationSeries:
        if marker in victims:
            raise RuntimeError("simulated station outage")
        return okada_loader(marker)

    summary = run_precompute(
        cfg, OKADA_REGION, flaky, tmp_path, "synthetic:flaky", seed=2
    )
    assert set(summary.stations_failed) == victims
    assert summary.deformation_failed is None
    result = SlipDistributionResult.model_validate_json(
        (tmp_path / "models" / f"{OKADA_REGION}_slip.json").read_text()
    )
    assert result.stations == sorted(set(OKADA_STATIONS) - victims)


def test_slip_stage_failure_is_recorded_not_fatal(
    gpsconfig_env: Path, tmp_path: Path
) -> None:
    """Below min_stations the stage fails — recorded, region products survive."""
    cfg = load_analysis_config()
    keep = set(sorted(OKADA_STATIONS)[:3])  # < min_stations=6

    def very_flaky(marker: str) -> StationSeries:
        if marker not in keep:
            raise RuntimeError("simulated station outage")
        return okada_loader(marker)

    summary = run_precompute(
        cfg, OKADA_REGION, very_flaky, tmp_path, "synthetic:very-flaky", seed=2
    )
    assert summary.deformation_failed is not None
    assert not (tmp_path / "models" / f"{OKADA_REGION}_slip.json").exists()
    # The region's other products survived the stage failure.
    assert (tmp_path / "velocities" / f"{OKADA_REGION}.geojson").is_file()
    meta = json.loads((tmp_path / "meta" / "run.json").read_text())
    assert meta["deformation_failed"] == summary.deformation_failed


# --- Config validation -------------------------------------------------------


def test_okada_source_requires_a_plane_block() -> None:
    with pytest.raises(ValueError, match="okada"):
        DeformationConfig(enabled_regions=("x",), source="okada")


def test_okada_plane_config_validates_geometry() -> None:
    base = dict(
        origin_lon=PLANE_LON,
        origin_lat=PLANE_LAT,
        strike=STRIKE,
        dip=DIP,
        length_km=LENGTH_KM,
        width_km=WIDTH_KM,
        top_depth_km=TOP_DEPTH_KM,
        n_strike=N_STRIKE,
        n_dip=N_DIP,
    )
    # A valid plane builds.
    plane = OkadaPlaneConfig(**base)  # type: ignore[arg-type]
    assert plane.centroid_depth_km() > TOP_DEPTH_KM
    with pytest.raises(ValueError, match="dip"):
        OkadaPlaneConfig(**{**base, "dip": 120.0})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="width_km"):
        OkadaPlaneConfig(**{**base, "width_km": 0.0})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="top_depth_km"):
        OkadaPlaneConfig(**{**base, "top_depth_km": -1.0})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="1x1"):
        OkadaPlaneConfig(**{**base, "n_dip": 0})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="components"):
        OkadaPlaneConfig(**{**base, "components": ("rake",)})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="smoothing"):
        OkadaPlaneConfig(**{**base, "smoothing": -1.0})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="edge"):
        OkadaPlaneConfig(**{**base, "edge": "periodic"})  # type: ignore[arg-type]


def test_okada_config_loads_from_yaml(gpsconfig_env: Path) -> None:
    cfg = load_analysis_config()
    dcfg = cfg.deformation
    assert dcfg.source == "okada"
    assert dcfg.enabled_for(OKADA_REGION)
    okada = dcfg.okada
    assert okada is not None
    assert okada.origin_lon == PLANE_LON
    assert okada.n_strike == N_STRIKE and okada.n_dip == N_DIP
    assert okada.components == ("opening",)
    assert okada.smoothing == pytest.approx(1.0e6)
    assert okada.nonneg is True


# --- Discriminated dispatch + store hygiene (mixed mogi/okada store) ---------


def _minimal_mogi(region: str) -> DeformationResult:
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    return DeformationResult(
        region=region,
        source_type="mogi",
        reference_time=now,
        origin_lon=0.0,
        origin_lat=0.0,
        series_kind="raw",
        stations=[],
        fits=[],
        posterior=None,
        fitted_at=now,
    )


def _minimal_slip(region: str) -> SlipDistributionResult:
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    return SlipDistributionResult(
        region=region,
        source_type="okada",
        reference_time=now,
        target_time=now,
        origin_lon=0.0,
        origin_lat=0.0,
        series_kind="raw",
        strike=0.0,
        dip=45.0,
        length_km=1.0,
        width_km=1.0,
        top_depth_km=1.0,
        n_strike=1,
        n_dip=1,
        components=["opening"],
        nonnegative=True,
        smoothing=1.0,
        smoothing_selected_by="fixed",
        edge="zero",
        stations=[],
        patches=[],
        potency_m3={"opening": 0.0},
        sigma_potency_m3={"opening": 0.0},
        residual_norm=0.0,
        roughness_norm=0.0,
        rms_mm=0.0,
        n_obs=0,
        n_stations=0,
        fitted_at=now,
    )


def _prov() -> Provenance:
    from datetime import UTC, datetime

    return Provenance(
        method="t", frame="ITRF2014", fitted_at=datetime.now(UTC), source="test"
    )


def test_mixed_store_dispatches_per_region(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One store, a mogi region and an okada region — each endpoint resolves
    to the right product by which file is present (production fleet shape)."""
    from gps_api.precompute import products

    products.write_deformation_json(
        tmp_path, "reg_mogi", _minimal_mogi("reg_mogi"), _prov()
    )
    products.write_slip_json(tmp_path, "reg_okada", _minimal_slip("reg_okada"), _prov())
    monkeypatch.setenv("GPS_API_STORE", str(tmp_path))
    c = TestClient(create_app())
    assert c.get("/v1/deformation/reg_mogi").json()["source_type"] == "mogi"
    assert c.get("/v1/deformation/reg_okada").json()["source_type"] == "okada"


def test_writer_removes_stale_sibling_on_source_flip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flipping a region mogi↔okada removes the stale sibling — the endpoint
    never serves a shadowing product from the previous source."""
    from gps_api.precompute import products

    models = tmp_path / "models"
    products.write_deformation_json(tmp_path, "reg", _minimal_mogi("reg"), _prov())
    assert (models / "reg_deformation.json").is_file()
    # Flip to okada: the mogi sibling must be gone.
    products.write_slip_json(tmp_path, "reg", _minimal_slip("reg"), _prov())
    assert not (models / "reg_deformation.json").exists()
    assert (models / "reg_slip.json").is_file()
    monkeypatch.setenv("GPS_API_STORE", str(tmp_path))
    c = TestClient(create_app())
    assert c.get("/v1/deformation/reg").json()["source_type"] == "okada"
    # Flip back to mogi: the slip sibling must be gone.
    products.write_deformation_json(tmp_path, "reg", _minimal_mogi("reg"), _prov())
    assert not (models / "reg_slip.json").exists()
    assert c.get("/v1/deformation/reg").json()["source_type"] == "mogi"
