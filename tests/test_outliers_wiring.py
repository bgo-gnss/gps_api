"""Outlier-detection wiring tests (slice 2 — design §5, contract A8).

End-to-end over the real chain (config via ``gps_parser`` temp gpsconfig,
the real ``gps_analysis.detect_outliers`` leaf, the store writers, the
wired series endpoint), with deterministic injected outliers so every
assertion is exact:

- non-destructive store: raw Parquet columns byte-identical to a
  ``--no-outliers`` run, flags purely additive;
- the empirical hard requirement: a station with a DECLARED step
  (``steps.csv``) is not over-flagged, an UNDECLARED step aborts loudly
  (``outliers_aborted``) instead of masking signal;
- per-station config overrides (magnitude floors) honored;
- ``meta/suspected_steps.csv`` carries the protected event clusters;
- GBIS4TS consumes the CLEANED (inlier) series and the breaks product
  records the outlier-params hash (BGÓ Q8);
- API A8: ``clean=false`` default serves everything + ``outlier`` flags;
  ``clean=true`` drops ONLY the flagged epochs, before LTTB;
- fault tolerance: a raising detection is recorded, the station's
  products survive unmasked.
"""

import csv
import json
import os
import zlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient
from gps_analysis import estimate_velocity as real_estimate_velocity

from gps_api.main import create_app
from gps_api.precompute import job
from gps_api.precompute.breaks import BreakDetectionOutcome, ComponentTask
from gps_api.precompute.config import load_analysis_config
from gps_api.precompute.outliers import (
    SUSPECTED_STEPS_COLUMNS,
    config_hash,
)
from gps_api.precompute.outliers import (
    detect_station_outliers as real_detect_outliers,
)
from gps_api.precompute.products import PROVENANCE_METADATA_KEY
from gps_api.precompute.sources import StationSeries
from gps_api.schemas import SeriesResponse

REGION = "reykjanes"
STATIONS = ("SENG", "ELDC", "SKSH", "VONC")

N_DAYS = 300
T0 = 2024.0
NOISE_MM = 1.5

#: Injected outliers/signals (station -> what the loader adds).
SENG_STEP_DAY = 150  # DECLARED in steps.csv (ALL components), 40 mm
#: Declared HALF A DAY before the first post-step sample: operator catalogs
#: are not sample-aligned, and a catalog epoch that rounds to just past the
#: sample would put the sample on the pre-step side (H(0)=1 convention).
SENG_STEP_YEARF = T0 + (SENG_STEP_DAY - 0.5) / 365.25
SENG_SPIKE_NORTH = 30  # +25 mm, north only
SENG_SPIKE_UP = 100  # -30 mm, up only
ELDC_SPIKE_NORTH = 30  # +25 mm — but ELDC's floors are overridden huge
SKSH_RUN_DAYS = range(120, 125)  # +15 mm sustained 5-day run (north)
VONC_STEP_DAY = 150  # 60 mm step, NOT declared -> abort

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

[SKSH]
station_id = SKSH
station_name = Skipastigshraun
latitude = 63.9105
longitude = -22.4801
height = 33.0

[VONC]
station_id = VONC
station_name = Vondufell C
latitude = 63.9500
longitude = -22.6000
height = 12.0
"""

POSTPROCESS_CFG = """\
[PATHS]
data_prepath = /nonexistent/unused-by-this-slice/
"""

ANALYSIS_YAML = f"""\
version: 0
regions:
  {REGION}:
    description: outlier wiring test region
    stations: [{", ".join(STATIONS)}]
    default_reference_frame: ITRF2014
velocity:
  default_window_years: 2.0
  default_method: wls
detrend:
  default_model: lineperiodic
  overrides: {{}}
breakpoints:
  enabled_regions: []
  n_breaks_default: 1
  n_runs: 420
  t_runs: 20
outliers:
  enabled: true
  min_outlier_mm:
    horizontal: 5.0
    vertical: 10.0
  overrides:
    ELDC:
      min_outlier_horizontal_mm: 1000.0
      min_outlier_vertical_mm: 1000.0
"""

STEPS_CSV = f"""\
# steps.csv test fixture — SENG's step is DECLARED; VONC's is not.
sta,epoch_yearf,component,kind,source,comment
SENG,{SENG_STEP_YEARF:.6f},ALL,equipment,manual,test antenna swap
"""


def _base_series(marker: str) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic linear + annual + white-noise base per station."""
    rng = np.random.default_rng(zlib.crc32(marker.encode()))
    t = T0 + np.arange(N_DAYS, dtype=np.float64) / 365.25
    y = np.vstack(
        [
            1.0 + 12.0 * (t - T0) + 2.0 * np.sin(2 * np.pi * t),
            -2.0 - 8.0 * (t - T0) + 1.5 * np.cos(2 * np.pi * t),
            0.5 + 3.0 * (t - T0) + 4.0 * np.sin(2 * np.pi * t),
        ]
    )
    return t, y + rng.normal(0.0, NOISE_MM, (3, N_DAYS))


def _loader(marker: str) -> StationSeries:
    """Fixture loader with per-station injections (module docstring)."""
    t, y = _base_series(marker)
    if marker == "SENG":
        y[:, SENG_STEP_DAY:] += 40.0  # declared step, all components
        y[0, SENG_SPIKE_NORTH] += 25.0
        y[2, SENG_SPIKE_UP] -= 30.0
    elif marker == "ELDC":
        y[0, ELDC_SPIKE_NORTH] += 25.0  # identical spike, huge floors
    elif marker == "SKSH":
        for day in SKSH_RUN_DAYS:
            # Return-to-baseline run -> blunder cluster, FLAGGED since the
            # leaf's run-protection release (gps_analysis 3d05812).
            y[0, day] += 15.0
    elif marker == "VONC":
        y[:, VONC_STEP_DAY:] += 60.0  # UNDECLARED step -> abort
    return StationSeries(
        marker=marker,
        t=t,
        y=y,
        sigma=np.full((3, N_DAYS), NOISE_MM),
        source=f"synthetic:outlier-fixture:{marker}",
    )


@pytest.fixture(scope="module")
def config_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    config_dir = tmp_path_factory.mktemp("gpsconfig-outliers")
    (config_dir / "stations.cfg").write_text(STATIONS_CFG)
    (config_dir / "postprocess.cfg").write_text(POSTPROCESS_CFG)
    (config_dir / "analysis.yaml").write_text(ANALYSIS_YAML)
    (config_dir / "steps.csv").write_text(STEPS_CSV)
    return config_dir


@pytest.fixture()
def gpsconfig_env(config_dir: Path) -> Iterator[Path]:
    """Point gps_parser at the fixture gpsconfig for one test."""
    old = os.environ.get("GPS_CONFIG_PATH")
    os.environ["GPS_CONFIG_PATH"] = str(config_dir)
    try:
        yield config_dir
    finally:
        if old is None:
            os.environ.pop("GPS_CONFIG_PATH", None)
        else:
            os.environ["GPS_CONFIG_PATH"] = old


def _run(config_dir: Path, store: Path, **kwargs: Any) -> job.RunSummary:
    """One region run against the fixture gpsconfig (env-scoped)."""
    old = os.environ.get("GPS_CONFIG_PATH")
    os.environ["GPS_CONFIG_PATH"] = str(config_dir)
    try:
        cfg = load_analysis_config()
        return job.run_precompute(
            cfg, REGION, _loader, store, "synthetic:outlier-fixture", **kwargs
        )
    finally:
        if old is None:
            os.environ.pop("GPS_CONFIG_PATH", None)
        else:
            os.environ["GPS_CONFIG_PATH"] = old


@pytest.fixture(scope="module")
def store_on(tmp_path_factory: pytest.TempPathFactory, config_dir: Path) -> Path:
    """Store with the outlier stage ON (outliers.enabled in the sidecar)."""
    store = tmp_path_factory.mktemp("store-outliers-on")
    summary = _run(config_dir, store)
    assert summary.stations_failed == {}
    return store


@pytest.fixture(scope="module")
def store_off(tmp_path_factory: pytest.TempPathFactory, config_dir: Path) -> Path:
    """Same data, --no-outliers (compute_outliers=False)."""
    store = tmp_path_factory.mktemp("store-outliers-off")
    summary = _run(config_dir, store, compute_outliers=False)
    assert summary.stations_failed == {}
    return store


@pytest.fixture()
def client_on(store_on: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("GPS_API_STORE", str(store_on))
    return TestClient(create_app())


def _flags(store: Path, marker: str, column: str) -> np.ndarray:
    table = pq.read_table(store / "series" / f"{marker}.parquet")
    return np.asarray(table.column(column).to_pylist(), dtype=bool)


# ---------------------------------------------------------------------------
# Store: non-destructive columns + provenance
# ---------------------------------------------------------------------------

RAW_COLUMNS = (
    "time",
    "north",
    "east",
    "up",
    "sigma_north",
    "sigma_east",
    "sigma_up",
)

FLAG_COLUMNS = tuple(
    f"{c}_outlier{suffix}"
    for c in ("north", "east", "up")
    for suffix in ("", "_reason", "_protected")
) + ("outlier_epoch",)


def test_raw_columns_byte_identical_and_flags_additive(
    store_on: Path, store_off: Path
) -> None:
    """Requirement 3: flags are ADDITIVE — the raw series is untouched."""
    for marker in STATIONS:
        table_on = pq.read_table(store_on / "series" / f"{marker}.parquet")
        table_off = pq.read_table(store_off / "series" / f"{marker}.parquet")
        for column in RAW_COLUMNS:
            assert table_on.column(column).equals(
                table_off.column(column)
            ), f"{marker}: raw column {column} changed by the outlier stage"
        assert set(FLAG_COLUMNS) <= set(table_on.column_names)
        assert not set(FLAG_COLUMNS) & set(table_off.column_names)
        # Typed as designed (§5.2): bool flags, uint8 bitmasks.
        schema = table_on.schema
        assert str(schema.field("north_outlier").type) == "bool"
        assert str(schema.field("north_outlier_reason").type) == "uint8"
        assert str(schema.field("north_outlier_protected").type) == "uint8"
        assert str(schema.field("outlier_epoch").type) == "bool"


def test_spikes_flagged_and_declared_step_survives(store_on: Path) -> None:
    """The SENG lesson: with steps.csv the step is absorbed, spikes flagged."""
    north = _flags(store_on, "SENG", "north_outlier")
    up = _flags(store_on, "SENG", "up_outlier")
    union = _flags(store_on, "SENG", "outlier_epoch")
    assert north[SENG_SPIKE_NORTH], "injected north spike not flagged"
    assert up[SENG_SPIKE_UP], "injected up spike not flagged"
    assert union[SENG_SPIKE_NORTH] and union[SENG_SPIKE_UP]
    # No over-flagging: nothing within ±30 d of the DECLARED step epoch.
    window = slice(SENG_STEP_DAY - 30, SENG_STEP_DAY + 30)
    assert not union[window].any(), "declared step over-flagged"
    # And only a handful of flags overall (the two spikes + at most noise).
    assert int(union.sum()) <= 4


def test_series_provenance_outliers_object(store_on: Path) -> None:
    table = pq.read_table(store_on / "series" / "SENG.parquet")
    provenance = json.loads(dict(table.schema.metadata)[PROVENANCE_METADATA_KEY])
    outliers = provenance["outliers"]
    assert outliers["method"] == "hampel-trajectory"
    assert outliers["aborted"] is False
    assert outliers["params"]["global_n_sigma"] == 5.0
    assert outliers["min_outlier_mm"] == {"north": 5.0, "east": 5.0, "up": 10.0}
    # The declared step epoch reached the model on every component.
    for component in ("north", "east", "up"):
        assert outliers["step_epochs"][component] == [
            pytest.approx(SENG_STEP_YEARF, abs=1e-5)
        ]
    assert outliers["n_flagged"]["north"] >= 1
    assert outliers["n_flagged"]["up"] >= 1
    assert isinstance(outliers["params_hash"], str)
    assert len(outliers["params_hash"]) == 16


def test_per_station_override_honored(store_on: Path) -> None:
    """BGÓ Q4/Q9: the identical spike is floor-protected under the override."""
    north = _flags(store_on, "ELDC", "north_outlier")
    assert not north[ELDC_SPIKE_NORTH], "override floors ignored"
    assert not north.any()
    table = pq.read_table(store_on / "series" / "ELDC.parquet")
    provenance = json.loads(dict(table.schema.metadata)[PROVENANCE_METADATA_KEY])
    assert provenance["outliers"]["min_outlier_mm"]["north"] == 1000.0
    assert provenance["outliers"]["min_outlier_mm"]["up"] == 1000.0


def test_undeclared_step_aborts_loud_never_masks(store_on: Path) -> None:
    """§3.5: an undeclared step looks like >f_max candidates — abort, unmasked."""
    union = _flags(store_on, "VONC", "outlier_epoch")
    assert not union.any(), "aborted station must proceed UNMASKED"
    meta = json.loads((store_on / "meta" / "run.json").read_text())
    assert meta["outliers_aborted"] == ["VONC"]
    assert meta["outliers_failed"] == {}
    table = pq.read_table(store_on / "series" / "VONC.parquet")
    provenance = json.loads(dict(table.schema.metadata)[PROVENANCE_METADATA_KEY])
    assert provenance["outliers"]["aborted"] is True
    # Diagnostics stay fully populated on abort (loud, never silent).
    assert provenance["outliers"]["n_candidates"]["north"] > 0


def test_suspected_steps_csv_written_for_operator_review(store_on: Path) -> None:
    """BGÓ requirement #4 / Q5: the protected clusters land in meta/.

    Re-pinned 2026-07-14 to the leaf's run-protection release
    (gps_analysis 3d05812, design §3.4 update): a sustained one-sided run
    that RETURNS to baseline is a blunder cluster and is now FLAGGED — so
    SKSH's 5-day run no longer appears as a protected event; the aborted
    VONC station's undeclared-step clusters remain the CSV's payload.
    """
    path = store_on / "meta" / "suspected_steps.csv"
    assert path.is_file()
    with path.open() as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == SUSPECTED_STEPS_COLUMNS
        rows = list(reader)
    assert rows, "no suspected events recorded at all"
    assert all(row["region"] == REGION for row in rows)
    assert all(row["start_time"].endswith("Z") for row in rows if row["start_time"])
    # VONC's undeclared step shows up, marked as an aborted station, and at
    # least one evidence-bearing cluster brackets the injected step epoch.
    vonc = [r for r in rows if r["sta"] == "VONC"]
    assert vonc, "aborted VONC left no suspected-event trace"
    assert all(r["station_aborted"] == "true" for r in vonc)
    step_yearf = T0 + VONC_STEP_DAY / 365.25
    evidenced = [r for r in vonc if r["step_evidence"]]
    assert evidenced, "no step-evidence clusters for the undeclared VONC step"
    assert any(
        float(r["t_start_yearf"]) - 5 / 365.25
        <= step_yearf
        <= float(r["t_end_yearf"]) + 5 / 365.25
        for r in evidenced
    )
    # SKSH's return-to-baseline run is a BLUNDER cluster under the released
    # run protection — flagged, not surfaced as a suspected event.
    north = _flags(store_on, "SKSH", "north_outlier")
    assert north[list(SKSH_RUN_DAYS)].all()


# ---------------------------------------------------------------------------
# Downstream estimates fit on the INLIERS
# ---------------------------------------------------------------------------


def test_breaks_consume_cleaned_series_and_record_hash(
    gpsconfig_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """BGÓ Q8: GBIS4TS gets the inlier series; provenance records the hash."""
    captured: dict[str, list[ComponentTask]] = {}

    def fake_detect(
        station_tasks: dict[str, list[ComponentTask]], **kwargs: object
    ) -> BreakDetectionOutcome:
        captured.update(station_tasks)
        return BreakDetectionOutcome(summaries={}, failures={}, triage=None)

    monkeypatch.setattr(job, "detect_station_breaks", fake_detect)
    cfg = load_analysis_config()
    job.run_precompute(
        cfg,
        REGION,
        _loader,
        tmp_path,
        "synthetic:outlier-fixture",
        detect_breaks=True,
    )
    # SENG: per-component inlier masks — each chain's input drops exactly
    # its own component's flagged epochs (never more, never fewer).
    full = _loader("SENG")
    for component in ("north", "east", "up"):
        flags = _flags(tmp_path, "SENG", f"{component}_outlier")
        task = next(t for t in captured["SENG"] if t.component == component)
        assert task.t.size == N_DAYS - int(flags.sum())
        np.testing.assert_array_equal(task.t, full.t[~flags])
    seng_north = next(t for t in captured["SENG"] if t.component == "north")
    assert seng_north.t.size < N_DAYS
    assert full.t[SENG_SPIKE_NORTH] not in seng_north.t
    # ELDC (floor-protected) and VONC (aborted) are unmasked.
    assert all(t.t.size == N_DAYS for t in captured["ELDC"])
    assert all(t.t.size == N_DAYS for t in captured["VONC"])
    # The breaks product records the outlier-params hash it consumed.
    payload = json.loads((tmp_path / "models" / f"{REGION}_breaks.json").read_text())
    outliers = payload["provenance"]["outliers"]
    assert outliers["cleaned_input"] is True
    assert outliers["params_hash"] == config_hash(cfg.outliers)


def test_velocity_estimates_fit_on_inliers(
    gpsconfig_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hard rule: parameter estimates never see the flagged epochs."""
    sizes: list[int] = []

    def spy(t: np.ndarray, y: np.ndarray, *args: Any, **kwargs: Any) -> Any:
        sizes.append(int(np.asarray(t).size))
        return real_estimate_velocity(t, y, *args, **kwargs)

    monkeypatch.setattr(job, "estimate_velocity", spy)
    cfg = load_analysis_config()
    job.run_precompute(cfg, REGION, _loader, tmp_path, "synthetic:outlier-fixture")
    observed = sorted(sizes)
    # SENG lost its two flagged epochs; the others (protected/aborted/full)
    # kept all 300. Any noise-level extra flags only shrink further.
    assert observed[0] <= N_DAYS - 2
    assert observed[-1] == N_DAYS


def test_detection_failure_is_fault_tolerant(
    gpsconfig_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A raising detection is recorded; the station proceeds UNMASKED."""

    def flaky(series: StationSeries, *args: Any, **kwargs: Any) -> Any:
        if series.marker == "SENG":
            raise RuntimeError("simulated detection blow-up")
        return real_detect_outliers(series, *args, **kwargs)

    monkeypatch.setattr(job, "detect_station_outliers", flaky)
    cfg = load_analysis_config()
    summary = job.run_precompute(
        cfg, REGION, _loader, tmp_path, "synthetic:outlier-fixture"
    )
    assert "SENG" in summary.stations_ok  # products survived
    assert set(summary.outliers_failed) == {"SENG"}
    assert "simulated detection blow-up" in summary.outliers_failed["SENG"]
    table = pq.read_table(tmp_path / "series" / "SENG.parquet")
    assert "outlier_epoch" not in table.column_names  # unmasked product
    meta = json.loads((tmp_path / "meta" / "run.json").read_text())
    assert set(meta["outliers_failed"]) == {"SENG"}


# ---------------------------------------------------------------------------
# API A8: clean parameter + outlier flags
# ---------------------------------------------------------------------------


def test_series_default_serves_raw_with_outlier_flags(client_on: TestClient) -> None:
    resp = client_on.get("/v1/stations/SENG/series")
    assert resp.status_code == 200
    series = SeriesResponse.model_validate(resp.json())
    assert series.clean is False
    assert len(series.time) == N_DAYS  # nothing hidden by default
    assert series.outlier is not None and len(series.outlier) == N_DAYS
    union = _flags_from_client(client_on)
    assert series.outlier == union.tolist()
    assert series.outlier[SENG_SPIKE_NORTH] is True
    assert series.outlier_provenance is not None
    assert series.outlier_provenance["method"] == "hampel-trajectory"


def _flags_from_client(client: TestClient) -> np.ndarray:
    store = Path(os.environ["GPS_API_STORE"])
    return _flags(store, "SENG", "outlier_epoch")


def test_series_clean_drops_only_flagged_epochs(client_on: TestClient) -> None:
    default = SeriesResponse.model_validate(
        client_on.get("/v1/stations/SENG/series").json()
    )
    clean = SeriesResponse.model_validate(
        client_on.get("/v1/stations/SENG/series", params={"clean": True}).json()
    )
    assert clean.clean is True
    assert default.outlier is not None
    n_flagged = sum(default.outlier)
    assert n_flagged >= 2
    assert len(clean.time) == N_DAYS - n_flagged
    flagged_times = {
        t for t, flag in zip(default.time, default.outlier, strict=True) if flag
    }
    assert set(default.time) - set(clean.time) == flagged_times
    assert clean.outlier is not None and not any(clean.outlier)


def test_series_clean_applies_before_lttb(client_on: TestClient) -> None:
    """A8: flagged epochs are dropped BEFORE downsampling."""
    resp = client_on.get(
        "/v1/stations/SENG/series", params={"clean": True, "max_points": 50}
    )
    series = SeriesResponse.model_validate(resp.json())
    assert len(series.time) == 50
    default = SeriesResponse.model_validate(
        client_on.get("/v1/stations/SENG/series").json()
    )
    assert default.outlier is not None
    flagged_times = {
        t for t, flag in zip(default.time, default.outlier, strict=True) if flag
    }
    assert not flagged_times & set(series.time)


def test_series_clean_composes_with_detrended(client_on: TestClient) -> None:
    resp = client_on.get(
        "/v1/stations/SENG/series", params={"clean": True, "detrended": True}
    )
    assert resp.status_code == 200
    series = SeriesResponse.model_validate(resp.json())
    assert series.detrended is True and series.clean is True
    # Detrended = residuals of the outlier-robust STEP-AUGMENTED fit: the
    # declared 40 mm step must not survive into the residuals.
    north = np.asarray(series.north)
    assert float(np.abs(north).max()) < 25.0  # step (40) gone, spikes dropped
