"""Detrend-parameter estimation writer — the geo_dataread handshake tests.

The point of this slice (BGÓ #2: "we need to test this well before we can
deliver actual detrended data"): the precompute job writes
``params/detrend_params.json`` in EXACTLY the schema
``geo_dataread.gps_views.read_detrend_params`` (on geo_dataread ``main``)
consumes, per ``gps_analysis/docs/DESIGN_live_detrending.md`` §0. Pinned
here:

- document shape: ``{"schema_version": 1, "stations": {STA: leaf record}}``
  where each record is ``gps_analysis.DetrendEstimate.to_record`` output,
- the ROUND TRIP through the real geo_dataread reader: ``read_detrend_params``
  → ``station_detrend_record`` → ``apply_detrend`` reproduces the leaf's
  detrended series (and stays exactly invertible),
- schema-version rejection (a v2 doc must be refused — the handshake guard),
- pinned records honored verbatim (decision 7, never refit),
- UseSTA borrowing with self-contained borrowed provenance (decision 6),
- graceful, LOUD degrade (decision 4): validity-gate failures and
  outlier-abort/rms-degraded fits produce NO silent record,
- frame tagging (decision 5, plate-first) + the downstream mismatch guard,
- real ``.NEU`` behavior (SENG pre-unrest window fits cleanly; the
  active-unrest window is caught by the rms degrade gate, not written).

The geo_dataread import is test-only (editable sibling install); the whole
module skips when it is absent so CI without the sibling stays green.
"""

import dataclasses
import datetime
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from gps_analysis import apply_detrend, estimate_detrend, evaluate_record

from gps_api.precompute.config import (
    DetrendConfig,
    load_analysis_config,
)
from gps_api.precompute.detrend import (
    estimate_station_record,
    run_detrend_estimation,
)
from gps_api.precompute.job import run_fleet, run_precompute
from gps_api.precompute.products import Provenance, write_detrend_params
from gps_api.precompute.sources import StationSeries, load_neu

gps_views = pytest.importorskip(
    "geo_dataread.gps_views",
    reason="geo_dataread (the reader half of the handshake) is not installed",
)

REGION = "reykjanes"
FRAME = "plate:ITRF2014"
STEP_EPOCH = 2021.2
FITTED_AT = datetime.datetime(2026, 7, 14, 12, 0, 0, tzinfo=datetime.UTC)

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
latitude = 63.9000
longitude = -22.5000
height = 30.0

[GATE]
station_id = GATE
station_name = Gate Failure
latitude = 63.9500
longitude = -22.6000
height = 10.0
"""

POSTPROCESS_CFG = "[PATHS]\ndata_prepath = /nonexistent/unused/\n"

ANALYSIS_YAML = f"""\
version: 0
regions:
  {REGION}:
    description: test region
    stations: [SENG, ELDC, SKSH, GATE]
    default_reference_frame: ITRF2014
velocity:
  default_window_years: 2.0
  default_method: wls
detrend:
  default_model: lineperiodic
  overrides: {{}}
  estimation:
    enabled: true
    method: step_augmented_robust
    use_sta:
      ELDC: SENG
    pinned:
      SKSH: keep
breakpoints:
  enabled_regions: []
  n_breaks_default: 1
  n_runs: 420
  t_runs: 20
"""

STEPS_CSV = (
    "sta,epoch_yearf,component,kind,source,comment\n"
    f"SENG,{STEP_EPOCH},ALL,equipment,manual,test antenna swap\n"
)


def _clean_series(
    marker: str,
    *,
    n_days: int = 900,
    t0: float = 2020.0,
    seed: int = 7,
    step_mm: float = 0.0,
    step_epoch: float = STEP_EPOCH,
) -> StationSeries:
    """Deterministic lineperiodic + noise series (optionally with a step)."""
    rng = np.random.default_rng([seed, len(marker)])
    t = t0 + np.arange(n_days, dtype=np.float64) / 365.25
    rows = []
    for base, rate in ((0.0, 20.0), (5.0, -12.0), (-3.0, 2.0)):
        row = (
            base
            + rate * (t - t0)
            + 3.0 * np.cos(2 * np.pi * t)
            + 1.5 * np.sin(2 * np.pi * t)
            + rng.normal(0.0, 1.0, t.size)
        )
        if step_mm:
            row = row + step_mm * (t >= step_epoch)
        rows.append(row)
    y = np.vstack(rows)
    return StationSeries(
        marker=marker, t=t, y=y, sigma=np.full_like(y, 1.0), source=f"test:{marker}"
    )


def _loader(marker: str) -> StationSeries:
    if marker == "GATE":  # too short for the min_span_years gate
        return _clean_series(marker, n_days=120)
    if marker == "SENG":  # known step from the steps.csv catalog
        return _clean_series(marker, step_mm=30.0)
    return _clean_series(marker)


def _sentinel_record() -> dict[str, Any]:
    """A valid, distinctive leaf record for the SKSH pin (never refit)."""
    series = _clean_series("SKSH", seed=99)
    estimate = estimate_detrend(
        "lineperiodic", series.t, series.y, series.sigma, frame=FRAME
    )
    return estimate.to_record(
        fitted_at="2000-01-01T00:00:00Z", refs={"note": "sentinel-pin"}
    )


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    """Temp gpsconfig + a store pre-seeded with the SKSH pinned record."""
    config_dir = tmp_path_factory.mktemp("gpsconfig")
    (config_dir / "stations.cfg").write_text(STATIONS_CFG)
    (config_dir / "postprocess.cfg").write_text(POSTPROCESS_CFG)
    (config_dir / "analysis.yaml").write_text(ANALYSIS_YAML)
    (config_dir / "steps.csv").write_text(STEPS_CSV)
    store = tmp_path_factory.mktemp("store")
    sentinel = _sentinel_record()
    params_dir = store / "params"
    params_dir.mkdir()
    (params_dir / "detrend_params.json").write_text(
        json.dumps({"schema_version": 1, "stations": {"SKSH": sentinel}})
    )
    old = os.environ.get("GPS_CONFIG_PATH")
    os.environ["GPS_CONFIG_PATH"] = str(config_dir)
    yield {"config_dir": config_dir, "store": store, "sentinel": sentinel}
    if old is None:
        os.environ.pop("GPS_CONFIG_PATH", None)
    else:
        os.environ["GPS_CONFIG_PATH"] = old


@pytest.fixture(scope="module")
def run(env: dict[str, Any]) -> dict[str, Any]:
    """One precompute run with the estimation stage on (module-shared)."""
    cfg = load_analysis_config()
    summary = run_precompute(cfg, REGION, _loader, env["store"], "test-source")
    doc_path = env["store"] / "params" / "detrend_params.json"
    return {
        "cfg": cfg,
        "summary": summary,
        "doc_path": doc_path,
        "doc": json.loads(doc_path.read_text()),
        **env,
    }


# ---------------------------------------------------------------------------
# document shape + run summary
# ---------------------------------------------------------------------------


def test_document_shape_and_membership(run: dict[str, Any]) -> None:
    doc = run["doc"]
    assert doc["schema_version"] == 1
    assert doc["frame"] == FRAME
    assert set(doc["stations"]) == {"SENG", "ELDC", "SKSH"}
    # GATE failed the min_span gate: absent = "no background model".
    assert "GATE" not in doc["stations"]
    provenance = doc["provenance"]
    assert provenance["method"] == "step_augmented_robust"
    assert provenance["frame"] == FRAME
    assert set(provenance["software"]) == {"gps_api", "gps_analysis"}


def test_run_summary_reports_every_outcome(run: dict[str, Any]) -> None:
    payload = run["summary"].as_dict()
    block = payload["detrend_params"]
    assert block["fitted"] == ["SENG"]
    assert block["pinned"] == ["SKSH"]
    assert block["borrowed"] == {"ELDC": "SENG"}
    assert block["degraded"] == {}
    assert "min_span_years" in block["skipped"]["GATE"]
    # The document is a registered product of the run.
    assert str(run["doc_path"]) in payload["products"]
    # And meta/run.json carries the same block.
    meta = json.loads((run["store"] / "meta" / "run.json").read_text())
    assert meta["detrend_params"] == block


def test_record_is_leaf_to_record_shape_with_step_and_frame(
    run: dict[str, Any],
) -> None:
    record = run["doc"]["stations"]["SENG"]
    assert record["record_version"] == 1
    assert record["model"] == "lineperiodic"
    assert record["detrend_method"] == "step_augmented_robust"
    assert record["frame"] == FRAME
    assert record["fitted_at"].endswith("Z")
    # steps.csv epoch augmented the model; the record is self-contained.
    assert record["step_epochs"] == [STEP_EPOCH]
    assert record["param_names"][-1] == "step_amp_1"
    assert len(record["components"]) == 3
    assert record["refs"]["region"] == REGION
    assert record["refs"]["source"] == "test:SENG"
    # The estimated step amplitude recovers the injected 30 mm offset.
    for component in record["components"]:
        assert component["params"][-1] == pytest.approx(30.0, abs=1.0)


# ---------------------------------------------------------------------------
# THE handshake: geo_dataread's real reader consumes the written document
# ---------------------------------------------------------------------------


def test_geo_dataread_reader_accepts_the_document(run: dict[str, Any]) -> None:
    doc = gps_views.read_detrend_params(run["doc_path"])
    assert set(doc["stations"]) == {"SENG", "ELDC", "SKSH"}
    record, source = gps_views.station_detrend_record(doc, "SENG")
    assert source == "SENG"
    assert record is not None and record["detrend_method"] == "step_augmented_robust"


def test_round_trip_apply_reproduces_the_leaf_detrend(run: dict[str, Any]) -> None:
    """estimate → write → read (geo_dataread) → apply == fit → apply."""
    series = _loader("SENG")
    detrended, provenance = gps_views.detrend_arrays(
        "SENG", series.t, series.y, params=run["doc_path"], frame=FRAME
    )
    assert provenance["applied"] is True
    assert provenance["degraded"] is False
    assert provenance["detrend_method"] == "step_augmented_robust"
    assert provenance["frame"] == FRAME
    assert provenance["borrowed"] is None
    # The stored background removes trend + seasonal + step: residual is at
    # the 1 mm noise level of the fixture.
    rms = np.sqrt(np.mean(detrended**2, axis=1))
    assert np.all(rms < 2.0)
    # Byte-compatible with the leaf: the written record applies identically
    # to a fresh fit → to_record → apply chain on the same inputs (incl. the
    # stage's outlier magnitude floors — 5/5/10 mm H/H/V from the resolved
    # outliers block, one threshold vocabulary for detection + estimation).
    estimate = estimate_detrend(
        "lineperiodic",
        series.t,
        series.y,
        series.sigma,
        step_epochs=[STEP_EPOCH],
        min_outlier=np.array([5.0, 5.0, 10.0]),
        frame=FRAME,
    )
    fresh = apply_detrend(
        estimate.to_record(fitted_at="x"), series.t, series.y, frame=FRAME
    )
    np.testing.assert_allclose(detrended, fresh, rtol=0.0, atol=1e-9)
    # Exactly invertible: raw = detrended + f(t; p̂) (design §4.1).
    record = run["doc"]["stations"]["SENG"]
    np.testing.assert_allclose(
        detrended + evaluate_record(record, series.t),
        series.y,
        rtol=0.0,
        atol=1e-9,
    )


def test_schema_version_2_is_refused_by_the_reader(
    run: dict[str, Any], tmp_path: Path
) -> None:
    """Guard the handshake: an unknown schema_version must be rejected."""
    doc = dict(run["doc"])
    doc["schema_version"] = 2
    path = tmp_path / "detrend_params.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="schema_version"):
        gps_views.read_detrend_params(path)


def test_frame_mismatch_is_a_hard_downstream_error(run: dict[str, Any]) -> None:
    """Decision 5: applying across frames is refused, never fudged."""
    series = _loader("SENG")
    record = run["doc"]["stations"]["SENG"]
    with pytest.raises(ValueError, match="frame mismatch"):
        apply_detrend(record, series.t, series.y, frame="plate:ITRF2008")


# ---------------------------------------------------------------------------
# pinning (decision 7) and borrowing (decision 6)
# ---------------------------------------------------------------------------


def test_pinned_record_is_honored_verbatim(run: dict[str, Any]) -> None:
    stored = run["doc"]["stations"]["SKSH"]
    sentinel = json.loads(json.dumps(run["sentinel"]))  # JSON-normalized
    assert stored == sentinel
    assert stored["fitted_at"] == "2000-01-01T00:00:00Z"
    assert stored["refs"]["note"] == "sentinel-pin"


def test_borrowed_record_is_self_contained_with_provenance(
    run: dict[str, Any],
) -> None:
    seng = run["doc"]["stations"]["SENG"]
    eldc = run["doc"]["stations"]["ELDC"]
    assert eldc["borrowed"] == {
        "from": "SENG",
        "terms": "all",
        "donor_fitted_at": seng["fitted_at"],
    }
    # Everything else is the donor's record, copied (self-contained: the
    # apply path never chases donor references — design §2.6).
    assert {k: v for k, v in eldc.items() if k != "borrowed"} == {
        k: v for k, v in seng.items() if k != "borrowed"
    }
    # The reader surfaces the borrowed provenance on the applied view.
    series = _clean_series("ELDC")
    _, provenance = gps_views.detrend_arrays(
        "ELDC", series.t, series.y, params=run["doc_path"], frame=FRAME
    )
    assert provenance["applied"] is True
    assert provenance["borrowed"]["from"] == "SENG"


# ---------------------------------------------------------------------------
# graceful, loud degrade (decision 4)
# ---------------------------------------------------------------------------


def _abort_series(marker: str = "BADS") -> StationSeries:
    """12 % huge blunders — trips the §3.5 excess-candidate abort."""
    series = _clean_series(marker, seed=3)
    rng = np.random.default_rng(3)
    idx = rng.choice(series.t.size, size=int(0.12 * series.t.size), replace=False)
    y = series.y.copy()
    y[:, idx] += rng.normal(0.0, 60.0, (3, idx.size))
    return dataclasses.replace(series, y=y)


def test_outlier_abort_writes_no_record_by_default() -> None:
    record, degrade = estimate_station_record(
        _abort_series(),
        "lineperiodic",
        DetrendConfig(),
        fitted_at=FITTED_AT,
        region=REGION,
        frame=FRAME,
    )
    assert record is None
    assert degrade is not None and "outlier stage aborted" in degrade


def test_outlier_abort_with_write_degraded_carries_explicit_marker() -> None:
    record, degrade = estimate_station_record(
        _abort_series(),
        "lineperiodic",
        DetrendConfig(write_degraded=True),
        fitted_at=FITTED_AT,
        region=REGION,
        frame=FRAME,
    )
    assert degrade is not None
    assert record is not None
    # Consistent with estimate_detrend's warn + plain-WLS fallback semantics.
    assert record["detrend_method"] == "plain_wls"
    assert record["refs"]["degraded"] is True
    assert "outlier stage aborted" in record["refs"]["degrade_reason"]


def test_rms_gate_degrades_a_bad_background() -> None:
    """max_rms_mm catches fits the abort does not (the SENG unrest lesson)."""
    series = _clean_series("RMSY", seed=11)
    record, degrade = estimate_station_record(
        series,
        "lineperiodic",
        DetrendConfig(max_rms_mm=0.5),  # below the 1 mm fixture noise
        fitted_at=FITTED_AT,
        region=REGION,
        frame=FRAME,
    )
    assert record is None
    assert degrade is not None and "max_rms_mm" in degrade


def test_missing_pin_and_missing_donor_are_skipped_loudly(
    run: dict[str, Any], tmp_path: Path
) -> None:
    cfg = run["cfg"]
    dcfg = DetrendConfig(
        pinned={"SKSH": "keep"},  # empty store below → pin unavailable
        use_sta={"ELDC": "GATE"},  # donor fails its gate → no record
    )
    result = run_detrend_estimation(
        cfg=dataclasses.replace(cfg, detrend_estimation=dcfg),
        region_name=REGION,
        frame=FRAME,
        stations=("SENG", "ELDC", "SKSH", "GATE"),
        series_map={m: _loader(m) for m in ("SENG", "ELDC", "GATE")},
        step_catalog={},
        store=tmp_path / "empty-store",
        fitted_at=FITTED_AT,
    )
    assert result.fitted == ("SENG",)
    assert "pinned" in result.skipped["SKSH"]
    assert "GATE" in result.skipped["ELDC"] and "donor" in result.skipped["ELDC"]
    assert "min_span_years" in result.skipped["GATE"]
    assert set(result.records) == {"SENG"}


# ---------------------------------------------------------------------------
# config parsing / validation
# ---------------------------------------------------------------------------


def test_absent_estimation_block_disables_the_stage(env: dict[str, Any]) -> None:
    yaml_path = env["config_dir"] / "analysis-no-estimation.yaml"
    yaml_path.write_text(
        ANALYSIS_YAML.replace(
            """\
  estimation:
    enabled: true
    method: step_augmented_robust
    use_sta:
      ELDC: SENG
    pinned:
      SKSH: keep
""",
            "",
        )
    )
    cfg = load_analysis_config(yaml_path)
    assert cfg.detrend_estimation.enabled is False


def test_config_loads_the_estimation_block(run: dict[str, Any]) -> None:
    dcfg = run["cfg"].detrend_estimation
    assert dcfg.enabled is True
    assert dcfg.method == "step_augmented_robust"
    assert dcfg.use_sta == {"ELDC": "SENG"}
    assert dcfg.pinned == {"SKSH": "keep"}
    assert dcfg.fit_window_years is None  # whole series, "as long as possible"


def test_detrend_config_validation() -> None:
    with pytest.raises(ValueError, match="method"):
        DetrendConfig(method="refit_on_read")
    with pytest.raises(ValueError, match="min_span_years"):
        DetrendConfig(min_span_years=0.5)  # decision 7: min 1-2 yr floor
    with pytest.raises(ValueError, match="cannot borrow from itself"):
        DetrendConfig(use_sta={"SENG": "SENG"})
    with pytest.raises(ValueError, match="borrow chains"):
        DetrendConfig(use_sta={"ELDC": "SENG", "SENG": "SKSH"})
    with pytest.raises(ValueError, match="both pinned and in use_sta"):
        DetrendConfig(use_sta={"ELDC": "SENG"}, pinned={"ELDC": "keep"})
    with pytest.raises(ValueError, match="end .* <= start"):
        DetrendConfig(fit_windows={"SENG": (2020.0, 2019.0)})
    with pytest.raises(ValueError, match="max_rms_mm"):
        DetrendConfig(max_rms_mm=0.0)


def test_window_policy_resolution() -> None:
    dcfg = DetrendConfig(fit_window_years=4.0, fit_windows={"SENG": (2017.1, 2019.9)})
    assert dcfg.window_for("SENG", 2026.5) == (2017.1, 2019.9)  # override wins
    assert dcfg.window_for("ELDC", 2026.5) == (2022.5, None)  # trailing policy
    open_cfg = DetrendConfig()
    assert open_cfg.window_for("ELDC", 2026.5) == (None, None)  # whole series


# ---------------------------------------------------------------------------
# fleet: one merged document across regions
# ---------------------------------------------------------------------------


def test_fleet_merges_one_document(
    env: dict[str, Any], tmp_path_factory: pytest.TempPathFactory
) -> None:
    config_dir = tmp_path_factory.mktemp("gpsconfig-fleet")
    (config_dir / "stations.cfg").write_text(STATIONS_CFG)
    (config_dir / "postprocess.cfg").write_text(POSTPROCESS_CFG)
    (config_dir / "analysis.yaml").write_text("""\
version: 0
regions:
  west:
    stations: [SENG]
    default_reference_frame: ITRF2014
  east:
    stations: [SKSH]
    default_reference_frame: ITRF2014
velocity:
  default_window_years: 2.0
  default_method: wls
detrend:
  default_model: lineperiodic
  estimation:
    enabled: true
breakpoints:
  enabled_regions: []
  n_breaks_default: 1
  n_runs: 420
  t_runs: 20
""")
    store = tmp_path_factory.mktemp("store-fleet")
    old = os.environ.get("GPS_CONFIG_PATH")
    os.environ["GPS_CONFIG_PATH"] = str(config_dir)
    try:
        cfg = load_analysis_config()
        fleet = run_fleet(cfg, lambda m: _clean_series(m), store, "test-fleet")
    finally:
        if old is None:
            os.environ.pop("GPS_CONFIG_PATH", None)
        else:
            os.environ["GPS_CONFIG_PATH"] = old
    doc = json.loads((store / "params" / "detrend_params.json").read_text())
    assert doc["schema_version"] == 1
    assert set(doc["stations"]) == {"SENG", "SKSH"}
    payload = fleet.as_dict()
    assert payload["regions"]["west"]["detrend_params"]["fitted"] == ["SENG"]
    assert payload["regions"]["east"]["detrend_params"]["fitted"] == ["SKSH"]
    # And the merged document round-trips through the reader.
    assert set(
        gps_views.read_detrend_params(store / "params" / "detrend_params.json")[
            "stations"
        ]
    ) == {"SENG", "SKSH"}


# ---------------------------------------------------------------------------
# real data (.NEU fixtures; skipped when the gitignored fixture is absent)
# ---------------------------------------------------------------------------

NEU_DIR = Path(__file__).parent / "fixtures" / "realdata" / "neu"

realdata = pytest.mark.skipif(
    not (NEU_DIR / "SENG.NEU").is_file(),
    reason="realdata fixture not fetched (tests/fixtures/realdata is gitignored)",
)


@realdata
def test_seng_whole_history_fails_the_gap_gate_loudly() -> None:
    """Default policy on real SENG: the 2016.3→2017.1 gap fails max_gap_years.

    "As long as possible" is gated, not blind — the station gets NO record
    (absent = no background model) with the gate named in the reason.
    """
    series = load_neu(NEU_DIR / "SENG.NEU")
    with pytest.raises(ValueError, match="max_gap_years"):
        estimate_station_record(
            series,
            "lineperiodic",
            DetrendConfig(),
            fitted_at=FITTED_AT,
            region=REGION,
            frame=FRAME,
        )


@realdata
def test_seng_pre_unrest_window_fits_and_roundtrips(tmp_path: Path) -> None:
    """Operator pre-unrest window on real SENG: clean robust fit, and the
    stored record detrends the FULL live series (transient preserved)."""
    series = load_neu(NEU_DIR / "SENG.NEU")
    record, degrade = estimate_station_record(
        series,
        "lineperiodic",
        DetrendConfig(fit_windows={"SENG": (2017.1, 2019.9)}),
        fitted_at=FITTED_AT,
        region=REGION,
        frame=FRAME,
    )
    assert degrade is None
    assert record is not None
    assert record["detrend_method"] == "step_augmented_robust"
    # Background is sane: pre-unrest secular rates are small, residuals mm.
    for component in record["components"]:
        assert abs(component["params"][1]) < 20.0  # rate [mm/yr]
    assert max(record["rms"]) < 8.0
    # Write → read through geo_dataread → apply at ALL epochs (incl. the
    # 2020+ unrest the window never saw): the Svartsengi transient survives
    # detrending — that is the point of the stored background.
    store = tmp_path / "store"
    write_detrend_params(
        store,
        {"SENG": record},
        Provenance(
            method="step_augmented_robust",
            frame=FRAME,
            fitted_at=FITTED_AT,
            source="neu:test",
        ),
    )
    detrended, provenance = gps_views.detrend_arrays(
        "SENG",
        series.t,
        series.y,
        params=store / "params" / "detrend_params.json",
        frame=FRAME,
    )
    assert provenance["applied"] is True
    pre = detrended[:, series.t < 2019.9]
    assert float(np.sqrt(np.mean(pre[:, pre.shape[1] // 2 :] ** 2))) < 10.0
    east_last = float(detrended[1, -1])
    assert east_last < -2000.0  # the ~-2.5 m unrest signal is preserved


@realdata
def test_seng_active_unrest_window_is_degraded_not_written() -> None:
    """A window spanning the unrest does NOT abort — the robust fit swallows
    the transient into a garbage background (rms 100s of mm). The rms
    degrade gate catches it: no bogus record is written (decision 4)."""
    series = load_neu(NEU_DIR / "SENG.NEU")
    record, degrade = estimate_station_record(
        series,
        "lineperiodic",
        # Relax the gap gate so the fit RUNS across the unrest — the point
        # is that the rms gate (not the gates, not the abort) rejects it.
        DetrendConfig(fit_windows={"SENG": (2017.1, None)}, max_gap_years=1.0),
        fitted_at=FITTED_AT,
        region=REGION,
        frame=FRAME,
    )
    assert record is None
    assert degrade is not None and "max_rms_mm" in degrade
