"""Parallel + triage GBIS4TS break detection (perf-audit #1/#6, plan §10.7).

Everything runs on synthetic fixtures with short dev chains (never the
production 1e6), in temp dirs, no network, no hand-listed real stations.
Covered here:

- pooled execution reproduces the serial path exactly (same seed ->
  same summary as an inline run and as a direct
  ``gps_analysis.detect_breakpoints`` call),
- workers return bounded scalar summaries, never the raw sample chain,
- the triage screen flags the synthetic break stations (and only those)
  and logs the flagged/screened counts,
- fault tolerance: one broken chain / station / region is recorded and
  skipped — the batch never sinks,
- the ``--fleet``-shaped run through the pool produces the same store
  layout, with triage stats recorded in the breaks product provenance.

Screen settings (600 kept runs, 20 annealing runs, threshold 6.0 sigma)
were probed empirically: with these fixtures and ``seed=1`` the break
stations score z >= 10 while quiet stations stay <= 4.4 — a
deterministic, comfortably separated margin (chains are fully seeded).
"""

import json
import os
import pickle
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from gps_api.precompute.breaks import (
    BreakSummary,
    ComponentTask,
    detect_station_breaks,
    resolve_workers,
    run_break_task,
)
from gps_api.precompute.config import load_analysis_config
from gps_api.precompute.job import run_fleet, run_precompute
from gps_api.precompute.sources import COMPONENTS, StationSeries, synthetic_station

# --- Fixture data -----------------------------------------------------------

BREAK_STATIONS = ("BRK1", "BRK2")  # synthetic_station: real velocity breaks
QUIET_STATION = "QUIE"  # linear trend + noise, no break
BROKEN_STATION = "BADS"  # loader raises — fault-tolerance probe
REGION_GATED = "reykjanes"
REGION_PLAIN = "vesturland"
N_DAYS = 240
SEED = 1

# Chain lengths: 16 * t_runs (annealing span) must stay below n_runs.
CONFIRM_RUNS = 900
CONFIRM_T_RUNS = 20
TRIAGE_RUNS = 600
TRIAGE_T_RUNS = 20
TRIAGE_SIGMA = 6.0


def quiet_station(
    marker: str, *, seed: int = SEED, n_days: int = N_DAYS
) -> StationSeries:
    """Break-free synthetic series: linear trend + white noise, mm."""
    rng = np.random.default_rng([seed, sum(map(ord, marker))])
    t = 2024.0 + np.arange(n_days, dtype=np.float64) / 365.25
    wn = 2.0
    rows = [v * (t - t[0]) + rng.normal(0.0, wn, t.size) for v in (12.0, -8.0, 3.0)]
    y = np.vstack(rows)
    return StationSeries(
        marker=marker, t=t, y=y, sigma=np.full_like(y, wn), source="synthetic:quiet"
    )


def series_for(marker: str) -> StationSeries:
    """Test loader: break stations, one quiet station, one that raises."""
    if marker == BROKEN_STATION:
        raise RuntimeError("simulated station outage")
    if marker == QUIET_STATION:
        return quiet_station(marker)
    return synthetic_station(marker, seed=SEED, n_days=N_DAYS)


def component_tasks(
    series: StationSeries, *, n_runs: int = CONFIRM_RUNS, t_runs: int = CONFIRM_T_RUNS
) -> list[ComponentTask]:
    """Confirm-shaped work items for one station (median-sigma wn_amp)."""
    assert series.sigma is not None
    return [
        ComponentTask(
            marker=series.marker,
            component=component,
            t=series.t,
            y=series.y[i],
            wn_amp=float(np.median(series.sigma[i])),
            n_breaks=1,
            n_runs=n_runs,
            t_runs=t_runs,
            seed=SEED,
        )
        for i, component in enumerate(COMPONENTS)
    ]


# --- Worker-level guarantees -------------------------------------------------


def test_worker_returns_bounded_scalar_summary() -> None:
    """Audit #6: nothing chain-sized crosses the process boundary."""
    task = component_tasks(synthetic_station("BRK1", seed=SEED, n_days=120))[0]
    summary = run_break_task(task)

    assert isinstance(summary, BreakSummary)
    # No arrays anywhere in the summary — scalars, strings, float tuples.
    for name, value in vars(summary).items():
        assert not isinstance(value, np.ndarray), f"{name} leaked an array"
    assert all(isinstance(v, float) for v in summary.optimal)
    # The full kept chain at these settings alone is n_params * n_runs * 8
    # bytes; the summary must stay orders of magnitude below one chain.
    assert len(pickle.dumps(summary)) < 2048
    assert summary.model == "BPD1"
    assert summary.n_runs == task.n_runs
    assert summary.trend_change_z >= 0.0


def test_pool_matches_serial_path_exactly() -> None:
    """Same seed -> pooled summaries identical to the serial ones."""
    from gps_analysis import detect_breakpoints

    series = synthetic_station("BRK1", seed=SEED, n_days=120)
    tasks = component_tasks(series, n_runs=420, t_runs=20)

    # Serial ground truth #1: the estimator called directly (the exact
    # call the pre-parallel job made), flattened the same way.
    direct = detect_breakpoints(
        tasks[0].t,
        tasks[0].y,
        tasks[0].wn_amp,
        n_breaks=1,
        n_runs=420,
        t_runs=20,
        seed=SEED,
    )
    # Serial ground truth #2: the worker function run inline.
    inline = [run_break_task(task) for task in tasks]

    outcome = detect_station_breaks(
        {series.marker: tasks},
        triage_n_runs=0,
        triage_t_runs=0,
        triage_sigma=TRIAGE_SIGMA,
        max_workers=2,
    )
    assert outcome.triage is None  # triage off -> straight to confirm
    assert outcome.failures == {}
    pooled = outcome.summaries[series.marker]

    assert pooled == tuple(inline)  # exact float equality, all components
    assert pooled[0].optimal == tuple(float(v) for v in direct.optimal)
    assert pooled[0].y_ref == float(direct.y_ref)
    assert pooled[0].model == direct.model


def test_resolve_workers_bounds() -> None:
    assert resolve_workers(None, 4) == min(os.cpu_count() or 1, 4)
    assert resolve_workers(2, 519) == 2
    assert resolve_workers(8, 3) == 3  # never more workers than chains
    assert resolve_workers(0, 10) == 0  # inline mode
    with pytest.raises(ValueError, match="max_workers"):
        resolve_workers(-1, 4)


# --- Triage -> confirm --------------------------------------------------------


def test_triage_flags_break_stations_and_logs_counts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    station_tasks = {
        marker: component_tasks(series_for(marker))
        for marker in (*BREAK_STATIONS, QUIET_STATION)
    }
    outcome = detect_station_breaks(
        station_tasks,
        triage_n_runs=TRIAGE_RUNS,
        triage_t_runs=TRIAGE_T_RUNS,
        triage_sigma=TRIAGE_SIGMA,
        max_workers=4,
    )

    assert outcome.failures == {}
    assert outcome.triage is not None
    assert outcome.triage.stations_screened == 3
    assert set(outcome.triage.stations_flagged) == set(BREAK_STATIONS)
    # Confirm ran only on the flagged candidates, at full confirm length.
    assert set(outcome.summaries) == set(BREAK_STATIONS)
    for summaries in outcome.summaries.values():
        assert len(summaries) == len(COMPONENTS)
        assert all(s.n_runs == CONFIRM_RUNS for s in summaries)
    # No silent caps: the flagged/screened counts are logged.
    out = capsys.readouterr().out
    assert "flagged 2 of 3 screened station(s)" in out


def test_pool_fault_tolerance_one_broken_chain() -> None:
    """A chain that raises records its station; the batch continues."""
    good = synthetic_station("BRK1", seed=SEED, n_days=120)
    bad_task = ComponentTask(
        marker="BADC",
        component="north",
        t=good.t,
        y=good.y[0][:-7],  # shape mismatch -> ValueError inside the worker
        wn_amp=2.0,
        n_breaks=1,
        n_runs=420,
        t_runs=20,
        seed=SEED,
    )
    outcome = detect_station_breaks(
        {"BADC": [bad_task], "BRK1": component_tasks(good, n_runs=420, t_runs=20)},
        triage_n_runs=0,
        triage_t_runs=0,
        triage_sigma=TRIAGE_SIGMA,
        max_workers=3,
    )
    assert set(outcome.failures) == {"BADC"}
    assert "ValueError" in outcome.failures["BADC"]
    assert set(outcome.summaries) == {"BRK1"}
    assert len(outcome.summaries["BRK1"]) == len(COMPONENTS)


# --- The fleet path through the pool -----------------------------------------

STATIONS_CFG = """\
[BRK1]
station_id = BRK1
station_name = Break One
latitude = 63.8721
longitude = -22.4353
height = 65.0

[BRK2]
station_id = BRK2
station_name = Break Two
latitude = 63.8412
longitude = -22.5501
height = 41.2

[QUIE]
station_id = QUIE
station_name = Quiet
latitude = 63.9105
longitude = -22.4801
height = 33.0

[BADS]
station_id = BADS
station_name = Broken Loader
latitude = 63.9330
longitude = -22.3999
height = 12.0
"""

POSTPROCESS_CFG = """\
[PATHS]
data_prepath = /nonexistent/unused-by-this-slice/
"""

ANALYSIS_YAML = f"""\
version: 0
regions:
  {REGION_GATED}:
    description: gated region (breaks + triage on)
    stations: [{", ".join((*BREAK_STATIONS, QUIET_STATION, BROKEN_STATION))}]
    default_reference_frame: ITRF2014
  {REGION_PLAIN}:
    description: ungated region (velocities only)
    stations: [{QUIET_STATION}]
    default_reference_frame: ITRF2014
velocity:
  default_window_years: 2.0
  default_method: wls
detrend:
  default_model: lineperiodic
  overrides: {{}}
breakpoints:
  enabled_regions: [{REGION_GATED}]
  n_breaks_default: 1
  n_runs: {CONFIRM_RUNS}
  t_runs: {CONFIRM_T_RUNS}
  triage_n_runs: {TRIAGE_RUNS}
  triage_t_runs: {TRIAGE_T_RUNS}
  triage_sigma: {TRIAGE_SIGMA}
  max_workers: 6
"""


@pytest.fixture()
def gpsconfig_env(tmp_path: Path) -> Iterator[Path]:
    """Temp gpsconfig dir (stations.cfg + triage-enabled analysis.yaml)."""
    config_dir = tmp_path / "gpsconfig"
    config_dir.mkdir()
    (config_dir / "stations.cfg").write_text(STATIONS_CFG)
    (config_dir / "postprocess.cfg").write_text(POSTPROCESS_CFG)
    (config_dir / "analysis.yaml").write_text(ANALYSIS_YAML)
    old = os.environ.get("GPS_CONFIG_PATH")
    os.environ["GPS_CONFIG_PATH"] = str(config_dir)
    try:
        yield config_dir
    finally:
        if old is None:
            os.environ.pop("GPS_CONFIG_PATH", None)
        else:
            os.environ["GPS_CONFIG_PATH"] = old


def test_breakpoint_config_reads_triage_keys(gpsconfig_env: Path) -> None:
    cfg = load_analysis_config()
    bp = cfg.breakpoints
    assert (bp.n_runs, bp.t_runs) == (CONFIRM_RUNS, CONFIRM_T_RUNS)
    assert (bp.triage_n_runs, bp.triage_t_runs) == (TRIAGE_RUNS, TRIAGE_T_RUNS)
    assert bp.triage_sigma == TRIAGE_SIGMA
    assert bp.max_workers == 6


def test_fleet_run_triage_confirm_through_pool(
    gpsconfig_env: Path, tmp_path: Path
) -> None:
    """Multi-region synthetic fleet: pool + triage + fault tolerance."""
    cfg = load_analysis_config()
    store = tmp_path / "store"

    fleet = run_fleet(cfg, series_for, store, "synthetic:parallel-test", seed=SEED)

    # Fault tolerance: the broken station is recorded, the batch survives.
    gated = fleet.regions[REGION_GATED]
    assert set(gated.stations_failed) == {BROKEN_STATION}
    assert "RuntimeError" in gated.stations_failed[BROKEN_STATION]
    assert set(gated.stations_ok) == {*BREAK_STATIONS, QUIET_STATION}
    assert fleet.regions[REGION_PLAIN].stations_ok == (QUIET_STATION,)
    assert fleet.regions_failed == {}

    # Store layout unchanged by the parallel path: per-region velocities,
    # combined catalog, gated-region breaks, fleet meta.
    assert (store / "stations.geojson").is_file()
    assert (store / "velocities" / f"{REGION_GATED}.geojson").is_file()
    assert (store / "velocities" / f"{REGION_PLAIN}.geojson").is_file()
    assert not (store / "models" / f"{REGION_PLAIN}_breaks.json").exists()
    for marker in (*BREAK_STATIONS, QUIET_STATION):
        assert (store / "series" / f"{marker}.parquet").is_file()

    # Triage: only the break stations were confirmed; the quiet station
    # stays in the velocity/series products but has no break entries.
    payload = json.loads((store / "models" / f"{REGION_GATED}_breaks.json").read_text())
    entries = payload["entries"]
    assert {e["marker"] for e in entries} == set(BREAK_STATIONS)
    assert len(entries) == len(BREAK_STATIONS) * len(COMPONENTS)
    for entry in entries:
        assert entry["method"] == "gbis"
        assert entry["model"] == "BPD1"
        assert entry["n_runs"] == CONFIRM_RUNS  # confirm length, not triage
        assert {
            "intercept_mm",
            "trend_mm_yr",
            "trend_change_mm_yr",
            "breakpoint_yearf",
            "kappa",
            "amp_mm",
        } <= entry["parameters"].keys()

    # No silent caps: the screen/flag counts ride in the provenance.
    triage = payload["provenance"]["triage"]
    assert triage["n_runs"] == TRIAGE_RUNS
    assert triage["t_runs"] == TRIAGE_T_RUNS
    assert triage["sigma"] == TRIAGE_SIGMA
    assert triage["stations_screened"] == 3
    assert sorted(triage["stations_flagged"]) == sorted(BREAK_STATIONS)

    # Fleet meta keeps the run-summary shape.
    meta = json.loads((store / "meta" / "run.json").read_text())
    assert meta["fleet"] is True
    assert meta["totals"]["stations_failed"] == 1
    assert meta["totals"]["stations_ok"] == 4  # 3 gated + QUIE again in B


def test_break_stage_failure_recorded_not_fatal(
    gpsconfig_env: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A station whose confirm chain raises is recorded; the region survives.

    Runs inline (``max_workers=0``) so the monkeypatched estimator applies
    inside the (would-be) worker code path.
    """
    from gps_analysis import detect_breakpoints as real_detect

    import gps_api.precompute.breaks as breaks_module

    def flaky_detect(t: object, y: object, wn_amp: object, **kwargs: object) -> object:
        # BRK2's east/up rows differ from BRK1's; poison BRK2 via its
        # north row values (arrays are positional here).
        if np.allclose(
            np.asarray(y)[:5],
            synthetic_station("BRK2", seed=SEED, n_days=N_DAYS).y[0][:5],
        ):
            raise RuntimeError("simulated chain blow-up")
        return real_detect(t, y, wn_amp, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(breaks_module, "detect_breakpoints", flaky_detect)
    cfg = load_analysis_config()

    summary = run_precompute(
        cfg,
        REGION_GATED,
        series_for,
        tmp_path / "store",
        "synthetic:flaky-chain",
        n_runs=420,
        t_runs=20,
        triage_n_runs=0,  # straight to confirm — the failure path under test
        max_workers=0,  # inline so the monkeypatch reaches the worker fn
        seed=SEED,
    )

    assert "BRK2" in summary.stations_failed
    assert "simulated chain blow-up" in summary.stations_failed["BRK2"]
    assert set(summary.stations_ok) == {"BRK1", QUIET_STATION}
    payload = json.loads(
        (tmp_path / "store" / "models" / f"{REGION_GATED}_breaks.json").read_text()
    )
    # BRK1 and QUIE were confirmed (triage off); BRK2 produced nothing.
    assert {e["marker"] for e in payload["entries"]} == {"BRK1", QUIET_STATION}
