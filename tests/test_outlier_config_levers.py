"""Outlier-config parity levers — despike + robust local-polynomial window.

Field-parity slice: the ``outliers:`` block (and its per-station
``overrides``) must be able to express the same active-station detection
levers geo_dataread's ``outlier_overrides.csv`` carries (``despike``,
``window_order``, ``window_robust_iterations``, ``despike_n_sigma``, plus
the pre-existing ``epoch_policy`` and magnitude floors), so the STORE's
cleaned series and the internal ``_cleaned.NEU`` path CAN be configured
identically. Pinned here:

- defaults mirror :class:`gps_analysis.OutlierParams` exactly — an absent
  key reproduces today's behavior (despike off, constant window);
- the new keys parse from ``analysis.yaml`` as fleet-wide GLOBALS, and a
  per-station ``outlier_overrides.csv`` row (the shared resolver — the SAME
  source geo_dataread reads, design §2) overrides them;
- a station configured ``despike: true, window_order: 1`` actually runs
  the leaf with those params (echoed in provenance, ``params_hash``
  differs from the default);
- ``window_order`` out of range fails at LOAD time — globally and inside
  a per-station override (never inside the fault-tolerant detection loop).
"""

import zlib
from pathlib import Path

import numpy as np
import pytest
from gps_analysis import OutlierParams, lineperiodic
from gps_parser.outlier_catalogs import StationOutlierOverride

from gps_api.precompute.config import (
    OutlierConfig,
    load_analysis_config,
)
from gps_api.precompute.outliers import (
    detect_station_outliers,
    station_outlier_params,
)
from gps_api.precompute.sources import StationSeries

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

ANALYSIS_YAML_LEVERS = """\
version: 0
regions:
  reykjanes:
    stations: [SENG, ELDC]
    default_reference_frame: ITRF2014
velocity:
  default_window_years: 2.0
  default_method: wls
detrend:
  default_model: lineperiodic
breakpoints:
  enabled_regions: []
  n_breaks_default: 1
  n_runs: 420
  t_runs: 20
outliers:
  enabled: true
  window_order: 1
  window_robust_iterations: 3
  despike: true
  despike_n_sigma: 6.0
"""

# Per-station levers now live in the deployed CSV (design §2), not the yaml —
# ELDC turns despike back off + bumps the window order (the same row shape
# geo_dataread's _cleaned.NEU path resolves).
ELDC_OVERRIDE = StationOutlierOverride(
    fields={
        "despike": False,
        "window_order": 2,
        "window_robust_iterations": 1,
        "despike_n_sigma": 12.0,
    },
    min_outlier=None,
)


@pytest.fixture()
def gpsconfig_levers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Temp gpsconfig whose sidecar sets the new levers globally + per-station."""
    (tmp_path / "stations.cfg").write_text(STATIONS_CFG)
    (tmp_path / "postprocess.cfg").write_text(POSTPROCESS_CFG)
    (tmp_path / "analysis.yaml").write_text(ANALYSIS_YAML_LEVERS)
    monkeypatch.setenv("GPS_CONFIG_PATH", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------
# Defaults: byte-parity with today's behavior
# ---------------------------------------------------------------------------


def test_defaults_mirror_leaf_outlier_params() -> None:
    """An untouched config maps to the leaf's OWN defaults — the parity pin.

    ``OutlierConfig()`` (and therefore every ``analysis.yaml`` that does not
    name the new keys) must resolve to exactly ``OutlierParams()``: despike
    off, ``window_order`` 0, ``window_robust_iterations`` 2 — today's masks.
    """
    params, floors = station_outlier_params(OutlierConfig(), "ANYSTA")
    assert params == OutlierParams()
    assert params.despike is False
    assert params.window_order == 0
    assert params.window_robust_iterations == 2
    assert params.despike_n_sigma == 10.0
    assert floors == (5.0, 5.0, 10.0)


# ---------------------------------------------------------------------------
# Parsing: analysis.yaml globals + per-station overrides
# ---------------------------------------------------------------------------


def test_levers_parse_from_yaml_globally_and_per_station(
    gpsconfig_levers: Path,
) -> None:
    cfg = load_analysis_config()
    ocfg = cfg.outliers
    assert ocfg.enabled is True
    # Global block values land on the config dataclass...
    assert ocfg.despike is True
    assert ocfg.window_order == 1
    assert ocfg.window_robust_iterations == 3
    assert ocfg.despike_n_sigma == 6.0
    # ...reach a station with no CSV override 1:1...
    seng, _ = station_outlier_params(ocfg, "SENG")
    assert seng.despike is True
    assert seng.window_order == 1
    assert seng.window_robust_iterations == 3
    assert seng.despike_n_sigma == 6.0
    # ...and the per-station CSV override wins on every new key (precedence).
    eldc, _ = station_outlier_params(ocfg, "ELDC", override=ELDC_OVERRIDE)
    assert eldc.despike is False
    assert eldc.window_order == 2
    assert eldc.window_robust_iterations == 1
    assert eldc.despike_n_sigma == 12.0


def test_new_keys_flow_into_config_dict_for_provenance(
    gpsconfig_levers: Path,
) -> None:
    """``as_dict`` (the config-hash / provenance input) carries the globals."""
    payload = load_analysis_config().outliers.as_dict()
    assert payload["despike"] is True
    assert payload["window_order"] == 1
    assert payload["window_robust_iterations"] == 3
    assert payload["despike_n_sigma"] == 6.0


# ---------------------------------------------------------------------------
# Validation: bad enum fails at LOAD time
# ---------------------------------------------------------------------------


def test_window_order_out_of_range_rejected_globally() -> None:
    with pytest.raises(ValueError, match="window_order"):
        OutlierConfig(window_order=3)
    with pytest.raises(ValueError, match="window_order"):
        OutlierConfig(window_order=-1)


def test_window_order_out_of_range_rejected_in_override() -> None:
    """A per-station typo fails the LOAD, not the fault-tolerant run loop."""
    with pytest.raises(ValueError, match=r"overrides\.SENG\.window_order"):
        OutlierConfig(overrides={"SENG": {"window_order": 5}})


def test_unknown_override_key_still_rejected() -> None:
    with pytest.raises(ValueError, match="unknown key"):
        OutlierConfig(overrides={"SENG": {"despike_gap_days": 2.0}})


def test_deprecated_per_station_keys_flagged_for_job_warning() -> None:
    """The guard the job uses to warn on stale yaml per-station config
    (authority moved to the CSVs — design §2)."""
    assert OutlierConfig().deprecated_per_station_keys() is False
    assert (
        OutlierConfig(overrides={"SENG": {"despike": True}})
        .deprecated_per_station_keys()
        is True
    )
    assert (
        OutlierConfig(protect_windows=((2023.9, 2024.1),))
        .deprecated_per_station_keys()
        is True
    )


# ---------------------------------------------------------------------------
# Detection actually runs with the levers (provenance + hash)
# ---------------------------------------------------------------------------


def _series(marker: str) -> StationSeries:
    """Deterministic linear+annual series with one gross spike (north)."""
    n_days = 300
    rng = np.random.default_rng(zlib.crc32(marker.encode()))
    t = 2024.0 + np.arange(n_days, dtype=np.float64) / 365.25
    y = np.vstack(
        [
            1.0 + 12.0 * (t - 2024.0) + 2.0 * np.sin(2 * np.pi * t),
            -2.0 - 8.0 * (t - 2024.0) + 1.5 * np.cos(2 * np.pi * t),
            0.5 + 3.0 * (t - 2024.0) + 4.0 * np.sin(2 * np.pi * t),
        ]
    ) + rng.normal(0.0, 1.5, (3, n_days))
    y[0, 150] += 30.0  # gross north spike, far above the 5 mm floor
    return StationSeries(
        marker=marker,
        t=t,
        y=y,
        sigma=np.full((3, n_days), 1.5),
        source=f"synthetic:lever-fixture:{marker}",
    )


def test_station_override_runs_leaf_with_levers_and_hash_differs() -> None:
    """despike=true + window_order=1 reach the leaf; params_hash moves."""
    default_cfg = OutlierConfig()
    lever = StationOutlierOverride(
        fields={"despike": True, "window_order": 1}, min_outlier=None
    )
    series = _series("SENG")

    baseline = detect_station_outliers(series, lineperiodic, default_cfg, ())
    levered = detect_station_outliers(
        series, lineperiodic, default_cfg, (), override=lever
    )

    # The leaf ran with the overridden params — echoed straight through.
    assert levered.params.despike is True
    assert levered.params.window_order == 1
    assert baseline.params == OutlierParams()
    # Both runs catch the gross spike; neither aborts.
    assert baseline.flags[0, 150] and levered.flags[0, 150]
    assert not baseline.aborted and not levered.aborted
    # Provenance a store consumer sees reflects what actually ran (BGÓ Q8).
    provenance = levered.provenance()
    assert provenance["params"]["despike"] is True
    assert provenance["params"]["window_order"] == 1
    assert provenance["params"]["window_robust_iterations"] == 2
    assert baseline.provenance()["params"]["despike"] is False
    assert levered.params_hash != baseline.params_hash
    # A station WITHOUT an override resolves to the default params — its
    # mask (and hash) is untouched by another station's lever.
    other = detect_station_outliers(_series("ELDC"), lineperiodic, default_cfg, ())
    assert other.params == OutlierParams()
