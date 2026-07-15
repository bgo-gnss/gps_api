"""Cross-repo forcing function for PER-STATION outlier config.

geo_dataread (the internal ``_cleaned.NEU`` path) and gps_api (the store) both
read the shared ``gps_parser.outlier_catalogs`` resolver and map a station's
CSV override → ``gps_analysis.OutlierParams`` + a per-component ``[N, E, U]``
floor. This pins that the two MAPPINGS agree (design
`gps_parser/docs/DESIGN_shared_outlier_config.md` §2) — including the N != E
floor the old ``(H, H, V)`` collapse could not express.

SCOPE (design §2, deliberate): only the PER-STATION config (levers, floors,
protect_windows, steps) is single-sourced. The fleet-wide GLOBAL thresholds
are two-sourced — gps_api from ``analysis.yaml``, geo_dataread from
``OutlierParams()`` defaults — so they agree ONLY while the yaml globals are
left at the leaf defaults. ``test_nondefault_global_is_a_known_divergence``
pins that boundary rather than implying it cannot happen.

Imports geo_dataread (an editable sibling); skipped if it is not installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("geo_dataread")

from geo_dataread.gps_views import resolve_outlier_detection  # noqa: E402
from gps_parser import outlier_catalogs as oc  # noqa: E402

from gps_api.precompute.config import OutlierConfig  # noqa: E402
from gps_api.precompute.outliers import station_outlier_params  # noqa: E402

CATALOG = (
    "sta,despike,window_order,window_robust_iterations,epoch_policy,"
    "despike_n_sigma,min_outlier_n,min_outlier_e,min_outlier_u\n"
    # active station: full lever set + independent N/E/U floor (the case
    # gps_api's old (H,H,V) collapse could NOT represent)
    "SENG,true,1,2,union,10,4,7,15\n"
    # quiet station: only a vertical floor bump, everything else default
    "HOFN,,,,,,,,12\n"
)


@pytest.mark.parametrize("marker", ["SENG", "HOFN"])
def test_geo_dataread_and_gps_api_agree_on_shared_catalog(tmp_path, marker):
    csv_path = tmp_path / "outlier_overrides.csv"
    csv_path.write_text(CATALOG)

    # geo_dataread's resolver (the _cleaned.NEU path)
    geo = resolve_outlier_detection(marker, outlier_overrides=csv_path)

    # gps_api maps the SAME shared-resolver override on top of the yaml globals
    override = oc.read_outlier_overrides(csv_path)[marker]
    api_params, api_floor = station_outlier_params(
        OutlierConfig(), marker, override=override
    )

    # Same OutlierParams on every field the CSV can set...
    assert api_params == geo.params, f"{marker}: OutlierParams diverge"
    # ...and the SAME per-component [N,E,U] floor (the collapse fix — N != E).
    assert api_floor == geo.min_outlier, f"{marker}: floor diverges"


def test_independent_n_e_floor_survives_both_paths(tmp_path):
    """The specific bug: N != E must survive both mappings, not collapse."""
    csv_path = tmp_path / "outlier_overrides.csv"
    csv_path.write_text(CATALOG)
    geo = resolve_outlier_detection("SENG", outlier_overrides=csv_path)
    override = oc.read_outlier_overrides(csv_path)["SENG"]
    _, api_floor = station_outlier_params(OutlierConfig(), "SENG", override=override)
    assert geo.min_outlier == (4.0, 7.0, 15.0)  # N != E preserved
    assert api_floor == (4.0, 7.0, 15.0)


def test_nondefault_global_is_a_known_divergence(tmp_path):
    """GLOBALS are two-sourced (design §2): a non-default gps_api yaml global
    is NOT seen by geo_dataread (which uses OutlierParams defaults). Pin this
    boundary so the per-station single-source claim is not mistaken for a
    global one — an operator who tunes a GLOBAL must do it on both sides.
    """
    csv_path = tmp_path / "outlier_overrides.csv"
    csv_path.write_text(CATALOG)
    override = oc.read_outlier_overrides(csv_path)["HOFN"]
    geo = resolve_outlier_detection("HOFN", outlier_overrides=csv_path)
    # gps_api operator sets a non-default GLOBAL threshold in analysis.yaml
    api_params, _ = station_outlier_params(
        OutlierConfig(window_n_sigma=3.5), "HOFN", override=override
    )
    assert api_params.window_n_sigma == 3.5
    assert geo.params.window_n_sigma == 4.0  # geo_dataread never saw the yaml
    assert api_params != geo.params  # globals diverge — the documented limit
