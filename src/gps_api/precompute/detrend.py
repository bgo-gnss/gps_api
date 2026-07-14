"""Detrend-parameter estimation stage — the WRITER half of the handshake.

``geo_dataread.gps_views.read_detrend_params`` (the reader half, on
geo_dataread ``main``) consumes a ``detrend_params.json`` document::

    {"schema_version": 1, "stations": {"<MARKER>": <leaf record>}}

where each leaf record is exactly
:meth:`gps_analysis.DetrendEstimate.to_record` output. This module makes
the precompute job produce that document (``DESIGN_live_detrending.md``
§0 locked decisions):

- **Estimate is deliberate and occasional; apply is cheap** (§0 preamble):
  this stage produces STORED parameters; geo_dataread later applies them
  purely — no re-fit on read, one background definition everywhere.
- **Plate-first (decision 5):** the ``.NEU`` inputs of the analysis lane
  are plate-removed products (the CDN ``{marker}-plate.NEU`` chain), so
  the estimate runs in — and the record is tagged with — the
  plate-removed processing frame (default tag ``plate:<region frame>``).
  A frame mismatch at apply time is a hard downstream error by design.
- **Method tag (decision 2):** ``estimate_detrend`` stamps
  ``detrend_method`` = ``"step_augmented_robust"`` (outlier stage on, the
  default) vs ``"plain_wls"``; this stage carries it through untouched.
- **UseSTA borrowing (decision 6):** the config's borrower → donor map is
  resolved here into SELF-CONTAINED borrower records (donor's coefficients
  copied, ``borrowed`` provenance set) — the apply path never chases donor
  references.
- **Pinning (decision 7):** a pinned station's record is honored verbatim
  (validated, never refit, never overwritten by a fresh fit).
- **Graceful, loud (decision 4):** a failed validity gate means NO record
  (absent = "no background model") with the reason in the run summary; an
  outlier abort or an excess residual RMS (``max_rms_mm`` — the real-data
  SENG finding: a window spanning active unrest does NOT abort, the robust
  fit swallows the transient into a garbage background) marks the station
  degraded and only writes the record when ``write_degraded`` says so,
  with an explicit ``refs.degraded`` marker. Nothing fails silently.
"""

from __future__ import annotations

import copy
import dataclasses
import datetime
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from gps_analysis import OutlierParams, estimate_detrend, trajectory_from_record

from gps_api import settings
from gps_api.precompute.config import (
    PIN_KEEP,
    AnalysisConfig,
    DetrendConfig,
    StepRecord,
)
from gps_api.precompute.outliers import component_step_epochs, station_outlier_params
from gps_api.precompute.products import DETREND_PARAMS_SCHEMA_VERSION, _iso_z
from gps_api.precompute.sources import COMPONENTS, StationSeries


@dataclasses.dataclass(frozen=True)
class DetrendStageResult:
    """Outcome of the per-region detrend-estimation stage.

    ``records`` holds every station record destined for the document
    (fresh fits + pinned + borrowed + degraded-but-written); the
    membership sets say how each record got there. ``degraded`` /
    ``skipped`` carry the loud reasons for the run summary — a station in
    ``skipped`` (and one in ``degraded`` under ``write_degraded: false``)
    is ABSENT from the document: "no background model", consumers degrade
    to the raw view (design §0.4).
    """

    records: dict[str, dict[str, Any]]
    fitted: tuple[str, ...]
    pinned: tuple[str, ...]
    borrowed: dict[str, str]
    degraded: dict[str, str]
    skipped: dict[str, str]

    def summary(self) -> dict[str, Any]:
        """The ``detrend_params`` member of the ``meta/run.json`` payloads."""
        return {
            "fitted": list(self.fitted),
            "pinned": list(self.pinned),
            "borrowed": dict(self.borrowed),
            "degraded": dict(self.degraded),
            "skipped": dict(self.skipped),
        }


def load_store_records(store: Path) -> dict[str, dict[str, Any]]:
    """Station records of the existing store document ({} when absent).

    Used to honor ``pinned: {STA: keep}`` entries across runs. A corrupt
    or unknown-schema document raises — a pin must fail loudly, never
    silently refit (decision 7).
    """
    path = store / settings.PARAMS_DIR / settings.DETREND_PARAMS_FILE
    if not path.is_file():
        return {}
    doc = json.loads(path.read_text())
    if not isinstance(doc, dict) or doc.get("schema_version") != (
        DETREND_PARAMS_SCHEMA_VERSION
    ):
        raise ValueError(
            f"{path}: not a schema-v{DETREND_PARAMS_SCHEMA_VERSION} "
            "detrend-parameter document"
        )
    stations = doc.get("stations")
    if not isinstance(stations, dict):
        raise ValueError(f"{path}: document has no 'stations' mapping")
    return {str(marker): dict(record) for marker, record in stations.items()}


def _pinned_record(
    marker: str,
    spec: str,
    store_records: dict[str, dict[str, Any]],
    config_dir: Path,
) -> dict[str, Any]:
    """Resolve one ``pinned:`` entry to a validated leaf record.

    ``spec == "keep"`` honors the record already in the store document;
    any other spec is a JSON file (absolute, or relative to the gpsconfig
    dir) holding either the bare leaf record or a schema-v1 document
    containing the station. The record is validated through the leaf's
    :func:`gps_analysis.trajectory_from_record` so an unapplyable pin
    fails HERE, loudly, not at every consumer's read.
    """
    if spec == PIN_KEEP:
        record = store_records.get(marker)
        if record is None:
            raise ValueError(
                f"pinned '{PIN_KEEP}' but station {marker!r} has no record "
                "in the existing store document"
            )
    else:
        path = Path(spec).expanduser()
        if not path.is_absolute():
            path = config_dir / path
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError(f"{path}: pinned record must be a JSON object")
        if "stations" in payload:
            stations = payload["stations"]
            if not isinstance(stations, dict) or marker not in stations:
                raise ValueError(f"{path}: no station {marker!r} in the document")
            record = dict(stations[marker])
        else:
            record = dict(payload)
    trajectory_from_record(record)  # raises ValueError on an unapplyable pin
    return dict(record)


def _borrowed_record(donor_record: dict[str, Any], donor: str) -> dict[str, Any]:
    """Self-contained borrower record from a donor's record (decision 6).

    The donor's coefficients are copied verbatim (a stored record applies
    cleanly to ANY station's epochs — design §2.6); only the ``borrowed``
    provenance slot changes, in exactly the shape the reader surfaces
    (``geo_dataread.gps_views`` reads ``record.get("borrowed")``).
    """
    record = copy.deepcopy(donor_record)
    record["borrowed"] = {
        "from": donor,
        "terms": "all",
        "donor_fitted_at": donor_record.get("fitted_at"),
    }
    return record


def estimate_station_record(
    series: StationSeries,
    model_name: str,
    dcfg: DetrendConfig,
    *,
    step_epochs: tuple[float, ...] = (),
    outlier_params: OutlierParams | None = None,
    min_outlier: tuple[float, float, float] | None = None,
    protect_windows: tuple[tuple[float, float], ...] = (),
    frame: str | None = None,
    fitted_at: datetime.datetime,
    region: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """Fit one station and package the leaf record (design §2).

    Chain: window policy (:meth:`DetrendConfig.window_for`) →
    :func:`gps_analysis.estimate_detrend` (validity gates → step
    augmentation → outlier stage per ``method`` → clean WLS) →
    degrade checks (outlier abort, ``max_rms_mm``) →
    :meth:`~gps_analysis.DetrendEstimate.to_record`.

    Leaf warnings (the abort ``RuntimeWarning``) are captured and re-said
    on stderr — loud in the job log, deterministic under test warning
    filters.

    Returns:
        ``(record, degrade_reason)`` — ``record`` is None for a degraded
        fit under ``write_degraded: false`` (the station must then be
        documented as absent); a written degraded record carries the
        explicit ``refs.degraded`` marker.

    Raises:
        ValueError: From the leaf's validity gates / input validation —
            the caller records the station as skipped (no record, loud).
    """
    window = dcfg.window_for(series.marker, float(series.t[-1]))
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        estimate = estimate_detrend(
            model_name,
            series.t,
            series.y,
            series.sigma,
            window=window,
            step_epochs=(
                np.asarray(step_epochs, dtype=np.float64) if step_epochs else None
            ),
            min_span_years=dcfg.min_span_years,
            min_epochs=dcfg.min_epochs,
            max_gap_years=dcfg.max_gap_years,
            detect=dcfg.method == "step_augmented_robust",
            outlier_params=outlier_params,
            protect_windows=protect_windows,
            min_outlier=(None if min_outlier is None else np.asarray(min_outlier)),
            names=COMPONENTS,
            frame=frame,
        )
    for entry in captured:
        print(
            f"[{series.marker}] detrend estimation: {entry.message}",
            file=sys.stderr,
            flush=True,
        )

    degrade: str | None = None
    if estimate.outlier_abort:
        degrade = "outlier stage aborted (excess-candidate rule); plain-WLS fallback"
    # Per-component RMS sanity gate (BGÓ 2026-07-14): horizontal (N, E)
    # against max_rms_mm, vertical (U) against the looser max_rms_mm_up —
    # GNSS vertical noise runs ~2-3x horizontal. rms rows are N/E/U
    # (sources.COMPONENTS order).
    rms = list(estimate.rms)
    horiz_rms = [rms[i] for i in (0, 1) if i < len(rms) and not math.isnan(rms[i])]
    worst_horiz = max(horiz_rms, default=float("nan"))
    up_rms = rms[2] if len(rms) > 2 else float("nan")
    rms_reason: str | None = None
    if (
        dcfg.max_rms_mm is not None
        and not math.isnan(worst_horiz)
        and worst_horiz > dcfg.max_rms_mm
    ):
        rms_reason = (
            f"horizontal inlier residual RMS {worst_horiz:.1f} mm exceeds "
            f"max_rms_mm {dcfg.max_rms_mm:g} (unmodeled signal in the fit window?)"
        )
    if (
        dcfg.max_rms_mm_up is not None
        and not math.isnan(up_rms)
        and up_rms > dcfg.max_rms_mm_up
    ):
        up_reason = (
            f"vertical inlier residual RMS {up_rms:.1f} mm exceeds "
            f"max_rms_mm_up {dcfg.max_rms_mm_up:g} (unmodeled signal in the fit window?)"
        )
        rms_reason = f"{rms_reason}; {up_reason}" if rms_reason else up_reason
    if rms_reason is not None:
        degrade = f"{degrade}; {rms_reason}" if degrade else rms_reason

    refs: dict[str, Any] = {"region": region, "source": series.source}
    if degrade is not None:
        if not dcfg.write_degraded:
            return None, degrade
        refs["degraded"] = True
        refs["degrade_reason"] = degrade
    record = estimate.to_record(fitted_at=_iso_z(fitted_at), refs=refs)
    return record, degrade


def run_detrend_estimation(
    *,
    cfg: AnalysisConfig,
    region_name: str,
    frame: str,
    stations: tuple[str, ...],
    series_map: dict[str, StationSeries],
    step_catalog: dict[str, tuple[StepRecord, ...]],
    store: Path,
    fitted_at: datetime.datetime,
) -> DetrendStageResult:
    """Run the estimation stage for one region's stations.

    Per station, in order: pinned records are honored verbatim (never
    refit); ``use_sta`` borrowers are deferred and resolved from THIS
    run's records (fresh or pinned — a donor outside the region/run means
    the borrower is skipped loudly); everything else gets a fresh
    :func:`estimate_station_record` fit over the configured window with
    the station's ``steps.csv`` epochs (union over components — the
    leaf's step list is shared across components; an amplitude near zero
    is estimated for components a step does not affect).

    The outlier-stage thresholds (``OutlierParams``, magnitude floors,
    protect windows) reuse the resolved ``outliers:`` block per station —
    one threshold vocabulary for detection and estimation.

    Every non-outcome is recorded on the result (``degraded`` /
    ``skipped`` with reasons) — loud, never silent (decision 4).
    """
    dcfg = cfg.detrend_estimation
    records: dict[str, dict[str, Any]] = {}
    fitted: list[str] = []
    pinned: list[str] = []
    borrowed: dict[str, str] = {}
    degraded: dict[str, str] = {}
    skipped: dict[str, str] = {}

    store_records: dict[str, dict[str, Any]] = {}
    store_records_error: str | None = None
    if any(spec == PIN_KEEP for spec in dcfg.pinned.values()):
        try:
            store_records = load_store_records(store)
        except (ValueError, json.JSONDecodeError, OSError) as exc:
            store_records_error = f"{type(exc).__name__}: {exc}"

    for marker in stations:
        spec = dcfg.pinned.get(marker)
        if spec is not None:
            if spec == PIN_KEEP and store_records_error is not None:
                skipped[marker] = (
                    f"pinned '{PIN_KEEP}' but the existing store document is "
                    f"unreadable: {store_records_error}"
                )
                print(
                    f"[{marker}] detrend params: {skipped[marker]}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            try:
                records[marker] = _pinned_record(
                    marker, spec, store_records, cfg.config_dir
                )
            except (ValueError, OSError, json.JSONDecodeError) as exc:
                skipped[marker] = f"pinned record unavailable: {exc}"
                print(
                    f"[{marker}] detrend params: {skipped[marker]}",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                pinned.append(marker)
                print(
                    f"[{marker}] detrend params: pinned record kept verbatim "
                    "(do-not-refit)",
                    flush=True,
                )
            continue
        if marker in dcfg.use_sta:
            continue  # resolved from this run's records below
        series = series_map.get(marker)
        if series is None:
            skipped[marker] = "no series available this run"
            continue
        outlier_params, floors = station_outlier_params(cfg.outliers, marker)
        step_lists = component_step_epochs(step_catalog.get(marker, ()))
        step_union = tuple(sorted({epoch for epochs in step_lists for epoch in epochs}))
        try:
            record, degrade = estimate_station_record(
                series,
                cfg.detrend_model_for(marker),
                dcfg,
                step_epochs=step_union,
                outlier_params=outlier_params,
                min_outlier=floors,
                protect_windows=cfg.outliers.protect_windows,
                frame=frame,
                fitted_at=fitted_at,
                region=region_name,
            )
        except ValueError as exc:
            skipped[marker] = str(exc)
            print(
                f"[{marker}] detrend params: no record - {exc}",
                file=sys.stderr,
                flush=True,
            )
            continue
        if degrade is not None:
            degraded[marker] = degrade
            print(
                f"[{marker}] detrend params DEGRADED - {degrade}"
                + ("" if record is not None else " (record NOT written)"),
                file=sys.stderr,
                flush=True,
            )
        if record is not None:
            records[marker] = record
            if degrade is None:
                fitted.append(marker)

    for borrower, donor in dcfg.use_sta.items():
        if borrower not in stations:
            continue
        donor_record = records.get(donor)
        if donor_record is None:
            skipped[borrower] = (
                f"use_sta donor {donor!r} produced no record this run "
                f"({skipped.get(donor) or degraded.get(donor) or 'not in this region'})"
            )
            print(
                f"[{borrower}] detrend params: {skipped[borrower]}",
                file=sys.stderr,
                flush=True,
            )
            continue
        records[borrower] = _borrowed_record(donor_record, donor)
        borrowed[borrower] = donor
        print(
            f"[{borrower}] detrend params: borrowed from {donor} (UseSTA)",
            flush=True,
        )

    return DetrendStageResult(
        records=records,
        fitted=tuple(fitted),
        pinned=tuple(pinned),
        borrowed=borrowed,
        degraded=degraded,
        skipped=skipped,
    )
