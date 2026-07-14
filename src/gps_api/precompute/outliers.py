"""Outlier-detection stage of the precompute job (design §5.1, slice 2).

Wires the ``gps_analysis.outliers`` leaf into the per-station chain,
non-destructively: the leaf returns a MASK plus diagnostics — this module
maps config → :class:`gps_analysis.OutlierParams`, builds the
**step-augmented** model inputs from the deployed per-station step catalog
(``steps.csv`` — TOS equipment changes + skjálftalísa coseismic offsets),
and packages the result for the writers. The raw series is never touched;
flags ride as additive Parquet columns (:mod:`gps_api.precompute.products`).

Hard requirement from the real-data verification (SENG vs HOFN): the
detection is only as safe as the trajectory model it is handed — a plain
``lineperiodic`` with no steps over-flags real signal on active stations.
The caller therefore always passes ``step_epochs`` from the station's step
catalog; component-specific catalog rows (``N``/``E``/``U``/``ALL``) map to
per-component step lists (design §5.4).

Hard rule (BGÓ): every downstream parameter estimate fits on the INLIERS
(:func:`mask_station_series` / per-component masks in the job), but the raw
series is always retrievable — flags are additive, never a filter.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any

import numpy as np
from gps_analysis import (
    OutlierDetection,
    OutlierParams,
    detect_outliers,
    remove_trend,
    with_steps,
)
from numpy.typing import NDArray

from gps_api.precompute.config import OutlierConfig, StepRecord
from gps_api.precompute.sources import (
    COMPONENTS,
    FloatArray,
    StationSeries,
    yearf_to_datetime,
)

#: Provenance method tag of the detection products (design §5.2).
OUTLIER_METHOD = "hampel-trajectory"

#: Column order of ``meta/suspected_steps.csv`` (BGÓ requirement #4 / Q5 —
#: the operator-review deliverable; fixed order so downstream tooling and
#: visual diffing stay stable).
SUSPECTED_STEPS_COLUMNS: tuple[str, ...] = (
    "region",
    "sta",
    "component",
    "kind",
    "sign",
    "step_evidence",
    "t_start_yearf",
    "t_end_yearf",
    "start_time",
    "end_time",
    "station_aborted",
)


@dataclasses.dataclass(frozen=True)
class SuspectedStep:
    """One protected candidate cluster, normalized for operator review.

    The precompute-side view of :class:`gps_analysis.SuspectedEvent`:
    component index resolved to its name, epochs carried both as
    fractional years and UTC timestamps. These rows become
    ``meta/suspected_steps.csv`` — candidate ``steps.csv`` entries and
    suspected-icing/transient hints for visual assessment (BGÓ Q5).
    """

    marker: str
    component: str
    kind: str
    sign: int
    step_evidence: float
    t_start: float
    t_end: float

    def as_row(self, region: str, *, station_aborted: bool) -> dict[str, Any]:
        """CSV row (:data:`SUSPECTED_STEPS_COLUMNS` order); NaN D → empty."""
        return {
            "region": region,
            "sta": self.marker,
            "component": self.component,
            "kind": self.kind,
            "sign": self.sign,
            "step_evidence": (
                "" if math.isnan(self.step_evidence) else f"{self.step_evidence:.3f}"
            ),
            "t_start_yearf": f"{self.t_start:.6f}",
            "t_end_yearf": f"{self.t_end:.6f}",
            "start_time": _iso_z(self.t_start),
            "end_time": _iso_z(self.t_end),
            "station_aborted": str(station_aborted).lower(),
        }

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready mapping (Parquet provenance ``suspected_events``)."""
        return {
            "component": self.component,
            "kind": self.kind,
            "sign": self.sign,
            "step_evidence": (
                None if math.isnan(self.step_evidence) else self.step_evidence
            ),
            "t_start_yearf": self.t_start,
            "t_end_yearf": self.t_end,
        }


def _iso_z(yearf: float) -> str:
    """Fractional year → contract ISO-8601 ``Z`` string."""
    return yearf_to_datetime(yearf).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclasses.dataclass(frozen=True)
class StationOutliers:
    """One station's detection outcome — masks + diagnostics, never data.

    All per-epoch arrays are shaped ``(3, N)`` in :data:`COMPONENTS` row
    order. ``detrended`` carries the residuals of the outlier-robust
    step-augmented inlier fit **evaluated at ALL epochs** (design §5.1 —
    flagged epochs keep a detrended value; they carry flags instead of
    disappearing). On ``aborted`` the flags are all-False by the leaf's
    §3.5 rule — loud (recorded in ``meta/run.json``), never a silent cap.
    """

    marker: str
    flags: NDArray[np.bool_]
    candidates: NDArray[np.bool_]
    reasons: NDArray[np.uint8]
    protected: NDArray[np.uint8]
    detrended: FloatArray
    events: tuple[SuspectedStep, ...]
    params: OutlierParams
    floors: tuple[float, float, float]
    step_epochs: tuple[tuple[float, ...], ...]
    aborted: bool
    converged: bool
    n_iterations: int

    @property
    def union_flags(self) -> NDArray[np.bool_]:
        """Epoch flagged in ANY component (the ``outlier_epoch`` column)."""
        return np.asarray(self.flags.any(axis=0), dtype=np.bool_)

    @property
    def params_hash(self) -> str:
        """Stable hash of the resolved detection inputs (BGÓ Q8 provenance).

        Covers the :class:`gps_analysis.OutlierParams` echo, the
        per-component floors and the step epochs actually used — everything
        that shaped this station's mask.
        """
        payload = {
            "params": dataclasses.asdict(self.params),
            "min_outlier_mm": list(self.floors),
            "step_epochs": [list(epochs) for epochs in self.step_epochs],
        }
        return _sha16(payload)

    def provenance(self) -> dict[str, Any]:
        """The Parquet ``outliers`` provenance object (design §5.2)."""
        return {
            "method": OUTLIER_METHOD,
            "params": dataclasses.asdict(self.params),
            "min_outlier_mm": dict(zip(COMPONENTS, self.floors, strict=True)),
            "step_epochs": {
                component: list(epochs)
                for component, epochs in zip(COMPONENTS, self.step_epochs, strict=True)
            },
            "n_flagged": self._per_component(self.flags),
            "n_candidates": self._per_component(self.candidates),
            "n_protected": self._per_component(self.protected != 0),
            "suspected_events": [event.as_dict() for event in self.events],
            "aborted": self.aborted,
            "converged": self.converged,
            "n_iterations": self.n_iterations,
            "params_hash": self.params_hash,
        }

    @staticmethod
    def _per_component(mask: NDArray[np.bool_]) -> dict[str, int]:
        return {
            component: int(np.count_nonzero(mask[i]))
            for i, component in enumerate(COMPONENTS)
        }


def _sha16(payload: dict[str, Any]) -> str:
    """First 16 hex chars of the SHA-256 of a canonical JSON encoding."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def config_hash(ocfg: OutlierConfig) -> str:
    """Stable hash of the whole resolved ``outliers:`` block.

    Recorded in the breaks-product provenance (BGÓ Q8: GBIS4TS consumes the
    cleaned series and records the outlier-params hash it consumed). Covers
    the global settings, the protect windows AND the per-station overrides —
    any change that could move a mask changes the hash.
    """
    return _sha16(ocfg.as_dict())


def station_outlier_params(
    ocfg: OutlierConfig, marker: str
) -> tuple[OutlierParams, tuple[float, float, float]]:
    """Map the config block to the leaf's thresholds for one station.

    Returns the :class:`gps_analysis.OutlierParams` (which re-validates
    every threshold) and the per-component magnitude floors a_min in
    :data:`COMPONENTS` order — ``(horizontal, horizontal, vertical)`` mm
    (design §3.4.1; defaults 5/5/10 mm, per-station overridable — BGÓ Q4).

    The active-station levers (Stage-0 ``despike`` + the robust
    local-polynomial identifier ``window_order``/``window_robust_iterations``
    + ``despike_n_sigma``) map through 1:1 — field parity with the
    ``outlier_overrides.csv`` path geo_dataread's ``_cleaned.NEU`` writer
    resolves, so the store's cleaned series can be configured identically.
    """
    settings = ocfg.settings_for(marker)
    params = OutlierParams(
        scale_estimator=str(settings["scale_estimator"]),
        global_n_sigma=float(settings["global_n_sigma"]),
        window_days=float(settings["window_days"]),
        window_n_sigma=float(settings["window_n_sigma"]),
        window_min_count=int(settings["window_min_count"]),
        window_order=int(settings["window_order"]),
        window_robust_iterations=int(settings["window_robust_iterations"]),
        despike=bool(settings["despike"]),
        despike_n_sigma=float(settings["despike_n_sigma"]),
        scale_floor=float(settings["scale_floor"]),
        max_run_days=float(settings["max_run_days"]),
        cluster_gap_days=float(settings["cluster_gap_days"]),
        run_sign_fraction=float(settings["run_sign_fraction"]),
        step_evidence_sigma=float(settings["step_evidence_sigma"]),
        step_window_days=float(settings["step_window_days"]),
        max_flag_fraction=float(settings["max_flag_fraction"]),
        max_iterations=int(settings["max_iterations"]),
        epoch_policy=str(settings["epoch_policy"]),
    )
    horizontal = float(settings["min_outlier_horizontal_mm"])
    vertical = float(settings["min_outlier_vertical_mm"])
    return params, (horizontal, horizontal, vertical)


def component_step_epochs(
    steps: tuple[StepRecord, ...],
) -> tuple[tuple[float, ...], ...]:
    """Per-component step-epoch lists from the station's catalog rows.

    ``ALL`` rows reach every component; ``N``/``E``/``U`` rows only their
    own (design §5.4). Epochs are sorted and de-duplicated per component —
    a duplicated epoch would make the step design rank-deficient.
    """
    return tuple(
        tuple(
            sorted(
                {record.epoch_yearf for record in steps if record.applies_to(component)}
            )
        )
        for component in COMPONENTS
    )


def _normalize_events(
    detection: OutlierDetection,
    marker: str,
    component_of: dict[int, str],
) -> list[SuspectedStep]:
    """Leaf ``SuspectedEvent``s → operator rows (component index → name)."""
    return [
        SuspectedStep(
            marker=marker,
            component=component_of[event.component],
            kind=event.kind,
            sign=int(event.sign),
            step_evidence=float(event.step_evidence),
            t_start=float(event.t_start),
            t_end=float(event.t_end),
        )
        for event in detection.suspected_events
    ]


def detect_station_outliers(
    series: StationSeries,
    model: Any,
    ocfg: OutlierConfig,
    steps: tuple[StepRecord, ...],
) -> StationOutliers:
    """Run the leaf detection for one station with its step catalog.

    The step-augmented model inputs are built here (the empirical hard
    requirement — see module docstring): when every component shares one
    step list (the common ``ALL``-rows case) a single ``(3, N)`` leaf call
    preserves the leaf's own ``epoch_policy`` handling; component-specific
    catalogs fall back to per-component calls with the union policy applied
    across them (an abort in any component aborts the station — the leaf's
    whole-call abort semantics, kept uniform).

    Returns:
        :class:`StationOutliers` — flags/reasons/protected masks, the
        step-augmented all-epoch detrended residuals, suspected-event rows,
        and the parameter echo for provenance.
    """
    params, floors = station_outlier_params(ocfg, series.marker)
    step_lists = component_step_epochs(steps)
    floors_arr = np.asarray(floors, dtype=np.float64)
    component_of = dict(enumerate(COMPONENTS))

    if len(set(step_lists)) == 1:
        epochs = np.asarray(step_lists[0], dtype=np.float64)
        detection = detect_outliers(
            model,
            series.t,
            series.y,
            series.sigma,
            step_epochs=epochs if epochs.size else None,
            protect_windows=ocfg.protect_windows,
            min_outlier=floors_arr,
            params=params,
            names=COMPONENTS,
        )
        fit_model = with_steps(model, epochs) if epochs.size else model
        detrended = np.asarray(
            remove_trend(fit_model, series.t, series.y, detection.fits),
            dtype=np.float64,
        )
        return StationOutliers(
            marker=series.marker,
            flags=detection.flags.copy(),
            candidates=detection.candidates.copy(),
            reasons=detection.reasons.copy(),
            protected=detection.protected.copy(),
            detrended=detrended,
            events=tuple(_normalize_events(detection, series.marker, component_of)),
            params=params,
            floors=floors,
            step_epochs=step_lists,
            aborted=detection.excess_flag_abort,
            converged=detection.converged,
            n_iterations=detection.n_iterations,
        )

    # Component-specific step catalogs: per-component leaf calls, then the
    # cross-component policy applied across them (mirrors the leaf's §3.4.4).
    n = series.t.size
    flags = np.zeros((len(COMPONENTS), n), dtype=np.bool_)
    candidates = np.zeros_like(flags)
    reasons = np.zeros((len(COMPONENTS), n), dtype=np.uint8)
    protected = np.zeros_like(reasons)
    detrended = np.zeros((len(COMPONENTS), n), dtype=np.float64)
    events: list[SuspectedStep] = []
    aborted = False
    converged = True
    n_iterations = 0
    for i, component in enumerate(COMPONENTS):
        epochs = np.asarray(step_lists[i], dtype=np.float64)
        detection = detect_outliers(
            model,
            series.t,
            series.y[i],
            None if series.sigma is None else series.sigma[i],
            step_epochs=epochs if epochs.size else None,
            protect_windows=ocfg.protect_windows,
            min_outlier=floors_arr[i],
            params=params,
            names=[component],
        )
        flags[i] = detection.flags
        candidates[i] = detection.candidates
        reasons[i] = detection.reasons
        protected[i] = detection.protected
        fit_model = with_steps(model, epochs) if epochs.size else model
        detrended[i] = np.asarray(
            remove_trend(fit_model, series.t, series.y[i], detection.fits[0]),
            dtype=np.float64,
        )
        events.extend(_normalize_events(detection, series.marker, {0: component}))
        aborted = aborted or detection.excess_flag_abort
        converged = converged and detection.converged
        n_iterations = max(n_iterations, detection.n_iterations)
    if aborted:
        # Whole-station abort, matching the leaf's single-call semantics
        # (any component over the candidate fraction ⇒ all flags all-False).
        flags[:] = False
        converged = False
    elif params.epoch_policy == "union":
        flags[:] = flags.any(axis=0)[np.newaxis, :]
    return StationOutliers(
        marker=series.marker,
        flags=flags,
        candidates=candidates,
        reasons=reasons,
        protected=protected,
        detrended=detrended,
        events=tuple(events),
        params=params,
        floors=floors,
        step_epochs=step_lists,
        aborted=aborted,
        converged=converged,
        n_iterations=n_iterations,
    )


def mask_station_series(
    series: StationSeries, keep: NDArray[np.bool_]
) -> StationSeries:
    """Inlier view of a series for downstream estimates (never the store).

    Hard rule (BGÓ): parameter estimates fit on the inliers. This builds
    the epoch-masked copy the velocity/deformation stages consume; the
    store keeps every epoch. ``keep=True`` epochs survive.
    """
    if keep.all():
        return series
    return dataclasses.replace(
        series,
        t=series.t[keep],
        y=series.y[:, keep],
        sigma=None if series.sigma is None else series.sigma[:, keep],
    )
