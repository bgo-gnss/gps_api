"""Parallel two-stage GBIS4TS break detection for the precompute job.

The fleet workload is embarrassingly parallel: every station x component
chain is an independent pure call ``detect_breakpoints(t, y[i], wn_amp,
...) -> InversionResult`` (RNG seeded per call, no shared state). Run
serially at production settings (``n_runs = 1e6``, ~2.5 h per chain) a
full 173-station fleet is 519 chains — ~54 days of wall clock. This
module makes that feasible (perf-audit findings #1 and #6):

- **Process-pool fan-out** — one :class:`ComponentTask` per station x
  component, executed by :func:`run_break_task` in a
  ``concurrent.futures.ProcessPoolExecutor`` (``spawn`` context — safe
  with a threaded parent). Near-linear speedup in cores.
- **Bounded memory** — the worker consumes the full ``InversionResult``
  (the kept chain ``m_keep`` is ~48–96 MB at 1e6 runs) *inside* the
  worker and returns only a scalar :class:`BreakSummary`; the chain is
  never pickled back, so peak RSS stays ~(workers x one chain), not
  (fleet x one chain).
- **Triage → confirm** (plan §10.7) — an optional cheap screening pass
  (short ``triage_n_runs`` chains) over every gated station flags
  *candidate* break stations by the posterior significance of the
  trend-change parameter(s); only flagged candidates get the full
  production-length confirm chains. Both chain lengths come from the
  ``breakpoints:`` block of ``analysis.yaml`` — no hardcoding, and the
  flagged/screened counts are logged and recorded in provenance (no
  silent caps).

Fault tolerance mirrors the job's station rule: one failed chain records
its station and is skipped — it never sinks the batch. A station is
atomic (as in the serial path): if any of its component chains fails, the
station produces no break entries at all.

No math is derived here (MATH_STANDARDS rule): the estimator is
``gps_analysis.detect_breakpoints``; this module only orchestrates calls
and summarizes the returned chain (posterior mean/std of the
trend-change rows over the post-annealing samples).
"""

from __future__ import annotations

import dataclasses
import multiprocessing
import os
import sys
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import (
    FIRST_COMPLETED,
    Executor,
    Future,
    ProcessPoolExecutor,
    wait,
)
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np
from gps_analysis import detect_breakpoints

from gps_api.precompute.sources import FloatArray

if TYPE_CHECKING:  # typing only — no runtime import needed
    from gps_analysis import InversionResult

#: Chain rows holding the velocity-change parameter(s) g_k, per model
#: (sampler layout == MATLAB order, ``transient.prepare_bounds``):
#: BPD1 ``[a, v, g, t_b, kappa, beta]`` -> row 2; BPD2 adds ``(g2, t_b2)``.
_TREND_CHANGE_ROWS: dict[str, tuple[int, ...]] = {"BPD1": (2,), "BPD2": (2, 4)}

#: BLAS/OpenMP thread caps exported to pool workers. The GBIS4TS hot loop
#: is scalar-bound (the O(N^2) Schur kernel), so per-worker BLAS threading
#: buys nothing and oversubscribes the node once the pool fans out.
_BLAS_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


@dataclasses.dataclass(frozen=True)
class ComponentTask:
    """One station x component break-detection work item (picklable).

    Attributes:
        marker: Station marker.
        component: Component name (``"north"`` / ``"east"`` / ``"up"``).
        t: Epochs [yr, fractional year], shape (N,).
        y: Displacement series [mm], shape (N,) — passed straight to the
            estimator (it zero-references internally).
        wn_amp: Fixed white-noise amplitude [mm].
        n_breaks: 1 (BPD1) or 2 (BPD2).
        n_runs: Kept MCMC iterations for this chain.
        t_runs: Kept iterations per annealing temperature.
        seed: RNG seed (same value the serial path would use, so a pooled
            run reproduces the serial summaries exactly).
    """

    marker: str
    component: str
    t: FloatArray
    y: FloatArray
    wn_amp: float
    n_breaks: int
    n_runs: int
    t_runs: int
    seed: int | None


@dataclasses.dataclass(frozen=True)
class BreakSummary:
    """Scalar summary of one chain — everything the products need.

    Deliberately holds **no arrays**: the worker reduces the full
    ``InversionResult`` (whose kept chain is ~48–96 MB at 1e6 runs) to
    this before returning, which is what keeps the process pool
    memory-bounded (audit finding #6).

    Attributes:
        marker / component / model: Identity of the chain.
        optimal: Best-likelihood parameter vector (MATLAB order), as
            plain floats.
        y_ref: Start baseline subtracted internally by the estimator [mm].
        wn_amp: Fixed white-noise amplitude used [mm].
        n_runs: Kept MCMC iterations of this chain.
        trend_change_z: max_k |mean(g_k)| / std(g_k) over the
            post-annealing samples — the triage significance of the
            velocity change(s). ``inf`` when the posterior collapsed to a
            nonzero point mass.
    """

    marker: str
    component: str
    model: str
    optimal: tuple[float, ...]
    y_ref: float
    wn_amp: float
    n_runs: int
    trend_change_z: float


@dataclasses.dataclass(frozen=True)
class TriageStats:
    """What the screening stage did (logged + recorded in provenance)."""

    n_runs: int
    t_runs: int
    sigma: float
    stations_screened: int
    stations_flagged: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class BreakDetectionOutcome:
    """Result of :func:`detect_station_breaks`.

    Attributes:
        summaries: Confirmed per-station summaries, one
            :class:`BreakSummary` per component in task order. Only
            stations whose every component chain succeeded appear
            (station atomicity, matching the serial path).
        failures: ``marker -> reason`` for stations whose screen or
            confirm chain raised — recorded, never re-raised.
        triage: Screening statistics, or ``None`` when triage was
            disabled (``triage_n_runs = 0`` — every station confirmed).
    """

    summaries: dict[str, tuple[BreakSummary, ...]]
    failures: dict[str, str]
    triage: TriageStats | None


def _summarize(result: InversionResult, task: ComponentTask) -> BreakSummary:
    """Reduce a full inversion result to the scalar summary (worker-side).

    The triage significance is the largest posterior ``|mean|/std`` over
    the trend-change row(s) of the kept chain, after discarding the
    simulated-annealing span (``16 * t_runs`` iterations, capped at half
    the chain so short screens still keep samples).
    """
    rows = _TREND_CHANGE_ROWS.get(result.model)
    if rows is None:
        raise ValueError(
            f"unsupported model {result.model!r} for break summaries "
            f"(expected one of {sorted(_TREND_CHANGE_ROWS)})"
        )
    n_kept = result.m_keep.shape[1]
    burn = min(16 * task.t_runs, n_kept // 2)
    z_max = 0.0
    for row in rows:
        samples = result.m_keep[row, burn:]
        mean = float(np.mean(samples))
        std = float(np.std(samples))
        if std > 0.0:
            z = abs(mean) / std
        else:  # point-mass posterior: significant iff away from zero
            z = float("inf") if mean != 0.0 else 0.0
        z_max = max(z_max, z)
    return BreakSummary(
        marker=task.marker,
        component=task.component,
        model=result.model,
        optimal=tuple(float(v) for v in result.optimal),
        y_ref=float(result.y_ref),
        wn_amp=task.wn_amp,
        n_runs=task.n_runs,
        trend_change_z=z_max,
    )


def run_break_task(task: ComponentTask) -> BreakSummary:
    """Execute one chain and return only its scalar summary (pool worker).

    The full ``InversionResult`` (kept chain included) lives and dies
    inside this call — nothing array-shaped crosses the process boundary.
    """
    result = detect_breakpoints(
        task.t,
        task.y,
        task.wn_amp,
        n_breaks=task.n_breaks,
        n_runs=task.n_runs,
        t_runs=task.t_runs,
        seed=task.seed,
    )
    summary = _summarize(result, task)
    del result  # drop m_keep/p_keep before returning (audit #6)
    return summary


@contextmanager
def _single_threaded_blas() -> Iterator[None]:
    """Cap BLAS/OpenMP threads for pool workers (parent env inherited).

    Workers are spawned lazily while the executor is alive, so the caps
    must stay exported for the pool's whole lifetime; previous values (or
    absence) are restored on exit. Explicit user settings win —
    ``setdefault`` only.
    """
    previous = {var: os.environ.get(var) for var in _BLAS_ENV_VARS}
    for var in _BLAS_ENV_VARS:
        os.environ.setdefault(var, "1")
    try:
        yield
    finally:
        for var, value in previous.items():
            if value is None:
                if os.environ.get(var) == "1":
                    del os.environ[var]
            else:
                os.environ[var] = value


def resolve_workers(max_workers: int | None, n_tasks: int) -> int:
    """Effective pool size: ``min(cpu_count, n_tasks)`` unless overridden.

    ``0`` means "run inline in this process" (serial debug/test mode).
    """
    if max_workers is not None:
        if max_workers < 0:
            raise ValueError(f"max_workers must be >= 0, got {max_workers}")
        return min(max_workers, n_tasks) if max_workers else 0
    return min(os.cpu_count() or 1, n_tasks)


def _run_stage(
    tasks: Sequence[ComponentTask],
    executor: Executor | None,
    stage: str,
) -> tuple[dict[tuple[str, str], BreakSummary], dict[str, str]]:
    """Run one stage's tasks (pooled or inline) with per-station tolerance.

    Returns ``(summaries by (marker, component), failures by marker)`` —
    a raising chain records its station and never propagates.
    """
    summaries: dict[tuple[str, str], BreakSummary] = {}
    failures: dict[str, str] = {}

    def _record(task: ComponentTask, outcome: BreakSummary | BaseException) -> None:
        if isinstance(outcome, BaseException):
            reason = f"{type(outcome).__name__}: {outcome}"
            failures.setdefault(
                task.marker, f"gbis {stage} ({task.component}) {reason}"
            )
            print(
                f"[{task.marker}] gbis {stage} FAILED ({task.component}) — {reason}",
                file=sys.stderr,
                flush=True,
            )
        else:
            summaries[(task.marker, task.component)] = outcome
            print(
                f"[{task.marker}] gbis {stage} chain done "
                f"({task.component}, n_runs={task.n_runs})",
                flush=True,
            )

    if executor is None:  # inline mode (max_workers == 0)
        for task in tasks:
            try:
                _record(task, run_break_task(task))
            except Exception as exc:  # noqa: BLE001 — batch survives one bad chain
                _record(task, exc)
        return summaries, failures

    pending: dict[Future[BreakSummary], ComponentTask] = {
        executor.submit(run_break_task, task): task for task in tasks
    }
    while pending:
        done, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            task = pending.pop(future)
            error = future.exception()
            _record(task, future.result() if error is None else error)
    return summaries, failures


def detect_station_breaks(
    station_tasks: Mapping[str, Sequence[ComponentTask]],
    *,
    triage_n_runs: int,
    triage_t_runs: int,
    triage_sigma: float,
    max_workers: int | None,
) -> BreakDetectionOutcome:
    """Run the (optionally two-stage) parallel break detection.

    Args:
        station_tasks: ``marker -> component tasks`` at **confirm**
            settings (the tasks' ``n_runs``/``t_runs`` are the full
            production chain lengths).
        triage_n_runs: Kept iterations of the screening chains; ``0``
            disables triage (every station goes straight to confirm —
            the pre-triage behavior).
        triage_t_runs: Annealing iterations of the screening chains.
        triage_sigma: Significance threshold — a station is flagged as a
            break *candidate* when any component's posterior
            ``|mean|/std`` of a trend-change parameter reaches it.
        max_workers: Pool size (``None`` -> ``min(cpu_count, n_tasks)``,
            ``0`` -> inline serial execution).

    Returns:
        :class:`BreakDetectionOutcome` — confirmed summaries per station,
        recorded failures, and the triage statistics (``None`` when
        triage was off).
    """
    markers = list(station_tasks)
    n_tasks = sum(len(tasks) for tasks in station_tasks.values())
    workers = resolve_workers(max_workers, n_tasks)
    failures: dict[str, str] = {}

    def _detect(executor: Executor | None) -> BreakDetectionOutcome:
        triage: TriageStats | None = None
        if triage_n_runs > 0:
            screen_tasks = [
                dataclasses.replace(task, n_runs=triage_n_runs, t_runs=triage_t_runs)
                for marker in markers
                for task in station_tasks[marker]
            ]
            screened, screen_failures = _run_stage(screen_tasks, executor, "triage")
            failures.update(screen_failures)
            flagged = tuple(
                marker
                for marker in markers
                if marker not in screen_failures
                and any(
                    screened[(marker, task.component)].trend_change_z >= triage_sigma
                    for task in station_tasks[marker]
                )
            )
            triage = TriageStats(
                n_runs=triage_n_runs,
                t_runs=triage_t_runs,
                sigma=triage_sigma,
                stations_screened=len(markers) - len(screen_failures),
                stations_flagged=flagged,
            )
            print(
                f"gbis triage: flagged {len(flagged)} of "
                f"{triage.stations_screened} screened station(s) "
                f"(threshold {triage_sigma}σ, screen n_runs={triage_n_runs}); "
                f"confirming flagged candidates at full n_runs",
                flush=True,
            )
        else:
            flagged = tuple(markers)

        confirm_tasks = [task for marker in flagged for task in station_tasks[marker]]
        confirmed, confirm_failures = _run_stage(confirm_tasks, executor, "confirm")
        failures.update(confirm_failures)
        summaries = {
            marker: tuple(
                confirmed[(marker, task.component)] for task in station_tasks[marker]
            )
            for marker in flagged
            if marker not in confirm_failures
        }
        return BreakDetectionOutcome(
            summaries=summaries, failures=failures, triage=triage
        )

    if workers == 0:
        return _detect(None)
    # spawn: fresh interpreters (safe with a threaded parent, and the
    # BLAS caps below reach the workers' import of numpy/gps_analysis).
    context = multiprocessing.get_context("spawn")
    with _single_threaded_blas():
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            print(
                f"gbis: fanning {n_tasks} chain(s) over {workers} workers",
                flush=True,
            )
            return _detect(pool)
