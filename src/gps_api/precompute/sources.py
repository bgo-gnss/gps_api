"""Displacement-series input for the precompute job (data source = parameter).

Two sources, same shape (mirrors ``gps_plot.dev_viz``, plan §2 thread C):

- :func:`load_neu` — a published ``.NEU`` product file
  (``date time dN DN dE DE dU DU`` rows, mm; the format the aflogun v0
  loader verified against the live CDN 2026-07-08). Point ``data.neu_dir``
  in ``analysis.yaml`` (or ``--neu-dir``) at a directory of
  ``<MARKER>.NEU`` files to run on real data.
- :func:`synthetic_station` — velocity-break trajectory + seasonal + white
  noise built entirely from ``gps_analysis`` forward models, so the job is
  runnable end-to-end without station data (``--synthetic``).

Epochs are fractional years (``yearf``, the lane convention);
:func:`yearf_to_datetime` converts to timezone-aware UTC datetimes for the
API-facing products (contract: ISO-8601 ``Z``).
"""

from __future__ import annotations

import dataclasses
import datetime
import zlib
from pathlib import Path

import numpy as np
from gps_analysis import BPD1Params, bpd1_forward, periodic
from gtimes.timefunc import TimefromYearf, dTimetoYearf
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

COMPONENTS: tuple[str, str, str] = ("north", "east", "up")


@dataclasses.dataclass(frozen=True)
class StationSeries:
    """One station's N/E/U displacement series.

    Attributes:
        marker: Station marker (e.g. ``"SENG"``).
        t: Epochs, shape (N,), fractional years [yr], ascending.
        y: Displacements, shape (3, N) [mm], rows in :data:`COMPONENTS` order.
        sigma: 1-σ observation uncertainties, shape (3, N) [mm], or None.
        source: Provenance tag (``"neu:<path>"`` / ``"synthetic:..."``).
    """

    marker: str
    t: FloatArray
    y: FloatArray
    sigma: FloatArray | None
    source: str

    def component_index(self, component: str) -> int:
        """Row index of ``component`` (case-insensitive)."""
        try:
            return [c.lower() for c in COMPONENTS].index(component.lower())
        except ValueError:
            raise ValueError(
                f"unknown component {component!r}; have {COMPONENTS}"
            ) from None


def yearf_to_datetime(yearf: float) -> datetime.datetime:
    """Fractional year → timezone-aware UTC datetime (via gtimes)."""
    value = TimefromYearf(float(yearf))
    if not isinstance(value, datetime.datetime):  # gtimes is untyped upstream
        raise TypeError(f"TimefromYearf returned {type(value).__name__}")
    if value.tzinfo is None:
        return value.replace(tzinfo=datetime.UTC)
    return value.astimezone(datetime.UTC)


def load_neu(path: str | Path, *, marker: str | None = None) -> StationSeries:
    """Load one published ``.NEU`` product file.

    Format (aflogun-verified): whitespace-separated
    ``date time dN DN dE DE dU DU`` rows in mm, time format
    ``%Y/%m/%d %H:%M:%S``; ``#`` comments and stray header lines skipped
    defensively. Rows are sorted by epoch.

    Args:
        path: ``.NEU`` file path.
        marker: Station marker; defaults to the file stem.

    Raises:
        ValueError: When no data rows parse.
    """
    path = Path(path)
    t_list: list[float] = []
    rows: list[list[float]] = []
    for line in path.read_text().splitlines():
        stripped = line.lstrip()
        if not stripped[:1].isdigit():  # headers, comments, junk lines
            continue
        fields = stripped.split()
        if len(fields) < 8:
            continue
        stamp = datetime.datetime.strptime(
            f"{fields[0]} {fields[1]}", "%Y/%m/%d %H:%M:%S"
        )
        t_list.append(float(dTimetoYearf(stamp)))
        rows.append([float(v) for v in fields[2:8]])
    if not rows:
        raise ValueError(f"no data rows parsed from {path}")
    data = np.asarray(rows, dtype=np.float64)  # columns dN DN dE DE dU DU
    order = np.argsort(np.asarray(t_list, dtype=np.float64))
    t = np.asarray(t_list, dtype=np.float64)[order]
    data = data[order]
    return StationSeries(
        marker=marker or path.stem,
        t=t,
        y=data[:, (0, 2, 4)].T.copy(),
        sigma=data[:, (1, 3, 5)].T.copy(),
        source=f"neu:{path}",
    )


def synthetic_station(
    marker: str,
    *,
    seed: int = 0,
    n_days: int = 365,
    t0: float = 2024.0,
    wn_amp: float = 2.0,
) -> StationSeries:
    """Deterministic synthetic N/E/U series with one velocity break.

    Composition per component (all forward models from ``gps_analysis`` —
    no math is derived here): ``y = bpd1_forward(params, t)
    + periodic(t, ...) + ε``, ε ~ N(0, wn_amp²), daily sampling. The RNG is
    seeded from ``(seed, crc32(marker))`` so each station gets a distinct
    but reproducible series. Intercepts are kept small (|a| ≤ 3 mm) so the
    series is compatible with the GBIS4TS intercept prior regardless of
    where zero-referencing is applied.

    Args:
        marker: Station marker stored on the result (and seeding the RNG).
        seed: Base RNG seed.
        n_days: Number of daily epochs N.
        t0: First epoch [yr, fractional year].
        wn_amp: White-noise standard deviation [mm].
    """
    rng = np.random.default_rng([seed, zlib.crc32(marker.encode())])
    t = t0 + np.arange(n_days, dtype=np.float64) / 365.25
    t_b = t0 + 0.55 * (t[-1] - t0)  # break past mid-series

    params = (
        BPD1Params(2.0, 12.0, 18.0, t_b, kappa=0.0, amp=0.0),
        BPD1Params(-3.0, -8.0, 25.0, t_b, kappa=0.0, amp=0.0),
        BPD1Params(0.0, 3.0, -30.0, t_b, kappa=0.0, amp=0.0),
    )
    # Annual/semiannual amplitudes [mm]: (cos_a, sin_a, cos_sa, sin_sa).
    seasonal = ((2.0, 1.0, 0.5, 0.3), (1.5, -0.8, 0.4, -0.2), (4.0, 2.5, 1.0, 0.5))

    rows = [
        bpd1_forward(p, t) + periodic(t, *amps) + rng.normal(0.0, wn_amp, t.size)
        for p, amps in zip(params, seasonal, strict=True)
    ]
    y = np.vstack(rows)
    return StationSeries(
        marker=marker,
        t=t,
        y=y,
        sigma=np.full_like(y, wn_amp),
        source=f"synthetic:seed={seed},n_days={n_days},t0={t0}",
    )
