"""Largest-Triangle-Three-Buckets (LTTB) downsampling for series responses.

Presentation layer only — this module *selects* a subset of the observed
points for plotting; it never estimates, smooths or aggregates anything
(the MATH_STANDARDS "no new math in gps_api" boundary: estimation stays in
``gps_analysis``; visually faithful reduction of an already-estimated
product is serving, not estimation). Because LTTB selects real observed
epochs, the served ``sigma_*`` values remain the observation uncertainties
of exactly the points shown — nothing is averaged.

Contract hook (docs/API_CONTRACT.md, Decisions #2): the ``max_points``
query parameter of ``GET /v1/stations/{marker}/series`` expresses a target
point count; the server guarantees a visually faithful reduction — peaks,
offsets and trend shape survive, which plain every-Nth decimation does not
(it can erase exactly the transients the portal exists to show).

Algorithm (Steinarsson 2013, "Downsampling Time Series for Visual
Representation", MSc thesis, University of Iceland, §4.2):

1. Always keep the first and the last point.
2. Split the interior points into ``n_out − 2`` near-equal buckets.
3. Walk the buckets left to right. In bucket *i*, select the point *B*
   maximizing the area of the triangle *A–B–C̄* where *A* is the point
   selected from bucket *i − 1* and *C̄* is the mean of bucket *i + 1*::

       area(B) = ½ |(x_A − x_C̄)(y_B − y_A) − (x_A − x_B)(y_A − y_C̄)|

Multichannel extension (this module): one GNSS series response carries a
*single* time base shared by the north/east/up components, so one shared
index set must serve all three. The bucket winner maximizes the *summed*
per-channel triangle area, ``area(B) = Σ_c area_c(B)`` — a transient in any
one component keeps its extremal point in the selection. No per-channel
normalization is applied: all components are in the same unit (mm), so the
sum weights a millimetre of structure equally wherever it appears.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IndexArray = NDArray[np.intp]


def lttb_indices(x: FloatArray, ys: FloatArray, n_out: int) -> IndexArray:
    """Select indices of a visually faithful ``n_out``-point subset.

    Computes the LTTB selection (module docstring) over one or more
    channels sharing the abscissa ``x``; the bucket winner maximizes the
    summed per-channel triangle area.

    Symbols → args:
        - ``x``: abscissa, shape (N,), float64, **ascending** (any monotone
          unit — the router passes POSIX seconds); asserted, not sorted.
        - ``ys``: channel values, shape (C, N), float64, common unit
          across channels (mm for N/E/U displacements).
        - ``n_out``: target point count, ≥ 2.

    Returns:
        Strictly increasing indices into ``x``/``ys`` columns; length
        ``min(n_out, N)``; always contains ``0`` and ``N − 1`` when N ≥ 2.

    Reference: Steinarsson 2013, §4.2 (LTTB); the summed-area multichannel
    rule is this module's documented extension (see module docstring).

    Numerical notes: areas are compared, never accumulated across buckets,
    so float64 is ample; the ½ factor and the sign are dropped (monotone
    under ``argmax`` of the absolute value). Bucket edges use rounding of
    an exactly representable arithmetic progression; since the step
    ``(N − 2)/(n_out − 2) > 1`` whenever downsampling happens, consecutive
    edges differ by ≥ 1 and no bucket is empty.

    Raises:
        ValueError: ``n_out < 2``, shape mismatch, or non-ascending ``x``.
    """
    if n_out < 2:
        raise ValueError(f"n_out must be >= 2, got {n_out}")
    if ys.ndim != 2 or ys.shape[1] != x.shape[0]:
        raise ValueError(
            f"ys must have shape (C, N={x.shape[0]}), got {ys.shape} — "
            "channels are rows, samples are columns"
        )
    n = int(x.shape[0])
    if n_out >= n:
        return np.arange(n, dtype=np.intp)
    if np.any(np.diff(x) < 0):
        raise ValueError("x must be ascending (series products are time-sorted)")
    if n_out == 2:
        return np.asarray([0, n - 1], dtype=np.intp)

    n_buckets = n_out - 2
    # Interior [1, n-1) split into n_buckets; edges[0] = 1, edges[-1] = n-1.
    edges = 1 + np.round(
        np.arange(n_buckets + 1, dtype=np.float64) * (n - 2) / n_buckets
    ).astype(np.intp)

    out = np.empty(n_out, dtype=np.intp)
    out[0] = 0
    a = 0  # index of the point selected from the previous bucket
    for i in range(n_buckets):
        lo, hi = int(edges[i]), int(edges[i + 1])
        # Next bucket (the last bucket looks at the fixed final point).
        nlo, nhi = (
            (int(edges[i + 1]), int(edges[i + 2]))
            if i < n_buckets - 1
            else (
                n - 1,
                n,
            )
        )
        x_c = float(x[nlo:nhi].mean())
        y_c = ys[:, nlo:nhi].mean(axis=1)  # (C,)
        x_a = float(x[a])
        y_a = ys[:, a]  # (C,)
        # Per-channel triangle areas for every candidate B in [lo, hi),
        # summed over channels (constant ½ dropped — argmax-invariant).
        areas = np.abs(
            (x_a - x_c) * (ys[:, lo:hi] - y_a[:, None])
            - (x_a - x[lo:hi])[None, :] * (y_a - y_c)[:, None]
        ).sum(axis=0)
        a = lo + int(np.argmax(areas))
        out[i + 1] = a
    out[-1] = n - 1
    return out
