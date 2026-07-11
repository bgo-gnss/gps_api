"""Unit tests for the LTTB downsampler (presentation layer).

Property-style checks per MATH_STANDARDS §4: exact output size, endpoint
preservation, identity below the target, and — the reason LTTB was chosen
over every-Nth decimation (contract Decisions #2) — retention of isolated
transients in *any* channel of the multichannel selection.
"""

import numpy as np
import pytest

from gps_api.downsample import lttb_indices


def _channels(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.arange(n, dtype=np.float64)
    ys = np.vstack(
        [
            0.05 * x + rng.normal(0, 0.5, n),
            np.sin(2 * np.pi * x / 90.0) + rng.normal(0, 0.5, n),
            -0.02 * x + rng.normal(0, 0.5, n),
        ]
    )
    return x, ys


def test_identity_when_target_covers_series() -> None:
    x, ys = _channels(50)
    for n_out in (50, 51, 10_000):
        idx = lttb_indices(x, ys, n_out)
        assert np.array_equal(idx, np.arange(50))


def test_exact_count_endpoints_and_strict_monotonicity() -> None:
    x, ys = _channels(1000)
    for n_out in (2, 3, 17, 100, 999):
        idx = lttb_indices(x, ys, n_out)
        assert idx.size == n_out
        assert idx[0] == 0 and idx[-1] == 999
        assert np.all(np.diff(idx) > 0), "indices must be strictly increasing"


def test_two_point_target_keeps_the_endpoints_only() -> None:
    x, ys = _channels(400)
    assert np.array_equal(lttb_indices(x, ys, 2), [0, 399])


def test_spike_survives_downsampling_in_every_channel() -> None:
    """An isolated transient must survive — in whichever channel it lives.

    Every-Nth decimation would drop a one-sample spike with probability
    (1 − 1/N); LTTB keeps it because the spike dominates its bucket's
    triangle area (summed over channels for the shared index set).
    """
    for channel in range(3):
        x, ys = _channels(1000, seed=channel)
        spike_at = 613
        ys = ys.copy()
        ys[channel, spike_at] += 50.0  # ≫ noise σ and trend within a bucket
        idx = lttb_indices(x, ys, 50)
        assert spike_at in idx, f"spike in channel {channel} was dropped"


def test_rejects_bad_inputs() -> None:
    x, ys = _channels(20)
    with pytest.raises(ValueError, match="n_out"):
        lttb_indices(x, ys, 1)
    with pytest.raises(ValueError, match="shape"):
        lttb_indices(x, ys[:, :-1], 5)
    x_bad = x.copy()
    x_bad[5], x_bad[6] = x_bad[6], x_bad[5]
    with pytest.raises(ValueError, match="ascending"):
        lttb_indices(x_bad, ys, 5)


def test_matches_reference_single_channel_lttb() -> None:
    """Multichannel code with C=1 reproduces a literal transcription of
    Steinarsson 2013 §4.2 (independent loop implementation)."""

    def reference_lttb(x: np.ndarray, y: np.ndarray, n_out: int) -> list[int]:
        n = x.size
        n_buckets = n_out - 2
        edges = [1 + round(i * (n - 2) / n_buckets) for i in range(n_buckets + 1)]
        out = [0]
        a = 0
        for i in range(n_buckets):
            lo, hi = edges[i], edges[i + 1]
            nlo, nhi = (
                (edges[i + 1], edges[i + 2])
                if i < n_buckets - 1
                else (
                    n - 1,
                    n,
                )
            )
            x_c = float(np.mean(x[nlo:nhi]))
            y_c = float(np.mean(y[nlo:nhi]))
            best, best_area = lo, -1.0
            for j in range(lo, hi):
                area = abs((x[a] - x_c) * (y[j] - y[a]) - (x[a] - x[j]) * (y[a] - y_c))
                if area > best_area:
                    best, best_area = j, area
            out.append(best)
            a = best
        out.append(n - 1)
        return out

    rng = np.random.default_rng(42)
    x = np.cumsum(rng.uniform(0.5, 1.5, 500))  # irregular sampling too
    y = np.sin(x / 20.0) + rng.normal(0, 0.3, 500)
    got = lttb_indices(x, y[None, :], 40)
    assert got.tolist() == reference_lttb(x, y, 40)
