"""Tests for the compensated-prefix-sum float path in join_agg.

See issue #1671: the forward `sum` range kernels for float32/float64 use
a Neumaier-compensated prefix sum instead of a Rust round-trip per range,
gated by a work-estimate heuristic (`_prefix_sum_is_worthwhile`) since the
one-time O(n) build only pays off when the total width of the queried
ranges is large relative to the array size - otherwise Rust's per-range
native scan is cheaper overall.

Numerical-correctness tests below exercise the compensated-prefix math
directly via `_prefix_range_sum` (bypassing the performance heuristic,
which is orthogonal to correctness) against `janitor_rs`'s own kernels -
not exact equality, since Rust computes a fresh Kahan sum per range while
this is a prefix-subtraction, but the agreed relative-error tolerance
(1e-9). Separate tests cover the heuristic itself and full `join_agg`
integration at a scale where the fast path is actually selected.
"""

import janitor_rs
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import janitor  # noqa: F401 - registers the join_agg DataFrame accessor
from janitor.functions._conditional_join import _agg_functions as af

RELATIVE_TOLERANCE = 1e-9

RUST_SUM_STARTS = {
    "float64": janitor_rs.compute_sum_start_f64,
    "float32": janitor_rs.compute_sum_start_f32,
}
RUST_SUM_ENDS = {
    "float64": janitor_rs.compute_sum_end_f64,
    "float32": janitor_rs.compute_sum_end_f32,
}
RUST_SUM_STARTS_ENDS = {
    "float64": janitor_rs.compute_sum_start_end_f64,
    "float32": janitor_rs.compute_sum_start_end_f32,
}


def _relative_error(actual: float, expected: float) -> float:
    denom = max(abs(expected), 1e-300)
    return abs(actual - expected) / denom


def _prefix_range_sum(arr, booleans, starts, ends):
    """Exercise the compensated-prefix path directly, bypassing the
    performance heuristic in _sum_starts/_sum_ends/_sum_starts_ends -
    for tests about the math, not about when it's chosen. Returns None
    if the safety guard legitimately rejects this array (e.g. dynamic
    range right at the threshold, which randomized test data can land on
    by chance) - callers should skip that trial rather than treat it as
    a failure."""
    prefix = af._build_prefix_if_safe(arr=arr, booleans=booleans)
    if prefix is None:
        return None
    hi, lo = prefix
    return af._range_sum_from_prefix(hi=hi, lo=lo, starts=starts, ends=ends)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_sum_starts_ends_within_tolerance_of_rust(dtype):
    """Randomized, mixed-magnitude ranges stay within the accuracy contract."""
    rng = np.random.default_rng(20260822)
    rust_fn = RUST_SUM_STARTS_ENDS[dtype]
    checked = 0
    for n in (1, 2, 5, 37, 500, 999, 4096, 10_007):
        for _ in range(25):
            arr = rng.standard_normal(n).astype(dtype)
            scale = rng.choice([1e-6, 1, 1e6, 1e9], size=n).astype(dtype)
            arr = (arr * scale).astype(dtype)
            booleans = rng.random(n) < 0.05
            start = int(rng.integers(0, n + 1))
            end = int(rng.integers(start, n + 1))
            starts = np.array([start], dtype=np.int64)
            ends = np.array([end], dtype=np.int64)

            actual = _prefix_range_sum(arr, booleans, starts, ends)
            if actual is None:
                continue  # safety guard legitimately rejected this draw
            expected = rust_fn(arr=arr, starts=starts, ends=ends, booleans=booleans)[0]
            assert _relative_error(actual[0], expected) <= RELATIVE_TOLERANCE, (
                dtype,
                n,
                start,
                end,
                expected,
                actual[0],
            )
            checked += 1
    assert checked > 0


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_sum_starts_within_tolerance_of_rust(dtype):
    """Suffix sums (single '<' join) stay within the accuracy contract."""
    rng = np.random.default_rng(20260823)
    rust_fn = RUST_SUM_STARTS[dtype]
    checked = 0
    for n in (1, 5, 500, 4096):
        for _ in range(10):
            arr = rng.standard_normal(n).astype(dtype)
            scale = rng.choice([1e-6, 1, 1e6], size=n).astype(dtype)
            arr = (arr * scale).astype(dtype)
            booleans = rng.random(n) < 0.05
            starts = rng.integers(0, n + 1, size=3).astype(np.int64)

            actual = _prefix_range_sum(arr, booleans, starts, arr.size)
            if actual is None:
                continue
            expected = rust_fn(arr=arr, starts=starts, booleans=booleans)
            for e, a in zip(expected, actual):
                assert _relative_error(a, e) <= RELATIVE_TOLERANCE
            checked += 1
    assert checked > 0


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_sum_ends_within_tolerance_of_rust(dtype):
    """Prefix sums (single '>' join) stay within the accuracy contract."""
    rng = np.random.default_rng(20260824)
    rust_fn = RUST_SUM_ENDS[dtype]
    checked = 0
    for n in (1, 5, 500, 4096):
        for _ in range(10):
            arr = rng.standard_normal(n).astype(dtype)
            scale = rng.choice([1e-6, 1, 1e6], size=n).astype(dtype)
            arr = (arr * scale).astype(dtype)
            booleans = rng.random(n) < 0.05
            ends = rng.integers(0, n + 1, size=3).astype(np.int64)

            actual = _prefix_range_sum(arr, booleans, 0, ends)
            if actual is None:
                continue
            expected = rust_fn(arr=arr, ends=ends, booleans=booleans)
            for e, a in zip(expected, actual):
                assert _relative_error(a, e) <= RELATIVE_TOLERANCE
            checked += 1
    assert checked > 0


def test_moderate_dynamic_range_still_uses_fast_path():
    """The dynamic-range guard shouldn't be so conservative that it defeats
    the point: everyday mixed-magnitude data must still take the fast
    path, not fall back to Rust for every call. Magnitudes are bounded
    away from zero (unlike a raw Gaussian*scale draw, whose tail can land
    arbitrarily close to zero by chance and blow out the *empirical*
    dynamic range regardless of the intended scale spread)."""
    rng = np.random.default_rng(2)
    magnitudes = rng.uniform(1.0, 1e9, size=500)
    signs = rng.choice([-1.0, 1.0], size=500)
    arr = signs * magnitudes
    booleans = np.zeros(arr.size, dtype=np.bool_)
    assert af._has_safe_dynamic_range(arr, booleans)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_well_scaled_arrays_match_rust_exactly(dtype):
    """Same order of magnitude throughout -> no meaningful cancellation,
    so the compensated prefix sum should agree with Rust bit-for-bit."""
    rng = np.random.default_rng(1)
    arr = rng.standard_normal(2000).astype(dtype)
    booleans = np.zeros(arr.size, dtype=np.bool_)
    dtype_name = np.dtype(dtype).name
    starts = np.array([0, 5, 1000], dtype=np.int64)
    ends = np.array([2000, 1800, 1999], dtype=np.int64)

    expected = RUST_SUM_STARTS_ENDS[dtype_name](
        arr=arr, starts=starts, ends=ends, booleans=booleans
    )
    actual = _prefix_range_sum(arr, booleans, starts, ends)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_null_mask_is_authoritative(dtype):
    """A `booleans=True` position is ignored regardless of its raw value -
    matches the Rust kernels' verified null-handling semantics."""
    arr = np.array([5.0, 999.0, 5.0], dtype=dtype)
    booleans = np.array([False, True, False])
    out = _prefix_range_sum(arr, booleans, 0, arr.size)
    assert out == pytest.approx(10.0)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_all_null_window_sums_to_zero(dtype):
    """An all-null window sums to 0.0, not NaN (matches Rust)."""
    arr = np.array([1.0, 2.0, 3.0], dtype=dtype)
    booleans = np.array([True, True, True])
    out = _prefix_range_sum(arr, booleans, 0, arr.size)
    assert out == 0.0


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_empty_range_sums_to_zero(dtype):
    """start == end -> an empty range sums to 0.0."""
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype)
    booleans = np.zeros(arr.size, dtype=np.bool_)
    out = _prefix_range_sum(arr, booleans, 2, 2)
    assert out == 0.0


def test_compensated_sum_more_accurate_than_naive_cumsum():
    """Sanity check that compensation is actually doing something: a
    repeated-0.1 sum should land on the exact value a naive cumsum misses."""
    n = 100_000
    arr = np.full(n, 0.1, dtype=np.float64)
    booleans = np.zeros(n, dtype=np.bool_)

    naive = np.cumsum(arr)[-1]
    compensated = _prefix_range_sum(arr, booleans, 0, n)

    assert naive != 10_000.0  # confirms the test actually stresses precision
    assert compensated == 10_000.0


def test_int_dtypes_still_use_rust_path():
    """Integer dtypes are untouched by this change (tracked in #1648)."""
    arr = np.array([1, 2, 3], dtype=np.int64)
    booleans = np.zeros(3, dtype=np.bool_)
    starts = np.array([0], dtype=np.int64)
    out = af._sum_starts(arr=arr, starts=starts, booleans=booleans)
    assert out[0] == 6


def test_overflow_during_summation_is_rejected_by_safety_guard():
    """Regression test: two large-but-finite values whose sum overflows
    float64 range must saturate to +/-inf (matching NumPy's own overflow
    behavior and Rust), not produce a spurious NaN from the compensation
    bookkeeping's inf - inf. Found via hypothesis property tests.
    `_build_prefix_if_safe` must refuse to hand back a prefix built over
    such an array, so callers fall back to Rust."""
    arr = np.array([-8.988466e307, -8.988466e307], dtype=np.float64)
    booleans = np.zeros(2, dtype=np.bool_)
    starts = np.array([0], dtype=np.int64)
    ends = np.array([2], dtype=np.int64)

    assert af._build_prefix_if_safe(arr=arr, booleans=booleans) is None
    expected = RUST_SUM_STARTS_ENDS["float64"](
        arr=arr, starts=starts, ends=ends, booleans=booleans
    )
    actual = af._sum_starts_ends(arr=arr, starts=starts, ends=ends, booleans=booleans)
    assert actual[0] == expected[0] == -np.inf


def test_extreme_dynamic_range_is_rejected_by_safety_guard():
    """Regression test: Neumaier compensation is itself just a float64, so
    once it has absorbed a moderate correction it can no longer represent
    a much later, much tinier one - that increment underflows below the
    compensation term's own ULP. A big excursion, another huge opposite-
    signed excursion, then a tiny value can silently lose 100% of that
    tiny value's true contribution, well before anything overflows to
    +/-inf (so the finiteness guard alone doesn't catch it). Found via
    randomized adversarial testing at extreme magnitude spreads
    (~1e99 down to ~1e-6) - a single-element window recovered 0.0 instead
    of the true ~1.06e-6. `_build_prefix_if_safe` must refuse this array."""
    arr = np.array(
        [
            3.39748102e99,
            1.98320141e12,
            -6.32453040e99,
            5.40393231e-07,
            1.06052966e-06,
        ],
        dtype=np.float64,
    )
    booleans = np.zeros(arr.size, dtype=np.bool_)
    starts = np.array([4], dtype=np.int64)
    ends = np.array([5], dtype=np.int64)

    assert not af._has_safe_dynamic_range(arr, booleans)
    assert af._build_prefix_if_safe(arr=arr, booleans=booleans) is None
    expected = RUST_SUM_STARTS_ENDS["float64"](
        arr=arr, starts=starts, ends=ends, booleans=booleans
    )
    actual = af._sum_starts_ends(arr=arr, starts=starts, ends=ends, booleans=booleans)
    assert actual[0] == expected[0] == arr[4]


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_range_excluding_earlier_infinity_matches_rust(dtype):
    """Regression test: once the running total hits +/-inf, naively
    subtracting two +/-inf prefix entries for a later range that never
    touched the infinity yields NaN. A range that excludes an earlier
    infinity must still fall back to Rust and get the correct finite sum."""
    arr = np.array([1.0, -np.inf, 2.0, 3.0], dtype=dtype)
    booleans = np.zeros(arr.size, dtype=np.bool_)
    starts = np.array([2], dtype=np.int64)
    ends = np.array([4], dtype=np.int64)

    assert af._build_prefix_if_safe(arr=arr, booleans=booleans) is None
    expected = RUST_SUM_STARTS_ENDS[np.dtype(dtype).name](
        arr=arr, starts=starts, ends=ends, booleans=booleans
    )
    actual = af._sum_starts_ends(arr=arr, starts=starts, ends=ends, booleans=booleans)
    assert np.isfinite(actual[0])
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_masked_infinity_still_passes_safety_guard(dtype):
    """An infinity at a null (`booleans=True`) position doesn't trip the
    safety guard, since it never contributes to the compensated prefix."""
    arr = np.array([1.0, np.inf, 2.0], dtype=dtype)
    booleans = np.array([False, True, False])
    out = _prefix_range_sum(arr, booleans, 0, arr.size)
    assert out == pytest.approx(3.0)


def test_prefix_sum_is_worthwhile_boundary():
    """Unit-tests the work-estimate heuristic directly at its threshold:
    the O(n) build should only be judged worthwhile once the total
    queried width clears `_MIN_TOTAL_WIDTH_RATIO * array_size`."""
    array_size = 1000
    threshold = af._MIN_TOTAL_WIDTH_RATIO * array_size
    assert not af._prefix_sum_is_worthwhile(array_size, threshold)
    assert af._prefix_sum_is_worthwhile(array_size, threshold + 1)


def test_small_query_count_against_large_array_defers_to_rust():
    """The scenario a prior review flagged as a performance regression: a
    single (or few) query against a large array. The one-time O(n) Python
    build (~250ns/element) would be far slower than Rust's native
    per-range scan (~2ns/element) here, so this must defer to Rust rather
    than pay the build cost for essentially no reuse."""
    n = 200_000
    arr = np.random.default_rng(5).standard_normal(n)
    booleans = np.zeros(n, dtype=np.bool_)
    starts = np.array([0], dtype=np.int64)

    total_width = n  # a single full-width suffix query
    assert not af._prefix_sum_is_worthwhile(n, total_width)

    expected = RUST_SUM_STARTS["float64"](arr=arr, starts=starts, booleans=booleans)
    actual = af._sum_starts(arr=arr, starts=starts, booleans=booleans)
    np.testing.assert_array_equal(actual, expected)


def test_join_agg_float_sum_end_to_end():
    """Small, deterministic end-to-end join_agg check for float `sum`,
    covering suffix ('<' -> _sum_starts), prefix ('>' -> _sum_ends), and
    range-join (-> _sum_starts_ends) aggregation. Deliberately uses
    well-scaled values rather than hypothesis-generated extremes: Rust's
    Kahan kernels are already known (independent of this change) to
    disagree with plain pandas/NumPy summation right at float64's overflow
    boundary, so pandas is only a valid oracle away from that edge. This
    dataset is far too small to trigger the fast path (see
    test_join_agg_large_overlapping_ranges_uses_fast_path below for that),
    but it exercises the full join_agg -> _agg_join_right -> _sum_* wiring."""
    left = pd.DataFrame({"a": [3, 7, 1], "b": [9, 2, 6]})
    right = pd.DataFrame(
        {
            "x": [1, 3, 5, 7, 9, 11],
            "y": [0, 2, 4, 6, 8, 10],
            "value": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )

    expected_lt = (
        left.reset_index(names="l")
        .merge(right, how="cross")
        .query("a < x")
        .groupby("l")
        .agg({"value": ["sum"]})
    )
    expected_lt.index.names = [None]
    actual_lt = left.join_agg(right, ("a", "x", "<"), aggfunc=[("value", "sum")])
    assert_frame_equal(expected_lt, actual_lt)

    expected_gt = (
        left.reset_index(names="l")
        .merge(right, how="cross")
        .query("a > x")
        .groupby("l")
        .agg({"value": ["sum"]})
    )
    expected_gt.index.names = [None]
    actual_gt = left.join_agg(right, ("a", "x", ">"), aggfunc=[("value", "sum")])
    assert_frame_equal(expected_gt, actual_gt)

    expected_range = (
        left.reset_index(names="l")
        .merge(right, how="cross")
        .query("b > y and a < x")
        .groupby("l")
        .agg({"value": ["sum"]})
    )
    expected_range.index.names = [None]
    actual_range = left.join_agg(
        right, ("b", "y", ">"), ("a", "x", "<"), aggfunc=[("value", "sum")]
    )
    actual_range = actual_range.loc[expected_range.index]
    assert_frame_equal(expected_range, actual_range)


def test_join_agg_large_overlapping_ranges_uses_fast_path():
    """Genuine end-to-end integration coverage of the fast path itself:
    a large right table with many overlapping suffix ranges (the case
    #1671 targets) should both trigger `_prefix_sum_is_worthwhile` and
    produce results matching a direct, per-row Rust computation."""
    n_right = 5000
    n_left = 2000
    rng = np.random.default_rng(11)
    left = pd.DataFrame({"key": rng.uniform(0, n_right, size=n_left)})
    right = pd.DataFrame(
        {
            "key": np.arange(n_right, dtype=float),
            "value": rng.standard_normal(n_right),
        }
    )

    total_width = int(n_left * n_right - left["key"].to_numpy().sum())
    assert af._prefix_sum_is_worthwhile(n_right, total_width)

    actual = left.join_agg(right, ("key", "key", "<"), aggfunc=[("value", "sum")])

    arr = right["value"].to_numpy()
    booleans = np.zeros(n_right, dtype=np.bool_)
    for l_idx, l_key in left["key"].items():
        start = int(np.searchsorted(right["key"].to_numpy(), l_key, side="right"))
        expected = RUST_SUM_STARTS["float64"](
            arr=arr, starts=np.array([start], dtype=np.int64), booleans=booleans
        )[0]
        if l_idx in actual.index:
            got = actual.loc[l_idx, ("value", "sum")]
            assert _relative_error(got, expected) <= RELATIVE_TOLERANCE
        else:
            assert start == n_right  # no matches for this row
