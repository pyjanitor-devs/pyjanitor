"""Focused unit tests for the prefix/suffix min/max kernels backing
`join_agg`.

Covers `_min_starts`, `_min_ends`, `_max_starts`, `_max_ends` in
`janitor.functions._conditional_join._agg_functions` (Issue #1653) -- the
O(n + m) NumPy replacements for the Rust `compute_min_start*`,
`compute_min_end*`, `compute_max_start*`, and `compute_max_end*` kernels,
used once the number/width of requested ranges makes precomputing worth it
(see `_use_argext`); sparse range requests continue to use the Rust
kernels directly.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from janitor.functions._conditional_join import _agg_functions

ALL_DTYPES = [
    "int64",
    "int32",
    "int16",
    "int8",
    "uint64",
    "uint32",
    "uint16",
    "uint8",
    "float64",
    "float32",
]
FLOAT_DTYPES = ["float64", "float32"]


def naive_suffix(arr, booleans, start, is_max):
    """Direct transliteration of the Rust min_starts/max_starts loop."""
    n = len(arr)
    if start >= n:
        return -1
    base = -1
    base_val = arr[start]
    for nn in range(start, n):
        if booleans[nn]:
            continue
        current = arr[nn]
        if base == -1 or (current > base_val if is_max else current < base_val):
            base_val = current
            base = nn
    return base


def naive_prefix(arr, booleans, end, is_max):
    """Direct transliteration of the Rust min_ends/max_ends loop."""
    n = len(arr)
    if n == 0:
        return -1
    base = -1
    base_val = arr[0]
    for nn in range(0, end):
        if booleans[nn]:
            continue
        current = arr[nn]
        if base == -1 or (current > base_val if is_max else current < base_val):
            base_val = current
            base = nn
    return base


def _dense_query(n, size=200):
    """Enough queries/width to clear `_use_argext`'s work-factor gate."""
    rng = np.random.default_rng(0)
    return rng.integers(0, n if n else 1, size=size).astype("int64")


@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("is_max", [True, False])
def test_starts_matches_naive_dense(dtype, is_max):
    """Dense queries (forces the NumPy path) match the naive Rust-loop
    reference, including ties and nulls."""
    rng = np.random.default_rng(hash((dtype, is_max)) % (2**32))
    n = 40
    if dtype.startswith("float"):
        arr = rng.integers(-5, 6, size=n).astype(dtype)
        arr[rng.random(n) < 0.15] = np.nan
    else:
        info = np.iinfo(dtype)
        choices = np.array(
            [info.min, info.min + 1, info.max - 1, info.max, 0, 1], dtype=dtype
        )
        arr = choices[rng.integers(0, len(choices), size=n)]
    booleans = rng.random(n) < 0.2
    starts = _dense_query(n)
    func = _agg_functions._max_starts if is_max else _agg_functions._min_starts

    expected = np.array(
        [naive_suffix(arr, booleans, int(s), is_max) for s in starts], dtype=np.int64
    )
    actual = func(arr=arr, starts=starts, booleans=booleans)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("is_max", [True, False])
def test_ends_matches_naive_dense(dtype, is_max):
    rng = np.random.default_rng(hash((dtype, is_max, "ends")) % (2**32))
    n = 40
    if dtype.startswith("float"):
        arr = rng.integers(-5, 6, size=n).astype(dtype)
        arr[rng.random(n) < 0.15] = np.nan
    else:
        info = np.iinfo(dtype)
        choices = np.array(
            [info.min, info.min + 1, info.max - 1, info.max, 0, 1], dtype=dtype
        )
        arr = choices[rng.integers(0, len(choices), size=n)]
    booleans = rng.random(n) < 0.2
    ends = rng.integers(0, n + 1, size=200).astype("int64")
    func = _agg_functions._max_ends if is_max else _agg_functions._min_ends

    expected = np.array(
        [naive_prefix(arr, booleans, int(e), is_max) for e in ends], dtype=np.int64
    )
    actual = func(arr=arr, ends=ends, booleans=booleans)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_all_null_range_is_minus_one(dtype):
    arr = np.array([1, 2, 3, 4] * 15, dtype=dtype)
    booleans = np.ones(60, dtype=bool)
    starts = _dense_query(60)
    ends = np.arange(60, dtype="int64")

    assert (
        _agg_functions._min_starts(arr=arr, starts=starts, booleans=booleans) == -1
    ).all()
    assert (_agg_functions._min_ends(arr=arr, ends=ends, booleans=booleans) == -1).all()
    assert (
        _agg_functions._max_starts(arr=arr, starts=starts, booleans=booleans) == -1
    ).all()
    assert (_agg_functions._max_ends(arr=arr, ends=ends, booleans=booleans) == -1).all()


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_empty_range_is_minus_one(dtype):
    # Mix a few empty-range queries into a dense batch of real ones, so the
    # whole call routes through the NumPy path (which handles start == n /
    # end == 0 gracefully). An all-empty batch would instead route to the
    # Rust fallback (total_width is always 0 there), and the existing Rust
    # kernel panics on `arr[start_]` when start_ == n -- a pre-existing
    # issue in the Rust kernel itself, unrelated to this change, and not a
    # shape of input this suite needs to force through that path.
    n = 60
    arr = np.array(list(range(60)), dtype=dtype)
    booleans = np.zeros(n, dtype=bool)
    starts = _dense_query(n)
    starts[:5] = n  # empty suffix
    ends = _dense_query(n)
    ends[:5] = 0  # empty prefix

    assert (
        _agg_functions._min_starts(arr=arr, starts=starts, booleans=booleans)[:5] == -1
    ).all()
    assert (
        _agg_functions._min_ends(arr=arr, ends=ends, booleans=booleans)[:5] == -1
    ).all()
    assert (
        _agg_functions._max_starts(arr=arr, starts=starts, booleans=booleans)[:5] == -1
    ).all()
    assert (
        _agg_functions._max_ends(arr=arr, ends=ends, booleans=booleans)[:5] == -1
    ).all()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_ties_keep_first_occurrence(dtype):
    """Duplicate extrema resolve to the smaller/earlier position."""
    arr = np.array([5, 1, 3, 1, 1, 9, 1], dtype=dtype)
    booleans = np.zeros(7, dtype=bool)
    starts = _dense_query(7)
    ends = np.arange(8, dtype="int64").repeat(30)[:200]

    expected_starts = np.array(
        [naive_suffix(arr, booleans, int(s), False) for s in starts], dtype=np.int64
    )
    expected_ends = np.array(
        [naive_prefix(arr, booleans, int(e), False) for e in ends], dtype=np.int64
    )
    np.testing.assert_array_equal(
        _agg_functions._min_starts(arr=arr, starts=starts, booleans=booleans),
        expected_starts,
    )
    np.testing.assert_array_equal(
        _agg_functions._min_ends(arr=arr, ends=ends, booleans=booleans), expected_ends
    )


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_nan_freezes_only_when_first_in_range(dtype):
    """A NaN that is the first non-null value of a *specific row's* range
    poisons that row's result (Rust: nothing compares less/greater than
    NaN); a NaN found later in the same row's range is just skipped."""
    arr = np.array([3.0, np.nan, 1.0], dtype=dtype)
    booleans = np.zeros(3, dtype=bool)
    # pad with dense filler queries so this exercises the NumPy path too
    filler = _dense_query(3)
    starts = np.concatenate([[0, 1, 2], filler])

    expected = np.array(
        [naive_suffix(arr, booleans, int(s), False) for s in starts], dtype=np.int64
    )
    actual = _agg_functions._min_starts(arr=arr, starts=starts, booleans=booleans)
    np.testing.assert_array_equal(actual, expected)
    # hand-verified contract for the first three (non-filler) queries:
    # start=0 -> real anchor (3.0), skips the NaN, finds 1.0 at index 2
    # start=1 -> anchor IS the NaN itself -> frozen there, index 1
    # start=2 -> real anchor (1.0), index 2
    assert actual[0] == 2
    assert actual[1] == 1
    assert actual[2] == 2


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_nan_first_in_prefix_poisons_rest(dtype):
    """Prefix scans always start at index 0, so if the array's first
    non-null value is NaN, every end past that point is frozen there."""
    arr = np.array([np.nan, 1.0, 2.0, 3.0], dtype=dtype)
    booleans = np.zeros(4, dtype=bool)
    ends = np.array([0, 1, 2, 3, 4], dtype="int64")
    filler = np.full(200, 4, dtype="int64")
    ends = np.concatenate([ends, filler])

    actual = _agg_functions._min_ends(arr=arr, ends=ends, booleans=booleans)
    assert actual[0] == -1  # end=0 -> empty prefix
    assert (actual[1:5] == 0).all()  # frozen at the NaN's own position
    assert (actual[5:] == 0).all()


@pytest.mark.parametrize(
    "func_name,rust_name,indexers",
    [
        ("_min_starts", "compute_min_start_int64", {"starts": [5]}),
        ("_min_ends", "compute_min_end_int64", {"ends": [1]}),
        ("_max_starts", "compute_max_start_int64", {"starts": [5]}),
        ("_max_ends", "compute_max_end_int64", {"ends": [1]}),
    ],
)
def test_sparse_ranges_use_rust(monkeypatch, func_name, rust_name, indexers):
    """A handful of narrow queries should not pay for the full precompute."""
    expected = np.array([7], dtype=np.int64)

    def fake_rust(**kwargs):
        return expected

    monkeypatch.setattr(_agg_functions.janitor_rs, rust_name, fake_rust)
    monkeypatch.setattr(
        _agg_functions,
        "_suffix_argext" if "start" in func_name else "_prefix_argext",
        lambda **kwargs: pytest.fail("sparse ranges should stay in Rust"),
    )
    actual = getattr(_agg_functions, func_name)(
        arr=np.arange(100, dtype=np.int64),
        booleans=np.zeros(100, dtype=bool),
        **{name: np.array(values, dtype=np.int64) for name, values in indexers.items()},
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "func_name,rust_name,indexers,expected",
    [
        ("_min_starts", "compute_min_start_int64", {"starts": [0] * 200}, 0),
        ("_min_ends", "compute_min_end_int64", {"ends": [100] * 200}, 0),
        ("_max_starts", "compute_max_start_int64", {"starts": [0] * 200}, 99),
        ("_max_ends", "compute_max_end_int64", {"ends": [100] * 200}, 99),
    ],
)
def test_dense_ranges_use_numpy(monkeypatch, func_name, rust_name, indexers, expected):
    """Heavily overlapping/wide ranges should use the O(n) precompute.

    `arr = arange(100)`, so over the full range [0, 100) the min is 0 (at
    index 0) and the max is 99 (at index 99).
    """

    def fail_rust(**kwargs):
        pytest.fail("dense ranges should use the NumPy prefix/suffix path")

    monkeypatch.setattr(_agg_functions.janitor_rs, rust_name, fail_rust)
    actual = getattr(_agg_functions, func_name)(
        arr=np.arange(100, dtype=np.int64),
        booleans=np.zeros(100, dtype=bool),
        **{name: np.array(values, dtype=np.int64) for name, values in indexers.items()},
    )
    assert (actual == expected).all()


@pytest.mark.parametrize(
    "func_name,indexer_name,indexer_value",
    [
        ("_min_starts", "starts", 0),
        ("_min_ends", "ends", 100),
        ("_max_starts", "starts", 0),
        ("_max_ends", "ends", 100),
    ],
)
@pytest.mark.parametrize("query_count", [1, 200])
def test_unsupported_dtype_error_does_not_depend_on_density(
    func_name, indexer_name, indexer_value, query_count
):
    """The performance gate must not silently expand supported dtypes."""
    arr = np.ones(100, dtype=bool)
    booleans = np.zeros(100, dtype=bool)
    indexers = np.full(query_count, indexer_value, dtype=np.int64)

    with pytest.raises(KeyError, match="Unsupported data type -> bool"):
        getattr(_agg_functions, func_name)(
            arr=arr,
            booleans=booleans,
            **{indexer_name: indexers},
        )


@pytest.mark.parametrize("is_max", [True, False])
def test_suffix_uses_min_or_max_accumulate_semantics(is_max):
    """Sanity check that _max_* actually picks the maximum, not the
    minimum (i.e. the is_max plumbing is wired correctly end to end)."""
    arr = np.array([1, 9, 2, 9, 3], dtype="int64")
    booleans = np.zeros(5, dtype=bool)
    starts = _dense_query(5)
    func = _agg_functions._max_starts if is_max else _agg_functions._min_starts
    expected = np.array(
        [naive_suffix(arr, booleans, int(s), is_max) for s in starts], dtype=np.int64
    )
    actual = func(arr=arr, starts=starts, booleans=booleans)
    np.testing.assert_array_equal(actual, expected)


# ---------------------------------------------------------------------------
# Property-based tests: the real correctness gate for the NaN/tie/null
# semantics above, which are easy to get subtly wrong by hand.
# ---------------------------------------------------------------------------

_HYP_DTYPES = ["int64", "int8", "uint64", "uint8", "float64", "float32"]


@st.composite
def _array_and_booleans(draw, dtype):
    n = draw(st.integers(min_value=0, max_value=25))
    if dtype.startswith("float"):
        values = draw(
            st.lists(
                st.one_of(
                    st.floats(
                        min_value=-5, max_value=5, allow_nan=False, allow_infinity=False
                    ),
                    st.just(float("nan")),
                ),
                min_size=n,
                max_size=n,
            )
        )
        arr = np.array(values, dtype=dtype)
    else:
        info = np.iinfo(dtype)
        values = draw(
            st.lists(
                st.sampled_from(
                    [info.min, info.min + 1, -2, -1, 0, 1, 2, info.max - 1, info.max]
                ),
                min_size=n,
                max_size=n,
            )
        )
        arr = np.array([v for v in values if info.min <= v <= info.max], dtype=dtype)
        n = arr.size
    booleans = np.array(draw(st.lists(st.booleans(), min_size=n, max_size=n)))
    return arr, booleans


@given(dtype=st.sampled_from(_HYP_DTYPES), is_max=st.booleans(), data=st.data())
@settings(max_examples=300, deadline=None)
def test_prefix_suffix_property_against_naive(dtype, is_max, data):
    arr, booleans = data.draw(_array_and_booleans(dtype))
    n = arr.size

    pre = _agg_functions._prefix_argext(arr=arr, booleans=booleans, is_max=is_max)
    suf = _agg_functions._suffix_argext(arr=arr, booleans=booleans, is_max=is_max)

    for e in range(n + 1):
        assert pre[e] == naive_prefix(arr, booleans, e, is_max), (
            arr,
            booleans,
            "end",
            e,
        )
    for s in range(n + 1):
        assert suf[s] == naive_suffix(arr, booleans, s, is_max), (
            arr,
            booleans,
            "start",
            s,
        )
