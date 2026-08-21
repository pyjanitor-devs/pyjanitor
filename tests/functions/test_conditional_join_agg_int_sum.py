"""Focused unit tests for the integer prefix-sum kernels backing `join_agg`.

Covers `_int64_prefix_sums`, `_sum_starts`, `_sum_ends`, and
`_sum_starts_ends` in `janitor.functions._conditional_join._agg_functions`
(Issue #1648) -- the O(n + m) NumPy replacements for the Rust
`compute_sum_start*`, `compute_sum_end*`, and `compute_sum_start_end*`
kernels, for integer dtypes only.
"""

import numpy as np
import pandas as pd
import pytest

from janitor.functions._conditional_join import _agg_functions

INTEGER_DTYPES = [
    "int64",
    "int32",
    "int16",
    "int8",
    "uint64",
    "uint32",
    "uint16",
    "uint8",
]

EXTENSION_DTYPE_NAMES = {
    "int64": "Int64",
    "int32": "Int32",
    "int16": "Int16",
    "int8": "Int8",
    "uint64": "UInt64",
    "uint32": "UInt32",
    "uint16": "UInt16",
    "uint8": "UInt8",
}


def naive_sum_starts(arr, starts, booleans):
    """Reference implementation matching the Rust kernel loop exactly."""
    out = np.zeros(starts.size, dtype="int64")
    n = arr.size
    for pos, start in enumerate(starts):
        total = 0
        for nn in range(start, n):
            if booleans[nn]:
                continue
            total += int(arr[nn])
        out[pos] = total
    return out


def naive_sum_ends(arr, ends, booleans):
    out = np.zeros(ends.size, dtype="int64")
    for pos, end in enumerate(ends):
        total = 0
        for nn in range(0, end):
            if booleans[nn]:
                continue
            total += int(arr[nn])
        out[pos] = total
    return out


def naive_sum_starts_ends(arr, starts, ends, booleans):
    out = np.zeros(starts.size, dtype="int64")
    for pos, (start, end) in enumerate(zip(starts, ends)):
        total = 0
        for nn in range(start, end):
            if booleans[nn]:
                continue
            total += int(arr[nn])
        out[pos] = total
    return out


@pytest.mark.parametrize("dtype", INTEGER_DTYPES)
def test_sum_starts_matches_naive_with_nulls(dtype):
    """Nulls scattered at start/middle/end should be skipped, not zeroed-in."""
    arr = pd.array([1, 2, 3, None, 5, None, 7, 8], dtype=EXTENSION_DTYPE_NAMES[dtype])
    booleans = pd.isna(arr)
    arr = arr.to_numpy(dtype=dtype, na_value=0, copy=False)
    starts = np.array([0, 1, 3, 5, 7, 8], dtype="int64")

    expected = naive_sum_starts(arr, starts, booleans)
    actual = _agg_functions._sum_starts(arr=arr, starts=starts, booleans=booleans)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", INTEGER_DTYPES)
def test_sum_ends_matches_naive_with_nulls(dtype):
    arr = pd.array([1, 2, 3, None, 5, None, 7, 8], dtype=EXTENSION_DTYPE_NAMES[dtype])
    booleans = pd.isna(arr)
    arr = arr.to_numpy(dtype=dtype, na_value=0, copy=False)
    ends = np.array([0, 1, 3, 5, 7, 8], dtype="int64")

    expected = naive_sum_ends(arr, ends, booleans)
    actual = _agg_functions._sum_ends(arr=arr, ends=ends, booleans=booleans)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", INTEGER_DTYPES)
def test_sum_starts_ends_matches_naive_with_nulls(dtype):
    arr = pd.array([1, 2, 3, None, 5, None, 7, 8], dtype=EXTENSION_DTYPE_NAMES[dtype])
    booleans = pd.isna(arr)
    arr = arr.to_numpy(dtype=dtype, na_value=0, copy=False)
    starts = np.array([0, 0, 2, 4, 8, 5], dtype="int64")
    ends = np.array([0, 8, 2, 3, 8, 2], dtype="int64")  # includes empty/inverted ranges

    expected = naive_sum_starts_ends(arr, starts, ends, booleans)
    actual = _agg_functions._sum_starts_ends(
        arr=arr, starts=starts, ends=ends, booleans=booleans
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", INTEGER_DTYPES)
def test_all_null_range_is_zero(dtype):
    arr = np.array([1, 2, 3, 4], dtype=dtype)
    booleans = np.ones(4, dtype=bool)
    starts = np.array([0], dtype="int64")
    ends = np.array([4], dtype="int64")

    assert _agg_functions._sum_starts(arr=arr, starts=starts, booleans=booleans)[0] == 0
    assert _agg_functions._sum_ends(arr=arr, ends=ends, booleans=booleans)[0] == 0
    assert (
        _agg_functions._sum_starts_ends(
            arr=arr, starts=starts, ends=ends, booleans=booleans
        )[0]
        == 0
    )


@pytest.mark.parametrize("dtype", INTEGER_DTYPES)
def test_empty_array(dtype):
    arr = np.array([], dtype=dtype)
    booleans = np.array([], dtype=bool)
    starts = np.array([0], dtype="int64")
    ends = np.array([0], dtype="int64")

    assert _agg_functions._sum_starts(arr=arr, starts=starts, booleans=booleans)[0] == 0
    assert _agg_functions._sum_ends(arr=arr, ends=ends, booleans=booleans)[0] == 0
    assert (
        _agg_functions._sum_starts_ends(
            arr=arr, starts=starts, ends=ends, booleans=booleans
        )[0]
        == 0
    )


def test_uint64_values_above_int64_max_reinterpret_like_rust():
    """uint64 values past i64::MAX must bit-reinterpret to negative int64,
    matching the Rust kernel's `current as i64` cast."""
    huge = np.iinfo("uint64").max  # 2**64 - 1 -> -1 as int64
    arr = np.array([huge, huge - 1, 5], dtype="uint64")
    booleans = np.zeros(3, dtype=bool)
    starts = np.array([0], dtype="int64")

    result = _agg_functions._sum_starts(arr=arr, starts=starts, booleans=booleans)
    # -1 + -2 + 5 == 2, computed in wrapping int64 arithmetic
    assert result[0] == 2


def test_int64_accumulation_wraps_like_rust_release_mode():
    """Overflowing the running total wraps (two's complement), it does not
    raise -- matching a Rust release build (overflow-checks disabled)."""
    n = 100
    arr = np.full(n, np.iinfo("int64").max // 2, dtype="int64")
    booleans = np.zeros(n, dtype=bool)
    starts = np.array([0], dtype="int64")

    result = _agg_functions._sum_starts(arr=arr, starts=starts, booleans=booleans)
    assert result[0] == -100


def test_starts_beyond_ends_is_empty_range():
    arr = np.array([1, 2, 3, 4, 5], dtype="int64")
    booleans = np.zeros(5, dtype=bool)
    starts = np.array([3], dtype="int64")
    ends = np.array([1], dtype="int64")

    result = _agg_functions._sum_starts_ends(
        arr=arr, starts=starts, ends=ends, booleans=booleans
    )
    assert result[0] == 0
