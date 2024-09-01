"""Various Functions powered by Numba"""

from __future__ import annotations

from math import ceil
from typing import Any, Union

import numpy as np
import pandas as pd
from numba import njit, prange
from pandas.api.types import is_datetime64_dtype, is_extension_array_dtype

# https://numba.discourse.group/t/uint64-vs-int64-indexing-performance-difference/1500
# indexing with unsigned integers offers more performance
from janitor.functions.utils import (
    _generic_func_cond_join,
    greater_than_join_types,
)


def _convert_to_numpy(
    left: np.ndarray, right: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Ensure array is a numpy array.
    """
    if is_extension_array_dtype(left):
        array_dtype = left.dtype.numpy_dtype
        left = left.astype(array_dtype)
        right = right.astype(array_dtype)
    if is_datetime64_dtype(left):
        left = left.view(np.int64)
        right = right.view(np.int64)
    return left, right


def _numba_equi_join(
    df: pd.DataFrame,
    right: pd.DataFrame,
    eqs: tuple,
    ge_gt: tuple,
    le_lt: tuple,
) -> Union[tuple[np.ndarray, np.ndarray], None]:
    """
    Compute indices when an equi join is present.
    """
    # the logic is to delay searching for actual matches
    # while reducing the search space
    # to get the smallest possible search area
    # this serves as an alternative to pandas' hash join
    # and in some cases,
    # usually for many to many joins,
    # can offer significant performance improvements.
    # it relies on binary searches, within the groups,
    # and relies on the fact that sorting ensures the first
    # two columns from the right dataframe are in ascending order
    # per group - this gives us the opportunity to
    # only do a linear search, within the groups,
    # for the last column (if any)
    # (the third column is applicable only for range joins)
    # Example :
    #     df1:
    #    id  value_1
    # 0   1        2
    # 1   1        5
    # 2   1        7
    # 3   2        1
    # 4   2        3
    # 5   3        4
    #
    #
    #  df2:
    #    id  value_2A  value_2B
    # 0   1         0         1
    # 1   1         3         5
    # 2   1         7         9
    # 3   1        12        15
    # 4   2         0         1
    # 5   2         2         4
    # 6   2         3         6
    # 7   3         1         3
    #
    #
    # join condition ->
    # ('id', 'id', '==') &
    # ('value_1', 'value_2A','>') &
    # ('value_1', 'value_2B', '<')
    #
    #
    # note how for df2, id and value_2A
    # are sorted per group
    # the third column (relevant for range join)
    # may or may not be sorted per group
    # (the group is determined by the values of the id column)
    # and as such, we do a linear search in that space, per group
    #
    # first we get the slice boundaries based on id -> ('id', 'id', '==')
    # value     start       end
    #  1         0           4
    #  1         0           4
    #  1         0           4
    #  2         4           7
    #  2         4           7
    #  3         7           8
    #
    # next step is to get the slice end boundaries,
    # based on the greater than condition
    # -> ('value_1', 'value_2A', '>')
    # the search will be within each boundary
    # so for the first row, value_1 is 2
    # the boundary search will be between 0, 4
    # for the last row, value_1 is 4
    # and its boundary search will be between 7, 8
    # since value_2A is sorted per group,
    # a binary search is employed
    # value     start       end      value_1   new_end
    #  1         0           4         2         1
    #  1         0           4         5         2
    #  1         0           4         7         2
    #  2         4           7         1         4
    #  2         4           7         3         6
    #  3         7           8         4         8
    #
    # next step is to get the start boundaries,
    # based on the less than condition
    # -> ('value_1', 'value_2B', '<')
    # note that we have new end boundaries,
    # and as such, our boundaries will use that
    # so for the first row, value_1 is 2
    # the boundary search will be between 0, 1
    # for the 5th row, value_1 is 3
    # and its boundary search will be between 4, 6
    # for value_2B, which is the third column
    # sinc we are not sure whether it is sorted or not,
    # a cumulative max array is used,
    # to get the earliest possible slice start
    # value     start       end      value_1   new_start   new_end
    #  1         0           4         2         -1           1
    #  1         0           4         5         -1           2
    #  1         0           4         7         -1           2
    #  2         4           7         1         -1           5
    #  2         4           7         3         5            6
    #  3         7           8         4         -1           8
    #
    # if there are no matches, boundary is reported as -1
    # from above, we can see that our search space
    # is limited to just 5, 6
    # we can then search for actual matches
    # 	id	value_1	id	value_2A	value_2B
    # 	2	  3	    2	   2	       4
    #
    left_column, right_column, _ = eqs
    # steal some perf here within the binary search
    # search for uniques
    # and later index them with left_positions
    left_positions, left_arr = df[left_column].factorize(sort=False)
    right_arr = right[right_column]._values
    left_index = df.index._values
    right_index = right.index._values
    slice_starts = right_arr.searchsorted(left_arr, side="left")
    slice_starts = slice_starts[left_positions]
    slice_ends = right_arr.searchsorted(left_arr, side="right")
    slice_ends = slice_ends[left_positions]
    # check if there is a search space
    # this also lets us know if there are equi matches
    keep_rows = slice_starts < slice_ends
    if not keep_rows.any():
        return None, None
    if not keep_rows.all():
        left_index = left_index[keep_rows]
        slice_starts = slice_starts[keep_rows]
        slice_ends = slice_ends[keep_rows]

    ge_arr1 = None
    ge_arr2 = None
    ge_strict = None
    if ge_gt:
        left_column, right_column, op = ge_gt
        ge_arr1 = df.loc[left_index, left_column]._values
        ge_arr2 = right[right_column]._values
        ge_arr1, ge_arr2 = _convert_to_numpy(left=ge_arr1, right=ge_arr2)
        ge_strict = True if op == ">" else False

    le_arr1 = None
    le_arr2 = None
    le_strict = None
    if le_lt:
        left_column, right_column, op = le_lt
        le_arr1 = df.loc[left_index, left_column]._values
        le_arr2 = right[right_column]._values
        le_arr1, le_arr2 = _convert_to_numpy(left=le_arr1, right=le_arr2)
        le_strict = True if op == "<" else False

    if le_lt and ge_gt:
        group = right.groupby(eqs[1])[le_lt[1]]
        # is the last column (le_lt) monotonic increasing?
        # fast path if it is
        all_monotonic_increasing = all(
            arr.is_monotonic_increasing for _, arr in group
        )
        if all_monotonic_increasing:
            cum_max_arr = le_arr2[:]
        else:
            cum_max_arr = group.cummax()._values
            if is_extension_array_dtype(cum_max_arr):
                array_dtype = cum_max_arr.dtype.numpy_dtype
                cum_max_arr = cum_max_arr.astype(array_dtype)
            if is_datetime64_dtype(cum_max_arr):
                cum_max_arr = cum_max_arr.view(np.int64)

        left_index, right_index = _numba_equi_join_range_join(
            left_index,
            right_index,
            slice_starts,
            slice_ends,
            ge_arr1,
            ge_arr2,
            ge_strict,
            le_arr1,
            le_arr2,
            le_strict,
            all_monotonic_increasing,
            cum_max_arr,
        )

    elif le_lt:
        left_index, right_index = _numba_equi_le_join(
            left_index,
            right_index,
            slice_starts,
            slice_ends,
            le_arr1,
            le_arr2,
            le_strict,
        )

    else:
        left_index, right_index = _numba_equi_ge_join(
            left_index,
            right_index,
            slice_starts,
            slice_ends,
            ge_arr1,
            ge_arr2,
            ge_strict,
        )

    return left_index, right_index


@njit(parallel=True)
def _numba_equi_le_join(
    left_index: np.ndarray,
    right_index: np.ndarray,
    slice_starts: np.ndarray,
    slice_ends: np.ndarray,
    le_arr1: np.ndarray,
    le_arr2: np.ndarray,
    le_strict: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get indices for an equi join
    and a less than join
    """
    length = left_index.size
    starts = np.empty(length, dtype=np.intp)
    booleans = np.ones(length, dtype=np.bool_)
    # sizes array is used to track the exact
    # number of matches per slice_start
    sizes = np.empty(length, dtype=np.intp)
    counts = 0
    for num in prange(length):
        l1 = le_arr1[num]
        slice_start = slice_starts[num]
        slice_end = slice_ends[num]
        r1 = le_arr2[slice_start:slice_end]
        start = np.searchsorted(r1, l1, side="left")
        if start < r1.size:
            if le_strict and (l1 == r1[start]):
                start = np.searchsorted(r1, l1, side="right")
        if start == r1.size:
            counts += 1
            booleans[num] = False
        else:
            starts[num] = slice_start + start
            sizes[num] = r1.size - start
    if counts == length:
        return None, None
    if counts > 0:
        left_index = left_index[booleans]
        starts = starts[booleans]
        slice_ends = slice_ends[booleans]
        sizes = sizes[booleans]

    slice_starts = starts
    starts = None
    # build the left and right indices
    # idea is to populate the indices
    # based on the number of matches
    # per slice_start and ensure alignment with the
    # slice_start during the iteration
    cum_sizes = np.cumsum(sizes)
    starts = np.empty(slice_ends.size, dtype=np.intp)
    starts[0] = 0
    starts[1:] = cum_sizes[:-1]
    r_index = np.empty(cum_sizes[-1], dtype=np.intp)
    l_index = np.empty(cum_sizes[-1], dtype=np.intp)
    for num in prange(slice_ends.size):
        start = starts[num]
        r_ind = slice_starts[num]
        l_ind = left_index[num]
        width = sizes[num]
        for n in range(width):
            indexer = start + n
            r_index[indexer] = right_index[r_ind + n]
            l_index[indexer] = l_ind

    return l_index, r_index


@njit(parallel=True)
def _numba_equi_ge_join(
    left_index: np.ndarray,
    right_index: np.ndarray,
    slice_starts: np.ndarray,
    slice_ends: np.ndarray,
    ge_arr1: np.ndarray,
    ge_arr2: np.ndarray,
    ge_strict: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get indices for an equi join
    and a greater than join
    """

    length = left_index.size
    ends = np.empty(length, dtype=np.intp)
    booleans = np.ones(length, dtype=np.bool_)
    # sizes array is used to track the exact
    # number of matches per slice_start
    sizes = np.empty(length, dtype=np.intp)
    counts = 0
    for num in prange(length):
        l1 = ge_arr1[num]
        slice_start = slice_starts[num]
        slice_end = slice_ends[num]
        r1 = ge_arr2[slice_start:slice_end]
        end = np.searchsorted(r1, l1, side="right")
        if end > 0:
            if ge_strict and (l1 == r1[end - 1]):
                end = np.searchsorted(r1, l1, side="left")
        if end == 0:
            counts += 1
            booleans[num] = False
        else:
            ends[num] = slice_start + end
            sizes[num] = end
    if counts == length:
        return None, None
    if counts > 0:
        left_index = left_index[booleans]
        ends = ends[booleans]
        slice_starts = slice_starts[booleans]
        sizes = sizes[booleans]
    slice_ends = ends
    ends = None
    # build the left and right indices
    # idea is to populate the indices
    # based on the number of matches
    # per slice_start and ensure alignment with the
    # slice_start during the iteration
    cum_sizes = np.cumsum(sizes)
    starts = np.empty(slice_ends.size, dtype=np.intp)
    starts[0] = 0
    starts[1:] = cum_sizes[:-1]
    r_index = np.empty(cum_sizes[-1], dtype=np.intp)
    l_index = np.empty(cum_sizes[-1], dtype=np.intp)
    for num in prange(slice_ends.size):
        start = starts[num]
        r_ind = slice_starts[num]
        l_ind = left_index[num]
        width = sizes[num]
        for n in range(width):
            indexer = start + n
            r_index[indexer] = right_index[r_ind + n]
            l_index[indexer] = l_ind

    return l_index, r_index


@njit(parallel=True)
def _numba_equi_join_range_join(
    left_index: np.ndarray,
    right_index: np.ndarray,
    slice_starts: np.ndarray,
    slice_ends: np.ndarray,
    ge_arr1: np.ndarray,
    ge_arr2: np.ndarray,
    ge_strict: bool,
    le_arr1: np.ndarray,
    le_arr2: np.ndarray,
    le_strict: bool,
    all_monotonic_increasing: bool,
    cum_max_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get indices for an equi join
    and a range join
    """
    length = left_index.size
    ends = np.empty(length, dtype=np.intp)
    booleans = np.ones(length, dtype=np.bool_)
    counts = 0
    for num in prange(length):
        l1 = ge_arr1[num]
        slice_start = slice_starts[num]
        slice_end = slice_ends[num]
        r1 = ge_arr2[slice_start:slice_end]
        end = np.searchsorted(r1, l1, side="right")
        if end > 0:
            if ge_strict and (l1 == r1[end - 1]):
                end = np.searchsorted(r1, l1, side="left")
        if end == 0:
            counts += 1
            booleans[num] = False
        else:
            ends[num] = slice_start + end
    if counts == length:
        return None, None

    if counts > 0:
        left_index = left_index[booleans]
        le_arr1 = le_arr1[booleans]
        ends = ends[booleans]
        slice_starts = slice_starts[booleans]
    slice_ends = ends
    ends = None

    length = left_index.size
    starts = np.empty(length, dtype=np.intp)
    booleans = np.ones(length, dtype=np.bool_)
    if all_monotonic_increasing:
        sizes = np.empty(length, dtype=np.intp)
    counts = 0
    for num in prange(length):
        l1 = le_arr1[num]
        slice_start = slice_starts[num]
        slice_end = slice_ends[num]
        r1 = cum_max_arr[slice_start:slice_end]
        start = np.searchsorted(r1, l1, side="left")
        if start < r1.size:
            if le_strict and (l1 == r1[start]):
                start = np.searchsorted(r1, l1, side="right")
        if start == r1.size:
            counts += 1
            booleans[num] = False
        else:
            starts[num] = slice_start + start
            if all_monotonic_increasing:
                sizes[num] = r1.size - start
    if counts == length:
        return None, None
    if counts > 0:
        left_index = left_index[booleans]
        le_arr1 = le_arr1[booleans]
        starts = starts[booleans]
        slice_ends = slice_ends[booleans]
        if all_monotonic_increasing:
            sizes = sizes[booleans]

    slice_starts = starts
    starts = None

    # no need to run a comparison
    # since all groups are monotonic increasing
    # simply create left and right indices
    if all_monotonic_increasing:
        cum_sizes = np.cumsum(sizes)
        starts = np.empty(slice_ends.size, dtype=np.intp)
        starts[0] = 0
        starts[1:] = cum_sizes[:-1]
        r_index = np.empty(cum_sizes[-1], dtype=np.intp)
        l_index = np.empty(cum_sizes[-1], dtype=np.intp)
        for num in prange(slice_ends.size):
            start = starts[num]
            r_ind = slice_starts[num]
            l_ind = left_index[num]
            width = sizes[num]
            for n in range(width):
                indexer = start + n
                r_index[indexer] = right_index[r_ind + n]
                l_index[indexer] = l_ind

        return l_index, r_index

    # get exact no of rows for left and right index
    # sizes is used to track the exact
    # number of matches per slice_start
    sizes = np.empty(slice_starts.size, dtype=np.intp)
    counts = 0
    for num in prange(slice_ends.size):
        l1 = le_arr1[num]
        start = slice_starts[num]
        end = slice_ends[num]
        r1 = le_arr2[start:end]
        internal_count = 0
        if le_strict:
            for n in range(r1.size):
                check = l1 < r1[n]
                internal_count += check
                counts += check
        else:
            for n in range(r1.size):
                check = l1 <= r1[n]
                internal_count += check
                counts += check
        sizes[num] = internal_count
    # populate the left and right index
    # idea is to populate the indices
    # based on the number of matches
    # per slice_start and ensure alignment with the
    # slice_start during the iteration
    r_index = np.empty(counts, dtype=np.intp)
    l_index = np.empty(counts, dtype=np.intp)
    starts = np.empty(sizes.size, dtype=np.intp)
    starts[0] = 0
    starts[1:] = np.cumsum(sizes)[:-1]
    for num in prange(sizes.size):
        l_ind = left_index[num]
        l1 = le_arr1[num]
        starter = slice_starts[num]
        slicer = slice(starter, slice_ends[num])
        r1 = le_arr2[slicer]
        start = starts[num]
        counter = sizes[num]
        if le_strict:
            for n in range(r1.size):
                if not counter:
                    break
                check = l1 < r1[n]
                if not check:
                    continue
                l_index[start] = l_ind
                r_index[start] = right_index[starter + n]
                counter -= 1
                start += 1
        else:
            for n in range(r1.size):
                if not counter:
                    break
                check = l1 <= r1[n]
                if not check:
                    continue
                l_index[start] = l_ind
                r_index[start] = right_index[starter + n]
                counter -= 1
                start += 1
    return l_index, r_index


@njit
def _numba_equals(arr: np.ndarray, value: Any):
    """
    Get earliest position in `arr`
    where arr[i] == `value`
    """
    min_idx = 0
    max_idx = len(arr)
    while min_idx < max_idx:
        # to avoid overflow
        mid_idx = min_idx + ((max_idx - min_idx) >> 1)
        if arr[mid_idx] == value:
            return mid_idx
        if arr[mid_idx] < value:
            min_idx = mid_idx + 1
        else:
            max_idx = mid_idx
    return -1


@njit
def _numba_less_than(arr: np.ndarray, value: Any, strict: bool = False):
    """
    Get earliest position in `arr`
    where arr[i] <= `value`
    """
    min_idx = 0
    max_idx = len(arr)
    while min_idx < max_idx:
        # to avoid overflow
        mid_idx = min_idx + ((max_idx - min_idx) >> 1)
        _mid_idx = np.uint64(mid_idx)
        if arr[_mid_idx] < value:
            min_idx = mid_idx + 1
        else:
            max_idx = mid_idx
    # it is greater than the max value in the array
    if min_idx == len(arr):
        return -1
    _value = arr[np.uint64(min_idx)]
    if strict & (value == _value):
        min_idx = _numba_greater_than(arr=arr, value=value)
    # check again
    if min_idx == len(arr):
        return -1
    return min_idx


@njit
def _numba_greater_than(arr: np.ndarray, value: Any, strict: bool = False):
    min_idx = 0
    max_idx = len(arr)
    """
    Get earliest position in `arr`
    where arr[i] > `value`
    """
    while min_idx < max_idx:
        # to avoid overflow
        mid_idx = min_idx + ((max_idx - min_idx) >> 1)
        _mid_idx = np.uint64(mid_idx)
        if value < arr[_mid_idx]:
            max_idx = mid_idx
        else:
            min_idx = mid_idx + 1
    # it is less than or equal
    # to the min value in the array
    if min_idx == 0:
        return -1
    _value = arr[np.uint64(min_idx - 1)]
    if strict & (value == _value):
        min_idx = _numba_less_than(arr=arr, value=value)
    # check again
    if min_idx == 0:
        return -1
    return min_idx


def _numba_single_non_equi_join(
    left: pd.Series, right: pd.Series, op: str, keep: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return matching indices for single non-equi join."""
    if op == "!=":
        outcome = _generic_func_cond_join(
            left=left, right=right, op=op, multiple_conditions=False, keep=keep
        )
        if outcome is None:
            return None, None
        return outcome
    outcome = _generic_func_cond_join(
        left=left, right=right, op=op, multiple_conditions=True, keep="all"
    )
    if outcome is None:
        return None, None
    left_index, right_index, starts = outcome
    if op in greater_than_join_types:
        right_index = right_index[::-1]
        starts = right_index.size - starts
    left_regions = np.empty(shape=(1, 0), dtype=np.intp)
    right_regions = np.empty(shape=(right_index.size, 0), dtype=np.intp)
    if keep == "first":
        return _numba_non_equi_join_monotonic_increasing_keep_first(
            left_regions=left_regions,
            right_regions=right_regions,
            left_index=left_index,
            right_index=right_index,
            starts=starts,
        )
    if keep == "last":
        return _numba_non_equi_join_monotonic_increasing_keep_last(
            left_regions=left_regions,
            right_regions=right_regions,
            left_index=left_index,
            right_index=right_index,
            starts=starts,
        )
    length = (right_index.size - starts).sum()
    left_indices = np.empty(length, dtype=np.intp)
    right_indices = np.empty(length, dtype=np.intp)
    return _numba_non_equi_join_monotonic_increasing_keep_all(
        left_regions=left_regions,
        right_regions=right_regions,
        left_index=left_index,
        right_index=right_index,
        left_indices=left_indices,
        right_indices=right_indices,
        starts=starts,
    )


def _numba_multiple_non_equi_join(
    df: pd.DataFrame, right: pd.DataFrame, gt_lt: list, keep: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    # https://www.scitepress.org/papers/2018/68268/68268.pdf
    An alternative to the _range_indices algorithm
    and more generalised - it covers any pair of non equi joins
    in >, >=, <, <=.
    Returns a tuple of left and right indices.
    """
    # https://www.scitepress.org/papers/2018/68268/68268.pdf

    # summary:
    # get regions for first and second conditions in the pair
    # (l_col1, r_col1, op1), (l_col2, r_col2, op2)
    # the idea is that r_col1 should always be ahead of the
    # appropriate value from lcol1; same applies to l_col2 & r_col2.
    # if the operator is in less than join types
    # the l_col should be in ascending order
    # if in greater than join types, l_col should be
    # in descending order
    # Example :
    #     df1:
    #    id  value_1
    # 0   1        2
    # 1   1        5
    # 2   1        7
    # 3   2        1
    # 4   2        3
    # 5   3        4
    #
    #
    #  df2:
    #    id  value_2A  value_2B
    # 0   1         0         1
    # 1   1         3         5
    # 2   1         7         9
    # 3   1        12        15
    # 4   2         0         1
    # 5   2         2         4
    # 6   2         3         6
    # 7   3         1         3
    #
    #
    # ('value_1', 'value_2A','>'), ('value_1', 'value_2B', '<')
    # for the first pair, since op is greater than
    # 'value_1' is sorted in descending order
    #  our pairing should be :
    # value  source      region number
    # 12   value_2A       0
    # 7    value_2A       1
    # 7    value_1        2
    # 5    value_1        2
    # 4    value_1        2
    # 3    value_2A       2
    # 3    value_2A       2
    # 3    value_1        3
    # 2    value_2A       3
    # 2    value_1        4
    # 1    value_2A       4
    # 1    value_1        5
    # 0    value_2A       5
    # 0    value_2A       5
    #
    # note that 7 for value_2A is not matched with 7 of value_1
    # because it is >, not >=, hence the different region numbers
    # looking at the output above, we can safely discard regions 0 and 1
    # since they do not have any matches with value_1
    # for the second pair, since op is <, value_1 is sorted
    # in ascending order, and our pairing should be:
    #   value    source    region number
    #     1    value_2B       0
    #     1    value_2B       1
    #     1    value_1        2
    #     2    value_1        2
    #     3    value_2B       2
    #     3    value_1        3
    #     4    value_2B       3
    #     4    value_1        4
    #     5    value_2B       4
    #     5    value_1        5
    #     6    value_2B       5
    #     7    value_1        6
    #     9    value_2B       6
    #     15   value_2B       6
    #
    # from the above we can safely discard regions 0 and 1, since there are
    # no matches with value_1 ... note that the index for regions 0 and 1
    # coincide with the index for region 5 values in value_2A(0, 0);
    # as such those regions will be discarded.
    # Similarly, the index for regions 0 and 1 of value_2A(12, 7)
    # coincide with the index for regions 6 for value_2B(9, 15);
    # these will be discarded as well.
    # let's create a table of the regions, paired with the index
    #
    #
    #  value_1 :
    ###############################################
    # index-->  2  1  5  4  0  3
    # pair1-->  2  2  2  3  4  5
    # pair2-->  6  5  4  3  2  2
    ###############################################
    #
    #
    # value_2A, value_2B
    ##############################################
    # index --> 1  6  5  7
    # pair1 --> 2  2  3  4
    # pair2 --> 4  5  3  2
    ##############################################
    #
    # To find matching indices, the regions from value_1 must be less than
    # or equal to the regions in value_2A/2B.
    # pair1 <= pair1 and pair2 <= pair2
    # Starting from the highest region in value_1
    # 5 in pair1 is not less than any in value_2A/2B, so we discard
    # 4 in pair1 is matched to 4 in pair1 of value_2A/2B
    # we look at the equivalent value in pair2 for 4, which is 2
    # 2 matches 2 in pair 2, so we have a match -> (0, 7)
    # 3 in pair 1 from value_1 matches 3 and 4 in pair1 for value_2A/2B
    # next we compare the equivalent value from pair2, which is 3
    # 3 matches only 3 in value_2A/2B, so our only match is  -> (4, 5)
    # next is 2 (we have 3 2s in value_1 for pair1)
    # they all match 2, 2, 3, 4 in pair1 of value_2A/2B
    # compare the first equivalent in pair2 -> 4
    # 4 matches only 4, 5 in pair2 of value_2A/2B
    # ->(5, 1), (5, 6)
    # the next equivalent is -> 5
    # 5 matches only 5 in pair2 of value_2A/2B
    # -> (1, 6)
    # the last equivalent is -> 6
    # 6 has no match in pair2 of value_2A/2B, so we discard
    # our final matching indices for the left and right pairs
    #########################################################
    # left_index      right_index
    #     0              7
    #     4              5
    #     5              1
    #     5              6
    #     1              6
    ########################################################
    # and if we index the dataframes, we should get the output below:
    #################################
    #    value_1  value_2A  value_2B
    # 0        2         1         3
    # 1        5         3         6
    # 2        3         2         4
    # 3        4         3         5
    # 4        4         3         6
    ################################
    left_df = df[:]
    right_df = right[:]
    left_column, right_column, _ = gt_lt[0]
    # sorting on the first column
    # helps to achieve more performance
    # when iterating to compare left and right regions for matches
    # note - keep track of the original index
    if not left_df[left_column].is_monotonic_increasing:
        left_df = df.sort_values(left_column)
        left_index = left_df.index._values
        left_df.index = range(len(left_df))
    else:
        left_index = left_df.index._values
    if not right_df[right_column].is_monotonic_increasing:
        right_df = right_df.sort_values(right_column)
        # original_index and right_is_sorted
        # is relevant where keep in {'first','last'}
        right_index = right_df.index._values
        right_df.index = range(len(right))
        right_is_sorted = False
    else:
        right_index = right_df.index._values
        right_is_sorted = True
    shape = (len(left_df), len(gt_lt))
    left_regions = np.empty(shape=shape, dtype=np.intp, order="F")
    left_regions[:] = -1
    shape = (len(right_df), len(gt_lt))
    right_regions = np.empty(shape=shape, dtype=np.intp, order="F")
    right_regions[:] = -1
    for position, (left_column, right_column, op) in enumerate(gt_lt):
        outcome = _generic_func_cond_join(
            left=left_df[left_column],
            right=right_df[right_column],
            op=op,
            multiple_conditions=True,
            keep="all",
        )
        if outcome is None:
            return None, None
        left_indexer, right_indexer, search_indices = outcome
        if op in greater_than_join_types:
            search_indices = right_indexer.size - search_indices
            right_indexer = right_indexer[::-1]
        r_region = np.zeros(right_indexer.size, dtype=np.intp)
        r_region[search_indices] = 1
        r_region[0] -= 1
        r_region = r_region.cumsum()
        left_regions[left_indexer, position] = r_region[search_indices]
        right_regions[right_indexer, position] = r_region
    r_region = None
    search_indices = None
    left_df = None
    right_df = None
    booleans = left_regions == -1
    booleans = booleans.any(axis=1)
    if booleans.any():
        booleans = np.logical_not(booleans)
        left_regions = left_regions[booleans]
        left_index = left_index[booleans]
    booleans = right_regions == -1
    booleans = booleans.any(axis=1)
    if booleans.any():
        booleans = np.logical_not(booleans)
        right_regions = right_regions[booleans]
        right_index = right_index[booleans]
    if gt_lt[0][-1] in greater_than_join_types:
        left_regions = left_regions[::-1]
        left_index = left_index[::-1]
        right_regions = right_regions[::-1]
        right_index = right_index[::-1]
        right_index_flipped = True
    else:
        right_index_flipped = False
    starts = right_regions[:, 0].searchsorted(left_regions[:, 0])
    booleans = starts == len(right_regions)
    if booleans.all():
        return None, None
    if booleans.any():
        booleans = np.logical_not(booleans)
        starts = starts[booleans]
        left_regions = left_regions[booleans]
        left_index = left_index[booleans]
    arr = pd.Index(right_regions[:, 1])

    check_increasing = arr.is_monotonic_increasing
    check_decreasing = arr.is_monotonic_decreasing
    arr = None
    if check_increasing:
        search_indices = right_regions[:, 1].searchsorted(left_regions[:, 1])
        booleans = search_indices == len(right_regions)
        if booleans.all():
            return None, None
        if booleans.any():
            booleans = np.logical_not(booleans)
            starts = starts[booleans]
            search_indices = search_indices[booleans]
            left_regions = left_regions[booleans]
            left_index = left_index[booleans]
        booleans = starts > search_indices
        starts = np.where(booleans, starts, search_indices)
        if right_is_sorted & (len(gt_lt) == 2):
            ends = np.empty(left_index.size, dtype=np.intp)
            ends[:] = len(right_regions)
    elif check_decreasing:
        ends = right_regions[::-1, 1].searchsorted(left_regions[:, 1])
        booleans = starts == len(right_regions)
        if booleans.all():
            return None, None
        if booleans.any():
            booleans = np.logical_not(booleans)
            starts = starts[booleans]
            left_regions = left_regions[booleans]
            left_index = left_index[booleans]
        ends = len(right_regions) - ends
        booleans = starts >= ends
        if booleans.all():
            return None, None
        if booleans.any():
            booleans = np.logical_not(booleans)
            starts = starts[booleans]
            left_regions = left_regions[booleans]
            left_index = left_index[booleans]
            ends = ends[booleans]
    if (
        (check_decreasing | check_increasing)
        & right_is_sorted
        & (keep == "first")
        & (len(gt_lt) == 2)
        & right_index_flipped
    ):
        return left_index, right_index[ends - 1]
    if (
        (check_decreasing | check_increasing)
        & right_is_sorted
        & (keep == "last")
        & (len(gt_lt) == 2)
        & right_index_flipped
    ):
        return left_index, right_index[starts]
    if (
        (check_decreasing | check_increasing)
        & right_is_sorted
        & (keep == "first")
        & (len(gt_lt) == 2)
    ):
        return left_index, right_index[starts]
    if (
        (check_decreasing | check_increasing)
        & right_is_sorted
        & (keep == "last")
        & (len(gt_lt) == 2)
    ):
        return left_index, right_index[ends - 1]
    if (check_increasing) & (keep == "first"):
        return _numba_non_equi_join_monotonic_increasing_keep_first(
            left_regions=left_regions[:, 2:],
            right_regions=right_regions[:, 2:],
            left_index=left_index,
            right_index=right_index,
            starts=starts,
        )
    if (check_increasing) & (keep == "last"):
        return _numba_non_equi_join_monotonic_increasing_keep_last(
            left_regions=left_regions[:, 2:],
            right_regions=right_regions[:, 2:],
            left_index=left_index,
            right_index=right_index,
            starts=starts,
        )
    if check_increasing:
        length = (right_index.size - starts).sum()
        left_indices = np.empty(length, dtype=np.intp)
        right_indices = np.empty(length, dtype=np.intp)
        return _numba_non_equi_join_monotonic_increasing_keep_all(
            left_regions=left_regions[:, 2:],
            right_regions=right_regions[:, 2:],
            left_index=left_index,
            right_index=right_index,
            left_indices=left_indices,
            right_indices=right_indices,
            starts=starts,
        )
    if (check_decreasing) & (keep == "first"):
        return _numba_non_equi_join_monotonic_keep_first(
            left_regions=left_regions[:, 2:],
            right_regions=right_regions[:, 2:],
            left_index=left_index,
            right_index=right_index,
            starts=starts,
            ends=ends,
        )

    if (check_decreasing) & (keep == "last"):
        return _numba_non_equi_join_monotonic_keep_last(
            left_regions=left_regions[:, 2:],
            right_regions=right_regions[:, 2:],
            left_index=left_index,
            right_index=right_index,
            starts=starts,
            ends=ends,
        )

    if check_decreasing:
        length = (ends - starts).sum()
        left_indices = np.empty(length, dtype=np.intp)
        right_indices = np.empty(length, dtype=np.intp)
        return _numba_non_equi_join_monotonic_keep_all(
            left_regions=left_regions[:, 2:],
            right_regions=right_regions[:, 2:],
            left_index=left_index,
            right_index=right_index,
            left_indices=left_indices,
            right_indices=right_indices,
            starts=starts,
            ends=ends,
        )

    load_factor = 3
    width = load_factor * 2
    length = ceil(right_index.size / load_factor)
    sorted_array = np.empty(
        (width, length), dtype=right_regions.dtype, order="F"
    )
    positions_array = np.empty(
        (width, length), dtype=right_regions.dtype, order="F"
    )
    # keep track of the max value per column
    maxxes = np.empty(length, dtype=np.intp)
    # keep track of the length of actual data for each column
    lengths = np.empty(length, dtype=np.intp)
    if keep == "all":
        return _numba_non_equi_join_not_monotonic_keep_all(
            left_regions=left_regions[:, 1:],
            right_regions=right_regions[:, 1:],
            left_index=left_index,
            right_index=right_index,
            maxxes=maxxes,
            lengths=lengths,
            sorted_array=sorted_array,
            positions_array=positions_array,
            starts=starts,
            load_factor=load_factor,
        )
    if keep == "first":
        return _numba_non_equi_join_not_monotonic_keep_first(
            left_regions=left_regions[:, 1:],
            right_regions=right_regions[:, 1:],
            left_index=left_index,
            right_index=right_index,
            maxxes=maxxes,
            lengths=lengths,
            sorted_array=sorted_array,
            positions_array=positions_array,
            starts=starts,
            load_factor=load_factor,
        )
    # keep == 'last'
    return _numba_non_equi_join_not_monotonic_keep_last(
        left_regions=left_regions[:, 1:],
        right_regions=right_regions[:, 1:],
        left_index=left_index,
        right_index=right_index,
        maxxes=maxxes,
        lengths=lengths,
        sorted_array=sorted_array,
        positions_array=positions_array,
        starts=starts,
        load_factor=load_factor,
    )


@njit(cache=True)
def _numba_non_equi_join_not_monotonic_keep_all(
    left_regions: np.ndarray,
    right_regions: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    maxxes: np.ndarray,
    lengths: np.ndarray,
    sorted_array: np.ndarray,
    positions_array: np.ndarray,
    starts: np.ndarray,
    load_factor: int,
):
    """
    Get indices for non-equi join,
    where the right regions are not monotonic
    """
    length = left_index.size
    end = right_index.size
    end -= 1
    region = right_regions[np.uint64(end), 0]
    sorted_array[0, 0] = region
    positions_array[0, 0] = end
    maxes_counter = 1
    maxxes[0] = region
    lengths[0] = 1
    total = (right_regions.size - starts).sum()
    left_indices = np.empty(total, dtype=np.intp)
    right_indices = np.empty(total, dtype=np.intp)
    begin = 0
    r_count = 0
    for indexer in range(length - 1, -1, -1):
        _indexer = np.uint64(indexer)
        start = starts[_indexer]
        for num in range(start, end):
            _num = np.uint64(num)
            region = right_regions[_num, 0]
            arr = maxxes[:maxes_counter]
            posn = _numba_less_than(arr=arr, value=region)
            if posn == -1:
                posn = maxes_counter - 1
                posn_ = np.uint64(posn)
                len_arr = lengths[posn_]
                len_arr_ = np.uint64(len_arr)
                sorted_array[len_arr_, posn_] = region
                positions_array[len_arr_, posn_] = num
                maxxes[posn_] = region
                lengths[posn_] += 1
            else:
                posn_ = np.uint64(posn)
                len_arr = lengths[posn_]
                arr = sorted_array[:len_arr, posn_]
                insort_posn = _numba_less_than(arr=arr, value=region)
                # shift downwards before inserting
                for ind in range(len_arr - 1, insort_posn - 1, -1):
                    ind_ = np.uint64(ind)
                    _ind = np.uint64(ind + 1)
                    sorted_array[_ind, posn_] = sorted_array[ind_, posn_]
                    positions_array[_ind, posn_] = positions_array[ind_, posn_]
                insort = np.uint64(insort_posn)
                sorted_array[insort, posn_] = region
                positions_array[insort, posn_] = num
                lengths[posn_] += 1
                maxxes[posn_] = sorted_array[np.uint64(len_arr), posn_]
            r_count += 1
            posn_ = np.uint64(posn)
            check = (lengths[posn_] == (load_factor * 2)) & (
                r_count < right_index.size
            )
            if check:
                # shift from left+1 to right
                for pos in range(maxes_counter - 1, posn, -1):
                    forward = np.uint64(pos + 1)
                    current = np.uint64(pos)
                    sorted_array[:, forward] = sorted_array[:, current]
                    positions_array[:, forward] = positions_array[:, current]
                    maxxes[forward] = maxxes[current]
                    lengths[forward] = lengths[current]
                # share half the load from left to left+1
                forward = np.uint64(posn + 1)
                current = np.uint64(posn)
                maxxes[forward] = sorted_array[-1, current]
                lengths[forward] = load_factor
                sorted_array[:load_factor, forward] = sorted_array[
                    load_factor:, current
                ]
                positions_array[:load_factor, forward] = positions_array[
                    load_factor:, current
                ]
                lengths[current] = load_factor
                maxxes[current] = sorted_array[
                    np.uint64(load_factor - 1), current
                ]
                maxes_counter += 1
        # now we do a binary search
        # for left region in right region
        l_region = left_regions[_indexer, 0]
        arr = maxxes[:maxes_counter]
        posn = _numba_less_than(arr=arr, value=l_region)
        if posn == -1:
            end = start
            continue
        posn_ = np.uint64(posn)
        len_arr = lengths[posn_]
        arr = sorted_array[:len_arr, posn_]
        _posn = _numba_less_than(arr=arr, value=l_region)
        l_index = left_index[_indexer]
        for ind in range(_posn, len_arr):
            ind_ = np.uint64(ind)
            counter = 1
            # move along the columns
            # and look for matches
            for loc in range(1, right_regions.shape[1]):
                loc_ = np.uint64(loc)
                next_left = left_regions[_indexer, loc_]
                r_pos = positions_array[ind_, posn_]
                r_pos = np.uint64(r_pos)
                next_right = right_regions[r_pos, loc_]
                if next_left > next_right:
                    counter = 0
                    break
            if counter == 0:
                continue
            begin_ = np.uint64(begin)
            r_pos = positions_array[ind_, posn_]
            r_pos = np.uint64(r_pos)
            r_index = right_index[r_pos]
            left_indices[begin_] = l_index
            right_indices[begin_] = r_index
            begin += 1
        for ind in range(posn + 1, maxes_counter):
            ind_ = np.uint64(ind)
            len_arr = lengths[ind_]
            for num in range(len_arr):
                _num = np.uint64(num)
                counter = 1
                for loc in range(1, right_regions.shape[1]):
                    loc_ = np.uint64(loc)
                    next_left = left_regions[_indexer, loc_]
                    r_pos = positions_array[_num, ind_]
                    r_pos = np.uint64(r_pos)
                    next_right = right_regions[r_pos, loc_]
                    if next_left > next_right:
                        counter = 0
                        break
                if counter == 0:
                    continue
                begin_ = np.uint64(begin)
                left_indices[begin_] = l_index
                r_pos = positions_array[_num, ind_]
                r_pos = np.uint64(r_pos)
                r_index = right_index[r_pos]
                right_indices[begin_] = r_index
                begin += 1
        end = start
    if begin == 0:
        return None, None
    return left_indices[:begin], right_indices[:begin]


@njit(cache=True)
def _numba_non_equi_join_not_monotonic_keep_first(
    left_regions: np.ndarray,
    right_regions: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    maxxes: np.ndarray,
    lengths: np.ndarray,
    sorted_array: np.ndarray,
    positions_array: np.ndarray,
    starts: np.ndarray,
    load_factor: int,
):
    length = left_index.size
    end = right_index.size
    end -= 1
    region = right_regions[np.uint64(end), 0]
    sorted_array[0, 0] = region
    positions_array[0, 0] = end
    maxes_counter = 1
    maxxes[0] = region
    lengths[0] = 1
    left_indices = np.empty(left_index.size, dtype=np.intp)
    right_indices = np.empty(left_index.size, dtype=np.intp)
    begin = 0
    r_count = 0
    for indexer in range(length - 1, -1, -1):
        _indexer = np.uint64(indexer)
        start = starts[_indexer]
        for num in range(start, end):
            _num = np.uint64(num)
            region = right_regions[_num, 0]
            arr = maxxes[:maxes_counter]
            posn = _numba_less_than(arr=arr, value=region)
            if posn == -1:
                posn = maxes_counter - 1
                posn_ = np.uint64(posn)
                len_arr = lengths[posn_]
                len_arr_ = np.uint64(len_arr)
                sorted_array[len_arr_, posn_] = region
                positions_array[len_arr_, posn_] = num
                maxxes[posn_] = region
                lengths[posn_] += 1
            else:
                posn_ = np.uint64(posn)
                len_arr = lengths[posn_]
                arr = sorted_array[:len_arr, posn_]
                insort_posn = _numba_less_than(arr=arr, value=region)
                # shift downwards before inserting
                for ind in range(len_arr - 1, insort_posn - 1, -1):
                    ind_ = np.uint64(ind)
                    _ind = np.uint64(ind + 1)
                    sorted_array[_ind, posn_] = sorted_array[ind_, posn_]
                    positions_array[_ind, posn_] = positions_array[ind_, posn_]
                insort = np.uint64(insort_posn)
                sorted_array[insort, posn_] = region
                positions_array[insort, posn_] = num
                lengths[posn_] += 1
                maxxes[posn_] = sorted_array[np.uint64(len_arr), posn_]
            r_count += 1
            posn_ = np.uint64(posn)
            check = (lengths[posn_] == (load_factor * 2)) & (
                r_count < right_index.size
            )
            if check:
                # shift from left+1 to right
                for pos in range(maxes_counter - 1, posn, -1):
                    forward = np.uint64(pos + 1)
                    current = np.uint64(pos)
                    sorted_array[:, forward] = sorted_array[:, current]
                    positions_array[:, forward] = positions_array[:, current]
                    maxxes[forward] = maxxes[current]
                    lengths[forward] = lengths[current]
                # share half the load from left to left+1
                forward = np.uint64(posn + 1)
                current = np.uint64(posn)
                maxxes[forward] = sorted_array[-1, current]
                lengths[forward] = load_factor
                sorted_array[:load_factor, forward] = sorted_array[
                    load_factor:, current
                ]
                positions_array[:load_factor, forward] = positions_array[
                    load_factor:, current
                ]
                lengths[current] = load_factor
                maxxes[current] = sorted_array[
                    np.uint64(load_factor - 1), current
                ]
                maxes_counter += 1
        l_region = left_regions[_indexer, 0]
        arr = maxxes[:maxes_counter]
        posn = _numba_less_than(arr=arr, value=l_region)
        if posn == -1:
            end = start
            continue
        posn_ = np.uint64(posn)
        len_arr = lengths[posn_]
        arr = sorted_array[:len_arr, posn_]
        _posn = _numba_less_than(arr=arr, value=l_region)
        matches = 0
        for ind in range(_posn, len_arr):
            ind_ = np.uint64(ind)
            counter = 1
            for loc in range(1, right_regions.shape[1]):
                loc_ = np.uint64(loc)
                next_left = left_regions[_indexer, loc_]
                r_pos = positions_array[ind_, posn_]
                r_pos = np.uint64(r_pos)
                next_right = right_regions[r_pos, loc_]
                if next_left > next_right:
                    counter = 0
                    break
            if counter == 0:
                continue
            r_pos = positions_array[ind_, posn_]
            r_pos = np.uint64(r_pos)
            r_index = right_index[r_pos]
            if matches == 0:
                base_index = r_index
                matches = 1
            elif r_index < base_index:
                base_index = r_index
        # step into the remaining columns
        for ind in range(posn + 1, maxes_counter):
            ind_ = np.uint64(ind)
            len_arr = lengths[ind_]
            # step into the rows for each column
            for num in range(len_arr):
                _num = np.uint64(num)
                counter = 1
                for loc in range(1, right_regions.shape[1]):
                    loc_ = np.uint64(loc)
                    next_left = left_regions[_indexer, loc_]
                    r_pos = positions_array[_num, ind_]
                    r_pos = np.uint64(r_pos)
                    next_right = right_regions[r_pos, loc_]
                    if next_left > next_right:
                        counter = 0
                        break
                if counter == 0:
                    continue
                r_pos = positions_array[_num, ind_]
                r_pos = np.uint64(r_pos)
                r_index = right_index[r_pos]
                if matches == 0:
                    base_index = r_index
                    matches = 1
                elif r_index < base_index:
                    base_index = r_index
        if matches == 0:
            end = start
            continue
        begin_ = np.uint64(begin)
        l_index = left_index[_indexer]
        left_indices[begin_] = l_index
        right_indices[begin_] = base_index
        begin += 1
        end = start
    if begin == 0:
        return None, None
    return left_indices[:begin], right_indices[:begin]


@njit(cache=True)
def _numba_non_equi_join_not_monotonic_keep_last(
    left_regions: np.ndarray,
    right_regions: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    maxxes: np.ndarray,
    lengths: np.ndarray,
    sorted_array: np.ndarray,
    positions_array: np.ndarray,
    starts: np.ndarray,
    load_factor: int,
):
    length = left_index.size
    end = right_index.size
    end -= 1
    region = right_regions[np.uint64(end), 0]
    sorted_array[0, 0] = region
    positions_array[0, 0] = end
    maxes_counter = 1
    maxxes[0] = region
    lengths[0] = 1
    left_indices = np.empty(left_index.size, dtype=np.intp)
    right_indices = np.empty(left_index.size, dtype=np.intp)
    begin = 0
    r_count = 0
    for indexer in range(length - 1, -1, -1):
        _indexer = np.uint64(indexer)
        start = starts[_indexer]
        for num in range(start, end):
            _num = np.uint64(num)
            region = right_regions[_num, 0]
            arr = maxxes[:maxes_counter]
            posn = _numba_less_than(arr=arr, value=region)
            if posn == -1:
                posn = maxes_counter - 1
                posn_ = np.uint64(posn)
                len_arr = lengths[posn_]
                len_arr_ = np.uint64(len_arr)
                sorted_array[len_arr_, posn_] = region
                positions_array[len_arr_, posn_] = num
                maxxes[posn_] = region
                lengths[posn_] += 1
            else:
                posn_ = np.uint64(posn)
                len_arr = lengths[posn_]
                arr = sorted_array[:len_arr, posn_]
                insort_posn = _numba_less_than(arr=arr, value=region)
                # shift downwards before inserting
                for ind in range(len_arr - 1, insort_posn - 1, -1):
                    ind_ = np.uint64(ind)
                    _ind = np.uint64(ind + 1)
                    sorted_array[_ind, posn_] = sorted_array[ind_, posn_]
                    positions_array[_ind, posn_] = positions_array[ind_, posn_]
                insort = np.uint64(insort_posn)
                sorted_array[insort, posn_] = region
                positions_array[insort, posn_] = num
                lengths[posn_] += 1
                maxxes[posn_] = sorted_array[np.uint64(len_arr), posn_]
            r_count += 1
            posn_ = np.uint64(posn)
            check = (lengths[posn_] == (load_factor * 2)) & (
                r_count < right_index.size
            )
            if check:
                # shift from left+1 to right
                for pos in range(maxes_counter - 1, posn, -1):
                    forward = np.uint64(pos + 1)
                    current = np.uint64(pos)
                    sorted_array[:, forward] = sorted_array[:, current]
                    positions_array[:, forward] = positions_array[:, current]
                    maxxes[forward] = maxxes[current]
                    lengths[forward] = lengths[current]
                # share half the load from left to left+1
                forward = np.uint64(posn + 1)
                current = np.uint64(posn)
                maxxes[forward] = sorted_array[-1, current]
                lengths[forward] = load_factor
                sorted_array[:load_factor, forward] = sorted_array[
                    load_factor:, current
                ]
                positions_array[:load_factor, forward] = positions_array[
                    load_factor:, current
                ]
                lengths[current] = load_factor
                maxxes[current] = sorted_array[
                    np.uint64(load_factor - 1), current
                ]
                maxes_counter += 1
        l_region = left_regions[_indexer, 0]
        arr = maxxes[:maxes_counter]
        posn = _numba_less_than(arr=arr, value=l_region)
        if posn == -1:
            end = start
            continue
        posn_ = np.uint64(posn)
        len_arr = lengths[posn_]
        arr = sorted_array[:len_arr, posn_]
        _posn = _numba_less_than(arr=arr, value=l_region)
        matches = 0
        for ind in range(_posn, len_arr):
            ind_ = np.uint64(ind)
            counter = 1
            for loc in range(1, right_regions.shape[1]):
                loc_ = np.uint64(loc)
                next_left = left_regions[_indexer, loc_]
                r_pos = positions_array[ind_, posn_]
                r_pos = np.uint64(r_pos)
                next_right = right_regions[r_pos, loc_]
                if next_left > next_right:
                    counter = 0
                    break
            if counter == 0:
                continue
            r_pos = positions_array[ind_, posn_]
            r_pos = np.uint64(r_pos)
            r_index = right_index[r_pos]
            if matches == 0:
                base_index = r_index
                matches = 1
            elif r_index > base_index:
                base_index = r_index
        # step into the remaining columns
        for ind in range(posn + 1, maxes_counter):
            ind_ = np.uint64(ind)
            len_arr = lengths[ind_]
            # step into the rows for each column
            for num in range(len_arr):
                _num = np.uint64(num)
                counter = 1
                for loc in range(1, right_regions.shape[1]):
                    loc_ = np.uint64(loc)
                    next_left = left_regions[_indexer, loc_]
                    r_pos = positions_array[_num, ind_]
                    r_pos = np.uint64(r_pos)
                    next_right = right_regions[r_pos, loc_]
                    if next_left > next_right:
                        counter = 0
                        break
                if counter == 0:
                    continue
                r_pos = positions_array[_num, ind_]
                r_pos = np.uint64(r_pos)
                r_index = right_index[r_pos]
                if matches == 0:
                    base_index = r_index
                    matches = 1
                elif r_index > base_index:
                    base_index = r_index
        if matches == 0:
            end = start
            continue
        begin_ = np.uint64(begin)
        l_index = left_index[_indexer]
        left_indices[begin_] = l_index
        right_indices[begin_] = base_index
        begin += 1
        end = start
    if begin == 0:
        return None, None
    return left_indices[:begin], right_indices[:begin]


@njit(cache=True)
def _numba_non_equi_join_monotonic_keep_first(
    left_regions: np.ndarray,
    right_regions: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
):
    """
    Get  indices for a non equi join.
    """
    length = left_index.size
    left_indices = np.empty(length, dtype=np.intp)
    right_indices = np.empty(length, dtype=np.intp)
    n = 0
    for ind in range(length):
        _ind = np.uint64(ind)
        start = starts[_ind]
        end = ends[_ind]
        matches = 0
        for num in range(start, end):
            _num = np.uint64(num)
            counter = 1
            for loc in range(right_regions.shape[1]):
                loc_ = np.uint64(loc)
                next_left = left_regions[_ind, loc_]
                next_right = right_regions[_num, loc_]
                if next_left > next_right:
                    counter = 0
                    break
            if counter == 0:
                continue
            rindex = right_index[_num]
            if matches == 0:
                base = rindex
                matches = 1
            elif rindex < base:
                base = rindex
        if matches == 0:
            continue
        _n = np.uint64(n)
        left_indices[_n] = left_index[_ind]
        right_indices[_n] = base
        n += 1
    if n == 0:
        return None, None
    return left_indices[:n], right_indices[:n]


@njit(cache=True)
def _numba_non_equi_join_monotonic_increasing_keep_first(
    left_regions: np.ndarray,
    right_regions: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    starts: np.ndarray,
):
    """
    Get  indices for a non equi join.
    """
    length = left_index.size
    left_indices = np.empty(length, dtype=np.intp)
    right_indices = np.empty(length, dtype=np.intp)
    end = len(right_regions)
    n = 0
    for ind in range(length):
        _ind = np.uint64(ind)
        start = starts[_ind]
        matches = 0
        for num in range(start, end):
            _num = np.uint64(num)
            counter = 1
            for loc in range(right_regions.shape[1]):
                loc_ = np.uint64(loc)
                next_left = left_regions[_ind, loc_]
                next_right = right_regions[_num, loc_]
                if next_left > next_right:
                    counter = 0
                    break
            if counter == 0:
                continue
            rindex = right_index[_num]
            if matches == 0:
                base = rindex
                matches = 1
            elif rindex < base:
                base = rindex
        if matches == 0:
            continue
        _n = np.uint64(n)
        left_indices[_n] = left_index[_ind]
        right_indices[_n] = base
        n += 1
    if n == 0:
        return None, None
    return left_indices[:n], right_indices[:n]


@njit(cache=True)
def _numba_non_equi_join_monotonic_keep_last(
    left_regions: np.ndarray,
    right_regions: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
):
    """
    Get  indices for a non equi join.
    """
    length = left_index.size
    left_indices = np.empty(length, dtype=np.intp)
    right_indices = np.empty(length, dtype=np.intp)
    n = 0
    for ind in range(length):
        _ind = np.uint64(ind)
        start = starts[_ind]
        end = ends[_ind]
        matches = 0
        for num in range(start, end):
            _num = np.uint64(num)
            counter = 1
            for loc in range(right_regions.shape[1]):
                loc_ = np.uint64(loc)
                next_left = left_regions[_ind, loc_]
                next_right = right_regions[_num, loc_]
                if next_left > next_right:
                    counter = 0
                    break
            if counter == 0:
                continue
            rindex = right_index[_num]
            if matches == 0:
                base = rindex
                matches = 1
            elif rindex > base:
                base = rindex
        if matches == 0:
            continue
        _n = np.uint64(n)
        left_indices[_n] = left_index[_ind]
        right_indices[_n] = base
        n += 1
    if n == 0:
        return None, None
    return left_indices[:n], right_indices[:n]


@njit(cache=True)
def _numba_non_equi_join_monotonic_increasing_keep_last(
    left_regions: np.ndarray,
    right_regions: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    starts: np.ndarray,
):
    """
    Get  indices for a non equi join.
    """
    length = left_index.size
    left_indices = np.empty(length, dtype=np.intp)
    right_indices = np.empty(length, dtype=np.intp)
    end = len(right_regions)
    n = 0
    for ind in range(length):
        _ind = np.uint64(ind)
        start = starts[_ind]
        matches = 0
        for num in range(start, end):
            _num = np.uint64(num)
            counter = 1
            for loc in range(right_regions.shape[1]):
                loc_ = np.uint64(loc)
                next_left = left_regions[_ind, loc_]
                next_right = right_regions[_num, loc_]
                if next_left > next_right:
                    counter = 0
                    break
            if counter == 0:
                continue
            rindex = right_index[_num]
            if matches == 0:
                base = rindex
                matches = 1
            elif rindex > base:
                base = rindex
        if matches == 0:
            continue
        _n = np.uint64(n)
        left_indices[_n] = left_index[_ind]
        right_indices[_n] = base
        n += 1
    if n == 0:
        return None, None
    return left_indices[:n], right_indices[:n]


# @njit(cache=True)
def _numba_non_equi_join_monotonic_keep_all(
    left_regions: np.ndarray,
    right_regions: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
):
    """
    Get  indices for a non equi join.
    """
    length = left_index.size
    n = 0
    for ind in range(length):
        _ind = np.uint64(ind)
        start = starts[_ind]
        end = ends[_ind]
        lindex = left_index[_ind]
        for num in range(start, end):
            _num = np.uint64(num)
            counter = 1
            for loc in range(right_regions.shape[1]):
                loc_ = np.uint64(loc)
                next_left = left_regions[_ind, loc_]
                next_right = right_regions[_num, loc_]
                if next_left > next_right:
                    counter = 0
                    break
            if counter == 0:
                continue
            rindex = right_index[_num]
            _n = np.uint64(n)
            left_indices[_n] = lindex
            right_indices[_n] = rindex
            n += 1
    if n == 0:
        return None, None
    return left_indices[:n], right_indices[:n]


@njit(cache=True)
def _numba_non_equi_join_monotonic_increasing_keep_all(
    left_regions: np.ndarray,
    right_regions: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
    starts: np.ndarray,
):
    """
    Get  indices for a non equi join.
    """
    length = left_index.size
    end = len(right_regions)
    n = 0
    for ind in range(length):
        _ind = np.uint64(ind)
        start = starts[_ind]
        lindex = left_index[_ind]
        for num in range(start, end):
            _num = np.uint64(num)
            counter = 1
            for loc in range(right_regions.shape[1]):
                loc_ = np.uint64(loc)
                next_left = left_regions[_ind, loc_]
                next_right = right_regions[_num, loc_]
                if next_left > next_right:
                    counter = 0
                    break
            if counter == 0:
                continue
            rindex = right_index[_num]
            _n = np.uint64(n)
            left_indices[_n] = lindex
            right_indices[_n] = rindex
            n += 1
    if n == 0:
        return None, None
    return left_indices[:n], right_indices[:n]
