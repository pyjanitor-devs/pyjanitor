"""Various Functions powered by Numba"""

from __future__ import annotations

from math import ceil
from typing import Any, Union

import numpy as np
import pandas as pd
from numba import njit, prange
from pandas.api.types import (
    is_datetime64_dtype,
    is_extension_array_dtype,
)

# https://numba.discourse.group/t/uint64-vs-int64-indexing-performance-difference/1500
# indexing with unsigned integers offers more performance
from janitor.functions.utils import (
    _null_checks_cond_join,
    less_than_join_types,
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


@njit(parallel=True, cache=True)
def _numba_less_than_indices(
    left: np.ndarray,
    right: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    strict: bool,
    right_is_sorted: bool,
    keep: str,
) -> tuple:
    """
    Use binary search to get indices where left
    is less than or equal to right.

    If strict is True, then only indices
    where `left` is less than
    (but not equal to) `right` are returned.

    A tuple of integer indexes
    for left and right is returned.

    Args:
        left: left array.
        right: right array. Should be sorted.
        Find positions within this array
        where values from the left should be inserted.
        left_index: Index of left array. Required to reconstruct DataFrame.
        right_index: Index of right array. Required to reconstruct DataFrame.
        strict: True if '<', else False.
        right_is_sorted: True if right array was already sorted
        keep: Determines if all rows are returned ('all'),
            or the first and last matches.

    Returns:
        A tuple of arrays.
    """
    search_indices = np.empty(left.size, dtype=np.intp)
    len_arr = right.size
    total = 0  # relevant when keep == 'all'
    counts = 0  # relevant when keep in {'first','last'}
    for indexer in prange(left.size):
        _indexer = np.uint64(indexer)
        value = left[_indexer]
        outcome = _numba_less_than(arr=right, value=value, strict=strict)
        search_indices[_indexer] = outcome
        if outcome == -1:
            continue
        counts += 1
        result = len_arr - outcome
        total += result
    if counts == 0:
        return None, None
    if right_is_sorted and (keep == "first"):
        new_right_index = np.empty(counts, dtype=right_index.dtype)
        new_left_index = np.empty(counts, dtype=left_index.dtype)
        new_indexer = 0
        for indexer in range(left.size):
            _indexer = np.uint64(indexer)
            value = search_indices[_indexer]
            if value == -1:
                continue
            _value = np.uint64(value)
            _new_indexer = np.uint64(new_indexer)
            new_left_index[_new_indexer] = left_index[_indexer]
            new_right_index[_new_indexer] = right_index[_value]
            new_indexer += 1
        return new_left_index, new_right_index
    if right_is_sorted and (keep == "last"):
        new_right_index = np.empty(counts, dtype=right_index.dtype)
        new_left_index = np.empty(counts, dtype=left_index.dtype)
        new_indexer = 0
        for indexer in range(left.size):
            _indexer = np.uint64(indexer)
            value = search_indices[_indexer]
            if value == -1:
                continue
            _value = np.uint64(len_arr - 1)
            _new_indexer = np.uint64(new_indexer)
            new_left_index[_new_indexer] = left_index[_indexer]
            new_right_index[_new_indexer] = right_index[_value]
            new_indexer += 1
        return new_left_index, new_right_index
    if keep == "first":
        new_left_index = np.empty(counts, dtype=left_index.dtype)
        new_right_index = np.empty(counts, dtype=right_index.dtype)
        new_indexer = 0
        for num in range(search_indices.size):
            _num = np.uint64(num)
            start = search_indices[_num]
            if start == -1:
                continue
            _start = np.uint64(start)
            minimum = right_index[_start]
            for indexer in range(start, len_arr):
                _indexer = np.uint64(indexer)
                value = right_index[_indexer]
                if value < minimum:
                    minimum = value
            _new_indexer = np.uint64(new_indexer)
            new_left_index[_new_indexer] = left_index[_num]
            new_right_index[_new_indexer] = minimum
            new_indexer += 1
        return new_left_index, new_right_index
    if keep == "last":
        new_left_index = np.empty(counts, dtype=left_index.dtype)
        new_right_index = np.empty(counts, dtype=right_index.dtype)
        new_indexer = 0
        for num in range(search_indices.size):
            _num = np.uint64(num)
            start = search_indices[_num]
            if start == -1:
                continue
            _start = np.uint64(start)
            maximum = right_index[_start]
            for indexer in range(start, len_arr):
                _indexer = np.uint64(indexer)
                value = right_index[_indexer]
                if value > maximum:
                    maximum = value
            _new_indexer = np.uint64(new_indexer)
            new_left_index[_new_indexer] = left_index[_num]
            new_right_index[_new_indexer] = maximum
            new_indexer += 1
        return new_left_index, new_right_index
    new_left_index = np.empty(total, dtype=left_index.dtype)
    new_right_index = np.empty(total, dtype=right_index.dtype)
    new_indexer = 0
    for num in range(search_indices.size):
        _num = np.uint64(num)
        start = search_indices[_num]
        if start == -1:
            continue
        l_index = left_index[_num]
        for indexer in range(start, len_arr):
            _indexer = np.uint64(indexer)
            value = right_index[_indexer]
            _new_indexer = np.uint64(new_indexer)
            new_left_index[_new_indexer] = l_index
            new_right_index[_new_indexer] = value
            new_indexer += 1
    return new_left_index, new_right_index


@njit(parallel=True, cache=True)
def _numba_greater_than_indices(
    left: np.ndarray,
    right: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    strict: bool,
    right_is_sorted: bool,
    keep: str,
) -> tuple:
    """
    Use binary search to get indices where left
    is greater than or equal to right.

    If strict is True, then only indices
    where `left` is greater than
    (but not equal to) `right` are returned.

    A tuple of integer indexes
    for left and right is returned.


    Args:
        left: left array.
        right: right array. Should be sorted.
        Find positions within this array
        where values from the left should be inserted.
        left_index: Index of left array. Required to reconstruct DataFrame.
        right_index: Index of right array. Required to reconstruct DataFrame.
        strict: True if '>', else False.
        right_is_sorted: True if right array was already sorted.
        keep: Determines if all rows are returned ('all'),
            or the first and last matches.

    Returns:
        A tuple of arrays.
    """
    search_indices = np.empty(left.size, dtype=np.intp)
    total = 0  # relevant when keep == 'all'
    counts = 0  # relevant when keep in {'first','last'}
    for indexer in prange(left.size):
        _indexer = np.uint64(indexer)
        value = left[_indexer]
        outcome = _numba_greater_than(arr=right, value=value, strict=strict)
        search_indices[_indexer] = outcome
        if outcome == -1:
            continue
        counts += 1
        total += outcome
    if counts == 0:
        return None, None
    if right_is_sorted and (keep == "first"):
        new_right_index = np.empty(counts, dtype=right_index.dtype)
        new_left_index = np.empty(counts, dtype=left_index.dtype)
        new_indexer = 0
        for indexer in range(left.size):
            _indexer = np.uint64(indexer)
            value = search_indices[_indexer]
            if value == -1:
                continue
            _new_indexer = np.uint64(new_indexer)
            _value = np.uint8(0)
            new_left_index[_new_indexer] = left_index[_indexer]
            r_val = right_index[_value]
            new_right_index[_new_indexer] = r_val
            new_indexer += 1
        return new_left_index, new_right_index
    if right_is_sorted and (keep == "last"):
        new_right_index = np.empty(counts, dtype=right_index.dtype)
        new_left_index = np.empty(counts, dtype=left_index.dtype)
        new_indexer = 0
        for indexer in range(left.size):
            _indexer = np.uint64(indexer)
            value = search_indices[_indexer]
            if value == -1:
                continue
            _new_indexer = np.uint64(new_indexer)
            _value = np.uint64(value - 1)
            new_left_index[_new_indexer] = left_index[_indexer]
            new_right_index[_new_indexer] = right_index[_value]
            new_indexer += 1
        return new_left_index, new_right_index
    if keep == "first":
        new_left_index = np.empty(counts, dtype=left_index.dtype)
        new_right_index = np.empty(counts, dtype=right_index.dtype)
        new_indexer = 0
        for num in range(search_indices.size):
            _num = np.uint64(num)
            start = search_indices[_num]
            if start == -1:
                continue
            minimum = right_index[np.uint64(start - 1)]
            for indexer in range(start):
                _indexer = np.uint64(indexer)
                value = right_index[_indexer]
                if value < minimum:
                    minimum = value
            _new_indexer = np.uint64(new_indexer)
            new_left_index[_new_indexer] = left_index[_num]
            new_right_index[_new_indexer] = minimum
            new_indexer += 1
        return new_left_index, new_right_index
    if keep == "last":
        new_left_index = np.empty(counts, dtype=left_index.dtype)
        new_right_index = np.empty(counts, dtype=right_index.dtype)
        new_indexer = 0
        for num in range(search_indices.size):
            _num = np.uint64(num)
            start = search_indices[_num]
            if start == -1:
                continue
            maximum = right_index[np.uint64(start - 1)]
            for indexer in range(start):
                _indexer = np.uint64(indexer)
                value = right_index[_indexer]
                if value > maximum:
                    maximum = value
            _new_indexer = np.uint64(new_indexer)
            new_left_index[_new_indexer] = left_index[_num]
            new_right_index[_new_indexer] = maximum
            new_indexer += 1
        return new_left_index, new_right_index
    new_left_index = np.empty(total, dtype=left_index.dtype)
    new_right_index = np.empty(total, dtype=right_index.dtype)
    new_indexer = 0
    for num in range(search_indices.size):
        _num = np.uint64(num)
        start = search_indices[_num]
        if start == -1:
            continue
        l_index = left_index[_num]
        for indexer in range(start):
            _indexer = np.uint64(indexer)
            value = right_index[_indexer]
            _new_indexer = np.uint64(new_indexer)
            new_left_index[_new_indexer] = l_index
            new_right_index[_new_indexer] = value
            new_indexer += 1
    return new_left_index, new_right_index


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
    if strict and (value == _value):
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
    if strict and (value == _value):
        min_idx = _numba_less_than(arr=arr, value=value)
    # check again
    if min_idx == 0:
        return -1
    return min_idx


def _numba_single_non_equi_join(
    left: pd.Series, right: pd.Series, op: str, keep: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return matching indices for single non-equi join."""
    outcome = _null_checks_cond_join(left=left, right=right)
    if not outcome:
        return None
    left, right, left_index, right_index, right_is_sorted, _ = outcome
    left, right = _convert_to_numpy(left=left, right=right)
    left_index, right_index = _convert_to_numpy(
        left=left_index, right=right_index
    )
    if op in less_than_join_types:
        result = _numba_less_than_indices(
            left=left,
            left_index=left_index,
            right_index=right_index,
            right=right,
            strict=op == "<",
            keep=keep,
            right_is_sorted=right_is_sorted,
        )
    else:
        result = _numba_greater_than_indices(
            left=left,
            left_index=left_index,
            right_index=right_index,
            right=right,
            strict=op == ">",
            keep=keep,
            right_is_sorted=right_is_sorted,
        )
    result = None if result[0] is None else result
    return result


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
    left_column, right_column, _ = gt_lt[0]
    # sorting the first column assists
    # in keeping the regions generated ordered
    # at least on the first region
    # this is essential for iterating
    # to get the matching indices
    if not df[left_column].is_monotonic_increasing:
        df = df.sort_values(left_column)
        df.index = range(len(df))
    if not right[right_column].is_monotonic_increasing:
        right = right.sort_values(right_column)
        # original_index and right_is_sorted
        # is relevant where keep in {'first','last'}
        original_index = right.index._values
        right.index = range(len(right))
        right_is_sorted = False
    else:
        original_index = right.index._values
        right_is_sorted = True
    arrays = []
    for left_on, right_on, op in gt_lt:
        left_array = df[left_on]
        right_array = right[right_on]
        outcome = _null_checks_cond_join(left=left_array, right=right_array)
        if outcome is None:
            return {
                "df": df,
                "right": right,
                "left_index": np.array([], dtype=np.intp),
                "right_index": np.array([], dtype=np.intp),
            }
        (
            left_array,
            right_array,
            left_index,
            right_index,
            *_,
        ) = outcome
        left_array, right_array = _convert_to_numpy(
            left=left_array, right=right_array
        )
        left_index, right_index = _convert_to_numpy(
            left=left_index, right=right_index
        )
        strict = op in {"<", ">"}
        op = op in less_than_join_types  # 1 or 0
        combo = (left_array, right_array, left_index, right_index, op, strict)
        # print('la', left_array)
        # print('ra', right_array)
        arrays.append(combo)
    shape = (len(df), len(gt_lt))
    left_regions = np.empty(shape=shape, dtype=np.intp)
    shape = (len(right), len(gt_lt))
    right_regions = np.empty(shape=shape, dtype=np.intp)
    indices = _get_indices_non_equi_joins(
        left_regions=left_regions,
        right_regions=right_regions,
        original_index=original_index,
        right_is_sorted=right_is_sorted,
        keep=keep,
        tuples=tuple(arrays),
    )
    return indices
    if indices[0] is None:
        left_index = np.array([], dtype=np.intp)
        right_index = np.array([], dtype=np.intp)
    else:
        left_index, right_index = indices
    return {
        "df": df,
        "right": right,
        "left_index": left_index,
        "right_index": right_index,
    }
    # if outcome[0] is not None:
    #     return pd.concat(
    #         [
    #             df.iloc[outcome[0]].reset_index(drop=True),
    #             right.iloc[outcome[1]].reset_index(drop=True),
    #         ],
    #         axis=1,
    #     )
    return 1  # outcome, df, right


# @njit
def _get_indices_non_equi_joins(
    left_regions: np.ndarray,
    right_regions: np.ndarray,
    original_index: np.ndarray,
    right_is_sorted: bool,
    keep: str,
    tuples: tuple,
):
    len_right_regions = len(right_regions)
    len_left_regions = len(left_regions)
    len_tuples = len(tuples)
    # use this to possibly trim the left and right regions
    l_counts = np.zeros(len_left_regions, dtype=np.intp)
    r_counts = np.zeros(len_right_regions, dtype=np.intp)
    l_booleans = np.zeros(len_left_regions, dtype=np.bool_)
    r_booleans = np.zeros(len_right_regions, dtype=np.bool_)
    # do we need to trim the regions
    # to only matched indexed positions?
    trim_left_region = False
    trim_right_region = False
    for position, entry in enumerate(tuples):
        # it is essential to keep track of positions
        # via the left and right index;
        # this ensures that the final regions
        # are properly aligned
        left, right, left_index, right_index, op, strict = entry
        len_left_arr = left.size
        len_right_arr = right.size
        boolean = len_right_arr < len_right_regions
        trim_right_region |= boolean
        boolean = len_left_arr < len_left_regions
        trim_left_region |= boolean
        r_region = np.zeros(len_right_arr, dtype=np.int8)
        counts = 0
        # step 1 - get the positions of left in right
        # op == True if less_than_join_types; else False
        # strict == True if in {'>','<'}; else False
        search_indices = np.empty(len_left_arr, dtype=np.intp)
        if op:
            for indexer in prange(len_left_arr):
                _indexer = np.uint64(indexer)
                value = left[_indexer]
                outcome = _numba_less_than(
                    arr=right, value=value, strict=strict
                )
                search_indices[_indexer] = outcome
                if outcome == -1:
                    trim_left_region = True
                    continue
                counts += 1
                r_region[np.uint64(outcome)] = 1
        else:
            for indexer in prange(len_left_arr):
                _indexer = np.uint64(indexer)
                value = left[_indexer]
                outcome = _numba_greater_than(
                    arr=right, value=value, strict=strict
                )
                if outcome == -1:
                    search_indices[_indexer] = -1
                    trim_left_region = True
                    continue
                counts += 1
                # since this is a greater than comparision
                # the order has to be descending order
                outcome = len_right_arr - outcome
                search_indices[_indexer] = outcome
                r_region[np.uint64(outcome)] = 1
            right_index = right_index[::-1]
        if counts == 0:
            return None, None
        # next step - build a cumulative ordered region
        # 0, 0, 1, 1, 2, 2, 2, etc
        # and update the left and right regions
        # based on left and right index
        # again, alignment should be done
        # on the left and right indexes to ensure correctness
        r_cum_region = np.empty(len_right_arr, dtype=np.intp)
        value = -1
        for num in range(len_right_arr):
            _num = np.uint64(num)
            current_value = r_region[_num]
            current_value += value
            r_cum_region[_num] = current_value
            boolean = current_value == -1
            if boolean:
                trim_right_region = True
                continue
            # ensure alignment on right_index
            # and the position in the join tuple
            # indexing into the right region
            # should be based on the above two only
            _position = np.uint64(position)
            _num = right_index[_num]
            _num = np.uint64(_num)
            r_counts[_num] += 1
            boolean = r_counts[_num] == len_tuples
            r_booleans[_num] = boolean
            right_regions[_num, _position] = current_value
            value = current_value
        for num in prange(len_left_arr):
            # grab position of left in right
            # before grabbing position in left_index
            _num = np.uint64(num)
            # this is used to get the ordered region
            # from the right ordered region
            value = search_indices[_num]
            # ensure alignment on the left_index
            # and the position in the join tuple
            # indexing into the left region
            # should be based on the above two only
            _num = left_index[_num]
            _num = np.uint64(_num)
            boolean = value == -1
            if boolean:
                trim_left_region = True
                continue
            l_counts[_num] += 1
            boolean = l_counts[_num] == len(tuples)
            l_booleans[_num] = boolean
            value = r_cum_region[np.uint64(value)]
            left_regions[_num, _position] = value
    left_index = np.arange(len_left_regions, dtype=np.intp)
    right_index = np.arange(len_right_regions, dtype=np.intp)
    print("l", left_regions)
    print("r", right_regions)
    print("lb", l_booleans)
    print("rb", r_booleans)
    if trim_left_region:
        left_regions = left_regions[l_booleans]
        left_index = left_index[l_booleans]
    if trim_right_region:
        right_regions = right_regions[r_booleans]
        right_index = right_index[r_booleans]
        original_index = original_index[r_booleans]
    print("lt", left_regions)
    print("rt", right_regions)
    print("li", left_index)
    print("ri", right_index)
    if len(tuples) == 2:
        return _numba_dual_non_equi_join(
            left_regions=left_regions,
            right_regions=right_regions,
            left_index=left_index,
            right_index=right_index,
            original_index=original_index,
            keep=keep,
            right_is_sorted=right_is_sorted,
        )
    return None, None


# @njit
def _numba_dual_join_monotonic_increasing_decreasing_keep_first(
    left_regions: np.ndarray,
    right_regions: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    original_index: np.ndarray,
    right_is_sorted: bool,
):
    """
    Get indices where the first right region is monotonic increasing,
    while the second right region is monotonic decreasing.
    """
    left_region1 = left_regions[:, 0]
    left_region2 = left_regions[:, 1]
    right_region1 = right_regions[:, 0]
    right_region2 = right_regions[:, 1]
    len_left_region = left_region1.size
    len_right_region = right_region1.size
    starts = np.empty(len_left_region, dtype=np.intp)
    ends = np.empty(len_left_region, dtype=np.intp)
    booleans = np.zeros(len_left_region, dtype=np.bool_)
    total_length = 0
    right_region2 = right_region2[::-1]
    for indexer in prange(len_left_region):
        indexer = np.uint64(indexer)
        value1 = left_region1[indexer]
        value1 = _numba_less_than(arr=right_region1, value=value1)
        if value1 == -1:
            continue
        starts[indexer] = value1
        value2 = left_region2[indexer]
        value2 = _numba_less_than(arr=right_region2, value=value2)
        if value2 == -1:
            continue
        value2 = len_right_region - value2
        ends[indexer] = value2
        boolean = value1 < value2
        if not boolean:
            continue
        booleans[indexer] = True
        total_length += 1
    if total_length == 0:
        return None, None
    left_indices = np.empty(total_length, dtype=left_index.dtype)
    right_indices = np.empty(total_length, dtype=right_index.dtype)
    if right_is_sorted:
        n = 0
        for indexer in range(len_left_region):
            _n = np.uint64(n)
            indexer = np.uint64(indexer)
            if not booleans[indexer]:
                continue
            start = starts[indexer]
            l_index = left_index[indexer]
            r_index = right_index[np.uint64(start)]
            _n = np.uint64(n)
            left_indices[_n] = l_index
            right_indices[_n] = r_index
            n += 1
        return left_indices, right_indices
    n = 0
    for indexer in range(len_left_region):
        indexer = np.uint64(indexer)
        if not booleans[indexer]:
            continue
        start = starts[indexer]
        end = ends[indexer]
        _start = np.uint64(start)
        first = original_index[_start]
        r_indexer = right_index[_start]
        for ind in range(start, end):
            l_ind = np.uint64(n)
            r_ind = np.uint64(ind)
            l_index = left_index[indexer]
            r_index = right_index[r_ind]
            orig_index = original_index[r_ind]
            if orig_index < first:
                r_indexer = r_index
                first = orig_index
            left_indices[l_ind] = l_index
            right_indices[l_ind] = r_indexer
        n += 1
    return left_indices, right_indices


# @njit
def _numba_dual_join_monotonic_increasing_decreasing_keep_last(
    left_regions: np.ndarray,
    right_regions: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    original_index: np.ndarray,
    right_is_sorted: bool,
):
    """
    Get indices where the first right region is monotonic increasing,
    while the second right region is monotonic decreasing.
    """
    left_region1 = left_regions[:, 0]
    left_region2 = left_regions[:, 1]
    right_region1 = right_regions[:, 0]
    right_region2 = right_regions[:, 1]
    len_left_region = left_region1.size
    len_right_region = right_region1.size
    starts = np.empty(len_left_region, dtype=np.intp)
    ends = np.empty(len_left_region, dtype=np.intp)
    booleans = np.zeros(len_left_region, dtype=np.bool_)
    total_length = 0
    right_region2 = right_region2[::-1]
    for indexer in prange(len_left_region):
        indexer = np.uint64(indexer)
        value1 = left_region1[indexer]
        value1 = _numba_less_than(arr=right_region1, value=value1)
        if value1 == -1:
            continue
        starts[indexer] = value1
        value2 = left_region2[indexer]
        value2 = _numba_less_than(arr=right_region2, value=value2)
        if value2 == -1:
            continue
        value2 = len_right_region - value2
        boolean = value1 < value2
        if not boolean:
            continue
        ends[indexer] = value2
        booleans[indexer] = True
        total_length += 1
    if total_length == 0:
        return None, None
    left_indices = np.empty(total_length, dtype=np.intp)
    right_indices = np.empty(total_length, dtype=np.intp)
    if right_is_sorted:
        n = 0
        for indexer in range(len_left_region):
            indexer = np.uint64(indexer)
            if not booleans[indexer]:
                continue
            end = ends[indexer]
            end -= 1
            l_index = left_index[indexer]
            r_index = right_index[np.uint64(end)]
            _n = np.uint64(n)
            left_indices[_n] = l_index
            right_indices[_n] = r_index
            n += 1
        return left_indices, right_indices
    n = 0
    for indexer in range(len_left_region):
        indexer = np.uint64(indexer)
        if not booleans[indexer]:
            continue
        start = starts[indexer]
        end = ends[indexer]
        _start = np.uint64(start)
        last = original_index[_start]
        r_indexer = right_index[_start]
        l_ind = np.uint64(n)
        for ind in range(start, end):
            r_ind = np.uint64(ind)
            l_index = left_index[indexer]
            r_index = right_index[r_ind]
            orig_index = original_index[r_ind]
            if orig_index > last:
                r_indexer = r_index
                last = orig_index
            left_indices[l_ind] = l_index
            right_indices[l_ind] = r_indexer
        n += 1
    return left_indices, right_indices


# @njit
def _numba_dual_join_monotonic_increasing_decreasing(
    left_regions: np.ndarray,
    right_regions: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
):
    """
    Get indices where the first right region is monotonic increasing,
    while the second right region is monotonic decreasing.
    """
    left_region1 = left_regions[:, 0]
    left_region2 = left_regions[:, 1]
    right_region1 = right_regions[:, 0]
    right_region2 = right_regions[:, 1]
    len_left_region = left_region1.size
    len_right_region = right_region1.size
    starts = np.empty(len_left_region, dtype=np.intp)
    ends = np.empty(len_left_region, dtype=np.intp)
    booleans = np.zeros(len_left_region, dtype=np.bool_)
    total_length = 0
    right_region2 = right_region2[::-1]
    for indexer in prange(len_left_region):
        indexer = np.uint64(indexer)
        value1 = left_region1[indexer]
        value1 = _numba_less_than(arr=right_region1, value=value1)
        if value1 == -1:
            continue
        starts[indexer] = value1
        value2 = left_region2[indexer]
        value2 = _numba_less_than(arr=right_region2, value=value2)
        if value2 == -1:
            continue
        value2 = len_right_region - value2
        ends[indexer] = value2
        boolean = value1 < value2
        if not boolean:
            continue
        booleans[indexer] = True
        diff = value2 - value1
        total_length += diff
    if total_length == 0:
        return None, None
    left_indices = np.empty(total_length, dtype=left_index.dtype)
    right_indices = np.empty(total_length, dtype=right_index.dtype)
    n = 0
    for indexer in range(len_left_region):
        indexer = np.uint64(indexer)
        if not booleans[indexer]:
            continue
        start = starts[indexer]
        end = ends[indexer]
        for ind in range(start, end):
            l_ind = np.uint64(n)
            r_ind = np.uint64(ind)
            l_index = left_index[indexer]
            left_indices[l_ind] = l_index
            right_indices[l_ind] = right_index[r_ind]
            n += 1
    return left_indices, right_indices


# @njit
def _numba_dual_join_monotonic_decreasing_increasing(
    left_regions: np.ndarray,
    right_regions: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
):
    """
    Get indices where the first right region is monotonic decreasing,
    while the second right region is monotonic increasing.
    """
    left_region1 = left_regions[:, 0]
    left_region2 = left_regions[:, 1]
    right_region1 = right_regions[:, 0]
    right_region2 = right_regions[:, 1]
    len_left_region = left_region1.size
    len_right_region = right_region1.size
    starts = np.empty(len_left_region, dtype=np.intp)
    ends = np.empty(len_left_region, dtype=np.intp)
    booleans = np.zeros(len_left_region, dtype=np.bool_)
    total_length = 0
    right_region1 = right_region1[::-1]
    for indexer in prange(len_left_region):
        indexer = np.uint64(indexer)
        value1 = left_region1[indexer]
        value1 = _numba_less_than(arr=right_region1, value=value1)
        if value1 == -1:
            continue
        value1 = len_right_region - value1
        ends[indexer] = value1
        value2 = left_region2[indexer]
        value2 = _numba_less_than(arr=right_region2, value=value2)
        if value2 == -1:
            continue
        starts[indexer] = value2
        boolean = value1 > value2
        if not boolean:
            continue
        booleans[indexer] = True
        total_length += value1 - value2
    if total_length == 0:
        return None, None
    left_indices = np.empty(total_length, dtype=left_index.dtype)
    right_indices = np.empty(total_length, dtype=right_index.dtype)
    n = 0
    for indexer in range(len_left_region):
        indexer = np.uint64(indexer)
        if not booleans[indexer]:
            continue
        start = starts[indexer]
        end = ends[indexer]
        for ind in range(start, end):
            l_ind = np.uint64(n)
            r_ind = np.uint64(ind)
            l_index = left_index[indexer]
            left_indices[l_ind] = l_index
            right_indices[l_ind] = right_index[r_ind]
            n += 1
    return left_indices, right_indices


# @njit
def _numba_dual_join_monotonic_decreasing_increasing_keep_first(
    left_regions: np.ndarray,
    right_regions: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    original_index: np.ndarray,
    right_is_sorted: bool,
):
    """
    Get indices where the first right region is monotonic decreasing,
    while the second right region is monotonic increasing.
    """
    left_region1 = left_regions[:, 0]
    left_region2 = left_regions[:, 1]
    right_region1 = right_regions[:, 0]
    right_region2 = right_regions[:, 1]
    len_left_region = left_region1.size
    len_right_region = right_region1.size
    starts = np.empty(len_left_region, dtype=np.intp)
    ends = np.empty(len_left_region, dtype=np.intp)
    booleans = np.zeros(len_left_region, dtype=np.bool_)
    total_length = 0
    right_region1 = right_region1[::-1]
    for indexer in prange(len_left_region):
        indexer = np.uint64(indexer)
        value1 = left_region1[indexer]
        value1 = _numba_less_than(arr=right_region1, value=value1)
        if value1 == -1:
            continue
        value1 = len_right_region - value1
        ends[indexer] = value1
        value2 = left_region2[indexer]
        value2 = _numba_less_than(arr=right_region2, value=value2)
        if value2 == -1:
            continue
        starts[indexer] = value2
        boolean = value1 > value2
        if not boolean:
            continue
        booleans[indexer] = True
        total_length += 1
    if total_length == 0:
        return None, None
    left_indices = np.empty(total_length, dtype=left_index.dtype)
    right_indices = np.empty(total_length, dtype=right_index.dtype)

    if right_is_sorted:
        n = 0
        for indexer in range(len_left_region):
            indexer = np.uint64(indexer)
            if not booleans[indexer]:
                continue
            start = starts[indexer]
            l_index = left_index[indexer]
            r_index = right_index[np.uint64(start)]
            _n = np.uint64(n)
            left_indices[_n] = l_index
            right_indices[_n] = r_index
            n += 1
        return left_indices, right_indices

    n = 0
    for indexer in range(len_left_region):
        indexer = np.uint64(indexer)
        if not booleans[indexer]:
            continue
        start = starts[indexer]
        end = ends[indexer]
        _start = np.uint64(start)
        first = original_index[_start]
        r_indexer = right_index[_start]
        for ind in range(start, end):
            l_ind = np.uint64(n)
            r_ind = np.uint64(ind)
            l_index = left_index[indexer]
            r_index = right_index[r_ind]
            orig_index = original_index[r_ind]
            if orig_index < first:
                r_indexer = r_index
                first = orig_index
            left_indices[l_ind] = l_index
            right_indices[l_ind] = r_indexer
        n += 1
    return left_indices, right_indices


# @njit
def _numba_dual_join_monotonic_decreasing_increasing_keep_last(
    left_regions: np.ndarray,
    right_regions: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    original_index: np.ndarray,
    right_is_sorted: bool,
):
    """
    Get indices where the first right region is monotonic decreasing,
    while the second right region is monotonic increasing.
    """
    left_region1 = left_regions[:, 0]
    left_region2 = left_regions[:, 1]
    right_region1 = right_regions[:, 0]
    right_region2 = right_regions[:, 1]
    len_left_region = left_region1.size
    len_right_region = right_region1.size
    starts = np.empty(len_left_region, dtype=np.intp)
    ends = np.empty(len_left_region, dtype=np.intp)
    booleans = np.zeros(len_left_region, dtype=np.bool_)
    total_length = 0
    right_region1 = right_region1[::-1]
    for indexer in prange(len_left_region):
        indexer = np.uint64(indexer)
        value1 = left_region1[indexer]
        value1 = _numba_less_than(arr=right_region1, value=value1)
        if value1 == -1:
            continue
        value1 = len_right_region - value1
        ends[indexer] = value1
        value2 = left_region2[indexer]
        value2 = _numba_less_than(arr=right_region2, value=value2)
        if value2 == -1:
            continue
        starts[indexer] = value2
        boolean = value1 > value2
        if not boolean:
            continue
        booleans[indexer] = True
        total_length += 1
    if total_length == 0:
        return None, None
    left_indices = np.empty(total_length, dtype=np.intp)
    right_indices = np.empty(total_length, dtype=np.intp)
    if right_is_sorted:
        n = 0
        for indexer in range(len_left_region):
            indexer = np.uint64(indexer)
            if not booleans[indexer]:
                continue
            end = ends[indexer]
            end -= 1
            l_index = left_index[indexer]
            r_index = right_index[np.uint64(end)]
            _n = np.uint64(n)
            left_indices[_n] = l_index
            right_indices[_n] = r_index
            n += 1
        return left_indices, right_indices
    n = 0
    for indexer in range(len_left_region):
        indexer = np.uint64(indexer)
        if not booleans[indexer]:
            continue
        start = starts[indexer]
        end = ends[indexer]
        _start = np.uint64(start)
        last = original_index[_start]
        r_indexer = right_index[_start]
        for ind in range(start, end):
            l_ind = np.uint64(n)
            r_ind = np.uint64(ind)
            l_index = left_index[indexer]
            r_index = right_index[r_ind]
            orig_index = original_index[r_ind]
            if orig_index > last:
                r_indexer = r_index
                last = orig_index
            left_indices[l_ind] = l_index
            right_indices[l_ind] = r_indexer
        n += 1
    return left_indices, right_indices


# @njit
def _numba_dual_join_monotonic_increasing_keep_first(
    left_regions: np.ndarray,
    right_regions: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    original_index: np.ndarray,
):
    """
    Get indices where the first right region is monotonic increasing
    """
    left_region1 = left_regions[:, 0]
    left_region2 = left_regions[:, 1]
    right_region1 = right_regions[:, 0]
    right_region2 = right_regions[:, 1]
    len_left_region = left_region1.size
    len_right_region = right_region1.size
    load_factor = 1_000
    length = load_factor * 2
    width = ceil(right_index.size / load_factor)
    sorted_array = np.empty((width, length), dtype=right_region2.dtype)
    right_index_array = np.empty((width, length), dtype=right_index.dtype)
    maxes = np.empty(width, dtype=right_region2.dtype)
    lengths = np.empty(width, dtype=np.intp)
    maxes_counter = 0
    starts = np.empty(len_left_region, dtype=np.intp)
    for indexer in prange(len_left_region):
        indexer = np.uint64(indexer)
        l_value = left_region1[indexer]
        start = _numba_less_than(arr=right_region1, value=l_value)
        starts[indexer] = start
    left_indices = np.empty(len_left_region, dtype=left_index.dtype)
    right_indices = np.empty(len_left_region, dtype=right_index.dtype)
    total = 0
    end = len_right_region
    for indexer in range(len_left_region - 1, -1, -1):
        indexer = np.uint64(indexer)
        start = starts[indexer]
        if start == -1:
            continue
        for pos_ in range(start, end):
            pos = np.uint64(pos_)
            r_value = right_region2[pos]
            r_index = pos_
            if maxes_counter == 0:
                sorted_array[0, 0] = r_value
                right_index_array[0, 0] = r_index
                lengths[0] = 1
                maxes_counter = 1
                maxes[0] = r_value
                posn = 0
            else:
                (
                    sorted_array,
                    right_index_array,
                    maxes,
                    lengths,
                    maxes_counter,
                    posn,
                ) = _numba_build_sorted_array(
                    sorted_array=sorted_array,
                    right_index_array=right_index_array,
                    maxes=maxes,
                    lengths=lengths,
                    maxes_counter=maxes_counter,
                    insert_value=r_value,
                    insert_index=r_index,
                )

            (
                sorted_array,
                right_index_array,
                maxes,
                lengths,
                posn,
                maxes_counter,
            ) = _expand_sorted_array(
                sorted_array=sorted_array,
                right_index_array=right_index_array,
                maxes=maxes,
                lengths=lengths,
                posn=posn,
                limit=length,
                load_factor=load_factor,
                maxes_counter=maxes_counter,
            )
        l_value = left_region2[indexer]
        maxes_trimmed = maxes[:maxes_counter]
        maxes_posn = _numba_less_than(arr=maxes_trimmed, value=l_value)
        if maxes_posn == -1:
            continue
        posn_ = np.uint64(maxes_posn)
        len_ = lengths[posn_]
        arr_to_search = sorted_array[posn_, :len_]
        _posn = _numba_less_than(arr=arr_to_search, value=l_value)
        _maxes_posn = np.uint64(maxes_posn)
        _first = right_index_array[_maxes_posn, np.uint64(_posn)]
        _first = np.uint64(_first)
        first_orig = original_index[_first]
        first_index = right_index[_first]
        for ind in range(_posn, len_):
            _ind = np.uint64(ind)
            r_pos = right_index_array[_maxes_posn, _ind]
            r_pos = np.uint64(r_pos)
            orig = original_index[r_pos]
            rindex = right_index[r_pos]
            if orig < first_orig:
                first_orig = orig
                first_index = rindex

        for ind in range(maxes_posn + 1, maxes_counter):
            _ind = np.uint64(ind)
            _len = lengths[_ind]
            for num in range(_len):
                _num = np.uint64(num)
                r_pos = right_index_array[_ind, _num]
                r_pos = np.uint64(r_pos)
                orig = original_index[r_pos]
                rindex = right_index[r_pos]
                if orig < first_orig:
                    first_orig = orig
                    first_index = rindex
        left_indices[indexer] = left_index[indexer]
        right_indices[indexer] = first_index
        end = start
        total += 1
    return left_indices[:total], right_indices[:total]


# @njit
def _numba_dual_join_monotonic_increasing_keep_last(
    left_regions: np.ndarray,
    right_regions: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    original_index: np.ndarray,
):
    """
    Get indices where the first right region is monotonic increasing
    """
    left_region1 = left_regions[:, 0]
    left_region2 = left_regions[:, 1]
    right_region1 = right_regions[:, 0]
    right_region2 = right_regions[:, 1]
    len_left_region = left_region1.size
    len_right_region = right_region1.size
    load_factor = 1_000
    length = load_factor * 2
    width = ceil(right_index.size / load_factor)
    sorted_array = np.empty((width, length), dtype=right_region2.dtype)
    right_index_array = np.empty((width, length), dtype=right_index.dtype)
    maxes = np.empty(width, dtype=right_region2.dtype)
    lengths = np.empty(width, dtype=np.intp)
    maxes_counter = 0
    starts = np.empty(len_left_region, dtype=np.intp)
    for indexer in prange(len_left_region):
        indexer = np.uint64(indexer)
        l_value = left_region1[indexer]
        start = _numba_less_than(arr=right_region1, value=l_value)
        starts[indexer] = start
    left_indices = np.empty(len_left_region, dtype=left_index.dtype)
    right_indices = np.empty(len_left_region, dtype=right_index.dtype)
    total = 0
    end = len_right_region
    for indexer in range(len_left_region - 1, -1, -1):
        indexer = np.uint64(indexer)
        start = starts[indexer]
        if start == -1:
            continue
        for pos_ in range(start, end):
            pos = np.uint64(pos_)
            r_value = right_region2[pos]
            r_index = pos_
            if maxes_counter == 0:
                sorted_array[0, 0] = r_value
                right_index_array[0, 0] = r_index
                lengths[0] = 1
                maxes_counter = 1
                maxes[0] = r_value
                posn = 0
            else:
                (
                    sorted_array,
                    right_index_array,
                    maxes,
                    lengths,
                    maxes_counter,
                    posn,
                ) = _numba_build_sorted_array(
                    sorted_array=sorted_array,
                    right_index_array=right_index_array,
                    maxes=maxes,
                    lengths=lengths,
                    maxes_counter=maxes_counter,
                    insert_value=r_value,
                    insert_index=r_index,
                )

            (
                sorted_array,
                right_index_array,
                maxes,
                lengths,
                posn,
                maxes_counter,
            ) = _expand_sorted_array(
                sorted_array=sorted_array,
                right_index_array=right_index_array,
                maxes=maxes,
                lengths=lengths,
                posn=posn,
                limit=length,
                load_factor=load_factor,
                maxes_counter=maxes_counter,
            )
        l_value = left_region2[indexer]
        maxes_trimmed = maxes[:maxes_counter]
        maxes_posn = _numba_less_than(arr=maxes_trimmed, value=l_value)
        if maxes_posn == -1:
            continue
        posn_ = np.uint64(maxes_posn)
        len_ = lengths[posn_]
        arr_to_search = sorted_array[posn_, :len_]
        _posn = _numba_less_than(arr=arr_to_search, value=l_value)
        _maxes_posn = np.uint64(maxes_posn)
        _last = right_index_array[_maxes_posn, np.uint64(_posn)]
        _last = np.uint64(_last)
        last_orig = original_index[_last]
        last_index = right_index[_last]
        for ind in range(_posn, len_):
            _ind = np.uint64(ind)
            r_pos = right_index_array[_maxes_posn, _ind]
            r_pos = np.uint64(r_pos)
            orig = original_index[r_pos]
            rindex = right_index[r_pos]
            if orig > last_orig:
                last_orig = orig
                last_index = rindex

        for ind in range(maxes_posn + 1, maxes_counter):
            _ind = np.uint64(ind)
            _len = lengths[_ind]
            for num in range(_len):
                _num = np.uint64(num)
                r_pos = right_index_array[_ind, _num]
                r_pos = np.uint64(r_pos)
                orig = original_index[r_pos]
                rindex = right_index[r_pos]
                if orig > last_orig:
                    last_orig = orig
                    last_index = rindex
        left_indices[indexer] = left_index[indexer]
        right_indices[indexer] = last_index
        end = start
        total += 1
    return left_indices[:total], right_indices[:total]


# @njit
def _numba_dual_join_monotonic_increasing(
    left_regions: np.ndarray,
    right_regions: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
):
    """
    Get indices where the first right region is monotonic increasing
    """
    left_region1 = left_regions[:, 0]
    left_region2 = left_regions[:, 1]
    right_region1 = right_regions[:, 0]
    right_region2 = right_regions[:, 1]
    len_left_region = left_region1.size
    len_right_region = right_region1.size
    # adaptation of grantjenks' sortedcontainers
    load_factor = 1_000
    length = load_factor * 2
    width = ceil(right_index.size / load_factor)
    shape = (width, length)
    sorted_array = np.empty(shape=shape, dtype=right_region2.dtype)
    right_index_array = np.empty(shape=shape, dtype=right_index.dtype)
    maxes = np.empty(width, dtype=right_region2.dtype)
    lengths = np.empty(width, dtype=np.intp)
    maxes_counter = 0
    total = 0
    starts = np.empty(len_left_region, dtype=np.intp)
    for indexer in prange(len_left_region):
        indexer = np.uint64(indexer)
        l_value = left_region1[indexer]
        start = _numba_less_than(arr=right_region1, value=l_value)
        starts[indexer] = start
        if start == -1:
            continue
        diff = len_right_region - start
        total += diff  # total possible index length
    left_indices = np.empty(total, dtype=left_index.dtype)
    right_indices = np.empty(total, dtype=right_index.dtype)
    total = 0
    begin = 0
    end = len_right_region
    for indexer in range(len_left_region - 1, -1, -1):
        indexer = np.uint64(indexer)
        start = starts[indexer]
        if start == -1:
            continue
        for pos in range(start, end):
            pos = np.uint64(pos)
            r_value = right_region2[pos]
            r_index = right_index[pos]
            if maxes_counter == 0:
                sorted_array[0, 0] = r_value
                right_index_array[0, 0] = r_index
                lengths[0] = 1
                maxes_counter = 1
                maxes[0] = r_value
                posn = 0
            else:
                (
                    sorted_array,
                    right_index_array,
                    maxes,
                    lengths,
                    maxes_counter,
                    posn,
                ) = _numba_build_sorted_array(
                    sorted_array=sorted_array,
                    right_index_array=right_index_array,
                    maxes=maxes,
                    lengths=lengths,
                    maxes_counter=maxes_counter,
                    insert_value=r_value,
                    insert_index=r_index,
                )

            (
                sorted_array,
                right_index_array,
                maxes,
                lengths,
                posn,
                maxes_counter,
            ) = _expand_sorted_array(
                sorted_array=sorted_array,
                right_index_array=right_index_array,
                maxes=maxes,
                lengths=lengths,
                posn=posn,
                limit=length,
                load_factor=load_factor,
                maxes_counter=maxes_counter,
            )
        l_value = left_region2[indexer]
        maxes_trimmed = maxes[:maxes_counter]
        maxes_posn = _numba_less_than(arr=maxes_trimmed, value=l_value)
        if maxes_posn == -1:
            continue
        l_index = left_index[indexer]
        posn_ = np.uint64(maxes_posn)
        len_ = lengths[posn_]
        arr_to_search = sorted_array[posn_, :len_]
        _posn = _numba_less_than(arr=arr_to_search, value=l_value)
        for ind in range(_posn, len_):
            _ind = np.uint64(ind)
            _begin = np.uint64(begin)
            _maxes_posn = np.uint64(maxes_posn)
            r_val = right_index_array[_maxes_posn, _ind]
            right_indices[_begin] = r_val
            left_indices[_begin] = l_index
            begin += 1
            total += 1
        for ind in range(maxes_posn + 1, maxes_counter):
            _ind = np.uint64(ind)
            _len = lengths[_ind]
            for num in range(_len):
                _begin = np.uint64(begin)
                _num = np.uint64(num)
                r_val = right_index_array[_ind, _num]
                right_indices[_begin] = r_val
                left_indices[_begin] = l_index
                begin += 1
                total += 1
        end = start
    if total == 0:
        return None, None
    return left_indices[:total], right_indices[:total]


# @njit
def _numba_dual_join_monotonic_decreasing(
    left_regions: np.ndarray,
    right_regions: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
):
    """
    Get indices where the first right region is monotonic decreasing
    """
    left_region1 = left_regions[:, 0]
    left_region2 = left_regions[:, 1]
    right_region1 = right_regions[:, 0]
    right_region2 = right_regions[:, 1]
    len_left_region = left_region1.size
    len_right_region = right_region1.size
    load_factor = 1_000
    length = load_factor * 2
    width = ceil(right_index.size / load_factor)
    sorted_array = np.empty((width, length), dtype=right_region2.dtype)
    right_index_array = np.empty((width, length), dtype=right_index.dtype)
    maxes = np.empty(width, dtype=right_region2.dtype)
    lengths = np.empty(width, dtype=np.intp)
    maxes_counter = 0
    total = 0
    ends = np.empty(len_left_region, dtype=np.intp)
    right_region1 = right_region1[::-1]
    for indexer in prange(len_left_region):
        indexer = np.uint64(indexer)
        l_value = left_region1[indexer]
        end = _numba_less_than(arr=right_region1, value=l_value)
        if end == -1:
            ends[indexer] = -1
            continue
        end = len_right_region - end
        ends[indexer] = end
        total += end
    left_indices = np.empty(total, dtype=left_index.dtype)
    right_indices = np.empty(total, dtype=right_index.dtype)
    total = 0
    start = 0
    begin = 0
    for indexer in range(len_left_region):
        indexer = np.uint64(indexer)
        end = ends[indexer]
        if end == -1:
            continue
        for pos in range(start, end):
            pos = np.uint64(pos)
            r_value = right_region2[pos]
            r_index = right_index[pos]
            if maxes_counter == 0:
                sorted_array[0, 0] = r_value
                right_index_array[0, 0] = r_index
                lengths[0] = 1
                maxes_counter = 1
                maxes[0] = r_value
                posn = 0
            else:
                (
                    sorted_array,
                    right_index_array,
                    maxes,
                    lengths,
                    maxes_counter,
                    posn,
                ) = _numba_build_sorted_array(
                    sorted_array=sorted_array,
                    right_index_array=right_index_array,
                    maxes=maxes,
                    lengths=lengths,
                    maxes_counter=maxes_counter,
                    insert_value=r_value,
                    insert_index=r_index,
                )

            (
                sorted_array,
                right_index_array,
                maxes,
                lengths,
                posn,
                maxes_counter,
            ) = _expand_sorted_array(
                sorted_array=sorted_array,
                right_index_array=right_index_array,
                maxes=maxes,
                lengths=lengths,
                posn=posn,
                limit=length,
                load_factor=load_factor,
                maxes_counter=maxes_counter,
            )

        l_value = left_region2[indexer]
        maxes_trimmed = maxes[:maxes_counter]
        maxes_posn = _numba_less_than(arr=maxes_trimmed, value=l_value)
        if maxes_posn == -1:
            continue

        l_index = left_index[indexer]
        posn_ = np.uint64(maxes_posn)
        len_ = lengths[posn_]
        arr_to_search = sorted_array[posn_, :len_]
        _posn = _numba_less_than(arr=arr_to_search, value=l_value)
        for ind in range(_posn, len_):
            _ind = np.uint64(ind)
            _begin = np.uint64(begin)
            _maxes_posn = np.uint64(maxes_posn)
            r_val = right_index_array[_maxes_posn, _ind]
            right_indices[_begin] = r_val
            left_indices[_begin] = l_index
            begin += 1
            total += 1
        for ind in range(maxes_posn + 1, maxes_counter):
            _ind = np.uint64(ind)
            _len = lengths[_ind]
            for num in range(_len):
                _begin = np.uint64(begin)
                _num = np.uint64(num)
                r_val = right_index_array[_ind, _num]
                right_indices[_begin] = r_val
                left_indices[_begin] = l_index
                begin += 1
                total += 1
        start = end
    if total == 0:
        return None, None
    return left_indices[:total], right_indices[:total]


# @njit
def _numba_dual_join_monotonic_decreasing_keep_first(
    left_regions: np.ndarray,
    right_regions: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    original_index: np.ndarray,
):
    """
    Get indices where the first right region is monotonic decreasing
    """
    left_region1 = left_regions[:, 0]
    left_region2 = left_regions[:, 1]
    right_region1 = right_regions[:, 0]
    right_region2 = right_regions[:, 1]
    len_left_region = left_region1.size
    len_right_region = right_region1.size
    load_factor = 1_000
    length = load_factor * 2
    width = ceil(right_index.size / load_factor)
    sorted_array = np.empty((width, length), dtype=right_region2.dtype)
    right_index_array = np.empty((width, length), dtype=right_index.dtype)
    maxes = np.empty(width, dtype=right_region2.dtype)
    lengths = np.empty(width, dtype=np.intp)
    maxes_counter = 0
    ends = np.empty(len_left_region, dtype=np.intp)
    right_region1 = right_region1[::-1]
    for indexer in prange(len_left_region):
        indexer = np.uint64(indexer)
        l_value = left_region1[indexer]
        end = _numba_less_than(arr=right_region1, value=l_value)
        if end == -1:
            ends[indexer] = end
            continue
        end = len_right_region - end
        ends[indexer] = end
    left_indices = np.empty(len_left_region, dtype=left_index.dtype)
    right_indices = np.empty(len_left_region, dtype=right_index.dtype)
    start = 0
    total = 0
    for indexer in range(len_left_region):
        indexer = np.uint64(indexer)
        end = ends[indexer]
        if end == -1:
            continue
        for pos_ in range(start, end):
            pos = np.uint64(pos_)
            r_value = right_region2[pos]
            r_index = pos_
            if maxes_counter == 0:
                sorted_array[0, 0] = r_value
                right_index_array[0, 0] = r_index
                lengths[0] = 1
                maxes_counter = 1
                maxes[0] = r_value
                posn = 0
            else:
                (
                    sorted_array,
                    right_index_array,
                    maxes,
                    lengths,
                    maxes_counter,
                    posn,
                ) = _numba_build_sorted_array(
                    sorted_array=sorted_array,
                    right_index_array=right_index_array,
                    maxes=maxes,
                    lengths=lengths,
                    maxes_counter=maxes_counter,
                    insert_value=r_value,
                    insert_index=r_index,
                )

            (
                sorted_array,
                right_index_array,
                maxes,
                lengths,
                posn,
                maxes_counter,
            ) = _expand_sorted_array(
                sorted_array=sorted_array,
                right_index_array=right_index_array,
                maxes=maxes,
                lengths=lengths,
                posn=posn,
                limit=length,
                load_factor=load_factor,
                maxes_counter=maxes_counter,
            )
        l_value = left_region2[indexer]
        maxes_trimmed = maxes[:maxes_counter]
        maxes_posn = _numba_less_than(arr=maxes_trimmed, value=l_value)
        if maxes_posn == -1:
            continue
        posn_ = np.uint64(maxes_posn)
        len_ = lengths[posn_]
        arr_to_search = sorted_array[posn_, :len_]
        _posn = _numba_less_than(arr=arr_to_search, value=l_value)
        _maxes_posn = np.uint64(maxes_posn)
        _first = right_index_array[_maxes_posn, np.uint64(_posn)]
        _first = np.uint64(_first)
        first_orig = original_index[_first]
        first_index = right_index[_first]
        for ind in range(_posn, len_):
            _ind = np.uint64(ind)
            r_pos = right_index_array[_maxes_posn, _ind]
            r_pos = np.uint64(r_pos)
            orig = original_index[r_pos]
            rindex = right_index[r_pos]
            if orig < first_orig:
                first_orig = orig
                first_index = rindex

        for ind in range(maxes_posn + 1, maxes_counter):
            _ind = np.uint64(ind)
            _len = lengths[_ind]
            for num in range(_len):
                _num = np.uint64(num)
                r_pos = right_index_array[_ind, _num]
                r_pos = np.uint64(r_pos)
                orig = original_index[r_pos]
                rindex = right_index[r_pos]
                if orig < first_orig:
                    first_orig = orig
                    first_index = rindex
        left_indices[indexer] = left_index[indexer]
        right_indices[indexer] = first_index
        start = end
        total += 1
    return left_indices[:total], right_indices[:total]


# @njit
def _numba_dual_join_monotonic_decreasing_keep_last(
    left_regions: np.ndarray,
    right_regions: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    original_index: np.ndarray,
):
    """
    Get indices where the first right region is monotonic decreasing
    """
    left_region1 = left_regions[:, 0]
    left_region2 = left_regions[:, 1]
    right_region1 = right_regions[:, 0]
    right_region2 = right_regions[:, 1]
    len_left_region = left_region1.size
    len_right_region = right_region1.size
    load_factor = 1_000
    length = load_factor * 2
    width = ceil(right_index.size / load_factor)
    sorted_array = np.empty((width, length), dtype=right_region2.dtype)
    right_index_array = np.empty((width, length), dtype=right_index.dtype)
    maxes = np.empty(width, dtype=right_region2.dtype)
    lengths = np.empty(width, dtype=np.intp)
    maxes_counter = 0
    ends = np.empty(len_left_region, dtype=np.intp)
    right_region1 = right_region1[::-1]
    for indexer in prange(len_left_region):
        indexer = np.uint64(indexer)
        l_value = left_region1[indexer]
        end = _numba_less_than(arr=right_region1, value=l_value)
        if end == -1:
            ends[indexer] = end
            continue
        end = len_right_region - end
        ends[indexer] = end
    left_indices = np.empty(len_left_region, dtype=left_index.dtype)
    right_indices = np.empty(len_left_region, dtype=right_index.dtype)
    start = 0
    total = 0
    for indexer in range(len_left_region):
        indexer = np.uint64(indexer)
        end = ends[indexer]
        if end == -1:
            continue
        for pos_ in range(start, end):
            pos = np.uint64(pos_)
            r_value = right_region2[pos]
            r_index = pos_
            if maxes_counter == 0:
                sorted_array[0, 0] = r_value
                right_index_array[0, 0] = r_index
                lengths[0] = 1
                maxes_counter = 1
                maxes[0] = r_value
                posn = 0
            else:
                (
                    sorted_array,
                    right_index_array,
                    maxes,
                    lengths,
                    maxes_counter,
                    posn,
                ) = _numba_build_sorted_array(
                    sorted_array=sorted_array,
                    right_index_array=right_index_array,
                    maxes=maxes,
                    lengths=lengths,
                    maxes_counter=maxes_counter,
                    insert_value=r_value,
                    insert_index=r_index,
                )

            (
                sorted_array,
                right_index_array,
                maxes,
                lengths,
                posn,
                maxes_counter,
            ) = _expand_sorted_array(
                sorted_array=sorted_array,
                right_index_array=right_index_array,
                maxes=maxes,
                lengths=lengths,
                posn=posn,
                limit=length,
                load_factor=load_factor,
                maxes_counter=maxes_counter,
            )
        l_value = left_region2[indexer]
        maxes_trimmed = maxes[:maxes_counter]
        maxes_posn = _numba_less_than(arr=maxes_trimmed, value=l_value)
        if maxes_posn == -1:
            continue
        posn_ = np.uint64(maxes_posn)
        len_ = lengths[posn_]
        arr_to_search = sorted_array[posn_, :len_]
        _posn = _numba_less_than(arr=arr_to_search, value=l_value)
        _maxes_posn = np.uint64(maxes_posn)
        _last = right_index_array[_maxes_posn, np.uint64(_posn)]
        _last = np.uint64(_last)
        last_orig = original_index[_last]
        last_index = right_index[_last]
        for ind in range(_posn, len_):
            _ind = np.uint64(ind)
            r_pos = right_index_array[_maxes_posn, _ind]
            r_pos = np.uint64(r_pos)
            orig = original_index[r_pos]
            rindex = right_index[r_pos]
            if orig > last_orig:
                last_orig = orig
                last_index = rindex

        for ind in range(maxes_posn + 1, maxes_counter):
            _ind = np.uint64(ind)
            _len = lengths[_ind]
            for num in range(_len):
                _num = np.uint64(num)
                r_pos = right_index_array[_ind, _num]
                r_pos = np.uint64(r_pos)
                orig = original_index[r_pos]
                rindex = right_index[r_pos]
                if orig > last_orig:
                    last_orig = orig
                    last_index = rindex
        left_indices[indexer] = left_index[indexer]
        right_indices[indexer] = last_index
        start = end
        total += 1
    return left_indices[:total], right_indices[:total]


# @njit
def _expand_sorted_array(
    sorted_array: np.ndarray,
    right_index_array: np.ndarray,
    maxes: np.ndarray,
    lengths: np.ndarray,
    posn: int,
    limit: int,
    load_factor: int,
    maxes_counter: int,
):
    """
    Adaptation of grantjenks' sortedcontainers.
    Expand sorted_array into a new array, if limit reached.
    """
    posn_ = np.uint64(posn)
    check = lengths[posn_] == limit
    if check:
        for ind_ in range(maxes_counter - 1, posn, -1):
            l_ind = np.uint64(ind_ + 1)
            r_ind = np.uint64(ind_)
            sorted_array[l_ind] = sorted_array[r_ind]
            right_index_array[l_ind] = right_index_array[r_ind]
            maxes[l_ind] = maxes[r_ind]
            lengths[l_ind] = lengths[r_ind]
        _posn = np.uint64(posn + 1)
        maxes[_posn] = sorted_array[posn_, -1]
        lengths[_posn] = load_factor
        arr = sorted_array[posn_, load_factor:]
        sorted_array[_posn, :load_factor] = arr
        arr = right_index_array[posn_, load_factor:]
        right_index_array[_posn, :load_factor] = arr
        lengths[posn_] = load_factor
        maxes[posn_] = sorted_array[posn_, np.uint64(load_factor - 1)]
        maxes_counter += 1
    return sorted_array, right_index_array, maxes, lengths, posn, maxes_counter


# @njit
def _numba_build_sorted_array(
    sorted_array: np.ndarray,
    right_index_array: np.ndarray,
    maxes: np.ndarray,
    lengths: np.ndarray,
    maxes_counter: int,
    insert_value: int,
    insert_index: int,
):
    """
    Adaptation of grantjenks' sortedcontainers.
    Builds and maintains a sorted array.
    """
    # get maxes with values
    maxes_trimmed = maxes[:maxes_counter]
    # get array to search in
    posn = _numba_less_than(arr=maxes_trimmed, value=insert_value)
    if posn == -1:
        posn = maxes_counter - 1
    posn_ = np.uint64(posn)
    # get the real length of that array
    len_arr = lengths[posn_]
    # grab array to search for value
    arr_to_search = sorted_array[posn_, :len_arr]
    insort_posn = _numba_less_than(arr=arr_to_search, value=insert_value)
    len_arr_ = np.uint64(len_arr)
    if insort_posn == -1:
        sorted_array[posn_, len_arr_] = insert_value
        right_index_array[posn_, len_arr_] = insert_index
    else:
        # run from back to front
        # to ensure downward shifting without slicing issues
        for ind in range(len_arr - 1, insort_posn - 1, -1):
            ind_ = np.uint64(ind)
            _ind = np.uint64(ind + 1)
            val = sorted_array[posn_, ind_]
            sorted_array[posn_, _ind] = val
            val = right_index_array[posn_, ind_]
            right_index_array[posn_, _ind] = val
        insort = np.uint64(insort_posn)
        sorted_array[posn_, insort] = insert_value
        right_index_array[posn_, insort] = insert_index

    lengths[posn_] += 1
    maxes[posn_] = sorted_array[posn_, len_arr_]
    return (
        sorted_array,
        right_index_array,
        maxes,
        lengths,
        maxes_counter,
        posn,
    )


# @njit
def _numba_dual_non_equi_join(
    left_regions: np.ndarray,
    right_regions: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    original_index: np.ndarray,
    keep: str,
    right_is_sorted: bool,
):
    """
    Get join indices for two non-equi join conditions
    """
    check1 = is_monotonic_increasing(right_regions[:, 0])
    check2 = is_monotonic_decreasing(right_regions[:, 1])
    if check1 and check2 and (keep == "all"):
        return _numba_dual_join_monotonic_increasing_decreasing(
            left_regions=left_regions,
            right_regions=right_regions,
            left_index=left_index,
            right_index=right_index,
        )
    if check1 and check2 and (keep == "first"):
        return _numba_dual_join_monotonic_increasing_decreasing_keep_first(
            left_regions=left_regions,
            right_regions=right_regions,
            left_index=left_index,
            right_index=right_index,
            original_index=original_index,
            right_is_sorted=right_is_sorted,
        )
    if check1 and check2 and (keep == "last"):
        return _numba_dual_join_monotonic_increasing_decreasing_keep_last(
            left_regions=left_regions,
            right_regions=right_regions,
            left_index=left_index,
            right_index=right_index,
            original_index=original_index,
            right_is_sorted=right_is_sorted,
        )

    check3 = is_monotonic_decreasing(right_regions[:, 0])
    check4 = is_monotonic_increasing(right_regions[:, 1])

    if check3 and check4 and (keep == "all"):
        return _numba_dual_join_monotonic_decreasing_increasing(
            left_regions=left_regions,
            right_regions=right_regions,
            left_index=left_index,
            right_index=right_index,
        )

    if check3 and check4 and (keep == "first"):
        return _numba_dual_join_monotonic_decreasing_increasing_keep_first(
            left_regions=left_regions,
            right_regions=right_regions,
            left_index=left_index,
            right_index=right_index,
            original_index=original_index,
            right_is_sorted=right_is_sorted,
        )
    if check3 and check4 and (keep == "last"):
        return _numba_dual_join_monotonic_decreasing_increasing_keep_last(
            left_regions=left_regions,
            right_regions=right_regions,
            left_index=left_index,
            right_index=right_index,
            original_index=original_index,
            right_is_sorted=right_is_sorted,
        )

    if check1 and (keep == "all"):
        return _numba_dual_join_monotonic_increasing(
            left_regions=left_regions,
            right_regions=right_regions,
            left_index=left_index,
            right_index=right_index,
        )
    if check1 and (keep == "first"):
        return _numba_dual_join_monotonic_increasing_keep_first(
            left_regions=left_regions,
            right_regions=right_regions,
            left_index=left_index,
            right_index=right_index,
            original_index=original_index,
        )
    if check1 and (keep == "last"):
        return _numba_dual_join_monotonic_increasing_keep_last(
            left_regions=left_regions,
            right_regions=right_regions,
            left_index=left_index,
            right_index=right_index,
            original_index=original_index,
        )
    if check3 and (keep == "all"):
        return _numba_dual_join_monotonic_decreasing(
            left_regions=left_regions,
            right_regions=right_regions,
            left_index=left_index,
            right_index=right_index,
        )
    if check3 and (keep == "first"):
        return _numba_dual_join_monotonic_decreasing_keep_first(
            left_regions=left_regions,
            right_regions=right_regions,
            left_index=left_index,
            right_index=right_index,
            original_index=original_index,
        )
    # check3 and (keep=='last')
    return _numba_dual_join_monotonic_decreasing_keep_last(
        left_regions=left_regions,
        right_regions=right_regions,
        left_index=left_index,
        right_index=right_index,
        original_index=original_index,
    )


# @njit
def is_monotonic_increasing(bounds: np.ndarray) -> bool:
    """Check if int64 values are monotonically increasing."""
    n = len(bounds)
    if n < 2:
        return True
    prev = bounds[0]
    for i in range(1, n):
        cur = bounds[i]
        if cur < prev:
            return False
        prev = cur
    return True


# @njit
def is_monotonic_decreasing(bounds: np.ndarray) -> bool:
    """Check if int64 values are monotonically decreasing."""
    n = len(bounds)
    if n < 2:
        return True
    prev = bounds[0]
    for i in range(1, n):
        cur = bounds[i]
        if cur > prev:
            return False
        prev = cur
    return True
