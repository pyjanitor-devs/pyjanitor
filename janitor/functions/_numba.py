"""Various Functions powered by Numba"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from numba import njit, prange
from pandas.api.types import (
    is_datetime64_dtype,
    is_extension_array_dtype,
)

from janitor.functions.utils import (
    _generic_func_cond_join,
    greater_than_join_types,
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
        return None
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

    if left_index is None:
        return None

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


def _numba_single_non_equi_join(
    left: pd.Series, right: pd.Series, op: str, keep: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return matching indices for single non-equi join."""
    if op == "!=":
        outcome = _generic_func_cond_join(
            left=left, right=right, op=op, multiple_conditions=False, keep=keep
        )
        if outcome is None:
            return None
        return outcome

    outcome = _generic_func_cond_join(
        left=left, right=right, op=op, multiple_conditions=True, keep="all"
    )
    if outcome is None:
        return None
    left_index, right_index, starts = outcome
    if op in less_than_join_types:
        counts = right_index.size - starts
    else:
        counts = starts[:]
        starts = np.zeros(starts.size, dtype=np.intp)
    if keep == "all":
        return _get_indices_monotonic_non_equi(
            left_index=left_index,
            right_index=right_index,
            starts=starts,
            counts=counts,
        )
    mapping = {"first": 1, "last": 0}
    return _get_indices_monotonic_non_equi_first_or_last(
        left_index=left_index,
        right_index=right_index,
        starts=starts,
        counts=counts,
        keep=mapping[keep],
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
    left_indices = []
    left_regions = []
    right_indices = []
    right_regions = []
    left_index = slice(None)
    right_index = slice(None)
    for left_on, right_on, op in gt_lt:
        left_on = df.columns.get_loc(left_on)
        right_on = right.columns.get_loc(right_on)
        left_arr = df.iloc[left_index, left_on]
        right_arr = right.iloc[right_index, right_on]
        result = _get_regions_non_equi(
            left=left_arr,
            right=right_arr,
            op=op,
        )
        if result is None:
            return None
        (
            left_index,
            right_index,
            left_region,
            right_region,
        ) = result
        left_index = pd.Index(left_index)
        right_index = pd.Index(right_index)
        left_indices.append(left_index)
        right_indices.append(right_index)
        left_regions.append(left_region)
        right_regions.append(right_region)

    right_index, right_regions = _align_indices_and_regions(
        indices=right_indices, regions=right_regions
    )

    left_index, left_regions = _align_indices_and_regions(
        indices=left_indices, regions=left_regions
    )
    left_regions = np.column_stack(left_regions)
    right_regions = np.column_stack(right_regions)

    left_region1 = left_regions[:, 0]
    # first column is already sorted
    right_region1 = right_regions[:, 0]
    starts = right_region1.searchsorted(left_region1)
    booleans = starts == right_regions.shape[0]
    # left_region should be <= right_region
    if booleans.all(axis=None):
        return None
    if booleans.any(axis=None):
        booleans = ~booleans
        starts = starts[booleans]
        left_index = left_index[booleans]
        left_regions = left_regions[booleans]
    # apply the same logic as above to the remaining columns
    # exclude points where the left_region is greater than
    # the max right_region at the search index
    # there is no point keeping those points,
    # since the left region should be <= right region
    # no need to include the first columns in the check,
    # since that has already been checked in the code above
    left_regions = left_regions[:, 1:]
    right_regions = right_regions[:, 1:]
    cum_max_arr = np.maximum.accumulate(right_regions[::-1])[::-1]

    booleans = left_regions > cum_max_arr[starts]

    booleans = booleans.any(axis=1)
    if booleans.all(axis=None):
        return None
    if booleans.any(axis=None):
        booleans = ~booleans
        left_regions = left_regions[booleans]
        left_index = left_index[booleans]
        starts = starts[booleans]
    if len(gt_lt) == 2:
        left_region = left_regions[:, 0]
        right_region = right_regions[:, 0]
        # there is a fast path if is_monotonic is True
        # we can get the matches via binary search
        if _is_monotonic_increasing(right_region):
            is_monotonic = True
            ends = right_region.searchsorted(left_region, side="left")
            starts = np.maximum(starts, ends)
            ends = right_index.size
        elif _is_monotonic_decreasing(right_region):
            is_monotonic = True
            ends = right_region[::-1].searchsorted(left_region, side="left")
            ends = right_region.size - ends
        else:
            is_monotonic = False
            # get the max end, beyond which left_region is > right_region
            cum_max_arr = cum_max_arr[:, 0]
            # the lowest left_region will have the largest coverage
            # that gives us the end
            # beyond which no left region is <= right_region
            ends = cum_max_arr[::-1].searchsorted(
                left_region.min(), side="left"
            )
            ends = right_region.size - ends
            right_index = right_index[:ends]
            right_region = right_region[:ends]
        if (ends - starts).max() == 1:
            # no point running a comparison op
            # if the width is all 1
            return left_index, right_index[starts]
        if is_monotonic and (keep == "all"):
            return _get_indices_monotonic_non_equi(
                left_index=left_index,
                right_index=right_index,
                starts=starts,
                counts=ends - starts,
            )
        if is_monotonic:
            mapping = {"first": 1, "last": 0}
            return _get_indices_monotonic_non_equi_first_or_last(
                left_index=left_index,
                right_index=right_index,
                starts=starts,
                counts=ends - starts,
                keep=mapping[keep],
            )

        if not _is_monotonic_increasing(starts):
            sorter = np.lexsort((left_region, starts))
            starts = starts[sorter]
            left_region = left_region[sorter]
            left_index = left_index[sorter]
            sorter = None
        positions, uniques = pd.factorize(right_region, sort=True)
        frequency = pd.Index(positions).value_counts()
        if keep == "all":
            return _get_indices_dual_non_monotonic_non_equi(
                left_region=left_region,
                uniques=uniques,
                right_region=right_region,
                left_index=left_index,
                right_index=right_index,
                starts=starts,
                ends=ends,
                max_arr=cum_max_arr,
                positions=positions,
                max_freq=frequency.array[0],
            )
        if keep == "first":
            return _get_indices_dual_non_monotonic_non_equi_first(
                left_region=left_region,
                uniques=uniques,
                right_region=right_region,
                left_index=left_index,
                right_index=right_index,
                starts=starts,
                ends=ends,
                max_arr=cum_max_arr,
                positions=positions,
                max_freq=frequency.array[0],
            )
        return _get_indices_dual_non_monotonic_non_equi_last(
            left_region=left_region,
            uniques=uniques,
            right_region=right_region,
            left_index=left_index,
            right_index=right_index,
            starts=starts,
            ends=ends,
            max_arr=cum_max_arr,
            positions=positions,
            max_freq=frequency.array[0],
        )
    # idea here is to iterate on the region
    # that has the lowest number of comparisions
    # for each left region in the right region
    # the region with the lowest number of comparisons
    # is moved to the front of the array
    # within this region, once we get a match,
    # we check the other regions on that same row
    # if they match, we keep the row, if not we discard
    # and move on to the next one
    if not _is_monotonic_increasing(starts):
        sorter = starts.argsort(kind="stable")
        starts = starts[sorter]
        left_regions = left_regions[sorter]
        left_index = left_index[sorter]
        sorter = None
    indices = cum_max_arr[starts] - left_regions
    indices = indices.sum(axis=0)
    indices = indices.argsort()
    left_regions = left_regions[:, indices]
    right_regions = right_regions[:, indices]
    cum_max_arr = cum_max_arr[:, indices[0]]
    positions, uniques = pd.factorize(right_regions[:, 0], sort=True)
    frequency = pd.Index(positions).value_counts()
    if keep == "all":
        return _get_indices_multiple_non_equi(
            left_regions=left_regions,
            uniques=uniques,
            right_regions=right_regions,
            left_index=left_index,
            right_index=right_index,
            starts=starts,
            ends=right_index.size,
            max_arr=cum_max_arr,
            positions=positions,
            max_freq=frequency.array[0],
        )
    if keep == "first":
        return _get_indices_multiple_non_equi_first(
            left_regions=left_regions,
            uniques=uniques,
            right_regions=right_regions,
            left_index=left_index,
            right_index=right_index,
            starts=starts,
            ends=right_index.size,
            max_arr=cum_max_arr,
            positions=positions,
            max_freq=frequency.array[0],
        )
    return _get_indices_multiple_non_equi_last(
        left_regions=left_regions,
        uniques=uniques,
        right_regions=right_regions,
        left_index=left_index,
        right_index=right_index,
        starts=starts,
        ends=right_index.size,
        max_arr=cum_max_arr,
        positions=positions,
        max_freq=frequency.array[0],
    )


def _is_monotonic_increasing(array: np.ndarray) -> np.ndarray:
    """
    numpy version of pandas' is_monotonic_increasing
    """
    return np.greater_equal(array[1:], array[:-1]).all()


def _is_monotonic_decreasing(array: np.ndarray) -> np.ndarray:
    """
    numpy version of pandas' is_monotonic_decreasing
    """
    return np.less_equal(array[1:], array[:-1]).all()


def _get_regions_non_equi(
    left: pd.Series,
    right: pd.Series,
    op: str,
) -> Union[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None]:
    """
    Get the regions for `left` and `right`.
    Strictly for non-equi joins,
    specifically the  >, >= , <, <= operators.

    Returns a tuple of indices and regions.
    """
    outcome = _generic_func_cond_join(
        left=left,
        right=right,
        op=op,
        multiple_conditions=True,
        keep="all",
    )
    if not outcome:
        return None
    left_index, right_index, search_indices = outcome
    if op in greater_than_join_types:
        right_index = right_index[::-1]
        search_indices = right_index.size - search_indices
    right_region = np.zeros(right_index.size, dtype=np.intp)
    right_region[search_indices] = 1
    right_region = right_region.cumsum()
    left_region = right_region[search_indices]
    right_region = right_region[search_indices.min() :]
    right_index = right_index[search_indices.min() :]
    return (
        left_index,
        right_index,
        left_region,
        right_region,
    )


def _align_indices_and_regions(
    indices, regions
) -> tuple[np.ndarray, np.ndarray]:
    """
    align the indices and regions
    obtained from _get_regions_non_equi.

    A single index is returned, with the regions
    properly aligned with the index.
    """
    *other_indices, index = indices
    *other_regions, region = regions
    outcome = [region]
    for _index, _region in zip(other_indices, other_regions):
        indexer = _index.get_indexer(index)
        booleans = indexer == -1
        if booleans.any():
            indexer = indexer[~booleans]
        _region = _region[indexer]
        outcome.append(_region)
    return index._values, outcome


@njit(parallel=True)
def _get_indices_monotonic_non_equi_first_or_last(
    left_index: np.ndarray,
    right_index: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
    keep: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Retrieves the matching indices
    for the left and right regions.
    This returns just the first or last matching index
    from the right, for each index in the left.
    Strictly for non-equi joins,
    where at most, only two join conditions are present.
    This is a fast path, compared to
    _get_indices_dual_non_monotonic_non_equi.
    """
    r_index = np.empty(starts.size, dtype=np.intp)
    if keep == 1:  # first
        for num in prange(starts.size):
            start = starts[np.uintp(num)]
            size = counts[np.uintp(num)]
            index = right_index[np.uintp(start)]
            for n in range(size):
                value = right_index[np.uintp(start + n)]
                index = min(index, value)
            r_index[np.uintp(num)] = index
        return left_index, r_index
    # keep == 'last'
    for num in prange(starts.size):
        start = starts[np.uintp(num)]
        size = counts[np.uintp(num)]
        index = right_index[np.uintp(start)]
        for n in range(size):
            value = right_index[np.uintp(start + n)]
            index = max(index, value)
        r_index[np.uintp(num)] = index
    return left_index, r_index


@njit(parallel=True)
def _get_indices_monotonic_non_equi(
    left_index: np.ndarray,
    right_index: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Retrieves the matching indices
    for the left and right regions.
    Strictly for non-equi joins,
    where at most, only two join conditions are present.
    This is a fast path, compared to
    _get_indices_dual_non_monotonic_non_equi.
    """
    # compute the starting positions in the left index
    # e.g if counts is [2, 4, 6]
    # the starting positions in the left_index(l_index)
    # will be 0, 2, 6, 12
    cumulative_starts = counts.cumsum()
    start_indices = np.zeros(starts.size, dtype=np.intp)
    start_indices[1:] = cumulative_starts[:-1]
    l_index = np.empty(cumulative_starts[-1], dtype=np.intp)
    r_index = np.empty(cumulative_starts[-1], dtype=np.intp)
    for num in prange(start_indices.size):
        ind = start_indices[np.uintp(num)]
        size = counts[np.uintp(num)]
        l_ind = left_index[np.uintp(num)]
        r_ind = starts[np.uintp(num)]
        for n in range(size):
            indexer = ind + n
            r_indexer = r_ind + n
            l_index[np.uintp(indexer)] = l_ind
            r_index[np.uintp(indexer)] = right_index[np.uintp(r_indexer)]
    return l_index, r_index


@njit()
def _get_indices_dual_non_monotonic_non_equi_first(
    left_region: np.ndarray,
    uniques: np.ndarray,
    right_region: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    starts: np.ndarray,
    positions: np.ndarray,
    max_arr: np.ndarray,
    ends: int,
    max_freq: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Retrieves the matching indices
    for the left and right regions.
    Strictly for non-equi joins,
    where only two join conditions are present.
    """
    nrows = positions.max() + 1
    value_counts = np.zeros(nrows, dtype=np.intp)
    indices = np.empty((nrows, max_freq), dtype=np.intp)
    r_index = np.empty(starts.size, dtype=np.intp)
    min_right_region = right_region.min()
    previous_start = ends
    previous_region = -1
    previous_index = -1
    # positions -----> [0 1 2 3 4 5]
    # left_region ---> [0 1 1 5 5 6]
    # right_region --> [7 7 7 7 7 7]
    # note how the last value in the left_region(6)
    # is less than the previous value(position 4) in the right_region (7)
    # there is no need to search from 5 to 7 for position 4
    # since that has already been captured in the iteration
    # for 6 to 7 (position 5)
    # the iterative search should not capture more than once
    # and only capture relevant values
    # in short, restrict the search space as much as permissible
    # search starts from the bottom/max
    for num in range(starts.size - 1, -1, -1):
        start = starts[np.uintp(num)]
        arr_max = max_arr[np.uintp(start)]
        end = min(previous_start, ends)
        l_region = left_region[np.uintp(num)]
        previous_start = start
        for n in range(start, end):
            pos = positions[np.uintp(n)]
            value_counts[np.uintp(pos)] += 1
            indices[
                np.uintp(pos), np.uintp(value_counts[np.uintp(pos)] - 1)
            ] = n
        if (l_region <= min_right_region) or ((ends - start) == 1):
            index = right_index[np.uintp(start)]
            counter = ends - start
            for n in range(counter):
                value = right_index[np.uintp(start + n)]
                index = min(index, value)
            r_index[np.uintp(num)] = index
            previous_region = l_region
            previous_index = index
            continue
        elif (start == end) and (previous_region == l_region):
            r_index[np.uintp(num)] = previous_index
            continue
        posn = np.searchsorted(uniques, l_region, side="left")
        max_posn = np.searchsorted(uniques, arr_max, side="right")
        diff1 = ends - start
        diff2 = max_posn - posn
        index = right_index.max()
        if diff1 < diff2:
            for sz in range(start, ends):
                pos = positions[np.uintp(sz)]
                r_region = uniques[np.uintp(pos)]
                status = l_region <= r_region
                if not status:
                    continue
                value = right_index[np.uintp(sz)]
                index = min(index, value)
            r_index[np.uintp(num)] = index
            previous_region = l_region
            previous_index = index
            continue
        for nn in range(posn, max_posn):
            counter = value_counts[np.uintp(nn)]
            if not counter:
                continue
            for sz in range(counter):
                pos = indices[np.uintp(nn), np.uintp(sz)]
                value = right_index[np.uintp(pos)]
                index = min(index, value)
        r_index[np.uintp(num)] = index
        previous_region = l_region
        previous_index = index
    return left_index, r_index


@njit()
def _get_indices_dual_non_monotonic_non_equi_last(
    left_region: np.ndarray,
    uniques: np.ndarray,
    right_region: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    starts: np.ndarray,
    positions: np.ndarray,
    max_arr: np.ndarray,
    ends: int,
    max_freq: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Retrieves the matching indices
    for the left and right regions.
    Strictly for non-equi joins,
    where only two join conditions are present.
    """
    nrows = positions.max() + 1
    value_counts = np.zeros(nrows, dtype=np.intp)
    indices = np.empty((nrows, max_freq), dtype=np.intp)
    r_index = np.empty(starts.size, dtype=np.intp)
    min_right_region = right_region.min()
    previous_start = ends
    previous_region = -1
    previous_index = -1
    # positions -----> [0 1 2 3 4 5]
    # left_region ---> [0 1 1 5 5 6]
    # right_region --> [7 7 7 7 7 7]
    # note how the last value in the left_region(6)
    # is less than the previous value(position 4) in the right_region (7)
    # there is no need to search from 5 to 7 for position 4
    # since that has already been captured in the iteration
    # for 6 to 7 (position 5)
    # the iterative search should not capture more than once
    # and only capture relevant values
    # in short, restrict the search space as much as permissible
    # search starts from the bottom/max
    for num in range(starts.size - 1, -1, -1):
        start = starts[np.uintp(num)]
        arr_max = max_arr[np.uintp(start)]
        end = min(previous_start, ends)
        l_region = left_region[np.uintp(num)]
        previous_start = start
        for n in range(start, end):
            pos = positions[np.uintp(n)]
            value_counts[np.uintp(pos)] += 1
            indices[
                np.uintp(pos), np.uintp(value_counts[np.uintp(pos)] - 1)
            ] = n
        if (l_region <= min_right_region) or ((ends - start) == 1):
            index = right_index[np.uintp(start)]
            counter = ends - start
            for n in range(counter):
                value = right_index[np.uintp(start + n)]
                index = max(index, value)
            r_index[np.uintp(num)] = index
            previous_region = l_region
            previous_index = index
            continue
        elif (start == end) and (previous_region == l_region):
            r_index[np.uintp(num)] = previous_index
            continue
        posn = np.searchsorted(uniques, l_region, side="left")
        max_posn = np.searchsorted(uniques, arr_max, side="right")
        diff1 = ends - start
        diff2 = max_posn - posn
        index = -1
        if diff1 < diff2:
            for sz in range(start, ends):
                pos = positions[np.uintp(sz)]
                r_region = uniques[np.uintp(pos)]
                status = l_region <= r_region
                if not status:
                    continue
                value = right_index[np.uintp(sz)]
                index = max(index, value)
            r_index[np.uintp(num)] = index
            previous_region = l_region
            previous_index = index
            continue
        for nn in range(posn, max_posn):
            counter = value_counts[np.uintp(nn)]
            if not counter:
                continue
            for sz in range(counter):
                pos = indices[np.uintp(nn), np.uintp(sz)]
                value = right_index[np.uintp(pos)]
                index = max(index, value)
        r_index[np.uintp(num)] = index
        previous_region = l_region
        previous_index = index
    return left_index, r_index


@njit()
def _get_indices_dual_non_monotonic_non_equi(
    left_region: np.ndarray,
    uniques: np.ndarray,
    right_region: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    starts: np.ndarray,
    positions: np.ndarray,
    max_arr: np.ndarray,
    ends: int,
    max_freq: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Retrieves the matching indices
    for the left and right regions.
    Strictly for non-equi joins,
    where only two join conditions are present.
    """
    # TODO: implement as a balanced binary tree
    # which should be more performant?
    # current implementation uses linear search

    # two step pass
    # first pass gets the length of the final indices

    # https://numba.discourse.group/t/uint64-vs-int64-indexing-performance-difference/1500
    # indexing with unsigned integers offers more performance
    nrows = positions.max() + 1
    value_counts = np.zeros(nrows, dtype=np.intp)
    count_indices = np.empty(starts.size, dtype=np.intp)
    min_right_region = right_region.min()
    total_length = 0
    previous_start = ends
    previous_region = -1
    previous_counter = 0
    # positions -----> [0 1 2 3 4 5]
    # left_region ---> [0 1 1 5 5 6]
    # right_region --> [7 7 7 7 7 7]
    # note how the last value in the left_region(6)
    # is less than the previous value(position 4) in the right_region (7)
    # there is no need to search from 5 to 7 for position 4
    # since that has already been captured in the iteration
    # for 6 to 7 (position 5)
    # the iterative search should not capture more than once
    # and only capture relevant values
    # in short, restrict the search space as much as permissible
    # search starts from the bottom/max
    for num in range(starts.size - 1, -1, -1):
        start = starts[np.uintp(num)]
        arr_max = max_arr[np.uintp(start)]
        end = min(previous_start, ends)
        l_region = left_region[np.uintp(num)]
        previous_start = start
        counter = 0
        for n in range(start, end):
            pos = positions[np.uintp(n)]
            value_counts[np.uintp(pos)] += 1
        if (l_region <= min_right_region) or ((ends - start) == 1):
            counter = ends - start
            total_length += counter
            count_indices[np.uintp(num)] = counter
            previous_region = l_region
            previous_counter = counter
            continue
        elif (start == end) and (previous_region == l_region):
            total_length += previous_counter
            count_indices[np.uintp(num)] = previous_counter
            continue
        posn = np.searchsorted(uniques, l_region, side="left")
        max_posn = np.searchsorted(uniques, arr_max, side="right")
        diff1 = ends - start
        diff2 = max_posn - posn
        if diff1 < diff2:
            for sz in range(start, ends):
                pos = positions[np.uintp(sz)]
                r_region = uniques[np.uintp(pos)]
                counter += l_region <= r_region
            total_length += counter
            count_indices[np.uintp(num)] = counter
            previous_region = l_region
            previous_counter = counter
            continue
        for nn in range(posn, max_posn):
            counter += value_counts[np.uintp(nn)]
        total_length += counter
        count_indices[np.uintp(num)] = counter
        previous_region = l_region
        previous_counter = counter
    # second pass populates the final indices with actual values
    start_indices = np.zeros(starts.size, dtype=np.intp)
    start_indices[1:] = count_indices.cumsum()[:-1]
    l_index = np.empty(total_length, dtype=np.intp)
    r_index = np.empty(total_length, dtype=np.intp)
    value_counts = np.zeros(nrows, dtype=np.intp)
    indices = np.empty((nrows, max_freq), dtype=np.intp)
    previous_start = ends
    previous_region = -1
    previous_index = -1
    for num in range(starts.size - 1, -1, -1):
        start = starts[np.uintp(num)]
        arr_max = max_arr[np.uintp(start)]
        end = min(previous_start, ends)
        l_region = left_region[np.uintp(num)]
        indexer = start_indices[np.uintp(num)]
        l_ind = left_index[np.uintp(num)]
        previous_start = start
        for n in range(start, end):
            pos = positions[np.uintp(n)]
            value_counts[np.uintp(pos)] += 1
            indices[
                np.uintp(pos), np.uintp(value_counts[np.uintp(pos)] - 1)
            ] = n
        # no need to compare
        if (l_region <= min_right_region) or ((ends - start) == 1):
            step = 0
            for cnt in range(start, ends):
                l_index[np.uintp(indexer + step)] = l_ind
                r_index[np.uintp(indexer + step)] = right_index[np.uintp(cnt)]
                step += 1
            previous_region = l_region
            previous_index = indexer
            continue
        # the work has already been done, no need to redo, just `steal`
        elif (start == end) and (previous_region == l_region):
            counter = count_indices[np.uintp(num)]
            for ct in range(counter):
                l_index[np.uintp(indexer + ct)] = l_ind
                r_index[np.uintp(indexer + ct)] = r_index[
                    np.uintp(previous_index + ct)
                ]
            previous_index = indexer
            continue
        posn = np.searchsorted(uniques, l_region, side="left")
        max_posn = np.searchsorted(uniques, arr_max, side="right")
        diff1 = ends - start
        diff2 = max_posn - posn
        if diff1 < diff2:
            step = 0
            counter = count_indices[np.uintp(num)]
            for sz in range(start, ends):
                if not counter:
                    break
                pos = positions[np.uintp(sz)]
                r_region = uniques[np.uintp(pos)]
                if l_region > r_region:
                    continue
                l_index[np.uintp(indexer + step)] = l_ind
                r_index[np.uintp(indexer + step)] = right_index[np.uintp(sz)]
                step += 1
                counter -= 1
            previous_region = l_region
            previous_index = indexer
            continue
        step = 0
        # builds off a variant of counting sort
        # still a linear search though
        for nn in range(posn, max_posn):
            counter = value_counts[np.uintp(nn)]
            if not counter:
                continue
            for sz in range(counter):
                pos = indices[np.uintp(nn), np.uintp(sz)]
                r_ind = right_index[np.uintp(pos)]
                l_index[np.uintp(indexer + step)] = l_ind
                r_index[np.uintp(indexer + step)] = r_ind
                step += 1
        previous_region = l_region
        previous_index = indexer
    return l_index, r_index


@njit()
def _get_indices_multiple_non_equi_first(
    left_regions: np.ndarray,
    uniques: np.ndarray,
    right_regions: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    starts: np.ndarray,
    positions: np.ndarray,
    ends: int,
    max_arr: np.ndarray,
    max_freq: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Retrieves the matching indices
    for the left and right regions.
    Strictly for non-equi joins,
    where the join conditions are more than two.
    """
    # same as _get_indices_dual_non_monotonic_non_equi
    # but with an extra check
    # on the remaining regions per row
    # to ensure they match
    nrows = positions.max() + 1
    value_counts = np.zeros(nrows, dtype=np.intp)
    indices = np.empty((nrows, max_freq), dtype=np.intp)
    r_index = np.empty(starts.size, dtype=np.intp)
    booleans = np.zeros(starts.size, dtype=np.bool_)
    previous_start = ends
    ncols = right_regions.shape[1]
    total_length = 0
    # positions -----> [0 1 2 3 4 5]
    # left_region ---> [0 1 1 5 5 6]
    # right_region --> [7 7 7 7 7 7]
    # note how the last value in the left_region(6)
    # is less than the previous value(position 4) in the right_region (7)
    # there is no need to search from 5 to 7 for position 4
    # since that has already been captured in the iteration
    # for 6 to 7 (position 5)
    # the iterative search should not capture more than once
    # and only capture relevant values
    # in short, restrict the search space as much as permissible
    # search starts from the bottom/max
    for num in range(starts.size - 1, -1, -1):
        start = starts[np.uintp(num)]
        arr_max = max_arr[np.uintp(start)]
        end = min(previous_start, ends)
        previous_start = start
        for n in range(start, end):
            pos = positions[np.uintp(n)]
            value_counts[np.uintp(pos)] += 1
            indices[
                np.uintp(pos), np.uintp(value_counts[np.uintp(pos)] - 1)
            ] = n
        posn = np.searchsorted(
            uniques, left_regions[np.uintp(num), 0], side="left"
        )
        max_posn = np.searchsorted(uniques, arr_max, side="right")
        index = right_index.max()
        any_match = False
        for nn in range(posn, max_posn):
            counts = value_counts[np.uintp(nn)]
            if not counts:
                continue
            for sz in range(counts):
                pos = indices[np.uintp(nn), np.uintp(sz)]
                status = True
                # check the remaining regions for this row
                # and ensure left_region<=right_region
                # for each one
                for col in range(1, ncols):
                    l_value = left_regions[np.uintp(num), np.uintp(col)]
                    r_value = right_regions[np.uintp(pos), np.uintp(col)]
                    if l_value > r_value:
                        status = False
                        break
                if not status:
                    continue
                any_match |= status
                value = right_index[np.uintp(pos)]
                index = min(index, value)
        r_index[np.uintp(num)] = index
        booleans[np.uintp(num)] = any_match
        total_length += any_match
    if total_length == starts.size:
        return left_index, r_index
    right_index = np.empty(total_length, dtype=np.intp)
    l_index = np.empty(total_length, dtype=np.intp)
    n = 0
    for num in range(starts.size):
        if n == total_length:
            break
        if not booleans[np.uintp(num)]:
            continue
        l_index[np.uintp(n)] = left_index[np.uintp(num)]
        right_index[np.uintp(n)] = r_index[np.uintp(num)]
        n += 1
    return l_index, right_index


@njit()
def _get_indices_multiple_non_equi_last(
    left_regions: np.ndarray,
    uniques: np.ndarray,
    right_regions: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    starts: np.ndarray,
    positions: np.ndarray,
    ends: int,
    max_arr: np.ndarray,
    max_freq: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Retrieves the matching indices
    for the left and right regions.
    Strictly for non-equi joins,
    where the join conditions are more than two.
    """
    # same as _get_indices_dual_non_monotonic_non_equi
    # but with an extra check
    # on the remaining regions per row
    # to ensure they match
    nrows = positions.max() + 1
    value_counts = np.zeros(nrows, dtype=np.intp)
    indices = np.empty((nrows, max_freq), dtype=np.intp)
    r_index = np.empty(starts.size, dtype=np.intp)
    booleans = np.zeros(starts.size, dtype=np.bool_)
    previous_start = ends
    ncols = right_regions.shape[1]
    total_length = 0
    # positions -----> [0 1 2 3 4 5]
    # left_region ---> [0 1 1 5 5 6]
    # right_region --> [7 7 7 7 7 7]
    # note how the last value in the left_region(6)
    # is less than the previous value(position 4) in the right_region (7)
    # there is no need to search from 5 to 7 for position 4
    # since that has already been captured in the iteration
    # for 6 to 7 (position 5)
    # the iterative search should not capture more than once
    # and only capture relevant values
    # in short, restrict the search space as much as permissible
    # search starts from the bottom/max
    for num in range(starts.size - 1, -1, -1):
        start = starts[np.uintp(num)]
        arr_max = max_arr[np.uintp(start)]
        end = min(previous_start, ends)
        previous_start = start
        for n in range(start, end):
            pos = positions[np.uintp(n)]
            value_counts[np.uintp(pos)] += 1
            indices[
                np.uintp(pos), np.uintp(value_counts[np.uintp(pos)] - 1)
            ] = n
        posn = np.searchsorted(
            uniques, left_regions[np.uintp(num), 0], side="left"
        )
        max_posn = np.searchsorted(uniques, arr_max, side="right")
        index = -1
        any_match = False
        for nn in range(posn, max_posn):
            counts = value_counts[np.uintp(nn)]
            if not counts:
                continue
            for sz in range(counts):
                pos = indices[np.uintp(nn), np.uintp(sz)]
                status = True
                # check the remaining regions for this row
                # and ensure left_region<=right_region
                # for each one
                for col in range(1, ncols):
                    l_value = left_regions[np.uintp(num), np.uintp(col)]
                    r_value = right_regions[np.uintp(pos), np.uintp(col)]
                    if l_value > r_value:
                        status = False
                        break
                if not status:
                    continue
                any_match |= status
                value = right_index[np.uintp(pos)]
                index = max(index, value)
        r_index[np.uintp(num)] = index
        booleans[np.uintp(num)] = any_match
        total_length += any_match
    if total_length == starts.size:
        return left_index, r_index
    right_index = np.empty(total_length, dtype=np.intp)
    l_index = np.empty(total_length, dtype=np.intp)
    n = 0
    for num in range(starts.size):
        if n == total_length:
            break
        if not booleans[np.uintp(num)]:
            continue
        l_index[np.uintp(n)] = left_index[np.uintp(num)]
        right_index[np.uintp(n)] = r_index[np.uintp(num)]
        n += 1
    return l_index, right_index


@njit()
def _get_indices_multiple_non_equi(
    left_regions: np.ndarray,
    uniques: np.ndarray,
    right_regions: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    starts: np.ndarray,
    positions: np.ndarray,
    ends: int,
    max_arr: np.ndarray,
    max_freq: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Retrieves the matching indices
    for the left and right regions.
    Strictly for non-equi joins,
    where the join conditions are more than two.
    """
    # same as _get_indices_dual_non_monotonic_non_equi
    # but with an extra check
    # on the remaining regions per row
    # to ensure they match
    nrows = positions.max() + 1
    value_counts = np.zeros(nrows, dtype=np.intp)
    count_indices = np.empty(starts.size, dtype=np.intp)
    indices = np.empty((nrows, max_freq), dtype=np.intp)
    total_length = 0
    true_count = 0
    previous_start = ends
    ncols = right_regions.shape[1]
    # positions -----> [0 1 2 3 4 5]
    # left_region ---> [0 1 1 5 5 6]
    # right_region --> [7 7 7 7 7 7]
    # note how the last value in the left_region(6)
    # is less than the previous value(position 4) in the right_region (7)
    # there is no need to search from 5 to 7 for position 4
    # since that has already been captured in the iteration
    # for 6 to 7 (position 5)
    # the iterative search should not capture more than once
    # and only capture relevant values
    # in short, restrict the search space as much as permissible
    # search starts from the bottom/max
    for num in range(starts.size - 1, -1, -1):
        start = starts[np.uintp(num)]
        arr_max = max_arr[np.uintp(start)]
        end = min(previous_start, ends)
        previous_start = start
        counter = 0
        for n in range(start, end):
            pos = positions[np.uintp(n)]
            value_counts[np.uintp(pos)] += 1
            indices[
                np.uintp(pos), np.uintp(value_counts[np.uintp(pos)] - 1)
            ] = n
        posn = np.searchsorted(
            uniques, left_regions[np.uintp(num), 0], side="left"
        )
        max_posn = np.searchsorted(uniques, arr_max, side="right")
        for nn in range(posn, max_posn):
            counts = value_counts[np.uintp(nn)]
            if not counts:
                continue
            for sz in range(counts):
                pos = indices[np.uintp(nn), np.uintp(sz)]
                status = 1
                # check the remaining regions for this row
                # and ensure left_region<=right_region
                # for each one
                for col in range(1, ncols):
                    l_value = left_regions[np.uintp(num), np.uintp(col)]
                    r_value = right_regions[np.uintp(pos), np.uintp(col)]
                    if l_value > r_value:
                        status = 0
                        break
                counter += status
                total_length += status
        true_count += counter != 0
        count_indices[np.uintp(num)] = counter
    # get actual index points
    start_indices = np.zeros(true_count, dtype=np.intp)
    true_indices = np.empty(true_count, dtype=np.intp)
    n = 0
    sz = 0
    cumcount = 0
    for num in range(starts.size):
        if n == true_count:
            break
        counts = count_indices[np.uintp(num)]
        if not counts:
            continue
        true_indices[np.uintp(n)] = num
        start_indices[np.uintp(sz)] = cumcount
        n += 1
        sz += 1
        cumcount += counts

    l_index = np.empty(total_length, dtype=np.intp)
    r_index = np.empty(total_length, dtype=np.intp)
    value_counts = np.zeros(nrows, dtype=np.intp)
    indices = np.empty((nrows, max_freq), dtype=np.intp)
    previous_start = ends
    for num in range(true_count - 1, -1, -1):
        true_num = true_indices[np.uintp(num)]
        start = starts[np.uintp(true_num)]
        arr_max = max_arr[np.uintp(start)]
        end = min(previous_start, ends)
        previous_start = start
        for n in range(start, end):
            pos = positions[np.uintp(n)]
            value_counts[np.uintp(pos)] += 1
            indices[np.uintp(pos), np.uintp(value_counts[pos] - 1)] = n
        posn = np.searchsorted(
            uniques, left_regions[np.uintp(true_num), 0], side="left"
        )
        max_posn = np.searchsorted(uniques, arr_max, side="right")
        size = count_indices[np.uintp(true_num)]
        l_ind = left_index[np.uintp(true_num)]
        indexer = start_indices[np.uintp(num)]
        for nn in range(posn, max_posn):
            counts = value_counts[np.uintp(nn)]
            if not counts:
                continue
            for sz in range(counts):
                pos = indices[np.uintp(nn), np.uintp(sz)]
                status = 1
                # check the remaining regions for this row
                # and ensure left_region<=right_region
                # for each one
                for col in range(1, ncols):
                    l_value = left_regions[np.uintp(true_num), np.uintp(col)]
                    r_value = right_regions[np.uintp(pos), np.uintp(col)]
                    if l_value > r_value:
                        status = 0
                        break
                if not status:
                    continue
                r_ind = right_index[np.uintp(pos)]
                l_index[np.uintp(indexer)] = l_ind
                r_index[np.uintp(indexer)] = r_ind
                indexer += 1
                size -= 1
                if not size:
                    break
    return l_index, r_index
