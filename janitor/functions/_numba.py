"""Various Functions powered by Numba"""

import numpy as np
import pandas as pd
from janitor.functions.utils import (
    _generic_func_cond_join,
    _JoinOperator,
    less_than_join_types,
    greater_than_join_types,
)
from numba import njit, prange
from pandas.api.types import is_extension_array_dtype, is_datetime64_dtype
from typing import Union


def _numba_dual_join(df: pd.DataFrame, right: pd.DataFrame, pair: list):
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

    # preparation phase
    # get rid of nulls,
    # sort if necessary
    # convert to numpy arrays
    left_columns, right_columns, (op1, op2) = zip(*pair)
    left_df = df.loc(axis=1)[set(left_columns)]
    any_nulls = left_df.isna().any(axis=1)
    if any_nulls.all(axis=None):
        return None
    if any_nulls.any(axis=None):
        left_df = left_df.loc[~any_nulls]
    right_df = right.loc(axis=1)[set(right_columns)]
    any_nulls = right_df.isna().any(axis=1)
    if any_nulls.all(axis=None):
        return None
    if any_nulls.any(axis=None):
        right_df = right_df.loc[~any_nulls]

    left_index = left_df.index._values
    left_arr1 = left_df[left_columns[0]]
    op1_strict = True if op1 in {"<", ">"} else False
    if op1 in less_than_join_types:
        op1 = True
        if not left_arr1.is_monotonic_increasing:
            left_arr1 = left_arr1.sort_values(ascending=True)
    else:
        op1 = False
        if not left_arr1.is_monotonic_decreasing:
            left_arr1 = left_arr1.sort_values(ascending=False)

    left_index1 = left_arr1.index._values
    left_arr1 = left_arr1._values
    right_arr1 = right_df[right_columns[0]]
    right_index1 = right_arr1.index._values
    right_arr1 = right_arr1._values
    if is_extension_array_dtype(left_arr1):
        arr_dtype = left_arr1.dtype.numpy_dtype
        left_arr1 = left_arr1.astype(arr_dtype)
        right_arr1 = right_arr1.astype(arr_dtype)
    if is_datetime64_dtype(left_arr1):
        left_arr1 = left_arr1.view(np.int64)
        right_arr1 = right_arr1.view(np.int64)

    left_arr2 = left_df[left_columns[1]]
    op2_strict = True if op2 in {"<", ">"} else False
    if op2 in less_than_join_types:
        op2 = True
        if not left_arr2.is_monotonic_increasing:
            left_arr2 = left_arr2.sort_values(ascending=True)
    else:
        op2 = False
        if not left_arr2.is_monotonic_decreasing:
            left_arr2 = left_arr2.sort_values(ascending=False)

    left_index2 = left_arr2.index._values
    left_arr2 = left_arr2._values
    right_arr2 = right_df[right_columns[1]]
    right_index2 = right_arr2.index._values
    right_arr2 = right_arr2._values
    if is_extension_array_dtype(left_arr2):
        arr_dtype = left_arr2.dtype.numpy_dtype
        left_arr2 = left_arr2.astype(arr_dtype)
        right_arr2 = right_arr2.astype(arr_dtype)
    if is_datetime64_dtype(left_arr2):
        left_arr2 = left_arr2.view(np.int64)
        right_arr2 = right_arr2.view(np.int64)

    left_index, right_index = _numba_non_equi_dual_join(
        left_arr1=left_arr1,
        right_arr1=right_arr1,
        left_index=left_index,
        left_index1=left_index1,
        right_index1=right_index1,
        op1=op1,
        op1_strict=op1_strict,
        left_arr2=left_arr2,
        right_arr2=right_arr2,
        left_index2=left_index2,
        right_index2=right_index2,
        op2=op2,
        op2_strict=op2_strict,
    )

    if left_index is None:
        return None
    return left_index, right_index


# the binary search functions below are modifications
# of python's bisect function, with a simple aim of
# getting regions.
# regions should always be at the far end left or right
# e.g for [2, 3, 3, 4], 3 (if <) should be position 0
# if (<=), position should be 1
# for [4, 3, 2, 2], 3 (if >) should be position 0
# if >=, position should be 1


@njit()
def _region_strict(
    arr: np.ndarray, value: Union[int, float], op_code: bool
) -> int:
    """
    Modification of Python's bisect_left function.
    Used to get the region where the operator is
    either > or <
    """
    high = len(arr)
    low = 0
    while low < high:
        mid = low + (high - low) // 2
        if op_code:
            check = arr[mid] < value
        else:
            check = arr[mid] > value
        if check:
            low = mid + 1
        else:
            high = mid
    return low - 1


@njit()
def _region_not_strict(
    arr: np.ndarray, value: Union[int, float], op_code: bool
) -> int:
    """
    Modification of Python's bisect_right function.
    Used to get the region where the operator is
    either >= or <=
    """
    high = len(arr)
    low = 0
    while low < high:
        mid = low + (high - low) // 2
        if op_code:
            check = value < arr[mid]
        else:
            check = value > arr[mid]
        if check:
            high = mid
        else:
            low = mid + 1
    return low - 1


@njit()
def _binary_search_exact_match(array: np.ndarray, value: Union[int, float]):
    """
    Modification of python's bisect_left.
    Aim is to get position where there is an exact match.

    Returns an integer.
    """
    high = array.size
    low = 0
    while low < high:
        mid = low + (high - low) // 2
        if array[mid] == value:
            return mid
        if array[mid] < value:
            low = mid + 1
        else:
            high = mid
    return -1


@njit(cache=True, parallel=True)
def _numba_non_equi_dual_join(
    left_arr1: np.ndarray,
    right_arr1: np.ndarray,
    left_index: np.ndarray,
    left_index1: np.ndarray,
    right_index1: np.ndarray,
    op1: bool,
    op1_strict: bool,
    left_arr2: np.ndarray,
    right_arr2: np.ndarray,
    left_index2: np.ndarray,
    right_index2: np.ndarray,
    op2: bool,
    op2_strict: bool,
):
    # get regions
    len_right_array = right_arr1.size
    len_left_array = left_arr1.size
    positions1 = np.empty(len_right_array, dtype=np.intp)
    positions2 = np.empty(len_right_array, dtype=np.intp)
    booleans1 = np.ones(len_right_array, dtype=np.bool_)
    booleans2 = np.ones(len_right_array, dtype=np.bool_)
    bools1 = np.zeros(len_left_array, dtype=np.int8)
    bools2 = np.zeros(len_left_array, dtype=np.int8)
    counts1 = 0
    counts2 = 0
    for num in prange(len_right_array):
        if op1_strict:
            position1 = _region_strict(left_arr1, right_arr1[num], op1)
        else:
            position1 = _region_not_strict(left_arr1, right_arr1[num], op1)
        if op2_strict:
            position2 = _region_strict(left_arr2, right_arr2[num], op2)
        else:
            position2 = _region_not_strict(left_arr2, right_arr2[num], op2)
        if position1 == -1:
            booleans1[num] = False
            counts1 += 1
        else:
            bools1[position1] = 1
        positions1[num] = position1
        if position2 == -1:
            booleans2[num] = False
            counts2 += 1
        else:
            bools2[position2] = 1
        positions2[num] = position2
    # no matches -> there is no value
    # from left_arr1 that is ahead of right_arr1
    if (counts1 == len_right_array) or (counts2 == len_right_array):
        return None, None

    if counts1 > 0:
        right_index1 = right_index1[booleans1]
        positions1 = positions1[booleans1]
    if counts2 > 0:
        right_index2 = right_index2[booleans2]
        positions2 = positions2[booleans2]
    # get rid of entries in the left_array
    # that have no match
    max_position = np.max(positions1) + 1
    if max_position < len_left_array:
        left_index1 = left_index1[:max_position]
        bools1 = bools1[:max_position]
    # generate left and right regions
    counter = np.sum(bools1)
    left_region1 = np.empty(max_position, dtype=np.intp)
    for num in range(len(bools1) - 1, -1, -1):
        counter = counter - bools1[num]
        left_region1[num] = counter
    right_region1 = np.empty(right_index1.size, dtype=np.intp)
    for num in prange(right_index1.size):
        right_region1[num] = left_region1[positions1[num]]

    max_position = np.max(positions2) + 1
    if max_position < len_left_array:
        left_index2 = left_index2[:max_position]
        bools2 = bools2[:max_position]
    # generate left and right regions
    counter = np.sum(bools2)
    left_region2 = np.empty(max_position, dtype=np.intp)
    for num in range(len(bools2) - 1, -1, -1):
        counter = counter - bools2[num]
        left_region2[num] = counter
    right_region2 = np.empty(right_index2.size, dtype=np.intp)
    for num in prange(right_index2.size):
        right_region2[num] = left_region2[positions2[num]]

    positions1 = None
    positions2 = None
    bools1 = None
    bools2 = None
    booleans1 = None
    booleans2 = None
    counts1 = None
    counts2 = None

    # realign left index and left regions
    booleans = np.zeros(left_index.size, dtype=np.bool_)
    positions = np.empty(left_index.size, dtype=np.intp)
    # get intersection of left_index and left_index1
    for num in prange(left_index1.size):
        position = _binary_search_exact_match(left_index, left_index1[num])
        booleans[position] = True
        positions[position] = num
    if left_index1.size < left_index.size:
        positions = positions[booleans]
        left_index = left_index[booleans]
    left_region1 = left_region1[positions]
    booleans = np.zeros(left_index2.size, dtype=np.bool_)
    positions = np.empty(left_index2.size, dtype=np.intp)
    counts = 0
    # get the intersection of left_index2 and trimmed left_index
    for num in prange(left_index2.size):
        position = _binary_search_exact_match(left_index, left_index2[num])
        if position == -1:
            counts += 1
        else:
            booleans[num] = True
        positions[num] = position
    if counts == left_index2.size:
        return None, None
    if counts > 0:
        positions = positions[booleans]
        left_index2 = left_index2[booleans]
        left_region2 = left_region2[booleans]
    left_region1 = left_region1[positions]
    left_index1 = None
    left_index = None

    # align right_index and right regions
    booleans = np.zeros(right_index2.size, dtype=np.bool_)
    positions = np.empty(right_index2.size, dtype=np.intp)
    counts = 0
    # get intersection of right_index1 and right_index2
    for num in prange(right_index2.size):
        position = _binary_search_exact_match(right_index1, right_index2[num])
        if position == -1:
            counts += 1
        else:
            booleans[num] = True
        positions[num] = position
    if counts == right_index2.size:
        return None, None
    if counts > 0:
        positions = positions[booleans]
        right_index2 = right_index2[booleans]
        right_region2 = right_region2[booleans]
    right_region1 = right_region1[positions]
    right_index1 = None

    # get positions where right_region2 is >= left_region2
    # use these positions to count backwards
    # and get positions where right_region1 >= left_region1
    positions = np.empty(right_region2.size, dtype=np.intp)
    for num in prange(right_region2.size):
        position = _region_not_strict(left_region2, right_region2[num], True)
        positions[num] = position

    # cumulative decreasing
    cummin_arr = np.empty(left_region1.size, dtype=np.intp)
    # is_monotonic_decreasing = True
    start = left_region1[0]
    cummin_arr[0] = start
    if left_region1.size > 1:
        for num in range(1, left_region1.size):
            new_value = left_region1[num]
            if start >= new_value:
                start = new_value
            # else:
            # is_monotonic_decreasing = False
            cummin_arr[num] = start

    # TODO: what if left_region1 is cumulative decreasing?
    # a faster option (binary search) is possible
    # I implemented it, however it triggers
    # a maximum recursion error in numba
    # cant explain why ...
    # besides a cumulative decreasing left_region1
    # might not occur often

    # find earliest point where left_region1 > right_region1
    # that serves as our end boundary, and should reduce search space

    # commented out
    # at the moment, I cant integrate this with parallel
    # it raises a maximum recursion error in numba
    # cant decipher the cause of the error
    # if is_monotonic_decreasing:
    #     counts = 0
    #     bool_count = 0
    #     booleans = np.ones(right_region1.size, dtype=np.bool_)
    #     sizes = np.empty(right_region1.size, dtype=np.intp)
    #     for num in prange(right_region1.size):
    #         value = right_region1[num]
    #         high = positions[num] + 1
    #         left_arr = left_region1[:high]
    #         low = 0
    #         while low < high:
    #             mid = low + (high - low) // 2
    #             if left_arr[mid] > value:
    #                 low = mid + 1
    #             else:
    #                 high = mid
    #         if low == left_arr.size:
    #             booleans[num] = False
    #             bool_count += 1
    #         else:
    #             size = left_arr.size - low
    #             sizes[num] = size
    #             counts += size

    #     if bool_count == right_region1.size:
    #         return None, None
    #     if bool_count > 0:
    #         right_region1 = right_region1[booleans]
    #         right_index2 = right_index2[booleans]
    #         positions = positions[booleans]
    #         sizes = sizes[booleans]

    #     starts = np.empty(right_region1.size, dtype=np.intp)
    #     starts[0] = 0
    #     starts[1:] = np.cumsum(sizes)[:-1]
    #     l_index = np.empty(counts, dtype=np.intp)
    #     r_index = np.empty(counts, dtype=np.intp)
    #     for num in prange(right_region1.size):
    #         ind = starts[num]
    #         start = positions[num]
    #         size = sizes[num]
    #         r_ind = right_index2[num]
    #         for n in range(size):
    #             indexer = ind + n
    #             l_index[indexer] = left_index2[start - n]
    #             r_index[indexer] = r_ind
    #     return l_index, r_index

    # find earliest point where left_region1 > right_region1
    # that serves as our end boundary, and should reduce search space
    countss = np.empty(right_region1.size, dtype=np.intp)
    ends = np.empty(right_region1.size, dtype=np.intp)
    total_length = 0
    for num in prange(right_region1.size):
        start = positions[num]
        value = right_region1[num]
        end = -1
        for n in range(start, -1, -1):
            check = cummin_arr[n] > value
            if check:
                end = n
                break
        # get actual counts
        counts = 0
        ends[num] = end
        for ind in range(start, end, -1):
            check = left_region1[ind] <= value
            counts += check
        countss[num] = counts
        total_length += counts

    # build left and right indices
    starts = np.empty(right_region1.size, dtype=np.intp)
    starts[0] = 0
    starts[1:] = np.cumsum(countss)[:-1]
    l_index = np.empty(total_length, dtype=np.intp)
    r_index = np.empty(total_length, dtype=np.intp)

    for num in prange(right_region1.size):
        ind = starts[num]
        pos = positions[num]
        end = ends[num]
        size = countss[num]
        value = right_region1[num]
        r_ind = right_index2[num]
        for n in range(pos, end, -1):
            if not size:
                break
            check = left_region1[n] > value
            if check:
                continue
            l_index[ind] = left_index2[n]
            r_index[ind] = r_ind
            ind += 1
            size -= 1

    return l_index, r_index


def _numba_single_join(
    left: pd.Series,
    right: pd.Series,
    op: str,
    keep: str,
) -> tuple:
    """Return matching indices for single non-equi join."""

    outcome = _generic_func_cond_join(
        left=left,
        right=right,
        op=op,
        multiple_conditions=True,
        keep=keep,
    )

    if (outcome is None) or (op == _JoinOperator.NOT_EQUAL.value):
        return outcome

    left_index, right_index, search_indices = outcome

    if op in greater_than_join_types:
        starts = np.zeros(shape=search_indices.size, dtype=np.int8)
        ends = search_indices
        counts = search_indices
    else:
        ends = np.full(
            shape=search_indices.size,
            dtype=np.intp,
            fill_value=right_index.size,
        )
        starts = search_indices
        counts = ends - starts
    if keep == "all":
        return _get_indices_single(
            l_index=left_index,
            r_index=right_index,
            counts=counts,
            starts=starts,
            ends=ends,
        )

    if (
        (keep == "first")
        and (op in less_than_join_types)
        and pd.Series(right_index).is_monotonic_increasing
    ):
        return left_index, right_index[search_indices]
    if (
        (keep == "last")
        and (op in greater_than_join_types)
        and pd.Series(right_index).is_monotonic_increasing
    ):
        return left_index, right_index[search_indices - 1]
    if keep == "first":
        right_index = _numba_single_non_equi_keep_first(
            right_index, starts, ends
        )
    else:
        right_index = _numba_single_non_equi_keep_last(
            right_index, starts, ends
        )
    return left_index, right_index


@njit(parallel=True)
def _numba_single_non_equi_keep_first(
    right_index: np.ndarray, starts, ends
) -> np.ndarray:
    """
    Generate all indices when keep = `first`
    Applies only to >, >= , <, <= operators.
    """
    r_index = np.empty(starts.size, np.intp)
    for num in prange(starts.size):
        indexer = slice(starts[num], ends[num])
        r_index[num] = right_index[indexer].min()
    return r_index


@njit(parallel=True)
def _numba_single_non_equi_keep_last(
    right_index: np.ndarray, starts, ends
) -> np.ndarray:
    """
    Generate all indices when keep = `last`
    Applies only to >, >= , <, <= operators.
    """
    r_index = np.empty(starts.size, np.intp)
    for num in prange(starts.size):
        indexer = slice(starts[num], ends[num])
        r_index[num] = right_index[indexer].max()
    return r_index


@njit(cache=True, parallel=True)
def _get_indices_single(l_index, r_index, counts, starts, ends):
    """ "Compute indices when starts and ends are already known"""
    lengths = np.cumsum(counts)
    left_index = np.empty(lengths[-1], np.intp)
    right_index = np.empty(lengths[-1], np.intp)
    start_indices = np.empty(lengths.size, np.intp)
    start_indices[0] = 0
    start_indices[1:] = lengths[:-1]
    for num in prange(lengths.size):
        start = start_indices[num]
        width = counts[num]
        l_indexer = slice(start, start + width)
        left_index[l_indexer] = l_index[num]
        r_indexer = slice(starts[num], ends[num])
        right_index[l_indexer] = r_index[r_indexer]

    return left_index, right_index
