"""Various Functions powered by Numba"""

from __future__ import annotations

from math import ceil
from typing import Any, Union

import numpy as np
import pandas as pd
from numba import njit, prange

# https://numba.discourse.group/t/uint64-vs-int64-indexing-performance-difference/1500
# indexing with unsigned integers offers more performance
from janitor.functions.utils import (
    _generic_func_cond_join,
    greater_than_join_types,
)


# TODO: revert to former implementation
# which is def. faster than this?
def _numba_equi_join(
    df: pd.DataFrame, right: pd.DataFrame, eqs: list, gt_lt: list, keep: str
) -> Union[tuple[np.ndarray, np.ndarray], None]:
    """
    Compute indices when an equi join is present.
    """
    # implementation is based on the algorithm described in this paper -
    # https://www.scitepress.org/papers/2018/68268/68268.pdf
    # the algorithm described in the paper focuses on non-equi joins
    # with a tweak however, this can be extended to include equi-joins
    # also, unlike the non-equi joins where the left region <= right region
    # the equi join is strict -> left_region == right_region
    left_df = df[:]
    right_df = right[:]
    left_column, right_column, _ = eqs[0]
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
        right_index = right_df.index._values
        right_df.index = range(len(right))
    else:
        right_index = right_df.index._values
    shape = (len(left_df), len(gt_lt) + len(eqs))
    left_regions = np.empty(shape=shape, dtype=np.intp, order="F")
    l_booleans = np.zeros(len(df), dtype=np.intp)
    shape = (len(right_df), len(gt_lt) + len(eqs))
    right_regions = np.empty(shape=shape, dtype=np.intp, order="F")
    r_booleans = np.zeros(len(right), dtype=np.intp)
    for position, (left_column, right_column, op) in enumerate(eqs + gt_lt):
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
        l_booleans[left_indexer] += 1
        right_regions[right_indexer, position] = r_region
        r_booleans[right_indexer] += 1
    r_region = None
    search_indices = None
    left_df = None
    right_df = None
    booleans = l_booleans == len(gt_lt) + len(eqs)
    if not booleans.any():
        return None, None
    if not booleans.all():
        left_regions = left_regions[booleans]
        left_index = left_index[booleans]
    booleans = r_booleans == len(gt_lt) + len(eqs)
    if not booleans.any():
        return None, None
    if not booleans.all():
        right_regions = right_regions[booleans]
        right_index = right_index[booleans]
    l_booleans = None
    r_booleans = None
    starts = right_regions[:, 0].searchsorted(left_regions[:, 0], side="left")
    ends = right_regions[:, 0].searchsorted(left_regions[:, 0], side="right")
    booleans = starts < ends
    if not booleans.any():
        return None, None
    if not booleans.all():
        starts = starts[booleans]
        ends = ends[booleans]
        left_regions = left_regions[booleans]
        left_index = left_index[booleans]
    if keep == "all":
        return _numba_equi_join_keep_all(
            left_regions=left_regions[:, 1:],
            right_regions=right_regions[:, 1:],
            left_index=left_index,
            right_index=right_index,
            starts=starts,
            ends=ends,
            len_eqs=len(eqs) - 1,
            len_gt_lt=len(gt_lt),
        )
    if keep == "first":
        return _numba_equi_join_keep_first(
            left_regions=left_regions[:, 1:],
            right_regions=right_regions[:, 1:],
            left_index=left_index,
            right_index=right_index,
            starts=starts,
            ends=ends,
            len_eqs=len(eqs) - 1,
            len_gt_lt=len(gt_lt),
        )
    return _numba_equi_join_keep_last(
        left_regions=left_regions[:, 1:],
        right_regions=right_regions[:, 1:],
        left_index=left_index,
        right_index=right_index,
        starts=starts,
        ends=ends,
        len_eqs=len(eqs) - 1,
        len_gt_lt=len(gt_lt),
    )


@njit(cache=True, parallel=True)
def _numba_equi_join_keep_all(
    left_regions: np.ndarray,
    right_regions: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    len_eqs: int,
    len_gt_lt: int,
):
    """
    Get  indices where an equi join is present.
    """
    # two-pass solution
    # first pass gets the actual length
    # second pass fills the arrays with indices
    length = left_index.size
    total = 0
    l_booleans = np.zeros(length, dtype=np.bool_)
    for ind in prange(length):
        _ind = np.uint64(ind)
        start = starts[_ind]
        end = ends[_ind]
        boolean = False
        for num in range(start, end):
            _num = np.uint64(num)
            counter = 1
            for loc in range(len_eqs):
                loc_ = np.uint64(loc)
                next_left = left_regions[_ind, loc_]
                next_right = right_regions[_num, loc_]
                if next_left != next_right:
                    counter = 0
                    break
            if counter == 0:
                continue
            for loc in range(len_eqs, len_gt_lt + len_eqs):
                loc_ = np.uint64(loc)
                next_left = left_regions[_ind, loc_]
                next_right = right_regions[_num, loc_]
                if next_left > next_right:
                    counter = 0
                    break
            if counter == 0:
                continue
            total += 1
            boolean = True
        l_booleans[_ind] = boolean
    left_indices = np.empty(total, dtype=np.intp)
    right_indices = np.empty(total, dtype=np.intp)
    n = 0
    for ind in range(length):
        _ind = np.uint64(ind)
        if not l_booleans[_ind]:
            continue
        start = starts[_ind]
        end = ends[_ind]
        lindex = left_index[_ind]
        for num in range(start, end):
            _num = np.uint64(num)
            counter = 1
            for loc in range(len_eqs):
                loc_ = np.uint64(loc)
                next_left = left_regions[_ind, loc_]
                next_right = right_regions[_num, loc_]
                if next_left != next_right:
                    counter = 0
                    break
            if counter == 0:
                continue
            for loc in range(len_eqs, len_gt_lt + len_eqs):
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
    return left_indices, right_indices


@njit(cache=True, parallel=True)
def _numba_equi_join_keep_first(
    left_regions: np.ndarray,
    right_regions: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    len_eqs: int,
    len_gt_lt: int,
):
    """
    Get  indices where an equi join is present.
    """
    # two-pass solution
    # first pass gets the actual length
    # second pass fills the arrays with indices
    length = left_index.size
    total = 0
    l_booleans = np.zeros(length, dtype=np.bool_)
    r_indices = np.empty(length, dtype=np.intp)
    for ind in prange(length):
        _ind = np.uint64(ind)
        start = starts[_ind]
        end = ends[_ind]
        matches = 0
        base = -1
        for num in range(start, end):
            _num = np.uint64(num)
            counter = 1
            for loc in range(len_eqs):
                loc_ = np.uint64(loc)
                next_left = left_regions[_ind, loc_]
                next_right = right_regions[_num, loc_]
                if next_left != next_right:
                    counter = 0
                    break
            if counter == 0:
                continue
            for loc in range(len_eqs, len_gt_lt + len_eqs):
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
        total += 1
        l_booleans[_ind] = True
        r_indices[_ind] = base
    if total == 0:
        return None, None
    left_indices = np.empty(total, dtype=np.intp)
    right_indices = np.empty(total, dtype=np.intp)
    n = 0
    for ind in range(length):
        _ind = np.uint64(ind)
        if not l_booleans[_ind]:
            continue
        _n = np.uint64(n)
        left_indices[_n] = left_index[_ind]
        right_indices[_n] = r_indices[_ind]
        n += 1
    return left_indices, right_indices


@njit(cache=True, parallel=True)
def _numba_equi_join_keep_last(
    left_regions: np.ndarray,
    right_regions: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    len_eqs: int,
    len_gt_lt: int,
):
    """
    Get  indices where an equi join is present.
    """
    # two-pass solution
    # first pass gets the actual length
    # second pass fills the arrays with indices
    length = left_index.size
    total = 0
    l_booleans = np.zeros(length, dtype=np.bool_)
    r_indices = np.empty(length, dtype=np.intp)
    for ind in prange(length):
        _ind = np.uint64(ind)
        start = starts[_ind]
        end = ends[_ind]
        matches = 0
        base = -1
        for num in range(start, end):
            _num = np.uint64(num)
            counter = 1
            for loc in range(len_eqs):
                loc_ = np.uint64(loc)
                next_left = left_regions[_ind, loc_]
                next_right = right_regions[_num, loc_]
                if next_left != next_right:
                    counter = 0
                    break
            if counter == 0:
                continue
            for loc in range(len_eqs, len_gt_lt + len_eqs):
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
        total += 1
        l_booleans[_ind] = True
        r_indices[_ind] = base
    if total == 0:
        return None, None
    left_indices = np.empty(total, dtype=np.intp)
    right_indices = np.empty(total, dtype=np.intp)
    n = 0
    for ind in range(length):
        _ind = np.uint64(ind)
        if not l_booleans[_ind]:
            continue
        _n = np.uint64(n)
        left_indices[_n] = left_index[_ind]
        right_indices[_n] = r_indices[_ind]
        n += 1
    return left_indices, right_indices


@njit
def _numba_less_than(arr: np.ndarray, value: Any):
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
    # it is greater than
    # the max value in the array
    if min_idx == len(arr):
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
    return _numba_non_equi_join_monotonic_increasing_keep_all(
        left_regions=left_regions,
        right_regions=right_regions,
        left_index=left_index,
        right_index=right_index,
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
    # implementation is based on the algorithm described in this paper -
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
    l_booleans = np.zeros(len(df), dtype=np.intp)
    shape = (len(right_df), len(gt_lt))
    right_regions = np.empty(shape=shape, dtype=np.intp, order="F")
    r_booleans = np.zeros(len(right), dtype=np.intp)
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
        l_booleans[left_indexer] += 1
        right_regions[right_indexer, position] = r_region
        r_booleans[right_indexer] += 1
    r_region = None
    search_indices = None
    left_df = None
    right_df = None
    booleans = l_booleans == len(gt_lt)
    if not booleans.any():
        return None, None
    if not booleans.all():
        left_regions = left_regions[booleans]
        left_index = left_index[booleans]
    booleans = r_booleans == len(gt_lt)
    if not booleans.any():
        return None, None
    if not booleans.all():
        right_regions = right_regions[booleans]
        right_index = right_index[booleans]
    l_booleans = None
    r_booleans = None
    if gt_lt[0][-1] in greater_than_join_types:
        left_regions = left_regions[::-1]
        left_index = left_index[::-1]
        right_regions = right_regions[::-1]
        right_index = right_index[::-1]
        right_index_flipped = True
    else:
        right_index_flipped = False
    starts = right_regions[:, 0].searchsorted(left_regions[:, 0])
    booleans = starts < len(right_regions)
    if not booleans.any():
        return None, None
    if not booleans.all():
        starts = starts[booleans]
        left_regions = left_regions[booleans]
        left_index = left_index[booleans]
    arr = pd.Index(right_regions[:, 1])
    check_increasing = arr.is_monotonic_increasing
    check_decreasing = arr.is_monotonic_decreasing
    arr = None
    if check_increasing:
        search_indices = right_regions[:, 1].searchsorted(left_regions[:, 1])
        booleans = search_indices < len(right_regions)
        if not booleans.any():
            return None, None
        if not booleans.all():
            starts = starts[booleans]
            search_indices = search_indices[booleans]
            left_regions = left_regions[booleans]
            left_index = left_index[booleans]
        booleans = starts > search_indices
        starts = np.where(booleans, starts, search_indices)
        if len(gt_lt) == 2:
            ends = np.empty(left_index.size, dtype=np.intp)
            ends[:] = len(right_regions)
    elif check_decreasing:
        ends = right_regions[::-1, 1].searchsorted(left_regions[:, 1])
        booleans = starts < len(right_regions)
        if not booleans.any():
            return None, None
        if not booleans.all():
            starts = starts[booleans]
            left_regions = left_regions[booleans]
            left_index = left_index[booleans]
        ends = len(right_regions) - ends
        booleans = starts < ends
        if not booleans.any():
            return None, None
        if not booleans.all():
            starts = starts[booleans]
            left_regions = left_regions[booleans]
            left_index = left_index[booleans]
            ends = ends[booleans]
    booleans = None
    # return check_increasing, check_decreasing
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
    if (
        (check_decreasing | check_decreasing)
        & (len(gt_lt) == 2)
        & (keep == "all")
    ):
        start_indices = np.empty(left_index.size, dtype=np.intp)
        start_indices[0] = 0
        indices = (ends - starts).cumsum()
        start_indices[1:] = indices[:-1]
        indices = indices[-1]
        left_indices = np.empty(indices, dtype=np.intp)
        right_indices = np.empty(indices, dtype=np.intp)
        return _numba_non_equi_join_monotonic_keep_all_dual(
            left_index=left_index,
            right_index=right_index,
            starts=starts,
            ends=ends,
            left_indices=left_indices,
            right_indices=right_indices,
            start_indices=start_indices,
        )

    if (
        (check_decreasing | check_decreasing)
        & (len(gt_lt) == 2)
        & (keep == "first")
    ):
        left_indices = np.empty(left_index.size, dtype=np.intp)
        right_indices = np.empty(left_index.size, dtype=np.intp)
        return _numba_non_equi_join_monotonic_keep_first_dual(
            left_index=left_index,
            right_index=right_index,
            starts=starts,
            ends=ends,
            left_indices=left_indices,
            right_indices=right_indices,
        )

    if (
        (check_decreasing | check_decreasing)
        & (len(gt_lt) == 2)
        & (keep == "last")
    ):
        left_indices = np.empty(left_index.size, dtype=np.intp)
        right_indices = np.empty(left_index.size, dtype=np.intp)
        return _numba_non_equi_join_monotonic_keep_last_dual(
            left_index=left_index,
            right_index=right_index,
            starts=starts,
            ends=ends,
            left_indices=left_indices,
            right_indices=right_indices,
        )

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
        return _numba_non_equi_join_monotonic_increasing_keep_all(
            left_regions=left_regions[:, 2:],
            right_regions=right_regions[:, 2:],
            left_index=left_index,
            right_index=right_index,
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
        return _numba_non_equi_join_monotonic_keep_all(
            left_regions=left_regions[:, 2:],
            right_regions=right_regions[:, 2:],
            left_index=left_index,
            right_index=right_index,
            starts=starts,
            ends=ends,
        )
    # logic here is based on grantjenks' sortedcontainers
    # https://github.com/grantjenks/python-sortedcontainers
    load_factor = 1_000
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
    # first pass - get actual length
    length = left_index.size
    end = right_index.size
    end -= 1
    region = right_regions[np.uint64(end), 0]
    sorted_array[0, 0] = region
    positions_array[0, 0] = end
    maxes_counter = 1
    maxxes[0] = region
    lengths[0] = 1
    r_count = 0
    total = 0
    l_booleans = np.zeros(length, dtype=np.bool_)
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
                # the sorted array is an adaptation
                # of grantjenks' sortedcontainers
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
            total += 1
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
                total += 1
        l_booleans[_indexer] = True
        end = start
    if total == 0:
        return None, None
    # second pass - fill arrays with indices
    length = left_index.size
    end = right_index.size
    end -= 1
    region = right_regions[np.uint64(end), 0]
    sorted_array[0, 0] = region
    positions_array[0, 0] = end
    maxes_counter = 1
    maxxes[0] = region
    lengths[0] = 1
    r_count = 0
    left_indices = np.empty(total, dtype=np.intp)
    right_indices = np.empty(total, dtype=np.intp)
    begin = 0
    for indexer in range(length - 1, -1, -1):
        _indexer = np.uint64(indexer)
        if not l_booleans[_indexer]:
            continue
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
    return left_indices, right_indices


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
    """
    Get indices for non-equi join - first match
    """
    # first pass - get the actual length
    length = left_index.size
    end = right_index.size
    end -= 1
    region = right_regions[np.uint64(end), 0]
    sorted_array[0, 0] = region
    positions_array[0, 0] = end
    maxes_counter = 1
    maxxes[0] = region
    lengths[0] = 1
    r_count = 0
    total = 0
    l_booleans = np.zeros(length, dtype=np.bool_)
    r_indices = np.empty(length, dtype=np.intp)
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
                # the sorted array is an adaptation
                # of grantjenks' sortedcontainers
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
        base_index = -1
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
        total += 1
        l_booleans[_indexer] = True
        r_indices[_indexer] = base_index
        end = start
    if total == 0:
        return None, None
    # second pass - fill arrays with indices
    left_indices = np.empty(total, dtype=np.intp)
    right_indices = np.empty(total, dtype=np.intp)
    n = 0
    for ind in range(length):
        _ind = np.uint64(ind)
        if not l_booleans[_ind]:
            continue
        _n = np.uint64(n)
        left_indices[_n] = left_index[_ind]
        right_indices[_n] = r_indices[_ind]
        n += 1
    return left_indices, right_indices


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
    """
    Get indices for non-equi join - last match
    """
    # first pass - get the actual length
    length = left_index.size
    end = right_index.size
    end -= 1
    region = right_regions[np.uint64(end), 0]
    sorted_array[0, 0] = region
    positions_array[0, 0] = end
    maxes_counter = 1
    maxxes[0] = region
    lengths[0] = 1
    r_count = 0
    total = 0
    l_booleans = np.zeros(length, dtype=np.bool_)
    r_indices = np.empty(length, dtype=np.intp)
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
                # the sorted array is an adaptation
                # of grantjenks' sortedcontainers
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
        base_index = -1
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
        total += 1
        l_booleans[_indexer] = True
        r_indices[_indexer] = base_index
        end = start
    if total == 0:
        return None, None
    # second pass - fill arrays with indices
    left_indices = np.empty(total, dtype=np.intp)
    right_indices = np.empty(total, dtype=np.intp)
    n = 0
    for ind in range(length):
        _ind = np.uint64(ind)
        if not l_booleans[_ind]:
            continue
        _n = np.uint64(n)
        left_indices[_n] = left_index[_ind]
        right_indices[_n] = r_indices[_ind]
        n += 1
    return left_indices, right_indices


@njit(cache=True, parallel=True)
def _numba_non_equi_join_monotonic_keep_first(
    left_regions: np.ndarray,
    right_regions: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
):
    """
    Get  indices for a non equi join - first match.
    """
    length = left_index.size
    total = 0
    l_booleans = np.zeros(length, dtype=np.bool_)
    r_indices = np.empty(length, dtype=np.intp)
    for ind in prange(length):
        _ind = np.uint64(ind)
        start = starts[_ind]
        end = ends[_ind]
        matches = 0
        base = -1
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
        total += 1
        l_booleans[_ind] = True
        r_indices[_ind] = base
    if total == 0:
        return None, None
    n = 0
    left_indices = np.empty(total, dtype=np.intp)
    right_indices = np.empty(total, dtype=np.intp)
    for ind in prange(length):
        _ind = np.uint64(ind)
        if not l_booleans[_ind]:
            continue
        _n = np.uint64(n)
        left_indices[_n] = left_index[_ind]
        right_indices[_n] = r_indices[_ind]
        n += 1
    return left_indices, right_indices


@njit(cache=True, parallel=True)
def _numba_non_equi_join_monotonic_increasing_keep_first(
    left_regions: np.ndarray,
    right_regions: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    starts: np.ndarray,
):
    """
    Get  indices for a non equi join - first match
    """
    length = left_index.size
    end = len(right_regions)
    total = 0
    l_booleans = np.zeros(length, dtype=np.bool_)
    r_indices = np.empty(length, dtype=np.intp)
    for ind in prange(length):
        _ind = np.uint64(ind)
        start = starts[_ind]
        matches = 0
        base = -1
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
        total += 1
        l_booleans[_ind] = True
        r_indices[_ind] = base
    if total == 0:
        return None, None
    n = 0
    left_indices = np.empty(total, dtype=np.intp)
    right_indices = np.empty(total, dtype=np.intp)
    for ind in prange(length):
        _ind = np.uint64(ind)
        if not l_booleans[_ind]:
            continue
        _n = np.uint64(n)
        left_indices[_n] = left_index[_ind]
        right_indices[_n] = r_indices[_ind]
        n += 1
    return left_indices, right_indices


@njit(cache=True, parallel=True)
def _numba_non_equi_join_monotonic_keep_last(
    left_regions: np.ndarray,
    right_regions: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
):
    """
    Get  indices for a non equi join - last match.
    """
    length = left_index.size
    total = 0
    l_booleans = np.zeros(length, dtype=np.bool_)
    r_indices = np.empty(length, dtype=np.intp)
    for ind in prange(length):
        _ind = np.uint64(ind)
        start = starts[_ind]
        end = ends[_ind]
        matches = 0
        base = -1
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
        total += 1
        l_booleans[_ind] = True
        r_indices[_ind] = base
    if total == 0:
        return None, None
    n = 0
    left_indices = np.empty(total, dtype=np.intp)
    right_indices = np.empty(total, dtype=np.intp)
    for ind in prange(length):
        _ind = np.uint64(ind)
        if not l_booleans[_ind]:
            continue
        _n = np.uint64(n)
        left_indices[_n] = left_index[_ind]
        right_indices[_n] = r_indices[_ind]
        n += 1
    return left_indices, right_indices


@njit(cache=True, parallel=True)
def _numba_non_equi_join_monotonic_increasing_keep_last(
    left_regions: np.ndarray,
    right_regions: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    starts: np.ndarray,
):
    """
    Get  indices for a non equi join - last match.
    """
    length = left_index.size
    end = len(right_regions)
    total = 0
    l_booleans = np.zeros(length, dtype=np.bool_)
    r_indices = np.empty(length, dtype=np.intp)
    for ind in prange(length):
        _ind = np.uint64(ind)
        start = starts[_ind]
        matches = 0
        base = -1
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
        total += 1
        l_booleans[_ind] = True
        r_indices[_ind] = base
    if total == 0:
        return None, None
    n = 0
    left_indices = np.empty(total, dtype=np.intp)
    right_indices = np.empty(total, dtype=np.intp)
    for ind in prange(length):
        _ind = np.uint64(ind)
        if not l_booleans[_ind]:
            continue
        _n = np.uint64(n)
        left_indices[_n] = left_index[_ind]
        right_indices[_n] = r_indices[_ind]
        n += 1
    return left_indices, right_indices


@njit(cache=True, parallel=True)
def _numba_non_equi_join_monotonic_keep_all_dual(
    left_index: np.ndarray,
    right_index: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    start_indices: np.ndarray,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
):
    """
    Get indices for a dual non equi join
    """
    for ind in prange(left_index.size):
        _ind = np.uint64(ind)
        start = starts[_ind]
        end = ends[_ind]
        indexer = start_indices[_ind]
        lindex = left_index[_ind]
        for num in range(start, end):
            _num = np.uint64(num)
            rindex = right_index[_num]
            _indexer = np.uint64(indexer)
            left_indices[_indexer] = lindex
            right_indices[_indexer] = rindex
            indexer += 1
    return left_indices, right_indices


@njit(cache=True, parallel=True)
def _numba_non_equi_join_monotonic_keep_first_dual(
    left_index: np.ndarray,
    right_index: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
):
    """
    Get indices for a dual non equi join
    """
    for ind in prange(left_index.size):
        _ind = np.uint64(ind)
        start = starts[_ind]
        end = ends[_ind]
        lindex = left_index[_ind]
        base_index = right_index[np.uint64(start)]
        for num in range(start, end):
            _num = np.uint64(num)
            rindex = right_index[_num]
            if rindex < base_index:
                base_index = rindex
        left_indices[_ind] = lindex
        right_indices[_ind] = base_index
    return left_indices, right_indices


@njit(cache=True, parallel=True)
def _numba_non_equi_join_monotonic_keep_last_dual(
    left_index: np.ndarray,
    right_index: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
):
    """
    Get indices for a dual non equi join
    """
    for ind in prange(left_index.size):
        _ind = np.uint64(ind)
        start = starts[_ind]
        end = ends[_ind]
        lindex = left_index[_ind]
        base_index = right_index[np.uint64(start)]
        for num in range(start, end):
            _num = np.uint64(num)
            rindex = right_index[_num]
            if rindex > base_index:
                base_index = rindex
        left_indices[_ind] = lindex
        right_indices[_ind] = base_index
    return left_indices, right_indices


@njit(cache=True, parallel=True)
def _numba_non_equi_join_monotonic_keep_all(
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
    total = 0
    l_booleans = np.zeros(length, dtype=np.bool_)
    for ind in prange(length):
        _ind = np.uint64(ind)
        start = starts[_ind]
        end = ends[_ind]
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
            total += 1
            l_booleans[_ind] = True
    if total == 0:
        return None, None
    n = 0
    left_indices = np.empty(total, dtype=np.intp)
    right_indices = np.empty(total, dtype=np.intp)
    for ind in range(length):
        _ind = np.uint64(ind)
        if not l_booleans[_ind]:
            continue
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
    return left_indices, right_indices


@njit(cache=True, parallel=True)
def _numba_non_equi_join_monotonic_increasing_keep_all(
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
    end = len(right_regions)
    total = 0
    l_booleans = np.zeros(length, dtype=np.bool_)
    for ind in prange(length):
        _ind = np.uint64(ind)
        start = starts[_ind]
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
            total += 1
            l_booleans[_ind] = True
    if total == 0:
        return None, None
    n = 0
    left_indices = np.empty(total, dtype=np.intp)
    right_indices = np.empty(total, dtype=np.intp)
    for ind in range(length):
        _ind = np.uint64(ind)
        if not l_booleans[_ind]:
            continue
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
    return left_indices, right_indices
