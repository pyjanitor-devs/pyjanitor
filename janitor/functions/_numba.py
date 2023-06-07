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
    def _get_regions(
        df: pd.DataFrame,
        right: pd.DataFrame,
        left_on: str,
        right_on: str,
        op: str,
    ):
        mapping = {">=": "<=", ">": "<", "<=": ">=", "<": ">"}
        left_index = df.index._values
        right_index = right.index._values
        left_c = df[left_on]
        right_c = right[right_on]
        outcome = _generic_func_cond_join(
            left=right_c,
            right=left_c,
            op=mapping[op],
            multiple_conditions=True,
            keep="all",
        )
        if not outcome:
            return None
        right_index, left_index, search_indices = outcome

        if op in greater_than_join_types:
            left_index = left_index[::-1]
            search_indices = left_index.size - search_indices
        # logic for computing regions
        # relies on binary search
        # subtract 1 from search indices
        # to align it with the lowest value for left region
        # say we had 2, 3, 5, 8, for the left region
        # and 6 for the right region
        # and is a < operation
        # a binary search returns 3
        # subtracting 1, yields 2,
        # which pairs it correctly with 5
        # since 2, 3, 5 are less than 6
        # if it was a > operation
        # first a subtraction from len(left) -> 4 - 3
        # which yields 1
        # flipping the left region in descending order
        # -> 8, 3, 5 ,2
        # subtract 1 from the search index yields 0
        # which correctly pairs with 8,
        # since 8 is the first closest number greater than 6
        # from here on we can compute the regions
        search_indices -= 1
        left_region = np.zeros(left_index.size, np.intp)
        left_region[search_indices] = 1
        max_position = np.max(search_indices) + 1
        # exclude values from left that are not ahead
        # of values from the right
        # e.g left -> [7,5, 3], right -> 4
        # 3 is not greater than 4, so we exclude it
        # left becomes -> [7,5]
        # if left -> [3,5,7], right -> 6
        # we exclude 7 since it is not less than 6
        # left becomes -> [3,5]
        if max_position < left_index.size:
            left_index = left_index[:max_position]
            left_region = left_region[:max_position]
        # compute regions from the end
        left_region[-1] = left_region.sum() - left_region[-1]
        left_region = np.subtract.accumulate(left_region[::-1])[::-1]
        right_region = left_region[search_indices]
        return left_index, left_region, right_index, right_region

    def _realign(
        index1: np.ndarray,
        index2: np.ndarray,
        region1: np.ndarray,
        region2: np.ndarray,
    ):
        """
        Realign the indices and regions
        obtained from _get_regions.
        """
        # we've got two regions, since we have two pairs
        # there might be region numbers missing from either
        # or misalignment of the arrays
        # this function ensures the regions are properly aligned
        indexer = pd.Index(index1).get_indexer(index2)
        mask = indexer == -1
        if mask.all():
            return None
        if mask.any():
            index2 = index2[~mask]
            region2 = region2[~mask]
            indexer = indexer[~mask]
        region1 = region1[indexer]
        return index2, region1, region2

    outcome1 = _get_regions(df, right, *pair[0])
    if outcome1 is None:
        return None

    outcome2 = _get_regions(df, right, *pair[1])
    if outcome2 is None:
        return None

    left_indices, left_regions, right_indices, right_regions = zip(
        outcome1, outcome2
    )
    outcome = _realign(*left_indices, *left_regions)
    if not outcome:
        return None
    left_index, left_region1, left_region2 = outcome
    outcome = _realign(*right_indices, *right_regions)
    if not outcome:
        return None
    right_index, right_region1, right_region2 = outcome

    # get positions where right_region2 is greater than left_region2
    # serves as end search point
    search_indices = left_region2.searchsorted(right_region2, side="right")
    # return left_region1, right_region1, search_indices
    # right_region2 should be greater than the minimum within that space
    # this is where numba comes in
    # to improve performance for the for-loop

    # shortcut if left_region1 is already cumulative_decreasing
    if pd.Series(left_region1).is_monotonic_decreasing:
        return _get_indices_dual_monotonic_decreasing(
            left_region=left_region1,
            right_region=right_region1,
            left_index=left_index,
            right_index=right_index,
            search_indices=search_indices,
        )

    return _get_indices_dual(
        left_region=left_region1,
        right_region=right_region1,
        left_index=left_index,
        right_index=right_index,
        search_indices=search_indices,
        cummin_arr=np.minimum.accumulate(left_region1),
    )


@njit(parallel=True)
def _get_indices_dual_monotonic_decreasing(
    left_region: np.ndarray,
    right_region: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    search_indices: np.ndarray,
):
    """
    Retrieve matching indices. Applies to dual non-equi joins.
    It is assumed that the left_region is cumulative decreasing.
    """
    length = right_region.size
    sizes = np.empty(length, dtype=np.intp)
    counts = 0
    for num in prange(length):
        end = search_indices[num]
        arr = left_region[:end]
        pos = arr.size - np.searchsorted(
            arr[::-1], right_region[num], side="right"
        )
        size = end - pos
        counts += size
        sizes[num] = size

    starts = np.empty(right_region.size, dtype=np.intp)
    starts[0] = 0
    starts[1:] = np.cumsum(sizes)[:-1]
    l_index = np.empty(counts, dtype=np.intp)
    r_index = np.empty(counts, dtype=np.intp)
    for num in prange(right_region.size):
        ind = starts[num]
        start = search_indices[num] - 1
        size = sizes[num]
        r_ind = right_index[num]
        for n in range(size):
            indexer = ind + n
            l_index[indexer] = left_index[start - n]
            r_index[indexer] = r_ind
    return l_index, r_index


@njit(parallel=True)
def _get_indices_dual(
    left_region: np.ndarray,
    right_region: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    search_indices: np.ndarray,
    cummin_arr: np.ndarray,
):
    """
    Retrieves the matching indices
    for the left and right regions.
    Strictly for non-equi joins (pairs).
    """
    # find earliest point where left_region1 > right_region1
    # that serves as our end boundary, and should reduce search space
    countss = np.empty(right_region.size, dtype=np.intp)
    ends = np.empty(right_region.size, dtype=np.intp)
    total_length = 0
    for num in prange(right_region.size):
        start = search_indices[num]
        value = right_region[num]
        end = -1
        for n in range(start - 1, -1, -1):
            check = cummin_arr[n] > value
            if check:
                end = n
                break
        # get actual counts
        counts = 0
        ends[num] = end
        for ind in range(start - 1, end, -1):
            check = left_region[ind] <= value
            counts += check
        countss[num] = counts
        total_length += counts

    # build left and right indices
    starts = np.empty(right_region.size, dtype=np.intp)
    starts[0] = 0
    starts[1:] = np.cumsum(countss)[:-1]
    l_index = np.empty(total_length, dtype=np.intp)
    r_index = np.empty(total_length, dtype=np.intp)

    for num in prange(right_region.size):
        ind = starts[num]
        pos = search_indices[num]
        end = ends[num]
        size = countss[num]
        value = right_region[num]
        r_ind = right_index[num]
        for n in range(pos - 1, end, -1):
            if not size:
                break
            check = left_region[n] > value
            if check:
                continue
            l_index[ind] = left_index[n]
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
    right_index: np.ndarray, starts: np.ndarray, ends: np.ndarray
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
    right_index: np.ndarray, starts: np.ndarray, ends: np.ndarray
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
def _get_indices_single(
    l_index: np.ndarray,
    r_index: np.ndarray,
    counts: int,
    starts: np.ndarray,
    ends: np.ndarray,
):
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
