"""Various Functions powered by Numba"""

import numpy as np
import pandas as pd
from janitor.functions.utils import _convert_to_numpy_array
from numba import njit, prange
from enum import Enum


class _JoinOperator(Enum):
    """
    List of operators used in conditional_join.
    """

    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_THAN_OR_EQUAL = ">="
    LESS_THAN_OR_EQUAL = "<="
    STRICTLY_EQUAL = "=="
    NOT_EQUAL = "!="


less_than_join_types = {
    _JoinOperator.LESS_THAN.value,
    _JoinOperator.LESS_THAN_OR_EQUAL.value,
}
greater_than_join_types = {
    _JoinOperator.GREATER_THAN.value,
    _JoinOperator.GREATER_THAN_OR_EQUAL.value,
}


def _numba_pair_le_lt(df: pd.DataFrame, right: pd.DataFrame, pair: list):
    """
    Numba implementation of algorithm in this paper:
    # https://www.scitepress.org/papers/2018/68268/68268.pdf
    Generally faster than the _range_indices algorithm
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
    # left_index      right_indes
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
    right_indices = []
    left_regions = []
    right_regions = []
    left_index = df.index
    right_index = right.index
    for num, (left_on, right_on, op) in enumerate(pair):
        # indexing is expensive; avoid it if possible
        if num:
            left_c = df.loc[left_index, left_on]
            right_c = right.loc[right_index, right_on]
        else:
            left_c = df[left_on]
            right_c = right[right_on]

        any_nulls = pd.isna(right_c)
        if any_nulls.all():
            return None
        if any_nulls.any():
            right_c = right_c[~any_nulls]
        any_nulls = pd.isna(left_c)
        if any_nulls.all():
            return None
        if any_nulls.any():
            left_c = left_c[~any_nulls]

        if op in less_than_join_types:
            left_is_sorted = pd.Series(left_c).is_monotonic_increasing
            if not left_is_sorted:
                left_c = left_c.sort_values(kind="stable", ascending=True)
        else:
            left_is_sorted = pd.Series(left_c).is_monotonic_decreasing
            if not left_is_sorted:
                left_c = left_c.sort_values(kind="stable", ascending=False)

        left_index = left_c.index._values
        right_index = right_c.index._values
        left_c, right_c = _convert_to_numpy_array(
            left_c._values, right_c._values
        )

        if op in {
            _JoinOperator.LESS_THAN.value,
            _JoinOperator.GREATER_THAN.value,
        }:
            strict = 1
        else:
            strict = 0
        if op in less_than_join_types:
            op_code = 1
        else:
            op_code = 0

        result = _get_regions(
            left_c, left_index, right_c, right_index, strict, op_code
        )
        if result is None:
            return None
        (
            left_index,
            right_index,
            left_region,
            right_region,
        ) = result

        left_indices.append(left_index)
        right_indices.append(right_index)
        left_regions.append(left_region)
        right_regions.append(right_region)

    def _realign(indices, regions):
        """
        Realign the indices and regions
        obtained from _get_regions.
        """
        # we've got two regions, since we have two pairs
        # there might be region numbers missing from either
        # or misalignment of the arrays
        # this function ensures the regions are properly aligned
        arr1, arr2 = indices
        region1, region2 = regions
        indexer = pd.Index(arr2).get_indexer(arr1)
        mask = indexer == -1
        if mask.any():
            arr1 = arr1[~mask]
            region1 = region1[~mask]
            indexer = indexer[~mask]
        region2 = region2[indexer]
        return arr1, region1, region2

    l_index, l_table1, l_table2 = _realign(left_indices, left_regions)
    r_index, r_table1, r_table2 = _realign(right_indices, right_regions)

    del (
        left_indices,
        left_regions,
        right_indices,
        right_regions,
        left_region,
        right_region,
    )

    # we'll be running a for loop to check sub arrays
    # to see if the region from the left is less than
    # the region on the right
    # sorting here allows us to search each first level
    # array more efficiently with a binary search
    if not pd.Series(r_table1).is_monotonic_increasing:
        indexer = np.lexsort((r_table2, r_table1))
        r_index, r_table1, r_table2 = (
            right_index[indexer],
            r_table1[indexer],
            r_table2[indexer],
        )

    indexer = None

    positions = r_table1.searchsorted(l_table1, side="left")
    # anything equal to the length of r_table1
    # implies exclusion
    bools = positions < r_table1.size
    if not bools.any():
        return None
    if not bools.all():
        positions = positions[bools]
        l_index = l_index[bools]
        l_table2 = l_table2[bools]

    # find the maximum from the bottom upwards
    # if value from l_table2 is greater than the maximum
    # there is no point searching within that space
    max_arr = np.maximum.accumulate(r_table2[::-1])[::-1]
    bools = l_table2 > max_arr[positions]
    # there is no match
    if bools.all():
        return None
    if bools.any():
        positions = positions[~bools]
        l_index = l_index[~bools]
        l_table2 = l_table2[~bools]

    return _get_matching_indices(
        l_index, l_table2, r_index, r_table2, positions, max_arr
    )


def _numba_single_join(
    left: pd.Series,
    right: pd.Series,
    strict: bool,
    keep: str,
    op_code: int,
) -> tuple:
    """Return matching indices for single non-equi join."""
    if op_code == -1:
        # for the not equal join, we combine indices
        # from strictly less than and strictly greater than indices
        # as well as indices for nulls, if any
        left_nulls, right_nulls = _numba_not_equal_indices(left, right)
        dummy = np.array([], dtype=int)
        result = _numba_less_than_indices(left, right)
        if result is None:
            lt_left = dummy
            lt_right = dummy
        else:
            lt_left, lt_right = _numba_generate_indices_ne(
                *result, strict, keep, op_code=1
            )
        result = _numba_greater_than_indices(left, right)
        if result is None:
            gt_left = dummy
            gt_right = dummy
        else:
            gt_left, gt_right = _numba_generate_indices_ne(
                *result, strict, keep, op_code=0
            )
        left = np.concatenate([lt_left, gt_left, left_nulls])
        right = np.concatenate([lt_right, gt_right, right_nulls])
        if (not left.size) & (not right.size):
            return None
        if keep == "all":
            return left, right
        indexer = np.argsort(left)
        left, pos = np.unique(left[indexer], return_index=True)
        if keep == "first":
            right = np.minimum.reduceat(right[indexer], pos)
        else:
            right = np.maximum.reduceat(right[indexer], pos)
        return left, right

    # convert Series to numpy arrays
    # get the regions for left and right
    # get the total count of indices
    # build the final left and right indices
    if op_code == 1:
        result = _numba_less_than_indices(left, right)
    else:
        result = _numba_greater_than_indices(left, right)
    if result is None:
        return None
    result = _get_regions(*result, strict, op_code)
    if result is None:
        return None
    left_index, right_index, left_region, right_region = result
    # numpy version of pandas monotonic increasing
    bools = np.all(right_region[1:] >= right_region[:-1])
    if not bools:
        indexer = np.lexsort((right_index, right_region))
        right_region = right_region[indexer]
        right_index = right_index[indexer]
    positions = right_region.searchsorted(left_region, side="left")
    if keep == "all":
        # get actual length of left and right indices
        counts = right_region.size - positions
        counts = counts.cumsum()
        return _numba_single_non_equi(
            left_index, right_index, counts, positions
        )
    return _numba_single_non_equi_keep_first_last(
        left_index, right_index, positions, keep
    )


def _numba_generate_indices_ne(
    left: np.ndarray,
    left_index: np.ndarray,
    right: np.ndarray,
    right_index: np.ndarray,
    strict: bool,
    keep: str,
    op_code: int,
) -> tuple:
    """
    Generate indices within a not equal join,
    for either greater or less than.
    if op_code is 1, that is a less than operation,
    if op_code is 0, that is a greater than operation.
    """
    dummy = np.array([], dtype=int)
    result = _get_regions(
        left, left_index, right, right_index, strict, op_code
    )
    if result is None:
        return dummy, dummy
    left_index, right_index, left_region, right_region = result
    # numpy version of pandas monotonic increasing
    bools = np.all(right_region[1:] >= right_region[:-1])
    if not bools:
        indexer = np.lexsort((right_index, right_region))
        right_region = right_region[indexer]
        right_index = right_index[indexer]
    positions = right_region.searchsorted(left_region, side="left")
    if keep == "all":
        # get actual length of left and right indices
        counts = right_region.size - positions
        counts = counts.cumsum()
        return _numba_single_non_equi(
            left_index, right_index, counts, positions
        )
    return _numba_single_non_equi_keep_first_last(
        left_index, right_index, positions, keep
    )


def _numba_not_equal_indices(left_c: pd.Series, right_c: pd.Series) -> tuple:
    """
    Preparatory function for _numba_single_join
    This retrieves the indices for nulls, if any.
    """

    dummy = np.array([], dtype=int)

    # deal with nulls
    l1_nulls = dummy
    r1_nulls = dummy
    l2_nulls = dummy
    r2_nulls = dummy
    any_left_nulls = left_c.isna()
    any_right_nulls = right_c.isna()
    if any_left_nulls.any():
        l1_nulls = left_c.index[any_left_nulls]
        l1_nulls = l1_nulls._values
        r1_nulls = right_c.index
        # avoid NAN duplicates
        if any_right_nulls.any():
            r1_nulls = r1_nulls[~any_right_nulls]
        r1_nulls = r1_nulls._values
        nulls_count = l1_nulls.size
        # blow up nulls to match length of right
        l1_nulls = np.tile(l1_nulls, r1_nulls.size)
        # ensure length of right matches left
        if nulls_count > 1:
            r1_nulls = np.repeat(r1_nulls, nulls_count)
    if any_right_nulls.any():
        r2_nulls = right_c.index[any_right_nulls]
        r2_nulls = r2_nulls._values
        l2_nulls = left_c.index
        nulls_count = r2_nulls.size
        # blow up nulls to match length of left
        r2_nulls = np.tile(r2_nulls, l2_nulls.size)
        # ensure length of left matches right
        if nulls_count > 1:
            l2_nulls = np.repeat(l2_nulls, nulls_count)
    l1_nulls = np.concatenate([l1_nulls, l2_nulls])
    r1_nulls = np.concatenate([r1_nulls, r2_nulls])

    return l1_nulls, r1_nulls


def _numba_less_than_indices(
    left: pd.Series,
    right: pd.Series,
) -> tuple:
    """
    Preparatory function for _numba_single_join
    """

    if left.min() > right.max():
        return None
    any_nulls = pd.isna(left)
    if any_nulls.all():
        return None
    if any_nulls.any():
        left = left[~any_nulls]
    if not left.is_monotonic_increasing:
        left = left.sort_values(kind="stable", ascending=True)
    any_nulls = pd.isna(right)
    if any_nulls.all():
        return None
    if any_nulls.any():
        right = right[~any_nulls]
    any_nulls = None
    left_index = left.index._values
    right_index = right.index._values
    left, right = _convert_to_numpy_array(left._values, right._values)
    return left, left_index, right, right_index


def _numba_greater_than_indices(
    left: pd.Series,
    right: pd.Series,
) -> tuple:
    """
    Preparatory function for _numba_single_join
    """
    if left.max() < right.min():
        return None

    any_nulls = pd.isna(left)
    if any_nulls.all():
        return None
    if any_nulls.any():
        left = left[~any_nulls]
    if not left.is_monotonic_decreasing:
        left = left.sort_values(kind="stable", ascending=False)
    any_nulls = pd.isna(right)
    if any_nulls.all():
        return None
    if any_nulls.any():
        right = right[~any_nulls]

    any_nulls = None
    left_index = left.index._values
    right_index = right.index._values
    left, right = _convert_to_numpy_array(left._values, right._values)
    return left, left_index, right, right_index


@njit(parallel=True)
def _numba_single_non_equi(
    left_index: np.ndarray,
    right_index: np.ndarray,
    counts: np.ndarray,
    positions: np.ndarray,
) -> tuple:
    """
    Generate all indices when keep = `all`.
    Applies only to >, >= , <, <= operators.
    """
    length = left_index.size
    len_right = right_index.size
    l_index = np.empty(counts[-1], np.intp)
    r_index = np.empty(counts[-1], np.intp)
    starts = np.empty(length, np.intp)
    # capture the starts and ends for each sub range
    starts[0] = 0
    starts[1:] = counts[:-1]
    # build the actual indices
    for num in prange(length):
        val = left_index[num]
        pos = positions[num]
        posn = starts[num]
        for ind in range(pos, len_right):
            r_index[posn] = right_index[ind]
            l_index[posn] = val
            posn += 1
    return l_index, r_index


@njit(parallel=True)
def _numba_single_non_equi_keep_first_last(
    left_index: np.ndarray,
    right_index: np.ndarray,
    positions: np.ndarray,
    keep: str,
) -> tuple:
    """
    Generate all indices when keep = `first` or `last`
    Applies only to >, >= , <, <= operators.
    """
    length = left_index.size
    l_index = np.empty(length, np.intp)
    r_index = np.empty(length, np.intp)

    len_right = right_index.size
    for num in prange(length):
        val = left_index[num]
        pos = positions[num]
        base_val = right_index[pos]
        for ind in range(pos + 1, len_right):
            value = right_index[ind]
            if keep == "first":
                bool_scalar = value < base_val
            else:
                bool_scalar = value > base_val
            if bool_scalar:
                base_val = value
        l_index[num] = val
        r_index[num] = base_val
    return l_index, r_index


@njit(cache=True, parallel=True)
def _get_matching_indices(
    l_index, l_table2, r_index, r_table2, positions, max_arr
):
    """
    Retrieves the matching indices
    for the left and right regions.
    Strictly for non-equi joins (pairs).
    """
    length = r_index.size
    counts = np.empty(positions.size, dtype=np.intp)
    ends = np.empty(positions.size, dtype=np.intp)

    # first let's get the total number of matches
    for num in prange(l_index.size):
        l2 = l_table2[num]
        pos = positions[num]
        end = 0
        pos_end = length
        # get the first point where l2
        # is less than the cumulative max
        # that will serve as the range
        # (pos, pos_end)
        # within which to search for actual matches
        for ind in range(pos + 1, length):
            val = max_arr[ind]
            if l2 > val:
                pos_end = ind
                break
        ends[num] = pos_end
        # get the total number of exact matches
        # for l2
        for ind in range(pos, pos_end):
            out = r_table2[ind]
            out = l2 <= out
            end += out
        counts[num] = end

    start_indices = np.cumsum(counts)
    starts = np.empty(counts.size, dtype=np.intp)
    starts[0] = 0
    starts[1:] = start_indices[:-1]

    # create left and right indexes
    left_index = np.empty(start_indices[-1], dtype=np.intp)
    right_index = np.empty(start_indices[-1], dtype=np.intp)
    start_indices = None

    for num in prange(l_index.size):
        pos = positions[num]
        pos_end = ends[num]
        l2 = l_table2[num]
        l3 = l_index[num]
        start = starts[num]
        counter = counts[num]
        for ind in range(pos, pos_end):
            if not counter:
                break
            if r_table2[ind] < l2:
                continue
            left_index[start] = l3
            right_index[start] = r_index[ind]
            start += 1
            counter -= 1

    return left_index, right_index


@njit()
def _get_regions(
    left_c: np.ndarray,
    left_index: np.ndarray,
    right_c: np.ndarray,
    right_index: np.ndarray,
    strict: int,
    op_code: int,
) -> tuple:
    """
    Get the regions where left_c and right_c converge.
    Strictly for non-equi joins,
    specifically  -->  >, >= , <, <= operators.
    """
    # The idea is to group values within regions.
    # An example:
    # left_array: [2, 5, 7]
    # right_array: [0, 3, 7]
    # if the join is left_array <= right_array
    # we should have pairs (0), (2,3),(5,7),(5,7)
    # since no value is less than 0, we can discard that
    # our final regions should be
    #  (2,3) --->  0
    #  (5,7) --->  1
    #  (7,7) --->  1
    #  based on the regions, we can see that any value in
    #  region 0 will be less than 1 ---> 2 <= 3, 7
    #  region 1 values are the end ---> 5 <= 7 & 7 <= 7
    #  if the join is left_array >= right_array
    #  then the left_array is sorted in descending order
    #  and the final pairs should be :
    #  (7, 7), (5, 3), (2, 0)
    # our final regions should be
    #  (7,7) --->  0
    #  (5,3) --->  1
    #  (2,0) --->  2
    #  based on the regions, we can see that any value in
    #  region 0 will be greater than 1 and 2 ---> 7 >= 7, 5, 0
    #  region 1 values will be greater than 2 ---> 5 >= 3, 0
    #  region 2 values are the end ----> 2 >= 0
    #  this concept becomes more relevant when two non equi conditions
    #  are present ---> l1 < r1 & l2 > r2
    #  For two non equi conditions, the matches are where
    #  the regions from group A (l1 < r1)
    #  are also lower than the regions from group B (l2 > r2)
    #  This implementation is based on the algorithm outlined here:
    #  https://www.scitepress.org/papers/2018/68268/68268.pdf
    indices = _search_indices(left_c, right_c, strict, op_code)
    left_region = np.empty(left_c.size, dtype=np.intp)
    left_region[:] = -1
    max_indices = indices.max() + 1
    if max_indices < left_index.size:
        left_region = left_region[:max_indices]
        left_index = left_index[:max_indices]
    mask = indices == -1
    if mask.all():
        return None
    if mask.any():
        right_index = right_index[~mask]
        indices = indices[~mask]
    left_region[indices] = 1
    mask = left_region == 1
    count_unique_indices = np.bincount(indices)
    count_unique_indices = np.count_nonzero(count_unique_indices)
    left_region[mask] = np.arange(count_unique_indices)
    start = left_region[-1]
    arr = np.arange(left_region.size)[::-1]
    for num in arr:
        if left_region[num] != -1:
            start = left_region[num]
        else:
            left_region[num] = start
    right_region = left_region[indices]
    return left_index, right_index, left_region, right_region


@njit(parallel=True)
def _search_indices(
    left_c: np.ndarray, right_c: np.ndarray, strict: int, op_code: int
) -> np.ndarray:
    """
    Get search indices for non-equi joins
    """
    indices = np.empty(right_c.size, dtype=np.intp)
    for num in prange(right_c.size):
        value = right_c[num]
        if strict:
            high = _searchsorted_left(left_c, value, op_code)
        else:
            high = _searchsorted_right(left_c, value, op_code)
        indices[num] = high

    return indices


# the binary search functions below are modifications
# of python's bisect function, with a simple aim of
# getting regions.
# region should always be at the far end left or right
# e.g for [2, 3, 3, 4], 3 (if <) should be position 0
# if (<=), position should be 1
# for [4, 3, 2, 2], 3 (if >) should be position 0
# if >=, position should be 1
@njit()
def _searchsorted_left(arr: np.ndarray, value: int, op_code: int) -> int:
    """
    Modification of Python's bisect_left function.
    Used to get the relevant region
    within the _get_regions function.
    """
    high = len(arr)
    low = 0
    while low < high:
        mid = (low + high) // 2
        if op_code:
            check = arr[mid] < value
        else:
            check = arr[mid] > value
        if check:
            low = mid + 1
        else:
            high = mid
    return high - 1


@njit()
def _searchsorted_right(arr: np.ndarray, value: int, op_code: int) -> int:
    """
    Modification of Python's bisect_right function.
    Used to get the relevant region
    within the _get_regions function.
    """
    high = len(arr)
    low = 0
    while low < high:
        mid = (low + high) // 2
        if op_code:
            check = value < arr[mid]
        else:
            check = value > arr[mid]
        if check:
            high = mid
        else:
            low = mid + 1
    return low - 1
