import janitor_rs
import numpy as np
import pandas as pd

from janitor.functions._conditional_join import (
    _greater_than_indices,
    _helpers,
    _less_than_indices,
)

# description below is based on multiple non-equi joins (>/>=/</<=)
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
# and if we index the dataframes,
# we should get the output below:
#################################
#    value_1  value_2A  value_2B
# 0        2         1         3
# 1        5         3         6
# 2        3         2         4
# 3        4         3         5
# 4        4         3         6
################################


def _get_indices(
    df: pd.DataFrame, right: pd.DataFrame, first_condition, second_condition
) -> dict:
    """
    Compute indices for a dual non-equi join
    """
    (left_on, right_on, op) = first_condition
    left_col = df[left_on]
    # sorting is done here to enable easy region filtering later on
    # by starting from the highest region
    left_col, _ = _helpers._sort_if_not_monotonic(series=left_col)
    right_col = right[right_on]
    outcome = _build_regions(left=left_col, right=right_col, op=op)
    if outcome is None:
        return None
    l1_index, l1_region, r1_index, r1_region = outcome
    (left_on, right_on, op) = second_condition
    left_col = df.loc[l1_index, left_on]
    right_col = right.loc[r1_index, right_on]
    outcome = _build_regions(left=left_col, right=right_col, op=op)
    if outcome is None:
        return None
    l2_index, l2_region, r2_index, r2_region = outcome
    outcome = _align_regions(
        left_index=l1_index,
        right_index=l2_index,
        left_region=l1_region,
        right_region=l2_region,
    )
    if outcome is None:
        return None
    l_index, l1_region, l2_region = outcome
    outcome = _align_regions(
        left_index=r1_index,
        right_index=r2_index,
        left_region=r1_region,
        right_region=r2_region,
    )
    if outcome is None:
        return None
    r_index, r1_region, r2_region = outcome
    # flip to ensure region filtering
    # starts from the highest search position
    if not pd.Index(l1_region).is_monotonic_decreasing:
        l1_region = l1_region[::-1]
        l_index = l_index[::-1]
        l2_region = l2_region[::-1]
    # due to our approach above,
    # r1_region is guaranteed to be sorted
    search_indices = r1_region.searchsorted(l1_region)
    # we keep only rows where
    # l_region <= r_region
    booleans = search_indices == r1_region.size
    if booleans.all():
        return None
    if booleans.any():
        booleans = ~booleans
        l_index = l_index[booleans]
        l2_region = l2_region[booleans]
        search_indices = search_indices[booleans]
    # exclude l2_region > r2_region's max
    max_region = r2_region.max()
    booleans = l2_region > max_region
    if booleans.all():
        return None
    if booleans.any():
        booleans = ~booleans
        l_index = l_index[booleans]
        l2_region = l2_region[booleans]
        search_indices = search_indices[booleans]
    positions, counts_array, total = janitor_rs.get_positions_where_left_le_right(
        left=l2_region,
        right=r2_region,
        starts=search_indices,
        max_right=max_region,
    )
    return {
        "left_index": l_index,
        "right_index": r_index,
        "positions": positions,
        "counts_array": counts_array,
        "total": total,
    }


def _build_regions(left: pd.Series, right: pd.Series, op: str) -> tuple:
    """Compute regions"""
    right, _ = _helpers._sort_if_not_monotonic(series=right)
    lt_or_le_check = op in _helpers.less_than_join_types
    if lt_or_le_check:
        outcome = _less_than_indices._le_lt_indices(
            left=left._values,
            left_index=left.index._values,
            right=right._values,
            strict=op == "<",
        )
    else:
        outcome = _greater_than_indices._ge_gt_indices(
            left=left._values,
            left_index=left.index._values,
            right=right._values,
            strict=op == ">",
        )
    if outcome is None:
        return None
    left_index, search_indices = outcome
    right_index = right.index._values
    if not lt_or_le_check:
        # computation here is to ensure alignment
        # since right should be in descending order
        # for >/>=
        search_indices = right_index.size - search_indices
        right_index = right_index[::-1]

    right_region = np.zeros(right_index.size, dtype=np.int64)
    right_region[search_indices] = 1
    right_region[0] -= 1
    right_region = right_region.cumsum()
    left_region = right_region[search_indices]
    # exclude regions that definitely do not have a match
    booleans = right_region > -1
    if not booleans.all():
        right_region = right_region[booleans]
        right_index = right_index[booleans]
    return left_index, left_region, right_index, right_region


def _align_regions(
    left_index: np.ndarray,
    right_index: np.ndarray,
    left_region: np.ndarray,
    right_region: np.ndarray,
) -> tuple:
    "Ensure regions are aligned"
    # right_index is always going to be <= left_index
    # based on previous computations,
    # right_index will always be a subset of left_index
    # as such, right_index becomes the control point
    positions = pd.Index(right_index).get_indexer(left_index)
    booleans = positions > -1
    if not booleans.any():
        return None
    left_index = left_index[booleans]
    left_region = left_region[booleans]
    positions = positions[booleans]
    right_index = right_index[positions]
    right_region = right_region[positions]
    return left_index, left_region, right_region


def _build_indices(
    left_index: np.ndarray,
    right_index: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    positions: np.ndarray,
    counts_array: np.ndarray,
    total: int,
    keep: str,
):
    """
    Build indices for a dual join
    """
    if keep == "all":
        left_index = janitor_rs.repeat_index(
            index=left_index, counts=counts_array, length=total
        )
        right_index = janitor_rs.build_positional_index(
            index=right_index, positions=positions, length=total
        )
    elif keep == "first":
        total = np.count_nonzero(counts_array)
        left_index = janitor_rs.trim_index(
            index=left_index, counts=counts_array, length=total
        )
        right_index = janitor_rs.build_positional_index_first(
            index=right_index,
            starts=starts,
            ends=ends,
            counts=counts_array,
            positions=positions,
            length=total,
        )
    else:
        total = np.count_nonzero(counts_array)
        left_index = janitor_rs.trim_index(
            index=left_index, counts=counts_array, length=total
        )
        right_index = janitor_rs.build_positional_index_last(
            index=right_index,
            starts=starts,
            ends=ends,
            counts=counts_array,
            positions=positions,
            length=total,
        )
    return {"left_index": left_index, "right_index": right_index}
