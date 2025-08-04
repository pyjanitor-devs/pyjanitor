import math

import numpy as np
import pandas as pd

from . import helpers


def _numba_multiple_non_equi_join(
    df: pd.DataFrame,
    right: pd.DataFrame,
    gt_lt: list,
    keep: str,
    is_range_join: bool,
    row_count: str = None,
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
    mapping = {">": 0, ">=": 1, "<": 2, "<=": 3, "!=": 4}
    first, second, *rest = gt_lt
    if right[first[1]].is_monotonic_increasing:
        right_is_sorted = True
    else:
        right_is_sorted = False
        right = right.sort_values([first[1], second[1]], ignore_index=False)
    if is_range_join & right[second[1]].is_monotonic_increasing:
        return _range_join_sorted(
            first=first,
            second=second,
            df=df,
            right=right,
            keep=keep,
            gt_lt=gt_lt,
            mapping=mapping,
            rest=rest,
            right_is_sorted=right_is_sorted,
            row_count=row_count,
        )
    if not df[first[0]].is_monotonic_increasing:
        df = df.sort_values(first[0], ignore_index=False)
    left_index = df.index._values
    right_index = right.index._values
    l_index = pd.RangeIndex(start=0, stop=left_index.size)
    df.index = l_index
    r_index = pd.RangeIndex(start=0, stop=right_index.size)
    right.index = r_index
    shape = (left_index.size, 2)
    # use the l_booleans and r_booleans
    # to track rows that have complete matches
    left_regions = np.empty(shape=shape, dtype=np.intp, order="F")
    l_booleans = np.zeros(left_index.size, dtype=np.intp)
    shape = (right_index.size, 2)
    right_regions = np.empty(shape=shape, dtype=np.intp, order="F")
    r_booleans = np.zeros(right_index.size, dtype=np.intp)
    for position, (left_column, right_column, op) in enumerate(
        (first, second)
    ):
        outcome = helpers._generic_func_cond_join(
            left=df[left_column],
            right=right[right_column],
            op=op,
            multiple_conditions=True,
            keep="all",
        )
        if outcome is None:
            return None
        left_indexer, right_indexer, search_indices = outcome
        if op in helpers.greater_than_join_types:
            search_indices = right_indexer.size - search_indices
            right_indexer = right_indexer[::-1]
        r_region = np.zeros(right_indexer.size, dtype=np.intp)
        r_region[search_indices] = 1
        r_region[0] -= 1
        r_region = r_region.cumsum()
        left_regions[left_indexer, position] = r_region[search_indices]
        l_booleans[left_indexer] += 1
        right_regions[right_indexer, position] = r_region
        r_booleans[right_indexer[search_indices.min() :]] += 1
    r_region = None
    search_indices = None
    booleans = l_booleans == 2
    if not booleans.any():
        return None
    if not booleans.all():
        left_regions = left_regions[booleans]
        left_index = left_index[booleans]
        l_index = l_index[booleans]
    booleans = r_booleans == 2
    if not booleans.any():
        return None
    if not booleans.all():
        right_regions = right_regions[booleans]
        right_index = right_index[booleans]
        r_index = r_index[booleans]
    l_booleans = None
    r_booleans = None
    if gt_lt[0][-1] in helpers.greater_than_join_types:
        left_regions = left_regions[::-1]
        left_index = left_index[::-1]
        l_index = l_index[::-1]
        right_regions = right_regions[::-1]
        right_index = right_index[::-1]
        r_index = r_index[::-1]
    starts = right_regions[:, 0].searchsorted(left_regions[:, 0])
    booleans = starts < len(right_regions)
    if not booleans.any():
        return None
    if not booleans.all():
        starts = starts[booleans]
        left_regions = left_regions[booleans]
        left_index = left_index[booleans]
        l_index = l_index[booleans]
    rest = tuple(
        (
            df.loc[l_index, left_on].to_numpy(),
            right.loc[r_index, right_on].to_numpy(),
            mapping[op],
        )
        for left_on, right_on, op in rest
    )
    # a range join will have > and <
    # > and < will be in opposite directions
    # if the first condition is >
    # and the second condition is <
    # and the second condition is monotonic increasing
    # then this kicks in
    if pd.Index(right_regions[:, 1]).is_monotonic_decreasing:
        return _range_join_right_region_monotonic_decreasing(
            left_regions=left_regions,
            right_regions=right_regions,
            left_index=left_index,
            right_index=right_index,
            keep=keep,
            rest=rest,
            starts=starts,
            gt_lt=gt_lt,
            right_is_sorted=right_is_sorted,
            row_count=row_count,
        )
    if pd.Index(right_regions[:, 1]).is_monotonic_increasing:
        return _numba_non_equi_join_monotonic_increasing(
            left_regions=left_regions,
            right_regions=right_regions,
            left_index=left_index,
            right_index=right_index,
            keep=keep,
            gt_lt=gt_lt,
            rest=rest,
            starts=starts,
            row_count=row_count,
        )
    from janitor.functions.conditional_join import _numba

    # logic here is based on grantjenks' sortedcontainers
    # https://github.com/grantjenks/python-sortedcontainers
    load_factor = 1_000
    width = load_factor * 2
    length = math.ceil(right_index.size / load_factor)
    # maintain a sorted array of the regions
    sorted_array = np.empty(
        (width, length), dtype=right_regions.dtype, order="F"
    )
    # keep track of the positions of each region
    # within the sorted array
    positions_array = np.empty(
        (width, length), dtype=right_regions.dtype, order="F"
    )
    # keep track of the max value per column
    maxxes = np.empty(length, dtype=np.intp)
    # keep track of the length of actual data for each column
    lengths = np.empty(length, dtype=np.intp)
    if (keep == "all") & (len(gt_lt) == 2):
        left_indices, right_indices = (
            _numba._numba_non_equi_join_not_monotonic_dual_keep_all(
                left_regions=left_regions[:, 1],
                right_regions=right_regions[:, 1],
                left_index=left_index,
                right_index=right_index,
                maxxes=maxxes,
                lengths=lengths,
                sorted_array=sorted_array,
                positions_array=positions_array,
                starts=starts,
                load_factor=load_factor,
                row_count=True if row_count else False,
            )
        )

        if row_count and (left_indices is None):
            return pd.Series(index=left_indices, data=0)
        if row_count:
            return pd.Series(index=left_indices, data=right_indices)
    elif (keep == "first") & (len(gt_lt) == 2):
        left_indices, right_indices = (
            _numba._numba_non_equi_join_not_monotonic_dual_keep_first(
                left_regions=left_regions[:, 1],
                right_regions=right_regions[:, 1],
                left_index=left_index,
                right_index=right_index,
                maxxes=maxxes,
                lengths=lengths,
                sorted_array=sorted_array,
                positions_array=positions_array,
                starts=starts,
                load_factor=load_factor,
            )
        )
    elif (keep == "last") & (len(gt_lt) == 2):
        left_indices, right_indices = (
            _numba._numba_non_equi_join_not_monotonic_dual_keep_last(
                left_regions=left_regions[:, 1],
                right_regions=right_regions[:, 1],
                left_index=left_index,
                right_index=right_index,
                maxxes=maxxes,
                lengths=lengths,
                sorted_array=sorted_array,
                positions_array=positions_array,
                starts=starts,
                load_factor=load_factor,
            )
        )

    elif keep == "all":
        left_indices, right_indices = (
            _numba._numba_non_equi_join_not_monotonic_keep_all(
                tupled=rest,
                left_index=left_index,
                right_index=right_index,
                left_regions=left_regions[:, 1],
                right_regions=right_regions[:, 1],
                maxxes=maxxes,
                lengths=lengths,
                sorted_array=sorted_array,
                positions_array=positions_array,
                load_factor=load_factor,
                starts=starts,
                row_count=True if row_count else False,
            )
        )
        if row_count and (left_indices is None):
            return pd.Series(index=left_index, data=0, name=row_count)
        if row_count:
            return pd.Series(
                index=left_indices, data=right_indices, name=row_count
            )
    elif keep == "first":
        left_indices, right_indices = (
            _numba._numba_non_equi_join_not_monotonic_keep_first(
                tupled=rest,
                left_index=left_index,
                right_index=right_index,
                left_regions=left_regions[:, 1],
                right_regions=right_regions[:, 1],
                maxxes=maxxes,
                lengths=lengths,
                sorted_array=sorted_array,
                positions_array=positions_array,
                load_factor=load_factor,
                starts=starts,
            )
        )
    else:
        left_indices, right_indices = (
            _numba._numba_non_equi_join_not_monotonic_keep_last(
                tupled=rest,
                left_index=left_index,
                right_index=right_index,
                left_regions=left_regions[:, 1],
                right_regions=right_regions[:, 1],
                maxxes=maxxes,
                lengths=lengths,
                sorted_array=sorted_array,
                positions_array=positions_array,
                load_factor=load_factor,
                starts=starts,
            )
        )
    if left_indices is None:
        return None
    return left_indices, right_indices


def _range_join_sorted(
    first: tuple,
    second: tuple,
    df: pd.DataFrame,
    right: pd.DataFrame,
    keep: str,
    gt_lt: tuple,
    mapping: dict,
    rest: list,
    right_is_sorted: bool,
    row_count: str | None,
) -> tuple:
    """
    Get indices for a  range join
    if both columns from the right
    are monotonically sorted
    """
    from janitor.functions.conditional_join import _numba

    left_on, right_on, op = first
    outcome = helpers._generic_func_cond_join(
        left=df[left_on],
        right=right[right_on],
        op=op,
        multiple_conditions=True,
        keep="all",
    )
    if not outcome:
        return None
    left_index, right_index, ends = outcome
    left_on, right_on, op = second
    outcome = helpers._generic_func_cond_join(
        left=df.loc[left_index, left_on],
        right=right.loc[right_index, right_on],
        op=op,
        multiple_conditions=True,
        keep="all",
    )
    if outcome is None:
        return None
    left_c, right_index, starts = outcome
    if left_c.size < left_index.size:
        keep_rows = pd.Index(left_c).get_indexer(left_index) != -1
        ends = ends[keep_rows]
        left_index = left_c
    # no point searching within (a, b)
    # if a == b
    # since range(a, b) yields none
    keep_rows = starts < ends
    if not keep_rows.any():
        return None
    if not keep_rows.all():
        left_index = left_index[keep_rows]
        starts = starts[keep_rows]
        ends = ends[keep_rows]
    repeater = ends - starts
    if (len(gt_lt) == 2) and row_count:
        return pd.Series(index=left_index, data=repeater, name=row_count)
    if (len(gt_lt) == 2) & (repeater.max() == 1):
        # no point running a comparison op
        # if the width is all 1
        # this also implies that the intervals
        # do not overlap on the right side
        return left_index, right_index[starts]
    if (len(gt_lt) == 2) & (keep == "first") & right_is_sorted:
        return left_index, right_index[starts]
    if (len(gt_lt) == 2) & (keep == "last") & right_is_sorted:
        return left_index, right_index[ends - 1]
    if (len(gt_lt) == 2) & (keep in {"first", "last"}):
        left_indices = np.empty(left_index.size, dtype=np.intp)
        right_indices = np.empty(left_index.size, dtype=np.intp)
        return _numba._numba_range_join_sorted_keep_first_or_last_dual(
            left_index=left_index,
            right_index=right_index,
            starts=starts,
            ends=ends,
            left_indices=left_indices,
            right_indices=right_indices,
            position=keep == "first",
        )
    if (len(gt_lt) == 2) & (keep == "all"):
        start_indices = np.empty(left_index.size, dtype=np.intp)
        start_indices[0] = 0
        indices = (ends - starts).cumsum()
        start_indices[1:] = indices[:-1]
        indices = indices[-1]
        left_indices = np.empty(indices, dtype=np.intp)
        right_indices = np.empty(indices, dtype=np.intp)
        return _numba._range_join_sorted_dual_keep_all(
            left_index=left_index,
            right_index=right_index,
            starts=starts,
            ends=ends,
            left_indices=left_indices,
            right_indices=right_indices,
            start_indices=start_indices,
        )

    rest = tuple(
        (
            df.loc[left_index, left_on].to_numpy(),
            right.loc[right_index, right_on].to_numpy(),
            mapping[op],
        )
        for left_on, right_on, op in rest
    )

    start_indices = np.empty(left_index.size, dtype=np.intp)
    start_indices[0] = 0
    indices = (ends - starts).cumsum()
    start_indices[1:] = indices[:-1]
    indices = indices[-1]
    indices = np.ones(indices, dtype=np.bool_)

    if keep == "all":
        left_indices, right_indices = (
            _numba._range_join_sorted_multiple_keep_all(
                rest,
                left_index=left_index,
                starts=starts,
                ends=ends,
                right_index=right_index,
                indices=indices,
                start_indices=start_indices,
                row_count=True if row_count else False,
            )
        )
        if row_count and (left_indices is None):
            return None
        if row_count:
            return pd.Series(
                index=left_indices, data=right_indices, name=row_count
            )
    else:
        left_indices, right_indices = (
            _numba._range_join_sorted_multiple_keep_first_or_last(
                rest,
                left_index=left_index,
                starts=starts,
                ends=ends,
                right_index=right_index,
                indices=indices,
                start_indices=start_indices,
                position=keep == "first",
            )
        )
    if left_indices is None:
        return None
    return left_indices, right_indices


def _range_join_right_region_monotonic_decreasing(
    left_regions: np.ndarray,
    right_regions: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    keep: str,
    gt_lt: tuple,
    rest: tuple,
    starts: np.ndarray,
    right_is_sorted: bool,
    row_count: str,
):
    """
    Get indices for a range join,
    if the second column in the right region
    is monotonic decreasing
    """
    from janitor.functions.conditional_join import _numba

    ends = right_regions[::-1, 1].searchsorted(left_regions[:, 1])
    ends = len(right_regions) - ends
    booleans = starts < ends
    if not booleans.any():
        return None
    if not booleans.all():
        starts = starts[booleans]
        left_regions = left_regions[booleans]
        left_index = left_index[booleans]
        ends = ends[booleans]
        rest = tuple(
            (left_arr[booleans], right_arr, op)
            for left_arr, right_arr, op in rest
        )
    booleans = None
    if (keep == "first") & (len(gt_lt) == 2) & right_is_sorted:
        return left_index, right_index[ends - 1]
    if (keep == "first") & (len(gt_lt) == 2):
        left_indices = np.empty(left_index.size, dtype=np.intp)
        right_indices = np.empty(left_index.size, dtype=np.intp)
        return _numba._numba_range_join_sorted_keep_first_dual(
            left_index=left_index,
            right_index=right_index,
            starts=starts,
            ends=ends,
            left_indices=left_indices,
            right_indices=right_indices,
        )
    if (keep == "last") & (len(gt_lt) == 2) & right_is_sorted:
        return left_index, right_index[starts]
    if (keep == "last") & (len(gt_lt) == 2):
        left_indices = np.empty(left_index.size, dtype=np.intp)
        right_indices = np.empty(left_index.size, dtype=np.intp)
        return _numba._numba_range_join_sorted_keep_first_or_last_dual(
            left_index=left_index,
            right_index=right_index,
            starts=starts,
            ends=ends,
            left_indices=left_indices,
            right_indices=right_indices,
            position=keep == "first",
        )
    if (keep == "all") & (len(gt_lt) == 2):
        if row_count:
            repeater = ends - starts
            return pd.Series(index=left_index, data=repeater, name=row_count)
        start_indices = np.empty(left_index.size, dtype=np.intp)
        start_indices[0] = 0
        indices = (ends - starts).cumsum()
        start_indices[1:] = indices[:-1]
        indices = indices[-1]
        left_indices = np.empty(indices, dtype=np.intp)
        right_indices = np.empty(indices, dtype=np.intp)
        return _numba._range_join_sorted_dual_keep_all(
            left_index=left_index,
            right_index=right_index,
            starts=starts,
            ends=ends,
            left_indices=left_indices,
            right_indices=right_indices,
            start_indices=start_indices,
        )
    start_indices = np.empty(left_index.size, dtype=np.intp)
    start_indices[0] = 0
    indices = (ends - starts).cumsum()
    start_indices[1:] = indices[:-1]
    indices = indices[-1]
    indices = np.ones(indices, dtype=np.bool_)
    if keep == "all":
        left_indices, right_indices = (
            _numba._range_join_sorted_multiple_keep_all(
                rest,
                left_index=left_index,
                starts=starts,
                ends=ends,
                right_index=right_index,
                indices=indices,
                start_indices=start_indices,
                row_count=row_count,
            )
        )
        if row_count and (left_indices is None):
            return None
        if row_count:
            return pd.Series(
                index=left_indices, data=right_indices, name=row_count
            )
    else:
        left_indices, right_indices = (
            _numba._range_join_sorted_multiple_keep_first_or_last(
                rest,
                left_index=left_index,
                starts=starts,
                ends=ends,
                right_index=right_index,
                indices=indices,
                start_indices=start_indices,
                position=keep == "first",
            )
        )

    if left_indices is None:
        return None
    return left_indices, right_indices


def _numba_non_equi_join_monotonic_increasing(
    left_regions: np.ndarray,
    right_regions: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    keep: str,
    gt_lt: tuple,
    rest: tuple,
    starts: np.ndarray,
    row_count: str,
):
    """
    Get indices for a non equi join,
    if the second column in the right region
    is monotonic increasing
    """
    from janitor.functions.conditional_join import _numba

    _starts = right_regions[:, 1].searchsorted(left_regions[:, 1])
    starts = np.where(starts > _starts, starts, _starts)
    booleans = starts == right_index.size
    if booleans.all():
        return None
    if booleans.any():
        booleans = ~booleans
        left_index = left_index[booleans]
        starts = starts[booleans]
        left_regions = left_regions[booleans]
        rest = tuple(
            (left_arr[booleans], right_arr, op)
            for left_arr, right_arr, op in rest
        )
    if (keep in {"first", "last"}) & (len(gt_lt) == 2):
        left_indices = np.empty(left_index.size, dtype=np.intp)
        right_indices = np.empty(left_index.size, dtype=np.intp)
        return _numba._numba_non_equi_join_monotonic_increasing_keep_first_or_last_dual(
            left_index=left_index,
            right_index=right_index,
            starts=starts,
            left_indices=left_indices,
            right_indices=right_indices,
            position=keep == "first",
        )
    if (keep == "all") & (len(gt_lt) == 2):
        if row_count:
            repeater = right_index.size - starts
            return pd.Series(index=left_index, data=repeater, name=row_count)
        start_indices = np.empty(left_index.size, dtype=np.intp)
        start_indices[0] = 0
        indices = (right_index.size - starts).cumsum()
        start_indices[1:] = indices[:-1]
        indices = indices[-1]
        left_indices = np.empty(indices, dtype=np.intp)
        right_indices = np.empty(indices, dtype=np.intp)
        return _numba._numba_non_equi_join_monotonic_increasing_keep_all_dual(
            left_index=left_index,
            right_index=right_index,
            starts=starts,
            left_indices=left_indices,
            right_indices=right_indices,
            start_indices=start_indices,
        )
    start_indices = np.empty(left_index.size, dtype=np.intp)
    start_indices[0] = 0
    indices = (right_index.size - starts).cumsum()
    start_indices[1:] = indices[:-1]
    indices = indices[-1]
    indices = np.ones(indices, dtype=np.bool_)
    if keep in {"first", "last"}:
        left_indices, right_indices = (
            _numba._numba_non_equi_join_monotonic_increasing_keep_first_or_last(
                rest,
                left_index=left_index,
                starts=starts,
                right_index=right_index,
                indices=indices,
                start_indices=start_indices,
                position=keep == "first",
            )
        )

    else:
        left_indices, right_indices = (
            _numba._numba_non_equi_join_monotonic_increasing_keep_all(
                rest,
                left_index=left_index,
                starts=starts,
                right_index=right_index,
                indices=indices,
                start_indices=start_indices,
                row_count=True if row_count else False,
            )
        )
        if row_count and (left_indices is None):
            return pd.Series(index=left_index, data=0)
        if row_count:
            return pd.Series(index=left_indices, data=right_indices)

    if left_indices is None:
        return None
    return left_indices, right_indices
