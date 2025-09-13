from __future__ import annotations

import math

import numpy as np
import pandas as pd

from janitor.functions.conditional_join import _numba

from . import helpers


def _numba_multiple_non_equi_join(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list,
    keep: str,
    row_count: str,
    booleans: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Build indices for joins where there is at least one >/>=/</<=
    """
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
    # and if we index the dataframes, we should get the output below:
    #################################
    #    value_1  value_2A  value_2B
    # 0        2         1         3
    # 1        5         3         6
    # 2        3         2         4
    # 3        4         3         5
    # 4        4         3         6
    ################################
    if conditions["non_equi_count"] > 1:
        len_df = len(df)
        len_right = len(right)
        left_regions = np.empty((len_df, 2), dtype=np.intp)
        positions = np.empty((len_df, 2), dtype=np.intp)
        right_regions = np.empty((len_right, 2), dtype=np.intp)
        starts = np.zeros(len_df, dtype=np.intp)
        ends = np.empty(len_df, dtype=np.intp)
        ends[:] = len_right
        sizes = np.zeros(len_df, dtype=np.intp)
        indices = {
            "starts": starts,
            "ends": ends,
            "sizes": sizes,
        }
        first, second, *rest = conditions["conditions"]
        lcol, rcol, op = first
        # sorting is done here to enable easy region filtering later on
        # by starting from the highest region
        if not df[lcol].is_monotonic_increasing:
            df = df.sort_values(lcol, ignore_index=False, kind="stable")
            if not booleans.all():
                booleans = booleans[df.index._values]
        if not right[rcol].is_monotonic_increasing:
            right = right.sort_values(rcol, ignore_index=False, kind="stable")
        indices["booleans"] = booleans
        outcome = _build_region(
            indices=indices, left=df[lcol], right=right[rcol], op=op
        )
        if outcome is None:
            return None
        left_regions[:, 0] = outcome["l_region"]
        right_regions[:, 0] = outcome["r_region"]
        right_index = outcome["r_index"]
        indices = _update_indices(
            indices=indices,
            booleans=outcome["booleans"],
            len_right=len_right,
        )
        lcol, rcol, op_ = second
        right_c = right[rcol]
        right_c, _ = helpers._sort_if_not_monotonic(series=right_c)
        outcome = _build_region(
            indices=indices, left=df[lcol], right=right_c, op=op_
        )
        if outcome is None:
            return None
        left_regions[:, 1] = outcome["l_region"]
        booleans = outcome["booleans"].astype(np.bool_, copy=False)
        # realign so that the two regions share a common index
        indexer = outcome["r_index"].get_indexer(right_index)
        right_regions[:, 1] = outcome["r_region"][indexer]
        right_index = right_index._values
        left_index = df.index._values
        # ensure highest values are at the top
        # monotonic decreasing
        if op in helpers.less_than_join_types:
            # this is necessary,
            # as the number of join conditions
            # may be > 2
            df = df.iloc[::-1]
            left_regions = left_regions[::-1]
            left_index = left_index[::-1]
            booleans = booleans[::-1]
        elif op in helpers.greater_than_join_types:
            # ensure right is aligned
            # since within build_region
            # we have flipped in monotonic decreasing order
            right = right.iloc[::-1]
        positions = right_regions[:, 0].searchsorted(left_regions[:, 0])
        bools = positions < len_right
        bools = bools & booleans
        # when searching, exclude regions that definitely
        # do not have a match
        booleans = np.where(bools, booleans, False)
        left_regions = left_regions[:, 1]
        right_regions = right_regions[:, 1]
        # logic here is based on grantjenks' sortedcontainers
        # https://github.com/grantjenks/python-sortedcontainers
        load_factor = len_right ** (1 / 3)
        load_factor = math.ceil(load_factor)
        load_factor = max(1_000, load_factor)
        width = load_factor * 2
        length = math.ceil(len_right / load_factor)
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
        tuples = helpers._generate_tuples(df=df, right=right, conditions=rest)

        if row_count:
            result = _numba._get_row_count_for_regions(
                booleans=booleans,
                starts=positions,
                left_region=left_regions,
                right_region=right_regions,
                sorted_array=sorted_array,
                positions_array=positions_array,
                maxxes=maxxes,
                lengths=lengths,
                load_factor=load_factor,
                tuples=tuples,
            )
            if result is None:
                return None
            return pd.Series(data=result, index=left_index, name=row_count)
        if keep == "all":
            left_index, right_index = _numba._get_indices_for_regions_keep_all(
                booleans=booleans,
                left_index=left_index,
                right_index=right_index,
                starts=positions,
                left_region=left_regions,
                right_region=right_regions,
                sorted_array=sorted_array,
                positions_array=positions_array,
                maxxes=maxxes,
                lengths=lengths,
                load_factor=load_factor,
                tuples=tuples,
            )
            if left_index is None:
                return None
            return left_index, right_index
        if keep == "first":
            left_index, right_index = (
                _numba._get_indices_for_regions_keep_first(
                    booleans=booleans,
                    left_index=left_index,
                    right_index=right_index,
                    starts=positions,
                    left_region=left_regions,
                    right_region=right_regions,
                    sorted_array=sorted_array,
                    positions_array=positions_array,
                    maxxes=maxxes,
                    lengths=lengths,
                    load_factor=load_factor,
                    tuples=tuples,
                )
            )
            if left_index is None:
                return None
            return left_index, right_index
        left_index, right_index = _numba._get_indices_for_regions_keep_last(
            booleans=booleans,
            left_index=left_index,
            right_index=right_index,
            starts=positions,
            left_region=left_regions,
            right_region=right_regions,
            sorted_array=sorted_array,
            positions_array=positions_array,
            maxxes=maxxes,
            lengths=lengths,
            load_factor=load_factor,
            tuples=tuples,
        )
        if left_index is None:
            return None
        return left_index, right_index
    right_on = conditions["conditions"][0][1]
    if not right[right_on].is_monotonic_increasing:
        right = right.sort_values(right_on, kind="stable", ignore_index=False)
    ge_gt = None
    le_lt = None
    condition, *rest = conditions["conditions"]
    left_index = df.index._values
    right_index = right.index._values
    if condition[-1] in helpers.greater_than_join_types:
        ge_gt = condition
        left_on, right_on, op = ge_gt
        left_c = df[left_on]._values
        right_c = right[right_on]._values
        left_c, right_c = helpers._convert_to_numpy(left=left_c, right=right_c)
        op = helpers.operator_mapping[op]
        op = np.array([op], dtype=np.intp)
        ge_gt = (left_c, right_c, op)
    else:
        le_lt = condition
        left_on, right_on, op = le_lt
        left_c = df[left_on]._values
        right_c = right[right_on]._values
        left_c, right_c = helpers._convert_to_numpy(left=left_c, right=right_c)
        op = helpers.operator_mapping[op]
        op = np.array([op], dtype=np.intp)
        le_lt = (left_c, right_c, op)
    length = len(df)
    starts = np.zeros(length, dtype=np.intp)
    ends = np.empty(length, dtype=np.intp)
    ends[:] = len(right)
    booleans = np.ones(length, dtype=np.int8)
    tuples = helpers._generate_tuples(df=df, right=right, conditions=rest)
    if row_count:
        result = _numba._get_row_count_equi_le_lt_or_ge_gt(
            tuples=tuples,
            starts=starts,
            ends=ends,
            le_lt=le_lt,
            ge_gt=ge_gt,
            booleans=booleans,
        )
        if result is None:
            return None
        return pd.Series(data=result, index=left_index, name=row_count)

    if keep == "all":
        left_index, right_index = (
            _numba._get_indices_equi_ge_gt_or_le_lt_join_keep_all(
                left_index=left_index,
                right_index=right_index,
                tuples=tuples,
                starts=starts,
                ends=ends,
                le_lt=le_lt,
                ge_gt=ge_gt,
                booleans=booleans,
            )
        )
        if left_index is None:
            return None
        return left_index, right_index
    if keep == "first":
        left_index, right_index = (
            _numba._get_indices_equi_ge_gt_or_le_lt_join_keep_first(
                left_index=left_index,
                right_index=right_index,
                tuples=tuples,
                starts=starts,
                ends=ends,
                le_lt=le_lt,
                ge_gt=ge_gt,
                booleans=booleans,
            )
        )
        if left_index is None:
            return None
        return left_index, right_index
    left_index, right_index = (
        _numba._get_indices_equi_ge_gt_or_le_lt_join_keep_last(
            left_index=left_index,
            right_index=right_index,
            tuples=tuples,
            starts=starts,
            ends=ends,
            le_lt=le_lt,
            ge_gt=ge_gt,
            booleans=booleans,
        )
    )
    if left_index is None:
        return None
    return left_index, right_index


def _update_indices(
    indices: dict,
    booleans: np.ndarray,
    len_right: int,
    reset_sizes: bool = False,
):
    """
    update indices
    """
    # build new starts to avoid mutation
    # which feeds into all start variables
    starts = np.zeros(indices["starts"].size, dtype=np.intp)
    ends = indices["ends"]
    ends[:] = len_right
    if reset_sizes:
        sizes = np.zeros(starts.size, dtype=np.intp)
        indices["sizes"] = sizes
    indices["starts"] = starts
    indices["ends"] = ends
    indices["booleans"] = booleans
    return indices


def _build_region(
    left: pd.Series,
    right: pd.series,
    op: str,
    indices: dict,
):
    """
    Build ordered regions
    """
    r_index = right.index
    left, right = helpers._convert_to_numpy(
        left=left._values, right=right._values
    )

    indices = helpers._update_search_indices(
        left=left,
        right=right,
        indices=indices,
        op=op,
    )
    if indices is None:
        return None
    len_right = len(right)
    if op in helpers.greater_than_join_types:
        positions = len_right - indices["ends"]
        r_index = r_index[::-1]
    else:
        positions = indices["starts"]
    booleans = indices["booleans"]
    r_region = np.zeros(len_right, dtype=np.intp)
    # ensure only matches have positive regions
    if not booleans.all():
        bools_ = booleans.astype(np.bool_, copy=False)
        positions = np.where(bools_, positions, 0)
        trimmed = positions[bools_]
    else:
        trimmed = positions[:]
    r_region[trimmed] = 1
    r_region[0] -= 1
    r_region = r_region.cumsum()
    l_region = r_region[positions]
    return {
        "l_region": l_region,
        "r_region": r_region,
        "r_index": r_index,
        "booleans": booleans,
    }
