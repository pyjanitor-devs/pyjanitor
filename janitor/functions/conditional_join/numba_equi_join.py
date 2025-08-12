from __future__ import annotations

from typing import Hashable

import numpy as np
import pandas as pd

from . import helpers


def _numba_equi_join(
    df: pd.DataFrame,
    right: pd.DataFrame,
    is_fastpath_range_join: bool,
    conditions: dict,
    row_count: Hashable | None,
    keep: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Compute indices when an equi join is present.
    """
    # this applies if there is a >/>=/</<= join present
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
    from janitor.functions.conditional_join import _numba

    equals = conditions["equals"]
    left_c, right_c, _ = equals[0]
    left_c = df[left_c]
    right_c = right[right_c]
    # steal some perf here within the binary search
    # search for uniques
    # and later index them with left_positions
    # it is assumed that users will only reach for this
    # if the data is reasonably duplicated; if not
    # pd.merge is superb especially if it's a one-to-one
    # or one-to-many
    left_index = left_c.index._values
    right_index = right_c.index._values
    right_c = right_c.array
    positions, left_c = pd.factorize(left_c, sort=False)
    left_c = left_c.array
    starts = right_c.searchsorted(left_c, side="left")
    starts = starts[positions]
    ends = right_c.searchsorted(left_c, side="right")
    ends = ends[positions]
    booleans = starts < ends
    if not booleans.any():
        return None
    equals = equals[1:]
    rest = conditions["conditions"]
    if equals or not conditions.get("less_than_or_greater_than"):
        sizes = ends - starts
        if not booleans.all():
            sizes = np.where(booleans, sizes, 0)
        start_indices = np.empty(sizes.size, dtype=np.intp)
        start_indices[0] = 0
        start_indices[1:] = sizes.cumsum()[:-1]
        rest = equals + rest
        counts_array = np.zeros(left_index.size, dtype=np.intp)
        matches = np.ones(sizes.sum(), dtype=np.bool_)
        tuples = _generate_tuples(df=df, right=right, conditions=rest)
        if row_count:
            data = _numba._get_rowcount_all_equi(
                tuples=tuples,
                starts=starts,
                ends=ends,
                start_indices=start_indices,
                matches=matches,
                booleans=booleans,
                counts_array=counts_array,
            )
            if data is None:
                return None
            return pd.Series(
                index=left_index,
                data=data,
                name=row_count,
            )
        if keep == "all":
            left_index, right_index = _numba._get_indices_all_equi_keep_all(
                left_index=left_index,
                right_index=right_index,
                tuples=tuples,
                starts=starts,
                start_indices=start_indices,
                ends=ends,
                matches=matches,
                booleans=booleans,
                counts_array=counts_array,
                sizes=sizes,
            )
            if left_index is None:
                return None
            return left_index, right_index
        if keep == "first":
            left_index, right_index = _numba._get_indices_all_equi_keep_first(
                left_index=left_index,
                right_index=right_index,
                tuples=tuples,
                starts=starts,
                start_indices=start_indices,
                ends=ends,
                matches=matches,
                booleans=booleans,
                counts_array=counts_array,
                sizes=sizes,
            )
            if left_index is None:
                return None
            return left_index, right_index
        left_index, right_index = _numba._get_indices_all_equi_keep_last(
            left_index=left_index,
            right_index=right_index,
            tuples=tuples,
            starts=starts,
            start_indices=start_indices,
            ends=ends,
            matches=matches,
            booleans=booleans,
            counts_array=counts_array,
            sizes=sizes,
        )
        if left_index is None:
            return None
        return left_index, right_index
    if is_fastpath_range_join:
        ge_gt, le_lt, *rest = conditions["conditions"]
        left_on, right_on, op1 = ge_gt
        left_c = df[left_on]._values
        right_c = right[right_on]._values
        left_c, right_c = helpers._convert_to_numpy(left=left_c, right=right_c)
        ge_gt = (left_c, right_c)
        left_on, right_on, op2 = le_lt
        left_c = df[left_on]._values
        right_c = right[right_on]._values
        left_c, right_c = helpers._convert_to_numpy(left=left_c, right=right_c)
        le_lt = (left_c, right_c)
        tuples = _generate_tuples(df=df, right=right, conditions=rest)
        if row_count:
            data = _numba._get_row_count_equi_join_range_join(
                starts=starts,
                ends=ends,
                booleans=booleans,
                tuples=tuples,
                ge_gt=ge_gt,
                le_lt=le_lt,
                op1=op1,
                op2=op2,
            )
            if data is None:
                return None
            return pd.Series(
                index=left_index,
                data=data,
                name=row_count,
            )
        if keep == "all":
            left_index, right_index = (
                _numba._build_indices_equi_join_range_join_keep_all(
                    left_index=left_index,
                    right_index=right_index,
                    starts=starts,
                    ends=ends,
                    booleans=booleans,
                    tuples=tuples,
                    ge_gt=ge_gt,
                    le_lt=le_lt,
                    op1=op1,
                    op2=op2,
                )
            )
            if left_index is None:
                return None
            return left_index, right_index
        if keep == "first":
            left_index, right_index = (
                _numba._build_indices_equi_join_range_join_keep_first(
                    left_index=left_index,
                    right_index=right_index,
                    starts=starts,
                    ends=ends,
                    booleans=booleans,
                    tuples=tuples,
                    ge_gt=ge_gt,
                    le_lt=le_lt,
                    op1=op1,
                    op2=op2,
                )
            )
            if left_index is None:
                return None
            return left_index, right_index
        left_index, right_index = (
            _numba._build_indices_equi_join_range_join_keep_last(
                left_index=left_index,
                right_index=right_index,
                starts=starts,
                ends=ends,
                booleans=booleans,
                tuples=tuples,
                ge_gt=ge_gt,
                le_lt=le_lt,
                op1=op1,
                op2=op2,
            )
        )
        if left_index is None:
            return None
        return left_index, right_index
    (left_on, right_on, op), *rest = conditions["conditions"]
    left_array = df[left_on]._values
    right_array = right[right_on]._values
    left_array, right_array = helpers._convert_to_numpy(
        left=left_array, right=right_array
    )
    tuples = _generate_tuples(df=df, right=right, conditions=rest)
    if row_count:
        data = _numba._get_row_count_equi_join(
            starts=starts,
            ends=ends,
            booleans=booleans,
            tuples=tuples,
            left_array=left_array,
            right_array=right_array,
            op=op,
        )
        if data is None:
            return None
        return pd.Series(
            index=left_index,
            data=data,
            name=row_count,
        )
    if keep == "all":
        left_index, right_index = _numba._build_indices_equi_join_keep_all(
            left_index=left_index,
            right_index=right_index,
            starts=starts,
            ends=ends,
            booleans=booleans,
            tuples=tuples,
            left_array=left_array,
            right_array=right_array,
            op=op,
        )
        if left_index is None:
            return None
        return left_index, right_index
    if keep == "first":
        left_index, right_index = _numba._build_indices_equi_join_keep_first(
            left_index=left_index,
            right_index=right_index,
            starts=starts,
            ends=ends,
            booleans=booleans,
            tuples=tuples,
            left_array=left_array,
            right_array=right_array,
            op=op,
        )
        if left_index is None:
            return None
        return left_index, right_index
    left_index, right_index = _numba._build_indices_equi_join_keep_last(
        left_index=left_index,
        right_index=right_index,
        starts=starts,
        ends=ends,
        booleans=booleans,
        tuples=tuples,
        left_array=left_array,
        right_array=right_array,
        op=op,
    )
    if left_index is None:
        return None
    return left_index, right_index


def _generate_tuples(df: pd.DataFrame, right: pd.DataFrame, conditions: list):
    """
    Build tuple of arrays to pass to numba
    """
    if not conditions:
        return None
    tuples = []
    left_booleans = np.empty(1, dtype=np.bool_)
    right_booleans = np.empty(1, dtype=np.bool_)
    for left_on, right_on, op in conditions:
        left_arr = df[left_on]
        is_extension_array = pd.api.types.is_extension_array_dtype(left_arr)
        left_arr = left_arr._values
        right_arr = right[right_on]._values
        if op == "!=":
            left_booleans = pd.isna(left_arr)
            right_booleans = pd.isna(right_arr)
        left_arr, right_arr = helpers._convert_to_numpy(
            left=left_arr, right=right_arr
        )
        condition = (
            left_arr,
            right_arr,
            helpers.operator_mapping[op],
            is_extension_array,
            left_booleans,
            right_booleans,
        )
        tuples.append(condition)
    return tuple(tuples)
