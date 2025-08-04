from typing import Union

import numpy as np
import pandas as pd

from .helpers import _convert_to_numpy


def _numba_equi_join(
    df: pd.DataFrame,
    right: pd.DataFrame,
    eqs: tuple,
    ge_gt: tuple,
    le_lt: tuple,
    rest: tuple,
    row_count: str,
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
    from janitor.functions.conditional_join import _numba

    mapping = {">": 0, ">=": 1, "<": 2, "<=": 3, "!=": 4}
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
    rest = tuple(
        (
            left.loc[left_index].to_numpy(),
            right.to_numpy(),
            mapping[op],
        )
        for left, right, op in rest
    )
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
        op = mapping[op]
    all_monotonic_increasing = False
    if le_lt and ge_gt:
        group = right.groupby(eqs[1])[le_lt[1]]
        # is the last column (le_lt) monotonic increasing?
        # fast path if it is
        all_monotonic_increasing = all(
            arr.is_monotonic_increasing for _, arr in group
        )

    if le_lt and ge_gt and all_monotonic_increasing and not rest:
        left_index, right_index = _numba._numba_equi_join_range_join_monotonic(
            left_index=left_index,
            right_index=right_index,
            slice_starts=slice_starts,
            slice_ends=slice_ends,
            ge_arr1=ge_arr1,
            ge_arr2=ge_arr2,
            ge_strict=ge_strict,
            le_arr1=le_arr1,
            le_arr2=le_arr2,
            le_strict=le_strict,
            row_count=True if row_count else False,
        )

    elif le_lt and ge_gt and all_monotonic_increasing:
        left_index, right_index = (
            _numba._numba_equi_join_range_join_multiple_monotonic(
                left_index=left_index,
                right_index=right_index,
                slice_starts=slice_starts,
                slice_ends=slice_ends,
                ge_arr1=ge_arr1,
                ge_arr2=ge_arr2,
                ge_strict=ge_strict,
                le_arr1=le_arr1,
                le_arr2=le_arr2,
                le_strict=le_strict,
                row_count=True if row_count else False,
                tupled=rest,
            )
        )

    elif le_lt and ge_gt:
        conditions = [(le_arr1, le_arr2, op)]
        conditions.extend(rest)
        left_index, right_index = (
            _numba._numba_equi_join_range_join_non_monotonic(
                left_index=left_index,
                right_index=right_index,
                slice_starts=slice_starts,
                slice_ends=slice_ends,
                ge_arr1=ge_arr1,
                ge_arr2=ge_arr2,
                ge_strict=ge_strict,
                row_count=True if row_count else False,
                tupled=conditions,
            )
        )

    elif le_lt and not rest:
        (
            left_index,
            right_index,
        ) = _numba._numba_equi_single_le_ge_join(
            left_index=left_index,
            right_index=right_index,
            slice_starts=slice_starts,
            slice_ends=slice_ends,
            arr1=le_arr1,
            arr2=le_arr2,
            strict=le_strict,
            less_than=True,
            row_count=True if row_count else False,
        )

    elif le_lt:
        (
            left_index,
            right_index,
        ) = _numba._numba_equi_single_le_ge_tupled_join(
            left_index=left_index,
            right_index=right_index,
            slice_starts=slice_starts,
            slice_ends=slice_ends,
            arr1=le_arr1,
            arr2=le_arr2,
            strict=le_strict,
            less_than=True,
            row_count=True if row_count else False,
            tupled=rest if rest else None,
        )

    elif ge_gt and not rest:
        (
            left_index,
            right_index,
        ) = _numba._numba_equi_single_le_ge_join(
            left_index=left_index,
            right_index=right_index,
            slice_starts=slice_starts,
            slice_ends=slice_ends,
            arr1=ge_arr1,
            arr2=ge_arr2,
            strict=ge_strict,
            less_than=False,
            row_count=True if row_count else False,
        )

    elif ge_gt:
        (
            left_index,
            right_index,
        ) = _numba._numba_equi_single_le_ge_tupled_join(
            left_index=left_index,
            right_index=right_index,
            slice_starts=slice_starts,
            slice_ends=slice_ends,
            arr1=ge_arr1,
            arr2=ge_arr2,
            strict=ge_strict,
            less_than=False,
            row_count=True if row_count else False,
            tupled=rest if rest else None,
        )
    if row_count and (left_index is None):
        return pd.Series(index=df.index, data=0, name=row_count)
    if row_count:
        return pd.Series(index=left_index, data=right_index, name=row_count)
    if left_index is None:
        return None

    return left_index, right_index
