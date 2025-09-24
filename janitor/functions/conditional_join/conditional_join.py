from __future__ import annotations

from typing import Any, Hashable, Literal, Optional

import numpy as np
import pandas as pd
import pandas_flavor as pf
from pandas.api.types import (
    is_datetime64_dtype,
    is_dtype_equal,
    is_numeric_dtype,
    is_timedelta64_dtype,
)
from pandas.core.dtypes.concat import concat_compat

from janitor.utils import check, check_column, deprecated_alias

from . import helpers, non_equi_join_single
from .equi_join import _multiple_conditional_join_eq
from .multiple_conditional_join_le_lt import _multiple_conditional_join_le_lt


@pf.register_dataframe_method
def conditional_join(
    df: pd.DataFrame,
    right: pd.DataFrame | pd.Series,
    *conditions: Any,
    how: Literal["inner", "left", "right", "outer"] = "inner",
    df_columns: Optional[Any] = slice(None),
    right_columns: Optional[Any] = slice(None),
    keep: Literal["first", "last", "all"] = "all",
    use_numba: bool = False,
    indicator: Optional[bool | str] = False,
    force: bool = False,
    sort_equi_join: bool = False,
    aggfunc: list[tuple] = None,
) -> pd.DataFrame:
    """

    !!!note

        Before reaching for `conditional_join`,
        see if `pd.merge`, `pd.merge_asof`, or `pd.IntervalIndex`
        meets your needs .

    The conditional_join function operates similarly to `pd.merge`,
    but supports joins on inequality operators,
    or a combination of equi and non-equi joins.

    Joins solely on equality are not supported.

    If the join is solely on equality, `pd.merge` function
    covers that; if you are interested in nearest joins, asof joins,
    or rolling joins, then `pd.merge_asof` covers that.
    There is also pandas' IntervalIndex, which is efficient for range joins,
    especially if the intervals do not overlap.

    Column selection in `df_columns` and `right_columns` is possible using the
    [`select`][janitor.functions.select.select] syntax.

    Performance might be improved by setting `use_numba` to `True` -
    this can be handy for equi joins that have lots of duplicated keys.
    This can also be handy for non-equi joins, where there are more than
    two join conditions,
    or there is significant overlap in the range join columns.
    This assumes that `numba` is installed.

    Noticeable performance can be observed for range joins,
    if both join columns from the right dataframe
    are monotonically increasing.

    This function returns rows, if any, where values from `df` meet the
    condition(s) for values from `right`. The conditions are passed in
    as a variable argument of tuples, where the tuple is of
    the form `(left_on, right_on, op)`; `left_on` is the column
    label from `df`, `right_on` is the column label from `right`,
    while `op` is the operator.

    For multiple conditions, the and(`&`)
    operator is used to combine the results of the individual conditions.

    In some scenarios there might be performance gains if the less than join,
    or the greater than join condition, or the range condition
    is executed before the equi join - pass `force=True` to force this.

    There may be scenarios for an equi-join, where pandas' merge
    may not be performant - if you have a range join combined
    with a single equi-join, and the columns involved in the equi-join
    have a lot of duplicate values;
    pass `sort_equi_join=True` for an alternative
    approach that may offer reduced computation time. This parameter
    is ignored if `use_numba=True`.

    Aggregations can be computed after the successful execution of a join;
    the aggregaton is computed on the right dataframe
    for each row of the left DataFrame (that has a match)
    based on the join keys.
    Supported aggregation functions are
    `sum`, `count`, `size`, `min`, `max`.

    The operator can be any of `==`, `!=`, `<=`, `<`, `>=`, `>`.

    The join is done only on the columns.

    For non-equi joins, only numeric, timedelta and date columns are supported.

    `inner`, `left`, `right` and `outer` joins are supported.

    If the columns from `df` and `right` have nothing in common,
    a single index column is returned; else, a MultiIndex column
    is returned.

    Examples:
        >>> import pandas as pd
        >>> import janitor
        >>> df1 = pd.DataFrame({"value_1": [2, 5, 7, 1, 3, 4]})
        >>> df2 = pd.DataFrame({"value_2A": [0, 3, 7, 12, 0, 2, 3, 1],
        ...                     "value_2B": [1, 5, 9, 15, 1, 4, 6, 3],
        ...                    })
        >>> df1
           value_1
        0        2
        1        5
        2        7
        3        1
        4        3
        5        4
        >>> df2
           value_2A  value_2B
        0         0         1
        1         3         5
        2         7         9
        3        12        15
        4         0         1
        5         2         4
        6         3         6
        7         1         3

        >>> df1.conditional_join(
        ...     df2,
        ...     ("value_1", "value_2A", ">"),
        ...     ("value_1", "value_2B", "<")
        ... )
           value_1  value_2A  value_2B
        0        2         1         3
        1        5         3         6
        2        3         2         4
        3        4         3         5
        4        4         3         6

        Select specific columns, after the join:
        >>> df1.conditional_join(
        ...     df2,
        ...     ("value_1", "value_2A", ">"),
        ...     ("value_1", "value_2B", "<"),
        ...     right_columns='value_2B',
        ...     how='left'
        ... )
           value_1  value_2B
        0        2       3.0
        1        5       6.0
        2        3       4.0
        3        4       5.0
        4        4       6.0
        5        7       NaN
        6        1       NaN

        Rename columns, before the join:
        >>> (df1
        ...  .rename(columns={'value_1':'left_column'})
        ...  .conditional_join(
        ...      df2,
        ...     ("left_column", "value_2A", ">"),
        ...     ("left_column", "value_2B", "<"),
        ...      right_columns='value_2B',
        ...      how='outer')
        ... )
            left_column  value_2B
        0           2.0       3.0
        1           5.0       6.0
        2           3.0       4.0
        3           4.0       5.0
        4           4.0       6.0
        5           7.0       NaN
        6           1.0       NaN
        7           NaN       1.0
        8           NaN       9.0
        9           NaN      15.0
        10          NaN       1.0

        Get the first match:
        >>> df1.conditional_join(
        ...     df2,
        ...     ("value_1", "value_2A", ">"),
        ...     ("value_1", "value_2B", "<"),
        ...     keep='first'
        ... )
           value_1  value_2A  value_2B
        0        2         1         3
        1        5         3         6
        2        3         2         4
        3        4         3         5

        Get the last match:
        >>> df1.conditional_join(
        ...     df2,
        ...     ("value_1", "value_2A", ">"),
        ...     ("value_1", "value_2B", "<"),
        ...     keep='last'
        ... )
           value_1  value_2A  value_2B
        0        2         1         3
        1        5         3         6
        2        3         2         4
        3        4         3         6

        Add an indicator column:
        >>> df1.conditional_join(
        ...     df2,
        ...     ("value_1", "value_2A", ">"),
        ...     ("value_1", "value_2B", "<"),
        ...     how='outer',
        ...     indicator=True
        ... )
            value_1  value_2A  value_2B      _merge
        0       2.0       1.0       3.0        both
        1       5.0       3.0       6.0        both
        2       3.0       2.0       4.0        both
        3       4.0       3.0       5.0        both
        4       4.0       3.0       6.0        both
        5       7.0       NaN       NaN   left_only
        6       1.0       NaN       NaN   left_only
        7       NaN       0.0       1.0  right_only
        8       NaN       7.0       9.0  right_only
        9       NaN      12.0      15.0  right_only
        10      NaN       0.0       1.0  right_only

        Compute an aggregation:
        >>> df1.conditional_join(
        ...     df2,
        ...     ("value_1", "value_2A", ">"),
        ...     ("value_1", "value_2B", "<"),
        ...     aggfunc=[("value_2B", "sum")],
        ... ).droplevel(level=0, axis="columns")
           value_1  sum
        0        2    3
        1        5    6
        4        3    4
        5        4   11


    !!! abstract "Version Changed"

        - 0.24.0
            - Added `df_columns`, `right_columns`, `keep` and `use_numba` parameters.
        - 0.24.1
            - Added `indicator` parameter.
        - 0.25.0
            - `col` class supported.
            - Outer join supported. `sort_by_appearance` deprecated.
            - Numba support for equi join
        - 0.27.0
            - Added support for timedelta dtype.
        - 0.28.0
            - `col` class deprecated.
        - 0.32.0
            - Added `aggfunc` parameter.

    Args:
        df: A pandas DataFrame.
        right: Named Series or DataFrame to join to.
        conditions: Variable argument of tuple(s) of the form
            `(left_on, right_on, op)`, where `left_on` is the column
            label from `df`, `right_on` is the column label from `right`,
            while `op` is the operator.
            The `col` class is also supported. The operator can be any of
            `==`, `!=`, `<=`, `<`, `>=`, `>`. For multiple conditions,
            the and(`&`) operator is used to combine the results
            of the individual conditions.
        how: Indicates the type of join to be performed.
            It can be one of `inner`, `left`, `right` or `outer`.
        df_columns: Columns to select from `df` in the final output dataframe.
            Column selection is based on the
            [`select`][janitor.functions.select.select] syntax.
        right_columns: Columns to select from `right` in the final output dataframe.
            Column selection is based on the
            [`select`][janitor.functions.select.select] syntax.
        use_numba: Use numba, if installed, to accelerate the computation.
        keep: Choose whether to return the first match, last match or all matches.
        indicator: If `True`, adds a column to the output DataFrame
            called `_merge` with information on the source of each row.
            The column can be given a different name by providing a string argument.
            The column will have a Categorical type with the value of `left_only`
            for observations whose merge key only appears in the left DataFrame,
            `right_only` for observations whose merge key
            only appears in the right DataFrame, and `both` if the observation’s
            merge key is found in both DataFrames.
        force: If `True`, force the non-equi join conditions to execute before the equi join.
        sort_equi_join: If True and an equi-join is present, sort the join columns.
            Default is False.
        aggfunc: Compute aggregates on the right dataframe
            for each row of the left DataFrame (that has a match)
            based on the join keys.
            Supported aggregation functions are
            `sum`, `count`, `size`, `min`, `max`.


    Returns:
        A pandas DataFrame of the two merged Pandas objects.
    """  # noqa: E501

    return _conditional_join_compute(
        df=df,
        right=right,
        conditions=conditions,
        how=how,
        df_columns=df_columns,
        right_columns=right_columns,
        keep=keep,
        use_numba=use_numba,
        indicator=indicator,
        force=force,
        sort_equi_join=sort_equi_join,
        aggfunc=aggfunc,
    )


def _check_operator(op: str):
    """
    Check that operator is one of
    `>`, `>=`, `==`, `!=`, `<`, `<=`.

    Used in `conditional_join`.
    """
    sequence_of_operators = {op.value for op in helpers._JoinOperator}
    if op not in sequence_of_operators:
        raise ValueError(
            "The conditional join operator "
            f"should be one of {sequence_of_operators}"
        )


def _conditional_join_preliminary_checks(
    df: pd.DataFrame,
    right: pd.DataFrame | pd.Series,
    conditions: list[tuple],
    how: str,
    df_columns: Any,
    right_columns: Any,
    keep: str,
    use_numba: bool,
    indicator: bool | str,
    force: bool,
    return_matching_indices: bool,
    return_ranges: bool,
    sort_equi_join: bool,
    aggfunc: list[tuple] | None,
) -> tuple:
    """
    Preliminary checks for conditional_join are conducted here.

    Checks include differences in number of column levels,
    length of conditions, existence of columns in dataframe, etc.
    """

    check("right", right, [pd.DataFrame, pd.Series])

    df = df[:]
    right = right[:]

    if isinstance(right, pd.Series):
        if not right.name:
            raise ValueError(
                "Unnamed Series are not supported for conditional_join."
            )
        right = right.to_frame()

    if df.columns.nlevels != right.columns.nlevels:
        raise ValueError(
            "The number of column levels "
            "from the left and right frames must match. "
            "The number of column levels from the left dataframe "
            f"is {df.columns.nlevels}, while the number of column levels "
            f"from the right dataframe is {right.columns.nlevels}."
        )

    if not conditions:
        raise ValueError("Kindly provide at least one join condition.")

    for condition in conditions:
        check("condition", condition, [tuple])
        len_condition = len(condition)
        if len_condition != 3:
            raise ValueError(
                "condition should have only three elements; "
                f"{condition} however is of length {len_condition}."
            )

    for left_on, right_on, op in conditions:
        check("left_on", left_on, [Hashable])
        check("right_on", right_on, [Hashable])
        check("operator", op, [str])
        check_column(df, [left_on])
        check_column(right, [right_on])
        _check_operator(op)

    if all(
        (op == helpers._JoinOperator.NOT_EQUAL.value for *_, op in conditions)
    ):
        raise ValueError("!= only joins are not supported")

    check("how", how, [str])

    if how not in {"inner", "left", "right", "outer"}:
        raise ValueError(
            "'how' should be one of 'inner', 'left', 'right' or 'outer'."
        )

    check("keep", keep, [str])

    if keep not in {"all", "first", "last"}:
        raise ValueError("'keep' should be one of 'all', 'first', 'last'.")

    check("use_numba", use_numba, [bool])

    if (len(conditions) == 1) and use_numba:
        raise ValueError("numba is not supported for single joins")

    check("indicator", indicator, [bool, str])

    check("force", force, [bool])

    check("return_ranges", return_ranges, [bool])

    check(
        "sort_equi_join",
        sort_equi_join,
        [bool],
    )

    if return_ranges and use_numba:
        raise ValueError("return_ranges applies only when use_numba is False.")
    if aggfunc is not None:
        check("aggfunc", aggfunc, [list])
        if use_numba:
            raise NotImplementedError(
                "aggfunc does not apply when use_numba is True"
            )
        for entry in aggfunc:
            check("entry in aggfunc", entry, [tuple])
            if len(entry) != 2:
                raise ValueError(
                    "The tuple in an aggfunc should be 2 elements; "
                    "The first element in the tuple should be the column name "
                    "in the right dataframe, while the second element "
                    "in the tuple should be an aggregation function"
                )
        if how != "inner":
            raise ValueError("aggregation applies only when `how=inner`")
        if keep != "all":
            raise ValueError("aggregation applies only when `keep=all`")
        r_cols = right.columns
        aggs = {"sum", "min", "max", "count", "size"}
        for column_name, agg in aggfunc:
            if column_name not in r_cols:
                raise KeyError(
                    f"{column_name} in aggfunc does not exist in the right dataframe"
                )
            if agg not in aggs:
                raise ValueError(
                    f"The aggregation function for {column_name} "
                    f"should be one of {','.join(aggs)}; "
                    f"instead got {agg}"
                )
            if (agg == "sum") and not pd.api.types.is_numeric_dtype(
                right[column_name]
            ):
                raise ValueError(
                    f"{agg} is supported only for numeric columns"
                )
    if all(
        (
            op == helpers._JoinOperator.STRICTLY_EQUAL.value
            for *_, op in conditions
        )
    ):
        if not (return_matching_indices or (aggfunc is not None)):
            raise ValueError(
                "Equality only joins are supported only "
                "if aggfunc is provided, "
                "or only indices are to be returned."
            )
        if return_matching_indices and use_numba:
            raise ValueError(
                "Equality only joins are supported only "
                "if indices are to be returned, "
                "and use_numba is False."
            )

    return (
        df,
        right,
        conditions,
        how,
        df_columns,
        right_columns,
        keep,
        use_numba,
        indicator,
        force,
        return_ranges,
        sort_equi_join,
        aggfunc,
    )


def _conditional_join_type_check(
    left_column: pd.Series,
    right_column: pd.Series,
    op: str,
    use_numba: bool,
    sort_equi_join: bool,
) -> None:
    """
    Dtype check for columns in the join.
    """
    if (
        sort_equi_join
        and not pd.api.types.is_numeric_dtype(left_column)
        and not pd.api.types.is_datetime64_dtype(left_column)
        and not pd.api.types.is_timedelta64_dtype(left_column)
    ):
        raise ValueError(
            "binary search is supported only "
            "for numeric, datetime and timedelta dtypes."
        )
    if (
        ((op != helpers._JoinOperator.STRICTLY_EQUAL.value) or use_numba)
        and not is_numeric_dtype(left_column)
        and not is_datetime64_dtype(left_column)
        and not is_timedelta64_dtype(left_column)
    ):
        raise TypeError(
            "Only numeric, timedelta and datetime types "
            "are supported in a non equi-join, "
            "or if use_numba is set to True. "
            f"{left_column.name} in condition "
            f"({left_column.name}, {right_column.name}, {op}) "
            f"has a dtype {left_column.dtype}."
        )

    if not is_dtype_equal(left_column, right_column):
        raise TypeError(
            f"Both columns should have the same type - "
            f"'{left_column.name}' has {left_column.dtype} type;"
            f"'{right_column.name}' has {right_column.dtype} type."
        )

    return None


def _conditional_join_compute(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list,
    how: str,
    df_columns: Any,
    right_columns: Any,
    keep: str,
    use_numba: bool,
    indicator: bool | str,
    force: bool,
    sort_equi_join: bool,
    return_matching_indices: bool = False,
    return_ranges: bool = False,
    aggfunc: tuple = None,
) -> pd.DataFrame:
    """
    This is where the actual computation
    for the conditional join takes place.
    """

    (
        df,
        right,
        conditions,
        how,
        df_columns,
        right_columns,
        keep,
        use_numba,
        indicator,
        force,
        return_ranges,
        sort_equi_join,
        aggfunc,
    ) = _conditional_join_preliminary_checks(
        df=df,
        right=right,
        conditions=conditions,
        how=how,
        df_columns=df_columns,
        right_columns=right_columns,
        keep=keep,
        use_numba=use_numba,
        indicator=indicator,
        force=force,
        return_matching_indices=return_matching_indices,
        return_ranges=return_ranges,
        sort_equi_join=sort_equi_join,
        aggfunc=aggfunc,
    )
    eq_check = False
    for condition in conditions:
        left_on, right_on, op = condition
        _conditional_join_type_check(
            left_column=df[left_on],
            right_column=right[right_on],
            op=op,
            use_numba=use_numba,
            sort_equi_join=sort_equi_join,
        )
        if op == helpers._JoinOperator.STRICTLY_EQUAL.value:
            eq_check = True
    df.index = range(len(df))
    right.index = range(len(right))
    if eq_check:
        result = _multiple_conditional_join_eq(
            df=df,
            right=right,
            conditions=conditions,
            keep=keep,
            use_numba=use_numba,
            force=force,
            return_ranges=return_ranges,
            sort_equi_join=sort_equi_join,
            aggfunc=aggfunc,
        )
    elif len(conditions) > 1:
        result = _multiple_conditional_join_le_lt(
            df=df,
            right=right,
            conditions=conditions,
            keep=keep,
            use_numba=use_numba,
            return_ranges=return_ranges,
            aggfunc=aggfunc,
        )
    else:
        result = non_equi_join_single._single_le_ge_join(
            df=df,
            right=right,
            condition=conditions[0],
            aggfunc=aggfunc,
            return_ranges=return_ranges,
            keep=keep,
        )
    if aggfunc:
        return _create_frame_agg(
            df=df,
            df_columns=df_columns,
            agg_result=result,
            aggfunc=aggfunc,
            right=right,
        )

    if result is None:
        result = np.array([], dtype=np.intp), np.array([], dtype=np.intp)
    if return_matching_indices:
        return result

    left_index, right_index = result
    return _create_frame(
        df=df,
        right=right,
        left_index=left_index,
        right_index=right_index,
        how=how,
        df_columns=df_columns,
        right_columns=right_columns,
        indicator=indicator,
    )


def _create_multiindex_column(df: pd.DataFrame, right: pd.DataFrame) -> tuple:
    """
    Create a MultiIndex column for conditional_join.
    """
    header = np.empty(df.columns.size, dtype="U4")
    header[:] = "left"
    header = [header]
    columns = [
        df.columns.get_level_values(n) for n in range(df.columns.nlevels)
    ]
    header.extend(columns)
    df.columns = pd.MultiIndex.from_arrays(header)
    header = np.empty(right.columns.size, dtype="U5")
    header[:] = "right"
    header = [header]
    columns = [
        right.columns.get_level_values(n) for n in range(right.columns.nlevels)
    ]
    header.extend(columns)
    right.columns = pd.MultiIndex.from_arrays(header)
    return df, right


def _create_frame_agg(
    df: pd.DataFrame,
    right: pd.DataFrame,
    df_columns: Any,
    agg_result: tuple,
    aggfunc: list[tuple],
) -> pd.DataFrame:
    """
    Create final dataframe for an aggregation
    """
    if (df_columns is not None) and (df_columns != slice(None)):
        df = df.select(columns=df_columns)
    if (agg_result is None) and (df.columns.nlevels == 1):
        dictionary = {}
        dtypes = df.dtypes
        for column_name in df:
            dtype = dtypes.loc[column_name]
            arr = pd.array([], dtype=dtype, copy=False)
            dictionary[("left", column_name)] = arr
        dtypes = right.dtypes
        for column_name, agg in aggfunc:
            if agg in {"size", "count"}:
                dtype = "int64"
            else:
                dtype = dtypes.loc[column_name]
            arr = pd.array([], dtype=dtype, copy=False)
            dictionary[(column_name, agg)] = arr
        return pd.DataFrame(dictionary, copy=False)
    if agg_result is None:
        dictionary = {}
        dtypes = df.dtypes
        for column_name in df:
            dtype = dtypes.loc[column_name]
            arr = pd.array([], dtype=dtype, copy=False)
            dictionary[("left", *column_name)] = arr
        dtypes = right.dtypes
        for column_name, agg in aggfunc:
            if agg in {"size", "count"}:
                dtype = "int64"
            else:
                dtype = dtypes.loc[column_name]
            arr = pd.array([], dtype=dtype, copy=False)
            dictionary[(*column_name, agg)] = arr
        return pd.DataFrame(dictionary, copy=False)
    df_index = agg_result["df_index"]
    agg_result = agg_result["aggregates"]
    if df.columns.nlevels == 1:
        dictionary = {}
        for key, value in df.items():
            series = value._values[df_index]
            dictionary[("left", key)] = series
        for (column_name, agg_name), agg_array in zip(aggfunc, agg_result):
            dictionary[(column_name, agg_name)] = agg_array
        return pd.DataFrame(dictionary, copy=False, index=df_index)
    dictionary = {}
    for key, value in df.items():
        series = value._values[df_index]
        dictionary[("left", *key)] = series
    for (column_name, agg_name), agg_array in zip(aggfunc, agg_result):
        dictionary[(*column_name, agg_name)] = agg_array
    return pd.DataFrame(dictionary, copy=False, index=df_index)


def _create_frame(
    df: pd.DataFrame,
    right: pd.DataFrame,
    left_index: np.ndarray,
    right_index: np.ndarray,
    how: str,
    df_columns: Any,
    right_columns: Any,
    indicator: bool | str,
) -> pd.DataFrame:
    """
    Create final dataframe
    """
    if (df_columns is None) and (right_columns is None):
        raise ValueError("df_columns and right_columns cannot both be None.")
    if (df_columns is not None) and (df_columns != slice(None)):
        df = df.select(columns=df_columns)
    if (right_columns is not None) and (right_columns != slice(None)):
        right = right.select(columns=right_columns)
    if df_columns is None:
        df = pd.DataFrame([])
    elif right_columns is None:
        right = pd.DataFrame([])

    if not df.columns.intersection(right.columns).empty:
        df, right = _create_multiindex_column(df, right)

    def _add_indicator(
        indicator: bool | str,
        how: str,
        column_length: int,
        columns: pd.Index,
    ):
        """Adds a categorical column to the DataFrame,
        mapping the rows to either the left or right source DataFrames.

        Args:
            indicator: Indicator column name or True for default name "_merge".
            how: Type of join operation ("inner", "left", "right").
            column_length: Length of the categorical column.
            columns: Columns of the final DataFrame.

        Returns:
            A tuple containing the indicator column name
            and a Categorical array
            representing the indicator values for each row.

        """
        mapping = {"left": "left_only", "right": "right_only", "inner": "both"}
        categories = ["left_only", "right_only", "both"]
        if isinstance(indicator, bool):
            indicator = "_merge"
        if indicator in columns:
            raise ValueError(
                "Cannot use name of an existing column for indicator column"
            )
        nlevels = columns.nlevels
        if nlevels > 1:
            indicator = [indicator] + [""] * (nlevels - 1)
            indicator = tuple(indicator)
        if not column_length:
            arr = pd.Categorical([], categories=categories)
        else:
            arr = pd.Categorical(
                [mapping[how]],
                categories=categories,
            )
            if column_length > 1:
                arr = arr.repeat(column_length)
        return indicator, arr

    def _inner(
        df: pd.DataFrame,
        right: pd.DataFrame,
        left_index: np.ndarray,
        right_index: np.ndarray,
        indicator: bool | str,
    ) -> pd.DataFrame:
        """Computes an inner joined DataFrame.

        Args:
            df: The left DataFrame to join.
            right: The right DataFrame to join.
            left_index: indices from df for rows that match right.
            right_index: indices from right for rows that match df.
            indicator: Indicator column name or True for default name "_merge".

        Returns:
            An inner joined DataFrame.
        """
        dictionary = {}
        for key, value in df.items():
            dictionary[key] = value._values[left_index]
        for key, value in right.items():
            dictionary[key] = value._values[right_index]
        if indicator:
            indicator, arr = _add_indicator(
                indicator=indicator,
                how="inner",
                column_length=left_index.size,
                columns=df.columns.union(right.columns),
            )
            dictionary[indicator] = arr
        return pd.DataFrame(dictionary, copy=False)

    if how == "inner":
        return _inner(
            df=df,
            right=right,
            left_index=left_index,
            right_index=right_index,
            indicator=indicator,
        )
    if how == "left":
        indexer = pd.unique(left_index)
        indexer = pd.Index(indexer).get_indexer(range(len(df)))
        indexer = (indexer < 0).nonzero()[0]
        length = indexer.size
        if not length:
            return _inner(
                df=df,
                right=right,
                left_index=left_index,
                right_index=right_index,
                indicator=indicator,
            )
        dictionary = {}
        for key, value in df.items():
            array = value._values
            top = array[left_index]
            bottom = array[indexer]
            value = concat_compat([top, bottom])
            dictionary[key] = value
        for key, value in right.items():
            array = value._values
            value = array[right_index]
            other = helpers.construct_1d_array_from_inferred_fill_value(
                value=array[:1], length=length
            )
            value = concat_compat([value, other])
            dictionary[key] = value
        if indicator:
            columns = df.columns.union(right.columns)
            name, arr1 = _add_indicator(
                indicator=indicator,
                how="inner",
                column_length=right_index.size,
                columns=columns,
            )
            name, arr2 = _add_indicator(
                indicator=indicator,
                how="left",
                column_length=length,
                columns=columns,
            )
            value = concat_compat([arr1, arr2])
            dictionary[name] = value
        return pd.DataFrame(dictionary, copy=False)

    if how == "right":
        indexer = pd.unique(right_index)
        indexer = pd.Index(indexer).get_indexer(range(len(right)))
        indexer = (indexer < 0).nonzero()[0]
        length = indexer.size
        if not length:
            return _inner(
                df=df,
                right=right,
                left_index=left_index,
                right_index=right_index,
                indicator=indicator,
            )
        dictionary = {}
        for key, value in df.items():
            array = value._values
            value = array[left_index]
            other = helpers.construct_1d_array_from_inferred_fill_value(
                value=array[:1], length=length
            )
            value = concat_compat([value, other])
            dictionary[key] = value
        for key, value in right.items():
            array = value._values
            top = array[right_index]
            bottom = array[indexer]
            value = concat_compat([top, bottom])
            dictionary[key] = value
        if indicator:
            columns = df.columns.union(right.columns)
            name, arr1 = _add_indicator(
                indicator=indicator,
                how="inner",
                column_length=left_index.size,
                columns=columns,
            )
            name, arr2 = _add_indicator(
                indicator=indicator,
                how="right",
                column_length=length,
                columns=columns,
            )
            value = concat_compat([arr1, arr2])
            dictionary[name] = value
        return pd.DataFrame(dictionary, copy=False)
    # how == 'outer'
    left_indexer = pd.unique(left_index)
    left_indexer = pd.Index(left_indexer).get_indexer(range(len(df)))
    left_indexer = (left_indexer < 0).nonzero()[0]
    right_indexer = pd.unique(right_index)
    right_indexer = pd.Index(right_indexer).get_indexer(range(len(right)))
    right_indexer = (right_indexer < 0).nonzero()[0]

    df_nulls_length = left_indexer.size
    right_nulls_length = right_indexer.size
    dictionary = {}
    for key, value in df.items():
        array = value._values
        top = array[left_index]
        top = [top]
        if df_nulls_length:
            middle = array[left_indexer]
            top.append(middle)
        if right_nulls_length:
            bottom = helpers.construct_1d_array_from_inferred_fill_value(
                value=array[:1], length=right_nulls_length
            )
            top.append(bottom)
        top = concat_compat(top)
        dictionary[key] = top
    for key, value in right.items():
        array = value._values
        top = array[right_index]
        top = [top]
        if df_nulls_length:
            middle = helpers.construct_1d_array_from_inferred_fill_value(
                value=array[:1], length=df_nulls_length
            )
            top.append(middle)
        if right_nulls_length:
            bottom = array[right_indexer]
            top.append(bottom)
        top = concat_compat(top)
        dictionary[key] = top
    if indicator:
        columns = df.columns.union(right.columns)
        name, arr1 = _add_indicator(
            indicator=indicator,
            how="inner",
            column_length=right_index.size,
            columns=columns,
        )
        arr1 = [arr1]
        if df_nulls_length:
            name, arr2 = _add_indicator(
                indicator=indicator,
                how="left",
                column_length=df_nulls_length,
                columns=columns,
            )
            arr1.append(arr2)
        if right_nulls_length:
            name, arr3 = _add_indicator(
                indicator=indicator,
                how="right",
                column_length=right_nulls_length,
                columns=columns,
            )
            arr1.append(arr3)
        arr1 = concat_compat(arr1)
        dictionary[name] = arr1

    return pd.DataFrame(dictionary, copy=False)


@deprecated_alias(return_ragged_arrays="return_ranges")
def get_join_indices(
    df: pd.DataFrame,
    right: pd.DataFrame | pd.Series,
    conditions: list[tuple[str]],
    keep: Literal["first", "last", "all"] = "all",
    use_numba: bool = False,
    force: bool = False,
    return_ranges: bool = False,
    sort_equi_join=False,
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience function to return the matching indices from an inner join.

    !!! info "New in version 0.27.0"

    !!! abstract "Version Changed"

        - 0.29.0
            - Add support for ragged array indices.
        - 0.32.0
            - ragged array indices deprecated.
            - return_ranges parameter added.

    Args:
        df: A pandas DataFrame.
        right: Named Series or DataFrame to join to.
        conditions: List of arguments of tuple(s) of the form
            `(left_on, right_on, op)`, where `left_on` is the column
            label from `df`, `right_on` is the column label from `right`,
            while `op` is the operator.
            The operator can be any of
            `==`, `!=`, `<=`, `<`, `>=`, `>`. For multiple conditions,
            the and(`&`) operator is used to combine the results
            of the individual conditions.
        use_numba: Use numba, if installed, to accelerate the computation.
        keep: Choose whether to return the first match, last match or all matches.
        force: If `True`, force the non-equi join conditions
            to execute before the equi join.
        return_ranges: If `True`, return ranges of matching right indices
            for each matching left index. Not applicable if `use_numba` is `True`.
            If `return_ranges` is `True`, the join condition
            should be a single join, or a range join,
            where the right columns are both monotonically increasing.
            If none of the above conditions are met, ranges are not returned;
            instead a tuple of indices for the rows in the dataframes that
            match is returned.
        sort_equi_join: If True and an equi-join is present, sort the join columns.
            Default is False.

    Returns:
        A tuple of indices for the rows in the dataframes that match.
    """
    return _conditional_join_compute(
        df=df,
        right=right,
        conditions=conditions,
        how="inner",
        df_columns=None,
        right_columns=None,
        keep=keep,
        use_numba=use_numba,
        indicator=False,
        force=force,
        return_matching_indices=True,
        return_ranges=return_ranges,
        sort_equi_join=sort_equi_join,
        aggfunc=None,
    )
