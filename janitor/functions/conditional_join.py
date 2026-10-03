"""Public conditional-join API and dispatch coordinator.

The public methods in this module intentionally describe pandas behavior and
do not expose the implementation language used by the kernels.  Internally,
the dispatcher normalizes both inputs to shallow working copies, validates
the complete predicate list, resets indexes to physical row positions, and
selects one specialized join family:

* equality-led joins use the equi-key dispatcher;
* all-``!=`` joins use the null-aware not-equal dispatcher;
* one range predicate, optionally followed by residuals, uses the
  single-range dispatcher;
* two or more range predicates use either the range or regions dispatcher.

Every dispatcher returns physical position arrays or an aggregation result.
The shared helpers materialize those positions back into pandas rows, which
keeps outer-join, indicator, column-selection, and positional-index behavior
consistent across all algorithms.
"""

from __future__ import annotations

import warnings
from typing import Any, Hashable, Literal, Optional

import pandas as pd
import pandas_flavor as pf
from pandas.api.types import (
    is_datetime64_dtype,
    is_dtype_equal,
    is_numeric_dtype,
    is_timedelta64_dtype,
)

from janitor.utils import check, check_column, deprecated_kwargs

from ._conditional_join import (
    _equi_join,
    _maybe_range_join,
    _not_equals_only,
    _regions,
    _single_range_predicate,
)
from ._conditional_join._helpers import (
    _JoinOperator,
    greater_than_join_types,
    less_than_join_types,
)


@pf.register_dataframe_method
def conditional_join(
    df: pd.DataFrame,
    right: pd.DataFrame | pd.Series,
    *conditions: tuple,
    how: Literal["inner", "left", "right", "outer"] = "inner",
    df_columns: Optional[Any] = slice(None),
    right_columns: Optional[Any] = slice(None),
    keep: Literal["first", "last", "any", "all"] = "all",
    indicator: Optional[bool | str] = False,
    force: bool = False,
    join_algorithm: str = "default",
    include_join_positions: bool = False,
) -> pd.DataFrame:
    """The conditional_join function operates similarly to `pd.merge`,
    but supports joins on inequality operators,
    or a combination of equi and non-equi joins.

    Joins solely on equality are not supported.

    If the join is solely on equality, `pd.merge` function
    covers that; if you are interested in nearest joins, asof joins,
    or rolling joins, then `pd.merge_asof` covers that.
    There is also pandas' IntervalIndex, which is efficient for range joins,
    especially if the intervals do not overlap.

    Column selection in `df_columns` and `right_columns` is possible using the
    [`select_columns`][janitor.functions.select.select_columns] syntax.

    !!! warning
        The `df_columns` and `right_columns` parameters are deprecated.
        Select or rename columns on the DataFrame before calling `conditional_join`.

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

    The operator can be any of `==`, `!=`, `<=`, `<`, `>=`, `>`.

    For a single `!=` condition with `keep="first"` or `keep="last"`,
    matching positions are selected without materializing all unequal pairs.
    For multiple all-`!=` conditions, candidate pairs are formed from the
    first condition and then filtered by the remaining conditions before
    applying `keep`.

    The join is done only on the columns.

    For non-equi joins, only numeric, timedelta and date columns are supported.

    `inner`, `left`, `right` and `outer` joins are supported.

    If the columns from `df` and `right` have nothing in common,
    a single index column is returned; else, a MultiIndex column
    is returned.

    If `include_join_positions` is `True`, the index of the returned dataframe
    will be a MultiIndex; the first level points to the original positions in `df`,
    while the second level points to the original positions in `right`.

    Examples:
        >>> import pandas as pd
        >>> import janitor
        >>> df1 = pd.DataFrame({"value_1": [2, 5, 7, 1, 3, 4]})
        >>> df2 = pd.DataFrame(
        ...     {
        ...         "value_2A": [0, 3, 7, 12, 0, 2, 3, 1],
        ...         "value_2B": [1, 5, 9, 15, 1, 4, 6, 3],
        ...     }
        ... )
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
        ...     df2, ("value_1", "value_2A", ">"), ("value_1", "value_2B", "<")
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
        ...     right_columns="value_2B",
        ...     how="left",
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
        >>> (
        ...     df1.rename(columns={"value_1": "left_column"}).conditional_join(
        ...         df2,
        ...         ("left_column", "value_2A", ">"),
        ...         ("left_column", "value_2B", "<"),
        ...         right_columns="value_2B",
        ...         how="outer",
        ...     )
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
        ...     keep="first",
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
        ...     keep="last",
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
        ...     how="outer",
        ...     indicator=True,
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

    !!! abstract "Version Changed"

        - 0.24.0
            - Added `df_columns`, `right_columns` and `keep` parameters.
        - 0.24.1
            - Added `indicator` parameter.
        - 0.25.0
            - `col` class supported.
            - Outer join supported. `sort_by_appearance` is deprecated.
        - 0.27.0
            - Added support for timedelta dtype.
        - 0.28.0
            - `col` class is deprecated.
        - 0.32.9
        - 0.32.10
            - Added `include_join_positions` parameter.
            - Added `join_algorithm` parameter.

    Args:
        df: A pandas DataFrame.
        right: Named Series or DataFrame to join to.
        conditions: Variable argument of tuple(s) of the form
            `(left_on, right_on, op)`, where `left_on` is the column
            label from `df`, `right_on` is the column label from `right`,
            while `op` is the operator.
            The operator can be any of
            `==`, `!=`, `<=`, `<`, `>=`, `>`. For multiple conditions,
            the and(`&`) operator is used to combine the results
            of the individual conditions.
        how: Indicates the type of join to be performed.
            It can be one of `inner`, `left`, `right` or `outer`.
        df_columns: Columns to select from `df` in the final output dataframe.
            Column selection is based on the
            [`select_columns`][janitor.functions.select.select_columns] syntax.
            !!! warning "Deprecated in 0.33.0"
                `df_columns` will be removed in a future release.
                Select or rename columns directly on the DataFrame before calling `conditional_join`.
        right_columns: Columns to select from `right` in the final output dataframe.
            Column selection is based on the
            [`select_columns`][janitor.functions.select.select_columns] syntax.
            !!! warning "Deprecated in 0.33.0"
                `right_columns` will be removed in a future release.
                Select or rename columns directly on the DataFrame before calling `conditional_join`.
            !!! warning "Deprecated in 0.33.0"
        keep: Choose whether to return the first match, last match, any match,
            or all matches.
        indicator: If `True`, adds a column to the output DataFrame
            called `_merge` with information on the source of each row.
            The column can be given a different name by providing a string argument.
            The column will have a Categorical type with the value of `left_only`
            for observations whose merge key only appears in the left DataFrame,
            `right_only` for observations whose merge key
            only appears in the right DataFrame, and `both` if the observation’s
            merge key is found in both DataFrames.
        force: If `True`, force the non-equi join conditions to execute before the equi join.
        join_algorithm: Determines what algorithm to use for multiple non-equi joins.
            Currently limited to `default` and `regions`.
        include_join_positions: Determines if the join positions of the left and right DataFrame
            should be included as an index of the final dataframe.



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
        indicator=indicator,
        force=force,
        aggfunc=None,
        include_join_positions=include_join_positions,
        return_building_blocks=False,
        reverse=False,
        return_matching_indices=False,
        join_algorithm=join_algorithm,
    )


def _check_operator(op: str):
    """Validate one public conditional-join operator.

    Args:
        op: Operator supplied in a three-element condition tuple.

    Raises:
        ValueError: If ``op`` is not one of ``>``, ``>=``, ``==``, ``!=``,
            ``<``, or ``<=``.
    """
    sequence_of_operators = {op.value for op in _JoinOperator}
    if op not in sequence_of_operators:
        raise ValueError(
            f"The conditional join operator should be one of {sequence_of_operators}"
        )


def _check_conditions(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: tuple,
) -> None:
    """Validate all public predicate tuples before dispatch.

    This is deliberately separate from dtype validation.  Shape and column
    errors can be reported without inspecting values, while dtype validation
    is performed later for each pair of referenced columns. Keeping these
    phases separate also lets equality-only aggregation requests reach the
    equi aggregation path without accidentally routing ordinary equality
    joins through a non-equi kernel.

    Args:
        df: Left dataframe whose columns are referenced by the conditions.
        right: Right dataframe whose columns are referenced by the conditions.
        conditions: Public ``(left_column, right_column, operator)`` tuples.

    Raises:
        ValueError: If no conditions are supplied, a tuple has the wrong
            length, or an operator is unsupported.
        TypeError: If a condition or column reference has the wrong type.
        KeyError: If a referenced column does not exist.
    """
    if not conditions:
        raise ValueError("Kindly provide at least one join condition.")

    for condition in conditions:
        check("condition", condition, [tuple])
        if len(condition) != 3:
            raise ValueError(
                "condition should have only three elements; "
                f"{condition} however is of length {len(condition)}."
            )

        left_on, right_on, op = condition
        check("left_on", left_on, [Hashable])
        check("right_on", right_on, [Hashable])
        check("operator", op, [str])
        check_column(df, [left_on])
        check_column(right, [right_on])
        _check_operator(op)


def _check_aggfunc(
    df: pd.DataFrame,
    right: pd.DataFrame,
    aggfunc: list[tuple] | None,
    reverse: bool,
) -> None:
    """Validate aggregation requests against their source dataframe.

    Aggregations read from the right dataframe for forward joins and from the
    left dataframe for reverse joins. Validation follows that same direction
    so a request cannot silently read from the wrong side after dispatch.

    Args:
        df: Left dataframe and reverse-aggregation source.
        right: Right dataframe and forward-aggregation source.
        aggfunc: ``(column, operation)`` requests, or ``None`` for index
            output. Supported operations are ``sum``, ``count``, ``min``,
            ``max``, ``size``, and ``prod``.
        reverse: Whether source values come from ``df`` instead of ``right``.

    Raises:
        ValueError: If a request is malformed, unsupported, or uses an
            operation incompatible with its source dtype.
        KeyError: If a requested aggregation column is absent.
    """
    if aggfunc is None:
        return

    check("aggfunc", aggfunc, [list])
    frame = df if reverse else right
    side = "left" if reverse else "right"
    supported = {"sum", "count", "min", "max", "size", "prod"}

    for entry in aggfunc:
        check("entry in aggfunc", entry, [tuple])
        if len(entry) != 2:
            raise ValueError(
                "The tuple in an aggfunc should be 2 elements; "
                "The first element in the tuple should be a column name "
                f"in the {side} dataframe, while the second element "
                "should be a supported aggregation function"
            )

        column_name, agg = entry
        if column_name not in frame.columns:
            raise KeyError(
                f"{column_name} in aggfunc does not exist in the {side} dataframe"
            )
        if agg not in supported:
            raise ValueError(
                f"The aggregation function for {column_name} should be one of "
                f"{','.join(supported)}; instead got {agg}"
            )

        series = frame[column_name]
        if agg in {"sum", "prod"} and not is_numeric_dtype(series):
            raise ValueError(f"{agg} is supported only for numeric columns")
        if (
            agg in {"min", "max"}
            and not is_numeric_dtype(series)
            and not is_datetime64_dtype(series)
            and not is_timedelta64_dtype(series)
        ):
            raise ValueError(
                f"{agg} is supported only for numeric, datetime, timedelta columns"
            )


def _conditional_join_preliminary_checks(
    df: pd.DataFrame,
    right: pd.DataFrame | pd.Series,
    conditions: tuple,
    how: str,
    df_columns: Any,
    right_columns: Any,
    keep: str,
    indicator: bool | str,
    force: bool,
    return_matching_indices: bool = False,
    aggfunc: list[tuple] = None,
    include_join_positions: bool = False,
    return_building_blocks: bool = False,
    reverse: bool = False,
    join_algorithm: str = "default",
    return_matched: bool = False,
) -> tuple:
    """Normalize and validate inputs shared by every conditional-join path.

    This function owns public argument validation and returns shallow copies
    so downstream preparation can reset indexes without mutating caller-owned
    dataframes. It intentionally does not choose a Rust/kernel path; that
    decision belongs to :func:`_conditional_join_compute` after dtype checks.

    Args:
        df: Left dataframe.
        right: Right dataframe or named Series, converted to a dataframe.
        conditions: Public predicate tuples.
        how: Requested join shape.
        df_columns: Deprecated left-column selection.
        right_columns: Deprecated right-column selection.
        keep: Match-selection policy.
        indicator: Whether to include a merge indicator column.
        force: Whether equality predicates may be handled by the non-equi
            preparation path.
        return_matching_indices: Whether callers want physical index arrays.
        aggfunc: Aggregation requests, when the caller is ``join_agg``.
        include_join_positions: Whether materialized output includes pair
            positions in its index.
        return_building_blocks: Whether to preserve kernel building blocks.
        reverse: Whether aggregation reads from the left side.
        join_algorithm: Multi-range algorithm selection.
        return_matched: Whether aggregation output includes a match mask.

    Returns:
        Shallow copies of the validated left and right dataframes.

    Raises:
        TypeError, ValueError, or KeyError: If a public argument is invalid.
    """

    check("right", right, [pd.DataFrame, pd.Series])

    if isinstance(right, pd.Series):
        if not right.name:
            raise ValueError("Unnamed Series are not supported for conditional_join.")
        right = right.to_frame()

    if df_columns != slice(None):
        warnings.warn(
            "The 'df_columns' parameter is deprecated and will be removed in a "
            "future release. Please select or rename columns on the left "
            "DataFrame before calling conditional_join.",
            DeprecationWarning,
            stacklevel=2,
        )

    if right_columns != slice(None):
        warnings.warn(
            "The 'right_columns' parameter is deprecated and will be removed in a "
            "future release. Please select or rename columns on the right "
            "DataFrame before calling conditional_join.",
            DeprecationWarning,
            stacklevel=2,
        )

    # Check MultiIndex column level mismatch first, before any column existence checks
    if df.columns.nlevels != right.columns.nlevels:
        raise ValueError(
            "The number of column levels "
            "from the left and right frames must match. "
            "The number of column levels from the left dataframe "
            f"is {df.columns.nlevels}, while the number of column levels "
            f"from the right dataframe is {right.columns.nlevels}."
        )

    _check_conditions(df, right, conditions)

    check("how", how, [str])

    if how not in {"inner", "left", "right", "outer"}:
        raise ValueError("'how' should be one of 'inner', 'left', 'right' or 'outer'.")

    check("keep", keep, [str])

    if keep not in {"all", "first", "last", "any"}:
        raise ValueError("'keep' should be one of 'all', 'first', 'last', 'any'.")

    check("indicator", indicator, [bool, str])

    check("force", force, [bool])

    check("reverse", reverse, [bool])

    _check_aggfunc(df, right, aggfunc, reverse)
    if all((op == _JoinOperator.STRICTLY_EQUAL.value for *_, op in conditions)):
        if not (return_matching_indices or aggfunc):
            raise ValueError("Equality only joins are not supported.")

    check("include_join_positions", include_join_positions, [bool])
    if include_join_positions and (how != "inner"):
        raise ValueError("include_join_positions is valid only if `how='inner'`")
    check("return_building_blocks", return_building_blocks, [bool])
    if all(op == "!=" for *_, op in conditions) and return_building_blocks:
        keep = "all"

    check("join_algorithm", join_algorithm, [str])
    if join_algorithm not in {"default", "regions"}:
        raise ValueError(
            f"join_algorithm should be either default or regions, "
            f"instead got {join_algorithm}"
        )
    check("return_matched", return_matched, [bool])

    # Only index and column metadata are reassigned downstream. Shallow copies
    # protect the caller's frames without duplicating every column buffer.
    return df.copy(deep=False), right.copy(deep=False)


def _conditional_join_type_check(
    left_column: pd.Series,
    right_column: pd.Series,
    op: str,
    force: bool,
) -> None:
    """Validate the dtype contract for one pair of join columns.

    Equality columns may use arbitrary pandas dtypes on the normal equi path.
    A forced equality or any inequality must use equal, numeric, datetime, or
    timedelta dtypes because those paths eventually use typed positional
    kernels.

    Args:
        left_column: Left condition column.
        right_column: Right condition column.
        op: Comparison operator for the condition.
        force: Whether an equality condition is being forced through the
            non-equi preparation path.

    Raises:
        TypeError: If an inequality-compatible dtype is unsupported or the
            two columns have unequal dtypes.
    """

    strictly_equal = op == _JoinOperator.STRICTLY_EQUAL.value
    if (
        ((not strictly_equal) or (force and strictly_equal))
        and not is_numeric_dtype(left_column)
        and not is_datetime64_dtype(left_column)
        and not is_timedelta64_dtype(left_column)
    ):
        raise TypeError(
            "Only numeric, timedelta and datetime types "
            "are supported in a non equi-join, "
            f"{left_column.name} in condition "
            f"({left_column.name}, {right_column.name}, {op}) "
            f"has a dtype {left_column.dtype}."
        )

    if ((not strictly_equal) or (force and strictly_equal)) and not is_dtype_equal(
        left_column, right_column
    ):
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
    indicator: bool | str,
    force: bool,
    return_matching_indices: bool = False,
    aggfunc: list[tuple] = None,
    include_join_positions: bool = False,
    return_building_blocks: bool = False,
    reverse: bool = False,
    join_algorithm: str = "default",
    return_matched: bool = False,
) -> pd.DataFrame:
    """Execute the validated conditional join or aggregation request.

    The dispatcher first resets working indexes to physical positions. Those
    positions are the only indexes sent to Rust and remain paired with sorted
    values throughout preparation. The selected family returns physical pairs
    or aggregation arrays; the shared materializer then restores the pandas
    result shape.

    Args:
        df: Validated left dataframe.
        right: Validated right dataframe.
        conditions: Validated predicate tuples.
        how: Requested join shape.
        df_columns: Deprecated left output selection.
        right_columns: Deprecated right output selection.
        keep: Match-selection policy.
        indicator: Indicator-column request.
        force: Whether equality conditions may use non-equi preparation.
        return_matching_indices: Return physical index arrays instead of rows.
        aggfunc: Aggregation requests, or ``None`` for index output.
        include_join_positions: Include physical pair positions in dataframe
            output.
        return_building_blocks: Preserve starts/ends or equivalent blocks.
        reverse: Aggregate left values into right output rows.
        join_algorithm: Multi-range algorithm, ``default`` or ``regions``.
        return_matched: Include aggregation match metadata.

    Returns:
        A dataframe, physical-index dictionary, or kernel building-block
        dictionary depending on the requested mode.
    """
    df, right = _conditional_join_preliminary_checks(
        df=df,
        right=right,
        conditions=conditions,
        how=how,
        df_columns=df_columns,
        right_columns=right_columns,
        keep=keep,
        indicator=indicator,
        force=force,
        return_matching_indices=return_matching_indices,
        aggfunc=aggfunc,
        include_join_positions=include_join_positions,
        return_building_blocks=return_building_blocks,
        reverse=reverse,
        join_algorithm=join_algorithm,
        return_matched=return_matched,
    )

    for condition in conditions:
        left_on, right_on, op = condition
        _conditional_join_type_check(
            left_column=df[left_on],
            right_column=right[right_on],
            op=op,
            force=force,
        )

    df.index = range(len(df))
    right.index = range(len(right))

    index_result_kwargs = {
        "how": how,
        "df_columns": df_columns,
        "right_columns": right_columns,
        "indicator": indicator,
        "include_join_positions": include_join_positions,
        "return_matching_indices": return_matching_indices,
    }

    eq_check = any(op == _JoinOperator.STRICTLY_EQUAL.value for *_, op in conditions)
    has_range = any(
        op in less_than_join_types.union(greater_than_join_types)
        for *_, op in conditions
    )
    use_equi_path = eq_check and (not force or not has_range)
    if use_equi_path and aggfunc:
        return _equi_join._aggregate(
            df=df,
            right=right,
            conditions=conditions,
            aggfunc=aggfunc,
            reverse=reverse,
            return_matched=return_matched,
        )
    if use_equi_path:
        return _equi_join._compute_equi_join(
            df=df,
            right=right,
            conditions=conditions,
            keep=keep,
            return_building_blocks=return_building_blocks,
            **index_result_kwargs,
        )

    # All-!= predicates have their own null-aware Rust ABI. Route them before
    # the mixed non-equi algorithms; join_algorithm is intentionally ignored
    # because this family never enters the regions/range dispatch.
    all_nes_check = all(op == "!=" for *_, op in conditions)
    if all_nes_check and aggfunc:
        # The dedicated all-!= aggregation path returns a schema-only empty
        # frame when no pair survives, preserving the requested index shape.
        return _not_equals_only._aggregate(
            df=df,
            right=right,
            conditions=conditions,
            aggfunc=aggfunc,
            reverse=reverse,
            return_matched=return_matched,
        )
    if all_nes_check and return_building_blocks:
        return _not_equals_only._compute_not_equals_join(
            df=df,
            right=right,
            conditions=conditions,
            keep="all",
            return_building_blocks=return_building_blocks,
            **index_result_kwargs,
        )
    if all_nes_check:
        return _not_equals_only._compute_not_equals_join(
            df=df,
            right=right,
            conditions=conditions,
            keep=keep,
            return_building_blocks=return_building_blocks,
            **index_result_kwargs,
        )
    single_join_check = len(conditions) == 1
    if single_join_check and aggfunc:
        return _single_range_predicate._aggregate_single_join(
            df=df,
            right=right,
            condition=conditions[0],
            aggfunc=aggfunc,
            return_matched=return_matched,
            reverse=reverse,
        )
    if single_join_check:
        return _single_range_predicate._compute_single_range_join(
            df=df,
            right=right,
            condition=conditions[0],
            keep=keep,
            return_building_blocks=return_building_blocks,
            **index_result_kwargs,
        )
    counter = 0
    for *_, op in conditions:
        if op not in less_than_join_types.union(greater_than_join_types):
            continue
        counter += 1
    # A join with exactly one range predicate uses the range-first Rust
    # boundary even when residual predicates are present. The first range
    # owns the sorted right search layout; residuals are filtered inside Rust
    # before keep/aggregation semantics are applied.
    if (counter == 1) and aggfunc:
        return _single_range_predicate._aggregate_multiple_join(
            df=df,
            right=right,
            conditions=conditions,
            aggfunc=aggfunc,
            return_matched=return_matched,
            reverse=reverse,
        )
    if (counter == 1) and return_building_blocks:
        return _single_range_predicate._compute_multi_range_join(
            df=df,
            right=right,
            conditions=conditions,
            keep="all",
            return_building_blocks=return_building_blocks,
            **index_result_kwargs,
        )
    if counter == 1:
        return _single_range_predicate._compute_multi_range_join(
            df=df,
            right=right,
            conditions=conditions,
            keep=keep,
            return_building_blocks=return_building_blocks,
            **index_result_kwargs,
        )

    if (counter > 1) and (join_algorithm == "regions") and aggfunc:
        return _regions._aggregate(
            df=df,
            right=right,
            conditions=conditions,
            aggfunc=aggfunc,
            return_matched=return_matched,
            reverse=reverse,
        )
    if (counter > 1) and (join_algorithm == "regions"):
        return _regions._compute_regions_join(
            df=df,
            right=right,
            conditions=conditions,
            keep=keep,
            return_building_blocks=return_building_blocks,
            **index_result_kwargs,
        )

    if (counter > 1) and aggfunc and (join_algorithm == "default"):
        return _maybe_range_join._aggregate(
            df=df,
            right=right,
            conditions=conditions,
            aggfunc=aggfunc,
            return_matched=return_matched,
            reverse=reverse,
        )
    if (counter > 1) and (join_algorithm == "default"):
        return _maybe_range_join._compute_multi_range_join(
            df=df,
            right=right,
            conditions=conditions,
            keep=keep,
            return_building_blocks=return_building_blocks,
            **index_result_kwargs,
        )


@deprecated_kwargs("return_ragged_arrays")
def get_join_indices(
    df: pd.DataFrame,
    right: pd.DataFrame | pd.Series,
    *conditions: tuple,
    keep: Literal["first", "last", "any", "all"] = "all",
    force: bool = False,
    return_building_blocks: bool = False,
    join_algorithm: str = "default",
) -> dict:
    """Return matching physical positions for an inner conditional join.

    Unlike :func:`conditional_join`, this helper does not gather dataframe
    rows. It returns zero-based physical positions in two parallel arrays;
    ``left_index[i]`` and ``right_index[i]`` identify one matched pair. The
    arrays are suitable for callers that need to perform their own material
    or aggregation step. With ``return_building_blocks=True``, the selected
    kernel may also return range windows such as ``starts`` and ``ends``.

    Args:
        df: Left dataframe.
        right: Right dataframe or named Series.
        conditions: ``(left_column, right_column, operator)`` predicates.
        keep: Return all matches, or one ``first``, ``last``, or ``any`` match
            per left row.
        force: Permit equality predicates to participate in a forced non-equi
            preparation path.
        return_building_blocks: Return the kernel's intermediate positional
            representation instead of only materialized pairs.
        join_algorithm: Algorithm for multiple range predicates.

    Returns:
        A dictionary containing parallel physical-position arrays. The result
        is empty when no pair satisfies every predicate.
    """
    return _conditional_join_compute(
        df=df,
        right=right,
        conditions=conditions,
        how="inner",
        df_columns=None,
        right_columns=None,
        keep=keep,
        indicator=False,
        force=force,
        return_matching_indices=True,
        aggfunc=None,
        include_join_positions=False,
        return_building_blocks=return_building_blocks,
        reverse=False,
        join_algorithm=join_algorithm,
    )


@pf.register_dataframe_method
def join_agg(
    df: pd.DataFrame,
    right: pd.DataFrame | pd.Series,
    *conditions,
    aggfunc: list[tuple],
    force: bool = False,
    reverse: bool = False,
    return_matched: bool = True,
    join_algorithm: str = "default",
) -> pd.DataFrame:
    """Compute aggregations over rows matched by a conditional join.

    ``aggfunc`` contains ``(column, operation)`` pairs. Supported operations
    are ``sum``, ``count``, ``size``, ``min``, ``max`` and ``prod``. The
    result retains one row for every physical row in the aggregation domain;
    when ``return_matched`` is true, its index also contains the match mask.

    Forward aggregation groups right-side values by left rows. Set
    ``reverse=True`` to group left-side values by right rows. The aggregation
    source arrays may be filtered or sorted internally, but their physical
    position maps remain aligned so extrema and residual predicates refer to
    the original dataframe rows.

    Args:
        df: Left dataframe and reverse-aggregation source.
        right: Right dataframe or named Series and forward-aggregation source.
        conditions: Conditional-join predicate tuples.
        aggfunc: Non-empty ``(column, operation)`` requests.
        force: Permit equality predicates to use the forced non-equi path.
        reverse: Group left-side values into right-side output rows.
        return_matched: Add a boolean ``matched`` level to the result index.
        join_algorithm: Algorithm for multiple range predicates.

    Returns:
        A dataframe whose columns are labelled ``(column, operation)`` and
        whose rows follow the physical output side.
    """
    return _conditional_join_compute(
        df=df,
        right=right,
        conditions=conditions,
        how="inner",
        df_columns=None,
        right_columns=None,
        keep="all",
        indicator=False,
        force=force,
        return_matching_indices=False,
        aggfunc=aggfunc,
        return_matched=return_matched,
        reverse=reverse,
        join_algorithm=join_algorithm,
    )
