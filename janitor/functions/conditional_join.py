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
    _multi_range_join,
    _not_equals_only,
    _regions,
    _single_range_predicate,
)
from ._conditional_join._helpers import (
    _JoinOperator,
    _normalize_conditions,
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
    use_numba: bool | None = None,
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

    In some scenarios there might be performance gains if a mixed
    equality/range join uses the non-equi path. The range predicates drive
    candidate generation before the equality predicates are applied as
    residual filters; pass ``force=True`` to request this. It has no effect on
    joins that mix equality and ``!=`` predicates without a range predicate.

    ``join_algorithm`` selects the strategy for joins with multiple range
    predicates. ``"default"`` uses the general range-join implementation;
    ``"regions"`` uses the region-based implementation, which can be useful
    when the predicates describe interval-like regions. The option is ignored
    for equality-only, single-range, and all-``!=`` joins.

    The operator can be any of `==`, `!=`, `<=`, `<`, `>=`, `>`.

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

        Get any match for each left row. The selected match is not ordered:
        >>> df1.conditional_join(
        ...     df2,
        ...     ("value_1", "value_2A", ">"),
        ...     ("value_1", "value_2B", "<"),
        ...     keep="any",
        ... )
           value_1  value_2A  value_2B
        0        2         1         3
        1        5         3         6
        2        3         2         4
        3        4         3         5

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

        Use ``force=True`` when a mixed equality/range join should use the
        non-equi preparation order, and select the regions algorithm for a
        multi-range join when desired:

        >>> forced = df1.conditional_join(
        ...     df2,
        ...     ("value_1", "value_2A", "=="),
        ...     ("value_1", "value_2B", "<"),
        ...     force=True,
        ... )
        >>> regional = df1.conditional_join(
        ...     df2,
        ...     ("value_1", "value_2A", ">"),
        ...     ("value_1", "value_2B", "<"),
        ...     join_algorithm="regions",
        ...     include_join_positions=True,
        ... )
        >>> isinstance(regional.index, pd.MultiIndex)
        True

        ``include_join_positions=True`` adds the physical left and right row
        positions to the result index. These are zero-based positions, not the
        original pandas index labels:

        >>> left = pd.DataFrame({"value": [1, 2]})
        >>> right = pd.DataFrame({"value": [2, 3]})
        >>> positioned = left.conditional_join(
        ...     right,
        ...     ("value", "value", "<"),
        ...     include_join_positions=True,
        ... )
        >>> print(positioned.to_string())
             left right
            value value
        0 0     1     2
          1     1     3
        1 1     2     3
        >>> positioned.index.tolist()
        [(0, 0), (0, 1), (1, 1)]

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
        - 0.32.10
            - Added `include_join_positions` parameter.
            - Added `join_algorithm` parameter.
        - 0.32.27
            - The `use_numba` parameter is deprecated and has no effect.

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
        keep: Choose whether to return the first match, last match, any match,
            or all matches.
        use_numba: Deprecated no-op retained for compatibility with older
            callers. Its value is ignored.
        indicator: If `True`, adds a column to the output DataFrame
            called `_merge` with information on the source of each row.
            The column can be given a different name by providing a string argument.
            The column will have a Categorical type with the value of `left_only`
            for observations whose merge key only appears in the left DataFrame,
            `right_only` for observations whose merge key
            only appears in the right DataFrame, and `both` if the observation’s
            merge key is found in both DataFrames.
        force: If ``True``, force mixed equality/range joins to use the
            non-equi path, with range predicates driving candidate generation
            before equality predicates are applied. It has no effect on joins
            that mix equality and ``!=`` predicates without a range predicate.
        join_algorithm: Strategy for joins with multiple range predicates.
            ``"default"`` uses the general range-join implementation and
            ``"regions"`` uses the region-based implementation. The option is
            ignored for equality-only, single-range, and all-``!=`` joins.
        include_join_positions: If ``True``, include the matched physical row
            positions in the result index as a two-level ``MultiIndex``. The
            first level contains zero-based positions from ``df`` and the
            second contains zero-based positions from ``right``; these are
            positions in the input row order, not the original index labels.
            This option is available only for inner joins.



    Returns:
        A pandas DataFrame of the two merged Pandas objects.
    """  # noqa: E501

    if use_numba is not None:
        warnings.warn(
            "The 'use_numba' parameter is deprecated and has no effect.",
            DeprecationWarning,
            stacklevel=2,
        )

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
        force: If ``True``, force mixed equality/range joins to use the
            non-equi path, with range predicates driving candidate generation
            before equality predicates are applied. It has no effect on joins
            that mix equality and ``!=`` predicates without a range predicate.
        return_matching_indices: Whether callers want physical index arrays.
        aggfunc: Aggregation requests, when the caller is ``join_agg``.
        include_join_positions: Whether materialized inner-join output includes
            matched physical left/right row positions in a two-level
            ``MultiIndex``. Positions are zero-based input-row positions, not
            the original pandas index labels.
        return_building_blocks: Experimental. Whether to preserve kernel
            building blocks.
        reverse: Whether aggregation reads from the left side.
        join_algorithm: Multi-range strategy: ``"default"`` uses the general
            range-join implementation and ``"regions"`` uses the region-based
            implementation. It is ignored when the join is not a multi-range
            join.
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
        force: If ``True``, force mixed equality/range joins to use the
            non-equi path, with range predicates driving candidate generation
            before equality predicates are applied. It has no effect on joins
            that mix equality and ``!=`` predicates without a range predicate.
        return_matching_indices: Return physical index arrays instead of rows.
        aggfunc: Aggregation requests, or ``None`` for index output.
        include_join_positions: Include matched physical left/right row
            positions in a two-level ``MultiIndex`` on dataframe output.
            Positions are zero-based input-row positions, not original pandas
            index labels; this is valid only for inner joins.
        return_building_blocks: Experimental. Preserve starts/ends or
            equivalent kernel building blocks.
        reverse: Aggregate left values into right output rows.
        join_algorithm: Multi-range strategy: ``"default"`` uses the general
            range-join implementation and ``"regions"`` uses the region-based
            implementation. It is ignored when the join is not a multi-range
            join.
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
    conditions = _normalize_conditions(conditions)

    for condition in conditions:
        _conditional_join_type_check(
            left_column=df[condition.left],
            right_column=right[condition.right],
            op=condition.op,
            force=force,
        )

    # Rust receives physical row positions, never the caller's labels.
    df.index = pd.RangeIndex(len(df))
    right.index = pd.RangeIndex(len(right))

    index_result_kwargs = {
        "how": how,
        "df_columns": df_columns,
        "right_columns": right_columns,
        "indicator": indicator,
        "include_join_positions": include_join_positions,
        "return_matching_indices": return_matching_indices,
    }

    eq_check = any(
        condition.op == _JoinOperator.STRICTLY_EQUAL.value for condition in conditions
    )
    has_range = any(
        condition.op in less_than_join_types.union(greater_than_join_types)
        for condition in conditions
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
    all_nes_check = all(condition.op == "!=" for condition in conditions)
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
    for condition in conditions:
        if condition.op not in less_than_join_types.union(greater_than_join_types):
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
        return _multi_range_join._aggregate(
            df=df,
            right=right,
            conditions=conditions,
            aggfunc=aggfunc,
            return_matched=return_matched,
            reverse=reverse,
        )
    if (counter > 1) and (join_algorithm == "default"):
        return _multi_range_join._compute_multi_range_join(
            df=df,
            right=right,
            conditions=conditions,
            keep=keep,
            return_building_blocks=return_building_blocks,
            **index_result_kwargs,
        )

    raise RuntimeError(
        "conditional_join reached an unsupported predicate dispatch combination"
    )


@deprecated_kwargs("return_ragged_arrays")
def get_join_indices(
    df: pd.DataFrame,
    right: pd.DataFrame | pd.Series,
    *conditions: tuple,
    keep: Literal["first", "last", "any", "all"] = "all",
    use_numba: bool | None = None,
    force: bool = False,
    return_building_blocks: bool = False,
    join_algorithm: str = "default",
) -> dict:
    """Return matching physical positions for an inner conditional join.

    Unlike [`conditional_join`][janitor.functions.conditional_join], this
    helper does not gather dataframe rows. It returns two parallel arrays of
    zero-based physical positions. At position ``i``,
    ``left_index[i]`` and ``right_index[i]`` identify one matching pair.

    The result is useful when callers need to materialize rows or perform
    additional processing themselves.

    !!! info "New in version 0.27.0"

    !!! abstract "Version changed"

        - **0.29.0:** Added support for ragged array indices.
        - **0.32.0:** Deprecated ragged array indices and changed the return
          value to a dictionary.
        - **0.32.9:** Deprecated ``use_numba``.
        - **0.32.10:** Added the experimental ``return_building_blocks`` and
          ``join_algorithm`` parameters.

    !!! warning "Experimental parameter"

        ``return_building_blocks=True`` may add implementation-level values,
        such as range-window ``starts`` and ``ends``, to the result. The keys
        and shape of this additional data are not part of the stable public
        API.

    Args:
        df: Left DataFrame.
        right: Right DataFrame or named Series.
        conditions: Predicates of the form
            ``(left_column, right_column, operator)``. The supported operators
            are ``==``, ``!=``, ``<``, ``<=``, ``>``, and ``>=``.
        keep: Return all matches, or one ``first``, ``last``, or ``any`` match
            per left row.
        use_numba: Deprecated no-op retained for compatibility with older
            callers. Its value is ignored.
        force: If ``True``, force mixed equality/range joins to use the
            non-equi path. Range predicates drive candidate generation before
            equality predicates are applied. It has no effect on joins that
            mix equality and ``!=`` predicates without a range predicate.
        return_building_blocks: If ``True``, include implementation-level
            values in the result.
            !!! warning "Experimental"
                This parameter may add range-window ``starts`` and ``ends``
                to the result. The keys and shape of this data are not part
                of the stable public API and may change without warning.
        join_algorithm: Strategy for multiple range predicates. ``"default"``
            uses the general range-join implementation and ``"regions"`` uses
            the region-based implementation. It is ignored for other joins.

    Returns:
        dict: Matching physical-position arrays and optional building blocks.

    !!! note "Return layout"

        - ``left_index`` and ``right_index`` are one-dimensional NumPy arrays
          of zero-based physical row positions from ``df`` and ``right``.
        - The arrays have equal length; values at position ``i`` form one
          matching pair.
        - ``keep="all"`` returns every pair. ``keep="first"``, ``"last"``,
          and ``"any"`` return at most one pair per left row.
        - When there are no matches, both required arrays are empty.
        - ``return_building_blocks=True`` may add experimental arrays such as
          ``starts`` and ``ends``. Their keys and shape are not stable.

    Examples:
        >>> import pandas as pd
        >>> import janitor
        >>> left = pd.DataFrame({"value": [1, 4, 7]})
        >>> right = pd.DataFrame({"value": [2, 5, 8]})
        >>> janitor.get_join_indices(left, right, ("value", "value", "<"), keep="first")
        {'left_index': array([0, 1, 2]), 'right_index': array([0, 1, 2])}

        With ``keep="all"`` (the default), every matching pair is returned:

        >>> all_matches = janitor.get_join_indices(left, right, ("value", "value", "<"))
        >>> all_matches["left_index"].tolist()
        [0, 0, 0, 1, 1, 2]
        >>> all_matches["right_index"].tolist()
        [0, 1, 2, 1, 2, 2]

        ``return_building_blocks`` is experimental and exposes implementation
        details—the range windows used by callers that need to perform their
        own materialization:

        >>> blocks = janitor.get_join_indices(
        ...     left,
        ...     right,
        ...     ("value", "value", "<"),
        ...     return_building_blocks=True,
        ... )
        >>> sorted(blocks)
        ['ends', 'left_index', 'right_index', 'starts']
    """
    if use_numba is not None:
        warnings.warn(
            "The 'use_numba' parameter is deprecated and has no effect.",
            DeprecationWarning,
            stacklevel=2,
        )

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
    """Aggregate values over rows matched by an inner conditional join.

    ``aggfunc`` contains ``(column, operation)`` pairs. Supported operations
    are ``sum``, ``prod``, ``size``, ``count``, ``min``, and ``max``.

    !!! info "Aggregation semantics"

        | Operation | Match and null behavior |
        | --- | --- |
        | `count` | Counts matched, non-null source values. |
        | `size` | Counts every matched source row, including null values. |
        | `sum` | Ignores nulls; uses `0` when no value contributes. |
        | `prod` | Ignores nulls; uses `1` when no value contributes. |
        | `min`, `max` | Ignore nulls; missing if no non-null value contributes. |

    !!! note "Dtypes and match status"

        - `sum` and `prod` require numeric source columns. Signed integer
          results use `int64`; unsigned results use `uint64`.
        - `min` and `max` support numeric, datetime, and timedelta columns.
          Integer results use nullable `Int64` or `UInt64` when an unmatched
          output row needs `pd.NA`; other results retain their source dtype.
        - `return_matched` identifies whether any row pair matched. This is
          independent of `count`, which can be zero when all matched source
          values are null.

    By default, right-side values are aggregated for each left row. Set
    ``reverse=True`` to aggregate left-side values for each right row.

    !!! note "Output and matching"

        The output contains one row for every eligible physical row on the
        output side. Rows with null values in any join-condition column are
        excluded. Eligible rows with no matching partner are retained with
        identity values and ``matched=False``. With ``return_matched=True``,
        the result index has a boolean ``matched`` level so identity-valued
        unmatched results can be distinguished from matched results with the
        same value.

    Args:
        df: Left DataFrame and reverse-aggregation source.
        right: Right DataFrame or named Series and forward-aggregation source.
        conditions: Conditional-join predicate tuples.
        aggfunc: Non-empty ``(column, operation)`` requests.
        force: If ``True``, force mixed equality/range joins to use the
            non-equi path. Range predicates drive candidate generation before
            equality predicates are applied. It has no effect on joins that
            mix equality and ``!=`` predicates without a range predicate.
        reverse: Aggregate left-side values into right-side output rows.
        return_matched: Add a boolean ``matched`` level to the result index.
        join_algorithm: Strategy for multiple range predicates. ``"default"``
            uses the general range-join implementation and ``"regions"`` uses
            the region-based implementation. It is ignored for other joins.

    Returns:
        pd.DataFrame: Aggregated values with one row per eligible physical
        output row. Null join-key rows are excluded; eligible unmatched rows
        are retained with identity values.

    Examples:
        >>> import pandas as pd
        >>> import janitor
        >>> left = pd.DataFrame({"key": [1, 2]})
        >>> right = pd.DataFrame({"key": [1, 2, 3], "amount": [10, 20, 30]})
        >>> result = left.join_agg(
        ...     right,
        ...     ("key", "key", "<"),
        ...     aggfunc=[("amount", "sum"), ("amount", "count")],
        ...     return_matched=True,
        ... )
        >>> print(result.to_string())  # doctest: +NORMALIZE_WHITESPACE
                  amount
                     sum count
              matched
        0 True        50     2
        1 True        30     1

        ``return_matched`` also distinguishes an unmatched row from a matched
        row whose aggregate happens to be zero:

        >>> left = pd.DataFrame({"key": [1, 3]})
        >>> right = pd.DataFrame({"key": [2, 3], "amount": [5, 7]})
        >>> partial = left.join_agg(
        ...     right,
        ...     ("key", "key", "<"),
        ...     aggfunc=[("amount", "sum")],
        ...     return_matched=True,
        ... )
        >>> print(partial.to_string())  # doctest: +NORMALIZE_WHITESPACE
                  amount
                     sum
              matched
        0 True        12
        1 False        0

        The same contract applies when no eligible row matches anywhere in
        the call:

        >>> left = pd.DataFrame({"key": [1, 2]})
        >>> right = pd.DataFrame({"key": [3], "amount": [10]})
        >>> all_unmatched = left.join_agg(
        ...     right,
        ...     ("key", "key", ">"),
        ...     aggfunc=[("amount", "sum"), ("amount", "prod")],
        ...     return_matched=True,
        ... )
        >>> print(all_unmatched.to_string())  # doctest: +NORMALIZE_WHITESPACE
                  amount
                     sum prod
              matched
        0 False         0    1
        1 False         0    1

        Set ``reverse=True`` to aggregate left-side values into right-side
        output rows:

        >>> left = pd.DataFrame({"key": [1, 2], "amount": [10, 20]})
        >>> right = pd.DataFrame({"key": [1, 2, 3]})
        >>> reverse_result = left.join_agg(
        ...     right,
        ...     ("key", "key", "<"),
        ...     aggfunc=[("amount", "sum")],
        ...     reverse=True,
        ...     return_matched=False,
        ... )
        >>> print(reverse_result.to_string())
          amount
             sum
        0      0
        1     10
        2     30
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
