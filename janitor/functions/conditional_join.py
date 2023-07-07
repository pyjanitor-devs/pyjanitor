from __future__ import annotations
import operator
from typing import Union, Any, Optional, Hashable, Literal
import numpy as np
import pandas as pd
import pandas_flavor as pf
import warnings
from pandas.core.dtypes.common import (
    is_datetime64_dtype,
    is_dtype_equal,
    is_extension_array_dtype,
    is_numeric_dtype,
    is_string_dtype,
)

from pandas.core.reshape.merge import _MergeOperation

from janitor.utils import check, check_column, find_stack_level
from janitor.functions.utils import (
    _JoinOperator,
    _generic_func_cond_join,
    _keep_output,
    less_than_join_types,
    greater_than_join_types,
    col,
)

warnings.simplefilter("always", DeprecationWarning)


@pf.register_dataframe_method
def conditional_join(
    df: pd.DataFrame,
    right: Union[pd.DataFrame, pd.Series],
    *conditions: Any,
    how: Literal["inner", "left", "right", "outer"] = "inner",
    sort_by_appearance: bool = False,
    df_columns: Optional[Any] = slice(None),
    right_columns: Optional[Any] = slice(None),
    keep: Literal["first", "last", "all"] = "all",
    use_numba: bool = False,
    indicator: Optional[Union[bool, str]] = False,
    force: bool = False,
) -> pd.DataFrame:
    """The conditional_join function operates similarly to `pd.merge`,
    but allows joins on inequality operators,
    or a combination of equi and non-equi joins.

    Joins solely on equality are not supported.

    If the join is solely on equality, `pd.merge` function
    covers that; if you are interested in nearest joins, asof joins,
    or rolling joins, then `pd.merge_asof` covers that.
    There is also pandas' IntervalIndex, which is efficient for range joins,
    especially if the intervals do not overlap.

    Column selection in `df_columns` and `right_columns` is possible using the
    [`select_columns`][janitor.functions.select.select_columns] syntax.

    Performance might be improved by setting `use_numba` to `True`.
    This assumes that `numba` is installed.

    This function returns rows, if any, where values from `df` meet the
    condition(s) for values from `right`. The conditions are passed in
    as a variable argument of tuples, where the tuple is of
    the form `(left_on, right_on, op)`; `left_on` is the column
    label from `df`, `right_on` is the column label from `right`,
    while `op` is the operator.

    The `col` class is also supported in the `conditional_join` syntax.

    For multiple conditions, the and(`&`)
    operator is used to combine the results of the individual conditions.

    In some scenarios there might be performance gains if the less than join,
    or the greater than join condition, or the range condition
    is executed before the equi join - pass `force=True` to force this.

    The operator can be any of `==`, `!=`, `<=`, `<`, `>=`, `>`.

    The join is done only on the columns.

    For non-equi joins, only numeric and date columns are supported.

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
        >>> df1.conditional_join(
        ...     df2,
        ...     col("value_1") > col("value_2A"),
        ...     col("value_1") < col("value_2B")
        ... )
           value_1  value_2A  value_2B
        0        2         1         3
        1        5         3         6
        2        3         2         4
        3        4         3         5
        4        4         3         6

    !!! abstract "Version Changed"

        - 0.24.0
            - Added `df_columns`, `right_columns`, `keep` and `use_numba` parameters.
        - 0.24.1
            - Added `indicator` parameter.
        - 0.25.0
            - `col` class supported.
            - Outer join supported. `sort_by_appearance` deprecated.
            - Numba support for equi join

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
        sort_by_appearance: If `how = inner` and
            `sort_by_appearance = False`, there
            is no guarantee that the original order is preserved.
            Usually, this offers more performance.
            If `how = left`, the row order from the left dataframe
            is preserved; if `how = right`, the row order
            from the right dataframe is preserved.
            !!!warning "Deprecated in 0.25.0"
        df_columns: Columns to select from `df` in the final output dataframe.
            Column selection is based on the
            [`select_columns`][janitor.functions.select.select_columns] syntax.
            It is also possible to rename the output columns via a dictionary.
        right_columns: Columns to select from `right` in the final output dataframe.
            Column selection is based on the
            [`select_columns`][janitor.functions.select.select_columns] syntax.
            It is also possible to rename the output columns via a dictionary.
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

    Returns:
        A pandas DataFrame of the two merged Pandas objects.
    """  # noqa: E501

    return _conditional_join_compute(
        df,
        right,
        conditions,
        how,
        sort_by_appearance,
        df_columns,
        right_columns,
        keep,
        use_numba,
        indicator,
        force,
    )


def _check_operator(op: str):
    """
    Check that operator is one of
    `>`, `>=`, `==`, `!=`, `<`, `<=`.

    Used in `conditional_join`.
    """
    sequence_of_operators = {op.value for op in _JoinOperator}
    if op not in sequence_of_operators:
        raise ValueError(
            "The conditional join operator "
            f"should be one of {sequence_of_operators}"
        )


def _conditional_join_preliminary_checks(
    df: pd.DataFrame,
    right: Union[pd.DataFrame, pd.Series],
    conditions: tuple,
    how: str,
    sort_by_appearance: bool,
    df_columns: Any,
    right_columns: Any,
    keep: str,
    use_numba: bool,
    indicator: Union[bool, str],
    force: bool,
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

    conditions = [
        cond.join_args if isinstance(cond, col) else cond
        for cond in conditions
    ]
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
        (op == _JoinOperator.STRICTLY_EQUAL.value for *_, op in conditions)
    ):
        raise ValueError("Equality only joins are not supported.")

    check("how", how, [str])

    if how not in {"inner", "left", "right", "outer"}:
        raise ValueError(
            "'how' should be one of 'inner', 'left', 'right' or 'outer'."
        )

    if sort_by_appearance:
        warnings.warn(
            "The keyword argument "
            "'sort_by_appearance' of 'conditional_join' is deprecated.",
            DeprecationWarning,
            stacklevel=find_stack_level(),
        )
    check("sort_by_appearance", sort_by_appearance, [bool])

    if (df.columns.nlevels > 1) and (
        isinstance(df_columns, dict) or isinstance(right_columns, dict)
    ):
        raise ValueError(
            "Column renaming with a dictionary is not supported "
            "for MultiIndex columns."
        )

    check("keep", keep, [str])

    if keep not in {"all", "first", "last"}:
        raise ValueError("'keep' should be one of 'all', 'first', 'last'.")

    check("use_numba", use_numba, [bool])

    check("indicator", indicator, [bool, str])

    check("force", force, [bool])

    return (
        df,
        right,
        conditions,
        how,
        sort_by_appearance,
        df_columns,
        right_columns,
        keep,
        use_numba,
        indicator,
        force,
    )


def _conditional_join_type_check(
    left_column: pd.Series, right_column: pd.Series, op: str, use_numba: bool
) -> None:
    """
    Raise error if column type is not any of numeric or datetime or string.
    """

    if (
        (op == _JoinOperator.STRICTLY_EQUAL.value)
        and use_numba
        and not is_numeric_dtype(left_column)
        and not is_datetime64_dtype(left_column)
    ):
        raise TypeError(
            "Only numeric and datetime types "
            "are supported in an equi-join "
            "when use_numba is set to True"
        )

    is_categorical_dtype = isinstance(left_column.dtype, pd.CategoricalDtype)

    if not is_categorical_dtype:
        permitted_types = {
            is_datetime64_dtype,
            is_numeric_dtype,
            is_string_dtype,
        }
        for func in permitted_types:
            if func(left_column.dtype):
                break
        else:
            raise TypeError(
                "conditional_join only supports "
                "string, category, numeric, or "
                "date dtypes (without timezone) - "
                f"'{left_column.name} is of type "
                f"{left_column.dtype}."
            )

    if is_categorical_dtype:
        if not left_column.array._categories_match_up_to_permutation(
            right_column.array
        ):
            raise TypeError(
                f"'{left_column.name}' and '{right_column.name}' "
                "should have the same categories, and the same order."
            )
    elif not is_dtype_equal(left_column, right_column):
        raise TypeError(
            f"Both columns should have the same type - "
            f"'{left_column.name}' has {left_column.dtype} type;"
            f"'{right_column.name}' has {right_column.dtype} type."
        )

    if (
        (op != _JoinOperator.STRICTLY_EQUAL.value)
        and not is_numeric_dtype(left_column)
        and not is_datetime64_dtype(left_column)
    ):
        raise TypeError(
            "non-equi joins are supported "
            "only for datetime and numeric dtypes. "
            f"{left_column.name} in condition "
            f"({left_column.name}, {right_column.name}, {op}) "
            f"has a dtype {left_column.dtype}."
        )

    return None


def _conditional_join_compute(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list,
    how: str,
    sort_by_appearance: bool,
    df_columns: Any,
    right_columns: Any,
    keep: str,
    use_numba: bool,
    indicator: Union[bool, str],
    force: bool,
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
        sort_by_appearance,
        df_columns,
        right_columns,
        keep,
        use_numba,
        indicator,
        force,
    ) = _conditional_join_preliminary_checks(
        df,
        right,
        conditions,
        how,
        sort_by_appearance,
        df_columns,
        right_columns,
        keep,
        use_numba,
        indicator,
        force,
    )

    eq_check = False
    le_lt_check = False
    for condition in conditions:
        left_on, right_on, op = condition
        _conditional_join_type_check(
            df[left_on], right[right_on], op, use_numba
        )
        if op == _JoinOperator.STRICTLY_EQUAL.value:
            eq_check = True
        elif op in less_than_join_types.union(greater_than_join_types):
            le_lt_check = True

    df.index = range(len(df))
    right.index = range(len(right))

    if len(conditions) > 1:
        if eq_check:
            result = _multiple_conditional_join_eq(
                df, right, conditions, keep, use_numba, force
            )
        elif le_lt_check:
            result = _multiple_conditional_join_le_lt(
                df, right, conditions, keep, use_numba
            )
        else:
            result = _multiple_conditional_join_ne(df, right, conditions, keep)
    else:
        left_on, right_on, op = conditions[0]
        if use_numba:
            from janitor.functions._numba import _numba_single_join

            result = _numba_single_join(
                left=df[left_on],
                right=right[right_on],
                op=op,
                keep=keep,
            )
        else:
            result = _generic_func_cond_join(
                left=df[left_on],
                right=right[right_on],
                op=op,
                multiple_conditions=False,
                keep=keep,
            )

    if result is None:
        result = np.array([], dtype=np.intp), np.array([], dtype=np.intp)

    return _create_frame(
        df,
        right,
        *result,
        how,
        df_columns,
        right_columns,
        indicator,
    )


operator_map = {
    _JoinOperator.STRICTLY_EQUAL.value: operator.eq,
    _JoinOperator.LESS_THAN.value: operator.lt,
    _JoinOperator.LESS_THAN_OR_EQUAL.value: operator.le,
    _JoinOperator.GREATER_THAN.value: operator.gt,
    _JoinOperator.GREATER_THAN_OR_EQUAL.value: operator.ge,
    _JoinOperator.NOT_EQUAL.value: operator.ne,
}


def _generate_indices(
    left_index: np.ndarray,
    right_index: np.ndarray,
    conditions: list[tuple[pd.Series, pd.Series, str]],
) -> tuple:
    """
    Run a for loop to get the final indices.
    This iteratively goes through each condition,
    builds a boolean array,
    and gets indices for rows that meet the condition requirements.
    `conditions` is a list of tuples, where a tuple is of the form:
    `(Series from df, Series from right, operator)`.
    """

    for condition in conditions:
        left, right, op = condition
        left = left._values[left_index]
        right = right._values[right_index]
        op = operator_map[op]
        mask = op(left, right)
        if not mask.any():
            return None
        if is_extension_array_dtype(mask):
            mask = mask.to_numpy(dtype=bool, na_value=False)
        if not mask.all():
            left_index = left_index[mask]
            right_index = right_index[mask]

    return left_index, right_index


def _multiple_conditional_join_ne(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list[tuple[pd.Series, pd.Series, str]],
    keep: str,
) -> tuple:
    """
    Get indices for multiple conditions,
    where all the operators are `!=`.

    Returns a tuple of (left_index, right_index)
    """
    # currently, there is no optimization option here
    # not equal typically combines less than
    # and greater than, so a lot more rows are returned
    # than just less than or greater than
    first, *rest = conditions
    left_on, right_on, op = first

    indices = _generic_func_cond_join(
        df[left_on],
        right[right_on],
        op,
        multiple_conditions=False,
        keep="all",
    )
    if indices is None:
        return None

    rest = (
        (df[left_on], right[right_on], op) for left_on, right_on, op in rest
    )

    indices = _generate_indices(*indices, rest)

    if not indices:
        return None

    return _keep_output(keep, *indices)


def _multiple_conditional_join_eq(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list,
    keep: str,
    use_numba: bool,
    force: bool,
) -> tuple:
    """
    Get indices for multiple conditions,
    if any of the conditions has an `==` operator.

    Returns a tuple of (left_index, right_index)
    """

    if force:
        return _multiple_conditional_join_le_lt(
            df=df,
            right=right,
            conditions=conditions,
            keep=keep,
            use_numba=use_numba,
        )

    if use_numba:
        from janitor.functions._numba import _numba_equi_join

        eqs = None
        for left_on, right_on, op in conditions:
            if op == _JoinOperator.STRICTLY_EQUAL.value:
                eqs = (left_on, right_on, op)

        le_lt = None
        ge_gt = None

        for condition in conditions:
            *_, op = condition
            if op in less_than_join_types:
                if le_lt:
                    continue
                le_lt = condition
            elif op in greater_than_join_types:
                if ge_gt:
                    continue
                ge_gt = condition
            if le_lt and ge_gt:
                break
        if not le_lt and not ge_gt:
            raise ValueError(
                "At least one less than or greater than "
                "join condition should be present when an equi-join "
                "is present, and use_numba is set to True."
            )
        rest = [
            condition
            for condition in conditions
            if condition not in {eqs, le_lt, ge_gt}
        ]

        right_columns = [eqs[1]]
        df_columns = [eqs[0]]
        if ge_gt:
            if ge_gt[1] not in right_columns:
                right_columns.append(ge_gt[1])
            if ge_gt[0] not in df_columns:
                df_columns.append(ge_gt[0])
        if le_lt:
            if le_lt[1] not in right_columns:
                right_columns.append(le_lt[1])
            if le_lt[0] not in df_columns:
                df_columns.append(le_lt[0])

        right_df = right.loc(axis=1)[right_columns]
        left_df = df.loc(axis=1)[df_columns]
        any_nulls = left_df.isna().any(axis=1)
        if any_nulls.all(axis=None):
            return None
        if any_nulls.any():
            left_df = left_df.loc[~any_nulls]
        any_nulls = right_df.isna().any(axis=1)
        if any_nulls.all(axis=None):
            return None
        if any_nulls.any():
            right_df = right.loc[~any_nulls]
        right_df = right_df.sort_values(right_columns)
        indices = _numba_equi_join(left_df, right_df, eqs, ge_gt, le_lt)

        if not rest or (indices is None):
            return indices

        rest = (
            (df[left_on], right[right_on], op)
            for left_on, right_on, op in rest
        )

        indices = _generate_indices(*indices, rest)

        if not indices:
            return None

        return _keep_output(keep, *indices)

    eqs = [
        (left_on, right_on)
        for left_on, right_on, op in conditions
        if op == _JoinOperator.STRICTLY_EQUAL.value
    ]
    left_on, right_on = zip(*eqs)
    left_on = [*left_on]
    right_on = [*right_on]

    rest = (
        (df[left_on], right[right_on], op)
        for left_on, right_on, op in conditions
        if op != _JoinOperator.STRICTLY_EQUAL.value
    )

    left_index, right_index = _MergeOperation(
        df,
        right,
        left_on=left_on,
        right_on=right_on,
        sort=False,
    )._get_join_indexers()

    if not left_index.size:
        return None

    indices = _generate_indices(left_index, right_index, rest)

    if not indices:
        return None

    return _keep_output(keep, *indices)


def _multiple_conditional_join_le_lt(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list,
    keep: str,
    use_numba: bool,
) -> tuple:
    """
    Get indices for multiple conditions,
    where `>/>=` or `</<=` is present,
    and there is no `==` operator.

    Returns a tuple of (df_index, right_index)
    """
    if use_numba:
        from janitor.functions._numba import (
            _numba_dual_join,
            _numba_single_join,
        )

        gt_lt = [
            condition
            for condition in conditions
            if condition[-1]
            in less_than_join_types.union(greater_than_join_types)
        ]
        conditions = [
            condition for condition in conditions if condition not in gt_lt
        ]
        if len(gt_lt) == 1:
            left_on, right_on, op = gt_lt[0]
            indices = _numba_single_join(
                df[left_on], right[right_on], op, keep="all"
            )
        else:
            indices = _numba_dual_join(df, right, gt_lt)
    else:
        # there is an opportunity for optimization for range joins
        # which is usually `lower_value < value < upper_value`
        # or `lower_value < a` and `b < upper_value`
        # intervalindex is not used here, as there are scenarios
        # where there will be overlapping intervals;
        # intervalindex does not offer an efficient way to get
        # the indices for overlaps
        # also, intervalindex covers only the first option
        # i.e => `lower_value < value < upper_value`
        # it does not extend to range joins for different columns
        # i.e => `lower_value < a` and `b < upper_value`
        # the option used for range joins is a simple form
        # dependent on sorting and extensible to overlaps
        # as well as the second option:
        # i.e =>`lower_value < a` and `b < upper_value`
        # range joins are also the more common types of non-equi joins
        # the other joins do not have an optimisation opportunity
        # within this space, as far as I know,
        # so a blowup of all the rows is unavoidable.

        # first step is to get two conditions, if possible
        # where one has a less than operator
        # and the other has a greater than operator
        # get the indices from that
        # and then build the remaining indices,
        # using _generate_indices function
        # the aim of this for loop is to see if there is
        # the possiblity of a range join, and if there is,
        # then use the optimised path
        le_lt = None
        ge_gt = None
        # keep the first match for le_lt or ge_gt
        for condition in conditions:
            *_, op = condition
            if op in less_than_join_types:
                if le_lt:
                    continue
                le_lt = condition
            elif op in greater_than_join_types:
                if ge_gt:
                    continue
                ge_gt = condition
            if le_lt and ge_gt:
                break

        # optimised path
        if le_lt and ge_gt:
            conditions = [
                condition
                for condition in conditions
                if condition not in (ge_gt, le_lt)
            ]

            indices = _range_indices(df, right, ge_gt, le_lt)

        # no optimised path
        # blow up the rows and prune
        else:
            if le_lt:
                conditions = [
                    condition for condition in conditions if condition != le_lt
                ]
                left_on, right_on, op = le_lt
            else:
                conditions = [
                    condition for condition in conditions if condition != ge_gt
                ]
                left_on, right_on, op = ge_gt

            indices = _generic_func_cond_join(
                df[left_on],
                right[right_on],
                op,
                multiple_conditions=False,
                keep="all",
            )
    if not indices:
        return None

    if conditions:
        conditions = (
            (df[left_on], right[right_on], op)
            for left_on, right_on, op in conditions
        )

        indices = _generate_indices(*indices, conditions)
        if not indices:
            return None

    return _keep_output(keep, *indices)


def _range_indices(
    df: pd.DataFrame,
    right: pd.DataFrame,
    first: tuple,
    second: tuple,
):
    """
    Retrieve index positions for range/interval joins.

    Idea inspired by article:
    https://www.vertica.com/blog/what-is-a-range-join-and-why-is-it-so-fastba-p223413/

    Returns a tuple of (left_index, right_index)
    """
    # summary of code for range join:
    # get the positions where start_left is >/>= start_right
    # then within the positions,
    # get the positions where end_left is </<= end_right
    # this should reduce the search space

    left_on, right_on, op = first
    left_c = df[left_on]
    right_c = right[right_on]
    left_on, right_on, _ = second
    # get rid of any nulls
    # this is helpful as we can convert extension arrays
    # to numpy arrays safely
    # and simplify the search logic below
    any_nulls = df[left_on].isna()
    if any_nulls.any():
        left_c = left_c[~any_nulls]
    any_nulls = right[right_on].isna()
    if any_nulls.any():
        right_c = right_c[~any_nulls]

    outcome = _generic_func_cond_join(
        left=left_c,
        right=right_c,
        op=op,
        multiple_conditions=True,
        keep="all",
    )

    if outcome is None:
        return None

    left_index, right_index, search_indices = outcome
    left_on, right_on, op = second
    right_c = right.loc[right_index, right_on]
    left_c = df.loc[left_index, left_on]
    # if True, we can use a binary search
    # for more performance, instead of a linear search
    fastpath = right_c.is_monotonic_increasing
    if fastpath:
        outcome = _generic_func_cond_join(
            left=left_c,
            right=right_c,
            op=op,
            multiple_conditions=False,
            keep="first",
        )
        if outcome is None:
            return None
        left_c, pos = outcome
    else:
        # the aim here is to get the first match
        # where the left array is </<= than the right array
        # this is solved by getting the cumulative max
        # thus ensuring that the first match is obtained
        # via a binary search
        # this allows us to avoid the less efficient linear search
        # of using a for loop with a break to get the first match
        outcome = _generic_func_cond_join(
            left=left_c,
            right=right_c.cummax(),
            op=op,
            multiple_conditions=True,
            keep="all",
        )
        if outcome is None:
            return None
        left_c, right_index, pos = outcome
    if left_c.size < left_index.size:
        keep_rows = np.isin(left_index, left_c, assume_unique=True)
        search_indices = search_indices[keep_rows]
        left_index = left_c
    # no point searching within (a, b)
    # if a == b
    # since range(a, b) yields none
    keep_rows = pos < search_indices

    if not keep_rows.any():
        return None

    if not keep_rows.all():
        left_index = left_index[keep_rows]
        pos = pos[keep_rows]
        search_indices = search_indices[keep_rows]

    repeater = search_indices - pos
    if (repeater == 1).all():
        # no point running a comparison op
        # if the width is all 1
        # this also implies that the intervals
        # do not overlap on the right side
        return left_index, right_index[pos]

    right_index = [
        right_index[start:end] for start, end in zip(pos, search_indices)
    ]

    right_index = np.concatenate(right_index)
    left_index = np.repeat(left_index, repeater)

    if fastpath:
        return left_index, right_index
    # here we search for actual positions
    # where left_c is </<= right_c
    # safe to index the arrays, since we are picking the positions
    # which are all in the original `df` and `right`
    # doing this allows some speed gains
    # while still ensuring correctness
    left_c = df[left_on]._values[left_index]
    right_c = right[right_on]._values[right_index]
    ext_arr = is_extension_array_dtype(left_c)
    op = operator_map[op]
    mask = op(left_c, right_c)

    if ext_arr:
        mask = mask.to_numpy(dtype=bool, na_value=False)

    if not mask.all():
        left_index = left_index[mask]
        right_index = right_index[mask]

    return left_index, right_index


def _cond_join_select_columns(columns: Any, df: pd.DataFrame):
    """
    Select columns in a DataFrame.
    Optionally rename the columns while selecting.
    Returns a Pandas DataFrame.
    """

    if isinstance(columns, dict):
        df = df.select_columns([*columns])
        df.columns = [columns.get(name, name) for name in df]
    else:
        df = df.select_columns(columns)

    return df


def _create_multiindex_column(df: pd.DataFrame, right: pd.DataFrame):
    """
    Create a MultiIndex column for conditional_join.
    """
    header = [np.array(["left"]).repeat(df.columns.size)]
    columns = [
        df.columns.get_level_values(n) for n in range(df.columns.nlevels)
    ]
    header.extend(columns)
    df.columns = pd.MultiIndex.from_arrays(header)
    header = [np.array(["right"]).repeat(right.columns.size)]
    columns = [
        right.columns.get_level_values(n) for n in range(right.columns.nlevels)
    ]
    header.extend(columns)
    right.columns = pd.MultiIndex.from_arrays(header)
    return df, right


def _create_frame(
    df: pd.DataFrame,
    right: pd.DataFrame,
    left_index: np.ndarray,
    right_index: np.ndarray,
    how: str,
    df_columns: Any,
    right_columns: Any,
    indicator: Union[bool, str],
):
    """
    Create final dataframe
    """
    if (df_columns is None) and (right_columns is None):
        raise ValueError("df_columns and right_columns cannot both be None.")
    if (df_columns is not None) and (df_columns != slice(None)):
        df = _cond_join_select_columns(df_columns, df)
    if (right_columns is not None) and (right_columns != slice(None)):
        right = _cond_join_select_columns(right_columns, right)
    if df_columns is None:
        df = pd.DataFrame([])
    elif right_columns is None:
        right = pd.DataFrame([])

    if not df.columns.intersection(right.columns).empty:
        df, right = _create_multiindex_column(df, right)

    def _add_indicator(
        indicator: Union[bool, str],
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
        indicator: Union[bool, str],
    ):
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
        frame = {key: value._values[left_index] for key, value in df.items()}
        r_frame = {
            key: value._values[right_index] for key, value in right.items()
        }
        frame.update(r_frame)
        if indicator:
            indicator, arr = _add_indicator(
                indicator=indicator,
                how="inner",
                column_length=left_index.size,
                columns=df.columns.union(right.columns),
            )
            frame[indicator] = arr
        return pd.DataFrame(frame, copy=False)

    if how == "inner":
        return _inner(df, right, left_index, right_index, indicator)

    if how != "outer":
        if how == "left":
            if not right.empty:
                right = right.take(right_index)
            right.index = left_index
        else:
            if not df.empty:
                df = df.take(left_index)
            df.index = right_index

        df = df.merge(
            right,
            left_index=True,
            right_index=True,
            indicator=indicator,
            how=how,
            copy=False,
            sort=False,
        )
        df.index = range(len(df))
        return df

    both = _inner(df, right, left_index, right_index, indicator)
    contents = []
    columns = df.columns.union(right.columns)
    left_index = np.setdiff1d(df.index, left_index)
    if left_index.size:
        df = df.take(left_index)
        if indicator:
            l_indicator, arr = _add_indicator(
                indicator=indicator,
                how="left",
                column_length=left_index.size,
                columns=columns,
            )
            df[l_indicator] = arr
        contents.append(df)

    contents.append(both)

    right_index = np.setdiff1d(right.index, right_index)
    if right_index.size:
        right = right.take(right_index)
        if indicator:
            r_indicator, arr = _add_indicator(
                indicator=indicator,
                how="right",
                column_length=right_index.size,
                columns=columns,
            )
            right[r_indicator] = arr
        contents.append(right)

    return pd.concat(
        contents, axis=0, copy=False, sort=False, ignore_index=True
    )
