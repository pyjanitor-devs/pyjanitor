from __future__ import annotations
import operator
from typing import Union, Any, Optional, Hashable, Literal
import numpy as np
import pandas as pd
import pandas_flavor as pf
from pandas.core.dtypes.common import (
    is_categorical_dtype,
    is_datetime64_dtype,
    is_dtype_equal,
    is_extension_array_dtype,
    is_numeric_dtype,
    is_string_dtype,
)

from pandas.core.reshape.merge import _MergeOperation

from janitor.utils import check, check_column
from janitor.functions.utils import (
    _convert_to_numpy_array,
    _JoinOperator,
    _generic_func_cond_join,
    _keep_output,
    less_than_join_types,
    greater_than_join_types,
)


@pf.register_dataframe_method
def conditional_join(
    df: pd.DataFrame,
    right: Union[pd.DataFrame, pd.Series],
    *conditions: Any,
    how: Literal["inner", "left", "right"] = "inner",
    sort_by_appearance: bool = False,
    df_columns: Optional[Any] = slice(None),
    right_columns: Optional[Any] = slice(None),
    keep: Literal["first", "last", "all"] = "all",
    use_numba: bool = False,
    indicator: Optional[bool, str] = False,
) -> pd.DataFrame:
    """The conditional_join function operates similarly to `pd.merge`,
    but allows joins on inequality operators,
    or a combination of equi and non-equi joins.

    Joins solely on equality are not supported.

    If the join is solely on equality, `pd.merge` function
    covers that; if you are interested in nearest joins, or rolling joins,
    then `pd.merge_asof` covers that.
    There is also pandas' IntervalIndex, which is efficient for range joins,
    especially if the intervals do not overlap.

    Column selection in `df_columns` and `right_columns` is possible using the
    [`select_columns`][janitor.functions.select.select_columns] syntax.

    For strictly non-equi joins, particularly range joins,
    involving either `>`, `<`, `>=`, `<=` operators,
    where the columns on the right are not both monotonically increasing,
    performance could be improved by setting `use_numba` to `True`.
    This assumes that `numba` is installed.

    To preserve row order, set `sort_by_appearance` to `True`.

    This function returns rows, if any, where values from `df` meet the
    condition(s) for values from `right`. The conditions are passed in
    as a variable argument of tuples, where the tuple is of
    the form `(left_on, right_on, op)`; `left_on` is the column
    label from `df`, `right_on` is the column label from `right`,
    while `op` is the operator. For multiple conditions, the and(`&`)
    operator is used to combine the results of the individual conditions.

    The operator can be any of `==`, `!=`, `<=`, `<`, `>=`, `>`.

    The join is done only on the columns.

    For non-equi joins, only numeric and date columns are supported.

    Only `inner`, `left`, and `right` joins are supported.

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

    !!! abstract "Version Changed"

        - 0.24.0
            - Added `df_columns`, `right_columns`, `keep` and `use_numba` parameters.
        - 0.24.1
            - Added `indicator` parameter.

    Args:
        df: A pandas DataFrame.
        right: Named Series or DataFrame to join to.
        conditions: Variable argument of tuple(s) of the form
            `(left_on, right_on, op)`, where `left_on` is the column
            label from `df`, `right_on` is the column label from `right`,
            while `op` is the operator. The operator can be any of
            `==`, `!=`, `<=`, `<`, `>=`, `>`. For multiple conditions,
            the and(`&`) operator is used to combine the results
            of the individual conditions.
        how: Indicates the type of join to be performed.
            It can be one of `inner`, `left`, `right`.
            Full outer join is not supported. Defaults to `inner`.
        sort_by_appearance: Default is `False`.
            This is useful for scenarios where the user wants
            the original order maintained.
            If `True` and `how = left`, the row order from the left dataframe
            is preserved; if `True` and `how = right`, the row order
            from the right dataframe is preserved.
        df_columns: Columns to select from `df`.
            It can be a single column or a list of columns.
            It is also possible to rename the output columns via a dictionary.
        right_columns: Columns to select from `right`.
            It can be a single column or a list of columns.
            It is also possible to rename the output columns via a dictionary.
        keep: Choose whether to return the first match,
            last match or all matches. Default is `all`.
        use_numba: Use numba, if installed, to accelerate the computation.
            Applicable only to strictly non-equi joins. Default is `False`.
        indicator: If `True`, adds a column to the output DataFrame
            called “_merge” with information on the source of each row.
            The column can be given a different name by providing a string argument.
            The column will have a Categorical type with the value of “left_only”
            for observations whose merge key only appears in the left DataFrame,
            “right_only” for observations whose merge key
            only appears in the right DataFrame, and “both” if the observation’s
            merge key is found in both DataFrames.

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
        (op == _JoinOperator.STRICTLY_EQUAL.value for *_, op in conditions)
    ):
        raise ValueError("Equality only joins are not supported.")

    check("how", how, [str])

    if how not in {"inner", "left", "right"}:
        raise ValueError("'how' should be one of 'inner', 'left' or 'right'.")

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
    )


def _conditional_join_type_check(
    left_column: pd.Series, right_column: pd.Series, op: str
) -> None:
    """
    Raise error if column type is not any of numeric or datetime or string.
    """

    permitted_types = {
        is_datetime64_dtype,
        is_numeric_dtype,
        is_string_dtype,
        is_categorical_dtype,
    }
    for func in permitted_types:
        if func(left_column):
            break
    else:
        raise ValueError(
            "conditional_join only supports "
            "string, category, numeric, or date dtypes (without timezone) - "
            f"'{left_column.name} is of type {left_column.dtype}."
        )

    lk_is_cat = is_categorical_dtype(left_column)
    rk_is_cat = is_categorical_dtype(right_column)

    if lk_is_cat & rk_is_cat:
        if not left_column.array._categories_match_up_to_permutation(
            right_column.array
        ):
            raise ValueError(
                f"'{left_column.name}' and '{right_column.name}' "
                "should have the same categories, and the same order."
            )
    elif not is_dtype_equal(left_column, right_column):
        raise ValueError(
            f"Both columns should have the same type - "
            f"'{left_column.name}' has {left_column.dtype} type;"
            f"'{right_column.name}' has {right_column.dtype} type."
        )

    number_or_date = is_numeric_dtype(left_column) or is_datetime64_dtype(
        left_column
    )
    if (op != _JoinOperator.STRICTLY_EQUAL.value) & (not number_or_date):
        raise ValueError(
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
) -> pd.DataFrame:
    """
    This is where the actual computation
    for the conditional join takes place.
    A pandas DataFrame is returned.
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
    )

    eq_check = False
    le_lt_check = False
    for condition in conditions:
        left_on, right_on, op = condition
        _conditional_join_type_check(df[left_on], right[right_on], op)
        if op == _JoinOperator.STRICTLY_EQUAL.value:
            eq_check = True
        elif op in less_than_join_types.union(greater_than_join_types):
            le_lt_check = True

    df.index = range(len(df))
    right.index = range(len(right))

    if len(conditions) > 1:
        if eq_check:
            result = _multiple_conditional_join_eq(df, right, conditions, keep)
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
        sort_by_appearance,
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
    df: pd.DataFrame, right: pd.DataFrame, conditions: list, keep: str
) -> tuple:
    """
    Get indices for multiple conditions,
    if any of the conditions has an `==` operator.

    Returns a tuple of (df_index, right_index)
    """
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
        from janitor.functions._numba import _numba_dual_join

        pairs = [
            condition
            for condition in conditions
            if condition[-1] != _JoinOperator.NOT_EQUAL.value
        ]
        conditions = [
            condition
            for condition in conditions
            if condition[-1] == _JoinOperator.NOT_EQUAL.value
        ]
        if len(pairs) > 2:
            patch = pairs[2:]
            conditions.extend(patch)
            pairs = pairs[:2]
        if len(pairs) < 2:
            # combine with != condition
            # say we have ('start', 'ID', '<='), ('end', 'ID', '!=')
            # we convert conditions to :
            # ('start', 'ID', '<='), ('end', 'ID', '>'), ('end', 'ID', '<')
            # subsequently we run the numba pair fn on the pairs:
            # ('start', 'ID', '<=') & ('end', 'ID', '>')
            # ('start', 'ID', '<=') & ('end', 'ID', '<')
            # finally unionize the outcome of the pairs
            # this only works if there is no null in the != condition
            # thanks to Hypothesis tests for pointing this out
            left_on, right_on, op = conditions[0]
            # check for nulls in the patch
            # and follow this path, only if there are no nulls
            if df[left_on].notna().all() & right[right_on].notna().all():
                patch = (
                    left_on,
                    right_on,
                    _JoinOperator.GREATER_THAN.value,
                ), (
                    left_on,
                    right_on,
                    _JoinOperator.LESS_THAN.value,
                )
                pairs.extend(patch)
                first, middle, last = pairs
                pairs = [(first, middle), (first, last)]
                indices = [_numba_dual_join(df, right, pair) for pair in pairs]
                indices = [arr for arr in indices if arr is not None]
                if not indices:
                    indices = None
                elif len(indices) == 1:
                    indices = indices[0]
                else:
                    indices = zip(*indices)
                    indices = map(np.concatenate, indices)
                conditions = conditions[1:]
            else:
                left_on, right_on, op = pairs[0]
                indices = _generic_func_cond_join(
                    df[left_on],
                    right[right_on],
                    op,
                    multiple_conditions=False,
                    keep="all",
                )
        else:
            indices = _numba_dual_join(df, right, pairs)
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

        # The numba version offers optimisations
        # for all types of non-equi joins
        # and is generally much faster

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
        if left_c.size < left_index.size:
            keep_rows = np.isin(left_index, left_c, assume_unique=True)
            search_indices = search_indices[keep_rows]
            left_index = left_c
    else:
        left_c = left_c._values
        right_c = right_c._values
        op = operator_map[op]
        left_c, right_c = _convert_to_numpy_array(left_c, right_c)
        pos = np.copy(search_indices)
        counter = np.arange(left_c.size)
        # better than np.outer memory wise?
        # using this for loop instead of np.outer
        # allows us to break early and reduce the
        # number of cartesian checks
        # since as we iterate, we reduce the size of left_c
        # speed wise, np.outer will be faster
        # alternatively, the user can just use the numba option
        # for more performance
        for ind in range(right_c.size):
            if not counter.size:
                break
            keep_rows = op(left_c, right_c[ind])
            if not keep_rows.any():
                continue
            pos[counter[keep_rows]] = ind
            counter = counter[~keep_rows]
            left_c = left_c[~keep_rows]

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
    sort_by_appearance: bool,
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

    if indicator:
        if isinstance(indicator, bool):
            indicator = "_merge"
        if indicator in df.columns.union(right.columns):
            raise ValueError(
                "Cannot use name of an existing column for indicator column"
            )

    if sort_by_appearance:
        if how in {"inner", "left"}:
            if not right.empty:
                right = right.take(right_index)
            right.index = left_index
        else:
            if not df.empty:
                df = df.take(left_index)
            df.index = right_index
        df = df.join(right, how=how)

        if indicator:
            if how == "right":
                df.loc[right_index, indicator] = 3
                df[indicator] = df[indicator].fillna(2)
            else:
                df.loc[left_index, indicator] = 3
                df[indicator] = df[indicator].fillna(1)
            df[indicator] = pd.Categorical(df[indicator], categories=[1, 2, 3])
            df[indicator] = df[indicator].cat.rename_categories(
                ["left_only", "right_only", "both"]
            )
        df.index = range(len(df))
        return df

    if how == "inner":
        return _inner(df, right, left_index, right_index, indicator)

    if how == "left":
        arr = df
        arr_ = np.delete(df.index._values, left_index)
    else:
        arr = right
        arr_ = np.delete(right.index._values, right_index)
    if not arr_.size:
        return _inner(df, right, left_index, right_index, indicator)
    if indicator:
        length = arr_.size
    arr_ = {key: value._values[arr_] for key, value in arr.items()}
    if indicator:
        arr_ = _add_indicator(
            df=arr_, indicator=indicator, how=how, length=length
        )
    arr_ = pd.DataFrame(arr_, copy=False)
    arr = _inner(df, right, left_index, right_index, indicator)
    df = pd.concat([arr, arr_], copy=False, sort=False)
    return df


def _add_indicator(
    df: dict, indicator: str, how: str, length: int
) -> pd.DataFrame:
    "Add indicator column to the DataFrame"
    mapping = {"inner": "both", "left": "left_only", "right": "right_only"}
    arr = pd.Categorical(
        [mapping[how]], categories=["left_only", "right_only", "both"]
    )
    if isinstance(next(iter(df)), tuple):
        indicator = (indicator, "")
    df[indicator] = arr.repeat(length)
    return df


def _inner(
    df: pd.DataFrame,
    right: pd.DataFrame,
    left_index: pd.DataFrame,
    right_index: pd.DataFrame,
    indicator: Union[bool, str],
) -> pd.DataFrame:
    """Create DataFrame for inner join"""

    df = {key: value._values[left_index] for key, value in df.items()}
    right = {key: value._values[right_index] for key, value in right.items()}
    df.update(right)
    if indicator:
        df = _add_indicator(df, indicator, "inner", left_index.size)
    return pd.DataFrame(df, copy=False)
