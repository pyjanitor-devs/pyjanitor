from pandas.core.construction import extract_array
from pandas.core.reshape.merge import _MergeOperation
from pandas.api.types import (
    is_datetime64_dtype,
    is_integer_dtype,
    is_float_dtype,
    is_string_dtype,
    is_extension_array_dtype,
    is_categorical_dtype,
)
import pandas_flavor as pf
import pandas as pd
from typing import Union
import operator
from janitor.utils import check, check_column
import numpy as np
from enum import Enum
from itertools import compress


@pf.register_dataframe_method
def conditional_join(
    df: pd.DataFrame,
    right: Union[pd.DataFrame, pd.Series],
    *conditions,
    how: str = "inner",
    sort_by_appearance: bool = False,
) -> pd.DataFrame:
    """

    This is a convenience function that operates similarly to ``pd.merge``,
    but allows joins on inequality operators,
    or a combination of equi and non-equi joins.

    If the join is solely on equality, `pd.merge` function
    is more efficient and should be used instead.

    If you are interested in nearest joins, or rolling joins,
    `pd.merge_asof` covers that. There is also the IntervalIndex,
    which is usually more efficient for range joins, especially if
    the intervals do not overlap.

    This function returns rows, if any, where values from `df` meet the
    condition(s) for values from `right`. The conditions are passed in
    as a variable argument of tuples, where the tuple is of
    the form `(left_on, right_on, op)`; `left_on` is the column
    label from `df`, `right_on` is the column label from `right`,
    while `op` is the operator.

    The operator can be any of `==`, `!=`, `<=`, `<`, `>=`, `>`.

    A binary search is used to get the relevant rows for non-equi joins;
    this avoids a cartesian join, and makes the process less memory intensive.

    For equi-joins, Pandas internal merge function (a hash join) is used.

    The join is done only on the columns.
    MultiIndex columns are not supported.

    For non-equi joins, only numeric and date columns are supported.

    Only `inner`, `left`, and `right` joins are supported.

    If the columns from `df` and `right` have nothing in common,
    a single index column is returned; else, a MultiIndex column
    is returned.

    Functional usage syntax:

    ```python
        import pandas as pd
        import janitor as jn

        df = pd.DataFrame(...)
        right = pd.DataFrame(...)

        df = jn.conditional_join(
                df,
                right,
                (col_from_df, col_from_right, join_operator),
                (col_from_df, col_from_right, join_operator),
                ...,
                how = 'inner' # or left/right
                sort_by_appearance = True # or False
                )
    ```

    Method chaining syntax:

    .. code-block:: python

        df.conditional_join(
            right,
            (col_from_df, col_from_right, join_operator),
            (col_from_df, col_from_right, join_operator),
            ...,
            how = 'inner' # or left/right
            sort_by_appearance = True # or False
            )


    :param df: A Pandas DataFrame.
    :param right: Named Series or DataFrame to join to.
    :param conditions: Variable argument of tuple(s) of the form
        `(left_on, right_on, op)`, where `left_on` is the column
        label from `df`, `right_on` is the column label from `right`,
        while `op` is the operator. The operator can be any of
        `==`, `!=`, `<=`, `<`, `>=`, `>`.
    :param how: Indicates the type of join to be performed.
        It can be one of `inner`, `left`, `right`.
        Full join is not supported. Defaults to `inner`.
    :param sort_by_appearance: Default is `False`. If True,
        values from `right` that meet the join condition will be returned
        in the final dataframe in the same order
        that they were before the join.
    :returns: A pandas DataFrame of the two merged Pandas objects.
    """

    return _conditional_join_compute(
        df, right, conditions, how, sort_by_appearance
    )


class _JoinOperator(Enum):
    """
    List of operators used in conditional_join.
    """

    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_THAN_OR_EQUAL = ">="
    LESS_THAN_OR_EQUAL = "<="
    STRICTLY_EQUAL = "=="
    NOT_EQUAL = "!="


class _JoinTypes(Enum):
    """
    List of join types for conditional_join.
    """

    INNER = "inner"
    LEFT = "left"
    RIGHT = "right"


operator_map = {
    _JoinOperator.STRICTLY_EQUAL.value: operator.eq,
    _JoinOperator.LESS_THAN.value: operator.lt,
    _JoinOperator.LESS_THAN_OR_EQUAL.value: operator.le,
    _JoinOperator.GREATER_THAN.value: operator.gt,
    _JoinOperator.GREATER_THAN_OR_EQUAL.value: operator.ge,
    _JoinOperator.NOT_EQUAL.value: operator.ne,
}


def _conditional_join_compute(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list,
    how: str,
    sort_by_appearance: bool,
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
    ) = _conditional_join_preliminary_checks(
        df, right, conditions, how, sort_by_appearance
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

    df.index = pd.RangeIndex(start=0, stop=len(df))
    right.index = pd.RangeIndex(start=0, stop=len(right))

    if len(conditions) == 1:
        left_on, right_on, op = conditions[0]

        result = _generic_func_cond_join(
            df, right, left_on, right_on, op, "single"
        )

        if result is None:
            return _create_conditional_join_empty_frame(df, right, how)

        left_c, right_c = result

        return _create_conditional_join_frame(
            df, right, left_c, right_c, how, sort_by_appearance
        )

    # multiple conditions
    if eq_check:
        result = _multiple_conditional_join_eq(df, right, conditions)
    elif le_lt_check:
        result = _multiple_conditional_join_le_lt(df, right, conditions)
    else:
        result = _multiple_conditional_join_ne(df, right, conditions)

    if result is None:
        return _create_conditional_join_empty_frame(df, right, how)

    left_c, right_c = result

    return _create_conditional_join_frame(
        df, right, left_c, right_c, how, sort_by_appearance
    )


def _conditional_join_preliminary_checks(
    df: pd.DataFrame,
    right: Union[pd.DataFrame, pd.Series],
    conditions: tuple,
    how: str,
    sort_by_appearance: bool,
) -> tuple:
    """
    Preliminary checks for conditional_join are conducted here.

    This function checks for conditions such as
    MultiIndexed dataframe columns,
    as well as unnamed Series.

    A tuple of
    (`df`, `right`, `left_on`, `right_on`, `operator`)
    is returned.
    """

    if isinstance(df.columns, pd.MultiIndex):
        raise ValueError(
            """
            MultiIndex columns are not
            supported for conditional_join.
            """
        )

    check("`right`", right, [pd.DataFrame, pd.Series])

    df = df.copy()
    right = right.copy()

    if isinstance(right, pd.Series):
        if not right.name:
            raise ValueError(
                """
                Unnamed Series are not supported
                for conditional_join.
                """
            )
        right = right.to_frame()

    if isinstance(right.columns, pd.MultiIndex):
        raise ValueError(
            """
            MultiIndex columns are not supported
            for conditional joins.
            """
        )

    if not conditions:
        raise ValueError(
            """
            Kindly provide at least one join condition.
            """
        )

    for condition in conditions:
        check("condition", condition, [tuple])
        len_condition = len(condition)
        if len_condition != 3:
            raise ValueError(
                f"""
                condition should have only three elements.
                {condition} however is of length {len_condition}.
                """
            )

    for left_on, right_on, op in conditions:
        check("left_on", left_on, [str])
        check("right_on", right_on, [str])
        check("operator", op, [str])
        check_column(df, left_on)
        check_column(right, right_on)
        _check_operator(op)

    check("how", how, [str])

    join_types = {jointype.value for jointype in _JoinTypes}
    if how not in join_types:
        raise ValueError(f"`how` should be one of {', '.join(join_types)}.")

    check("sort_by_appearance", sort_by_appearance, [bool])

    return df, right, conditions, how, sort_by_appearance


def _conditional_join_type_check(
    left_column: pd.Series, right_column: pd.Series, op: str
) -> None:
    """
    Raise error if column type is not
    any of numeric or datetime or string.
    """

    permitted_types = {
        is_datetime64_dtype,
        is_integer_dtype,
        is_float_dtype,
        is_string_dtype,
        is_categorical_dtype,
    }
    for func in permitted_types:
        if func(left_column):
            break
    else:
        raise ValueError(
            """
            conditional_join only supports
            string, category, integer,
            float or date dtypes.
            """
        )
    cols = (left_column, right_column)
    for func in permitted_types:
        if all(map(func, cols)):
            break
    else:
        raise ValueError(
            f"""
             Both columns should have the same type.
             `{left_column.name}` has {left_column.dtype} type;
             `{right_column.name}` has {right_column.dtype} type.
             """
        )

    if (
        is_string_dtype(left_column)
        and op != _JoinOperator.STRICTLY_EQUAL.value
    ):
        raise ValueError(
            """
            For string columns,
            only the `==` operator is supported.
            """
        )

    if (
        is_categorical_dtype(left_column)
        and op != _JoinOperator.STRICTLY_EQUAL.value
    ):
        raise ValueError(
            """
            For categorical columns,
            only the `==` operator is supported.
            """
        )

    return None


def _generic_func_cond_join(
    df: pd.DataFrame,
    right: pd.DataFrame,
    left_on: Union[str, list],
    right_on: Union[str, list],
    op: str,
    join_type: str,
):
    """
    Generic function to call any of the individual functions
    (_less_than_indices, _greater_than_indices, _equal_indices,
    or _not_equal_indices).
    """
    strict = False

    if op in {
        _JoinOperator.GREATER_THAN.value,
        _JoinOperator.LESS_THAN.value,
        _JoinOperator.NOT_EQUAL.value,
    }:
        strict = True

    if op in less_than_join_types:
        return _less_than_indices(
            df, right, left_on, right_on, strict, join_type
        )
    elif op in greater_than_join_types:
        return _greater_than_indices(
            df, right, left_on, right_on, strict, join_type
        )
    elif op == _JoinOperator.NOT_EQUAL.value:
        return _not_equal_indices(df, right, left_on, right_on)
    else:
        return _equal_indices(df, right, left_on, right_on)


def _multiple_conditional_join_ne(
    df: pd.DataFrame, right: pd.DataFrame, conditions: list
) -> tuple:
    """
    Get indices for multiple conditions,
    where all the operators are `!=`.

    Returns a tuple of (df_index, right_index)
    """

    first, *rest = conditions
    left_on, right_on, op = first
    result = _generic_func_cond_join(
        df, right, left_on, right_on, op, "single"
    )
    if result is None:
        return None

    df_index, right_index = result

    mask = None
    for left_on, right_on, op in rest:
        left_c = extract_array(df[left_on], extract_numpy=True)
        left_c = left_c[df_index]
        right_c = extract_array(right[right_on], extract_numpy=True)
        right_c = right_c[right_index]
        op = operator_map[op]

        if mask is None:
            mask = op(left_c, right_c)
        else:
            mask &= op(left_c, right_c)

    if not mask.any():
        return None

    if mask.all():
        return df_index, right_index

    if is_extension_array_dtype(mask):
        mask = mask.to_numpy(dtype=bool, na_value=False)

    return df_index[mask], right_index[mask]


def _multiple_conditional_join_eq(
    df: pd.DataFrame, right: pd.DataFrame, conditions: list
) -> tuple:
    """
    Get indices for multiple conditions,
    if any of the conditions has an `==` operator.

    Returns a tuple of (df_index, right_index)
    """

    eq_cond = [
        cond
        for cond in conditions
        if cond[-1] == _JoinOperator.STRICTLY_EQUAL.value
    ]
    rest = [
        cond
        for cond in conditions
        if cond[-1] != _JoinOperator.STRICTLY_EQUAL.value
    ]

    left_on, right_on, _ = zip(*eq_cond)

    result = _generic_func_cond_join(
        df,
        right,
        [*left_on],
        [*right_on],
        _JoinOperator.STRICTLY_EQUAL.value,
        "single",
    )

    if result is None:
        return None

    if not rest:
        return result

    df_index, right_index = result

    # non-equi conditions are present
    mask = None
    for left_on, right_on, op in rest:
        left_c = extract_array(df[left_on], extract_numpy=True)
        left_c = left_c[df_index]
        right_c = extract_array(right[right_on], extract_numpy=True)
        right_c = right_c[right_index]
        op = operator_map[op]

        if mask is None:
            mask = op(left_c, right_c)
        else:
            mask &= op(left_c, right_c)

    if not mask.any():
        return None

    if mask.all():
        return df_index, right_index

    if is_extension_array_dtype(mask):
        mask = mask.to_numpy(dtype=bool, na_value=False)

    return df_index[mask], right_index[mask]


def _multiple_conditional_join_le_lt(
    df: pd.DataFrame, right: pd.DataFrame, conditions: list
) -> tuple:
    """
    Get indices for multiple conditions,
    where `>/>=` or `</<=` is present,
    and there is no `==` operator.

    Returns a tuple of (df_index, right_index)
    """

    check1 = len(conditions) == 2
    check2 = (conditions[0][-1] in less_than_join_types) & (
        conditions[-1][-1] in greater_than_join_types
    )
    check3 = (conditions[0][-1] in greater_than_join_types) & (
        conditions[-1][-1] in less_than_join_types
    )
    if check1 & (check2 | check3):
        (
            (start_left, end_left),
            (start_right, end_right),
            (left_op, right_op),
        ) = zip(*conditions)
        if check2:
            start_left, end_left = end_left, start_left
            start_right, end_right = end_right, start_right
            left_op, right_op = right_op, left_op
        return _range_indices(
            df,
            right,
            start_left,
            end_left,
            start_right,
            end_right,
            left_op,
            right_op,
        )

    # find minimum df_index and right_index
    # aim is to reduce search space
    df_index = df.index
    right_index = right.index
    arrs = []
    for left_on, right_on, op in conditions:
        # no point checking for `!=`, since best case scenario
        # they'll have the same no of rows as the other operators
        if op == _JoinOperator.NOT_EQUAL.value:
            continue

        result = _generic_func_cond_join(
            df.loc[df_index],
            right.loc[right_index],
            left_on,
            right_on,
            op,
            join_type="multiple",
        )

        if result is None:
            return None

        df_index, right_index, arr, lengths = result
        arrs.append((df_index, arr, lengths))

    new_arrs = []

    # trim to the minimum index for df
    # and the smallest indices for right
    # the minimum index for df should be available
    # to all conditions; we achieve this via boolean indexing
    for l_index, r_index, repeats in arrs:
        bools = np.isin(l_index, df_index)
        if not np.all(bools):
            l_index = l_index[bools]
            r_index = compress(r_index, bools)
            repeats = repeats[bools]
        new_arrs.append((l_index, r_index, repeats))
    _, arr, repeats = zip(*new_arrs)

    # with the aim of reducing the search space
    # we get the smallest indices for each index in df
    # e.g if df_index is [1, 2, 3]
    # and there are two outcomes for the conditions for right:
    # [[1,2,3], [4]], [[2], [4, 6]]
    # the reconstituted indices will be the smallest per pairing
    # which turns out to : [ [2], [4]]
    # we achieve this by getting the minimum size in `repeats`
    # and use that to index into `arr`
    repeats = np.vstack(repeats)
    positions = np.argmin(repeats, axis=0)
    repeats = np.minimum.reduce(repeats)
    arrays = []
    arr = zip(*arr)  # pair all the indices for right obtained per condition
    for row, pos in zip(arr, positions):
        arrays.append(row[pos])
    right_index = np.concatenate(arrays)
    df_index = df_index.repeat(repeats)

    mask = None
    for left_on, right_on, op in conditions:
        left_c = extract_array(df[left_on], extract_numpy=True)
        left_c = left_c[df_index]
        right_c = extract_array(right[right_on], extract_numpy=True)
        right_c = right_c[right_index]
        op = operator_map[op]

        if mask is None:
            mask = op(left_c, right_c)
        else:
            mask &= op(left_c, right_c)

    if not mask.any():
        return None

    if mask.all():
        return df_index, right_index

    if is_extension_array_dtype(mask):
        mask = mask.to_numpy(dtype=bool, na_value=False)

    return df_index[mask], right_index[mask]


def _create_conditional_join_empty_frame(
    df: pd.DataFrame, right: pd.DataFrame, how: str
):
    """
    Create final dataframe for conditional join,
    if there are no matches.
    """

    if set(df.columns).intersection(right.columns):
        df.columns = pd.MultiIndex.from_product([["left"], df.columns])
        right.columns = pd.MultiIndex.from_product([["right"], right.columns])

    if how == _JoinTypes.INNER.value:
        df = df.dtypes.to_dict()
        right = right.dtypes.to_dict()
        df = {**df, **right}
        df = {key: pd.Series([], dtype=value) for key, value in df.items()}
        return pd.DataFrame(df)

    if how == _JoinTypes.LEFT.value:
        right = right.dtypes.to_dict()
        right = {
            key: float if dtype.kind == "i" else dtype
            for key, dtype in right.items()
        }
        right = {
            key: pd.Series([], dtype=value) for key, value in right.items()
        }
        right = pd.DataFrame(right)
        return df.join(right, how=how, sort=False)

    if how == _JoinTypes.RIGHT.value:
        df = df.dtypes.to_dict()
        df = {
            key: float if dtype.kind == "i" else dtype
            for key, dtype in df.items()
        }
        df = {key: pd.Series([], dtype=value) for key, value in df.items()}
        df = pd.DataFrame(df)
        return df.join(right, how=how, sort=False)


def _create_conditional_join_frame(
    df: pd.DataFrame,
    right: pd.DataFrame,
    left_index: pd.Index,
    right_index: pd.Index,
    how: str,
    sort_by_appearance: bool,
):
    """
    Create final dataframe for conditional join,
    if there are matches.
    """
    if sort_by_appearance:
        sorter = np.lexsort((right_index, left_index))
        right_index = right_index[sorter]
        left_index = left_index[sorter]
        sorter = None

    if set(df.columns).intersection(right.columns):
        df.columns = pd.MultiIndex.from_product([["left"], df.columns])
        right.columns = pd.MultiIndex.from_product([["right"], right.columns])

    if how == _JoinTypes.INNER.value:
        df = {
            key: extract_array(value, extract_numpy=True)[left_index]
            for key, value in df.items()
        }
        right = {
            key: extract_array(value, extract_numpy=True)[right_index]
            for key, value in right.items()
        }
        return pd.DataFrame({**df, **right})

    if how == _JoinTypes.LEFT.value:
        right = right.loc[right_index]
        right.index = left_index
        return df.join(right, how=how, sort=False).reset_index(drop=True)

    if how == _JoinTypes.RIGHT.value:
        df = df.loc[left_index]
        df.index = right_index
        return df.join(right, how=how, sort=False).reset_index(drop=True)


less_than_join_types = {
    _JoinOperator.LESS_THAN.value,
    _JoinOperator.LESS_THAN_OR_EQUAL.value,
}
greater_than_join_types = {
    _JoinOperator.GREATER_THAN.value,
    _JoinOperator.GREATER_THAN_OR_EQUAL.value,
}


def _check_operator(op: str):
    """
    Check that operator is one of
    `>`, `>=`, `==`, `!=`, `<`, `<=`.

    Used in `conditional_join`.
    """
    sequence_of_operators = {op.value for op in _JoinOperator}
    if op not in sequence_of_operators:
        raise ValueError(
            f"""
             The conditional join operator
             should be one of {", ".join(sequence_of_operators)}
             """
        )


def _range_indices(
    df: pd.DataFrame,
    right: pd.DataFrame,
    start_left: str,
    end_left: str,
    start_right: str,
    end_right: str,
    left_op: str,
    right_op: str,
):
    """
    Retrieve index positions for a range/between join.

    Idea inspired by article:
    https://www.vertica.com/blog/what-is-a-range-join-and-why-is-it-so-fastba-p223413/

    Returns a tuple of (left_index, right_index)
    """

    # summary of code:
    # get the positions where start_left is >/>= start_right
    # then within the positions,
    # get the positions where end_left is </<= end_right
    # this should reduce the search space significantly
    strict = False
    if left_op == _JoinOperator.GREATER_THAN.value:
        strict = True

    outcome = _greater_than_indices(
        df, right, start_left, start_right, strict, join_type="dual"
    )

    if outcome is None:
        return None

    search_indices, right_index, left_index = outcome
    right_c = right.loc[right_index, end_right]
    right_c = extract_array(right_c, extract_numpy=True)
    left_c = df.loc[left_index, end_left]
    left_c = extract_array(left_c, extract_numpy=True)
    top_pos = np.copy(search_indices)
    bottom_pos = np.copy(search_indices)
    counter = np.arange(left_index.size)
    right_op = operator_map[right_op]

    for ind in range(np.max(search_indices)):
        # get rid of rows if it exceeds
        # the range for the search_indices
        # e.g search_indices is [2, 4, 6]
        # our search space for 2( the first index)
        # is 0, 1. if ind is 2, there is no point
        # searching within the first index space anymore
        exclude_rows = search_indices <= ind
        if exclude_rows.any():
            counter = counter[~exclude_rows]
            search_indices = search_indices[~exclude_rows]
            left_c = left_c[~exclude_rows]
        keep_rows = right_op(left_c, right_c[ind])
        # get the index positions where left_c is </<= right_c
        # that minimum position combined with the equivalent position
        # from search_indices becomes our search space
        # for the equivalent left_c index
        if keep_rows.any():
            top_pos[counter[keep_rows]] = ind
            counter = counter[~keep_rows]
            search_indices = search_indices[~keep_rows]
            left_c = left_c[~keep_rows]
        if not counter.size > 0:
            break

    # no point searching within (a, b)
    # if a == b
    # since range(a, b) yields none
    keep_rows = top_pos < bottom_pos

    if not keep_rows.any():
        return None

    if not keep_rows.all():
        left_index = left_index[keep_rows]
        top_pos = top_pos[keep_rows]
        bottom_pos = bottom_pos[keep_rows]
    repeater = bottom_pos - top_pos
    right_index = [
        right_index[start:end] for start, end in zip(top_pos, bottom_pos)
    ]
    right_index = np.concatenate(right_index)
    left_index = np.repeat(left_index, repeater)

    # here we search for actual positions
    # where left_c is </<= right_c
    left_c = extract_array(df[end_left], extract_numpy=True)[left_index]
    right_c = extract_array(right[end_right], extract_numpy=True)[right_index]

    mask = right_op(left_c, right_c)

    if not mask.any():
        return None
    if not mask.all():
        return left_index[mask], right_index[mask]

    return left_index, right_index


def _less_than_indices(
    df: pd.DataFrame,
    right: pd.DataFrame,
    left_on: str,
    right_on: str,
    strict: bool,
    join_type: str = "single",
) -> tuple:
    """
    Use binary search to get indices where left_c
    is less than or equal to right_c.

    If strict is True,then only indices
    where `left_c` is less than
    (but not equal to) `right_c` are returned.

    Returns a tuple of (left_c, right_c)
    """

    left_c = df[left_on]
    right_c = right[right_on]

    # no point going through all the hassle
    if left_c.min() > right_c.max():
        return None

    any_nulls = pd.isna(right_c.array)
    if any_nulls.any():
        right_c = right_c[~any_nulls]
    any_nulls = pd.isna(left_c.array)
    if any_nulls.any():
        left_c = left_c[~any_nulls]

    if not right_c.is_monotonic_increasing:
        right_c = right_c.sort_values(kind="stable")

    left_index = left_c.index.to_numpy(dtype=int, copy=False)
    left_c = extract_array(left_c, extract_numpy=True)
    right_index = right_c.index.to_numpy(dtype=int, copy=False)
    right_c = extract_array(right_c, extract_numpy=True)

    search_indices = right_c.searchsorted(left_c, side="left")
    # if any of the positions in `search_indices`
    # is equal to the length of `right_keys`
    # that means the respective position in `left_c`
    # has no values from `right_c` that are less than
    # or equal, and should therefore be discarded
    len_right = right_c.size
    rows_equal = search_indices == len_right

    if rows_equal.any():
        left_c = left_c[~rows_equal]
        left_index = left_index[~rows_equal]
        search_indices = search_indices[~rows_equal]

    if search_indices.size == 0:
        return None

    # the idea here is that if there are any equal values
    # shift upwards to the immediate next position
    # that is not equal
    if strict:
        rows_equal = right_c[search_indices]
        rows_equal = left_c == rows_equal
        # replace positions where rows are equal
        # with positions from searchsorted('right')
        # positions from searchsorted('right') will never
        # be equal and will be the furthermost in terms of position
        # example : right_c -> [2, 2, 2, 3], and we need
        # positions where values are not equal for 2;
        # the furthermost will be 3, and searchsorted('right')
        # will return position 3.
        if rows_equal.any():
            replacements = right_c.searchsorted(left_c, side="right")
            # now we can safely replace values
            # with strictly less than positions
            search_indices = np.where(rows_equal, replacements, search_indices)
        # check again if any of the values
        # have become equal to length of right_c
        # and get rid of them
        rows_equal = search_indices == len_right

        if rows_equal.any():
            left_c = left_c[~rows_equal]
            left_index = left_index[~rows_equal]
            search_indices = search_indices[~rows_equal]

    if search_indices.size == 0:
        return None

    right_c = [right_index[ind:len_right] for ind in search_indices]

    if join_type == "multiple":
        return (
            left_index,
            right_index[search_indices.min() :],  # noqa: E203
            right_c,
            len_right - search_indices,
        )

    search_indices = len_right - search_indices
    left_c = np.repeat(left_index, search_indices)
    right_c = np.concatenate(right_c)
    return left_c, right_c


def _greater_than_indices(
    df: pd.DataFrame,
    right: pd.DataFrame,
    left_on: str,
    right_on: str,
    strict: bool,
    join_type: str = "single",
) -> tuple:
    """
    Use binary search to get indices where left_c
    is greater than or equal to right_c.

    If strict is True,then only indices
    where `left_c` is greater than
    (but not equal to) `right_c` are returned.

    Returns a tuple of (left_c, right_c).
    """
    left_c = df[left_on]
    right_c = right[right_on]

    # quick break, avoiding the hassle
    if left_c.max() < right_c.min():
        return None

    any_nulls = pd.isna(right_c.array)
    if any_nulls.any():
        right_c = right_c[~any_nulls]
    any_nulls = pd.isna(left_c.array)
    if any_nulls.any():
        left_c = left_c[~any_nulls]

    if not right_c.is_monotonic_increasing:
        right_c = right_c.sort_values(kind="stable")

    left_index = left_c.index.to_numpy(dtype=int, copy=False)
    left_c = extract_array(left_c, extract_numpy=True)
    right_index = right_c.index.to_numpy(dtype=int, copy=False)
    right_c = extract_array(right_c, extract_numpy=True)

    search_indices = right_c.searchsorted(left_c, side="right")
    # if any of the positions in `search_indices`
    # is equal to 0 (less than 1), it implies that
    # left_c[position] is not greater than any value
    # in right_c
    rows_equal = search_indices < 1
    if rows_equal.any():
        left_c = left_c[~rows_equal]
        left_index = left_index[~rows_equal]
        search_indices = search_indices[~rows_equal]
    if search_indices.size == 0:
        return None

    # the idea here is that if there are any equal values
    # shift downwards to the immediate next position
    # that is not equal
    if strict:
        rows_equal = right_c[search_indices - 1]
        rows_equal = left_c == rows_equal
        # replace positions where rows are equal with
        # searchsorted('left');
        # however there can be scenarios where positions
        # from searchsorted('left') would still be equal;
        # in that case, we shift down by 1
        if rows_equal.any():
            replacements = right_c.searchsorted(left_c, side="left")
            # return replacements
            # `left` might result in values equal to len right_c
            replacements = np.where(
                replacements == right_c.size, replacements - 1, replacements
            )
            # now we can safely replace values
            # with strictly greater than positions
            search_indices = np.where(rows_equal, replacements, search_indices)
        # any value less than 1 should be discarded
        # since the lowest value for binary search
        # with side='right' should be 1
        rows_equal = search_indices < 1
        if rows_equal.any():
            left_c = left_c[~rows_equal]
            left_index = left_index[~rows_equal]
            search_indices = search_indices[~rows_equal]

    if search_indices.size == 0:
        return None

    if join_type == "dual":
        return search_indices, right_index, left_index

    right_c = [right_index[:ind] for ind in search_indices]

    if join_type == "multiple":
        return (
            left_index,
            right_index[: search_indices.max()],
            right_c,
            search_indices,
        )

    left_c = np.repeat(left_index, search_indices)
    right_c = np.concatenate(right_c)
    return left_c, right_c


def _equal_indices(
    df: pd.DataFrame,
    right: pd.DataFrame,
    left_on: Union[str, list],
    right_on: Union[str, list],
) -> tuple:
    """
    Use Pandas' merge internal functions
    to find the matches, if any.

    Returns a tuple of (left_c, right_c)
    """

    # select_columns will take care of duplicates
    exclude_rows = df.select_columns(left_on).isna().any(axis=1)
    if exclude_rows.any(axis=None):
        df = df.loc[~exclude_rows]

    exclude_rows = right.select_columns(right_on).isna().any(axis=1)
    if exclude_rows.any(axis=None):
        right = right.loc[~exclude_rows]

    outcome = _MergeOperation(
        left=df,
        right=right,
        left_on=left_on,
        right_on=right_on,
        sort=False,
    )

    left_index, right_index = outcome._get_join_indexers()
    left_c = df.index.to_numpy(dtype=int, copy=False)
    right_c = right.index.to_numpy(dtype=int, copy=False)

    if not left_index.size > 0:
        return None

    return left_c[left_index], right_c[right_index]


def _not_equal_indices(
    df: pd.DataFrame,
    right: pd.DataFrame,
    left_on: str,
    right_on: str,
) -> tuple:
    """
    Use binary search to get indices where
    `left_c` is exactly  not equal to `right_c`.

    It is a combination of strictly less than
    and strictly greater than indices.

    Returns a tuple of (left_c, right_c)
    """

    dummy = np.array([], dtype=int)

    outcome = _less_than_indices(
        df, right, left_on, right_on, True, join_type="single"
    )

    if outcome is None:
        lt_left = dummy
        lt_right = dummy
    else:
        lt_left, lt_right = outcome

    outcome = _greater_than_indices(
        df, right, left_on, right_on, True, join_type="single"
    )

    if outcome is None:
        gt_left = dummy
        gt_right = dummy
    else:
        gt_left, gt_right = outcome

    if (not lt_left.size > 0) and (not gt_left.size > 0):
        return None
    left_c = np.concatenate([lt_left, gt_left])
    right_c = np.concatenate([lt_right, gt_right])

    return left_c, right_c
