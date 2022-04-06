import operator
from enum import Enum
from typing import Union

import numpy as np
import pandas as pd
import pandas_flavor as pf
from pandas.core.construction import extract_array
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


@pf.register_dataframe_method
def conditional_join(
    df: pd.DataFrame,
    right: Union[pd.DataFrame, pd.Series],
    *conditions,
    how: str = "inner",
    sort_by_appearance: bool = False,
) -> pd.DataFrame:
    """

    This is a convenience function that operates similarly to `pd.merge`,
    but allows joins on inequality operators,
    or a combination of equi and non-equi joins.

    Join solely on equality are not supported.

    If the join is solely on equality, `pd.merge` function
    covers that; if you are interested in nearest joins, or rolling joins,
    or the first match (lowest or highest) - `pd.merge_asof` covers that.
    There is also the IntervalIndex, which is usually more efficient
    for range joins, especially if the intervals do not overlap.

    This function returns rows, if any, where values from `df` meet the
    condition(s) for values from `right`. The conditions are passed in
    as a variable argument of tuples, where the tuple is of
    the form `(left_on, right_on, op)`; `left_on` is the column
    label from `df`, `right_on` is the column label from `right`,
    while `op` is the operator. For multiple conditions, the and(`&`)
    operator is used to combine the results of the individual conditions.

    The operator can be any of `==`, `!=`, `<=`, `<`, `>=`, `>`.

    A binary search is used to get the relevant rows for non-equi joins;
    this avoids a cartesian join, and makes the process less memory intensive.

    For equi-joins, Pandas internal merge function is used.

    The join is done only on the columns.
    MultiIndex columns are not supported.

    For non-equi joins, only numeric and date columns are supported.

    Only `inner`, `left`, and `right` joins are supported.

    If the columns from `df` and `right` have nothing in common,
    a single index column is returned; else, a MultiIndex column
    is returned.

    Example:

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
        ...     ("value_1", "value_2A", ">="),
        ...     ("value_1", "value_2B", "<=")
        ... )
            value_1  value_2A  value_2B
        0         2         1         3
        1         2         2         4
        2         5         3         5
        3         5         3         6
        4         7         7         9
        5         1         0         1
        6         1         0         1
        7         1         1         3
        8         3         1         3
        9         3         2         4
        10        3         3         5
        11        3         3         6
        12        4         2         4
        13        4         3         5
        14        4         3         6


    :param df: A pandas DataFrame.
    :param right: Named Series or DataFrame to join to.
    :param conditions: Variable argument of tuple(s) of the form
        `(left_on, right_on, op)`, where `left_on` is the column
        label from `df`, `right_on` is the column label from `right`,
        while `op` is the operator. The operator can be any of
        `==`, `!=`, `<=`, `<`, `>=`, `>`. For multiple conditions,
        the and(`&`) operator is used to combine the results
        of the individual conditions.
    :param how: Indicates the type of join to be performed.
        It can be one of `inner`, `left`, `right`.
        Full join is not supported. Defaults to `inner`.
    :param sort_by_appearance: Default is `False`.
        This is useful for strictly non-equi joins,
        where the user wants the original order maintained.
        If True, values from `df` and `right`
        that meet the join condition will be returned
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
            "The conditional join operator "
            f"should be one of {sequence_of_operators}"
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
            "MultiIndex columns are not supported for conditional joins."
        )

    check("right", right, [pd.DataFrame, pd.Series])

    df = df.copy()
    right = right.copy()

    if isinstance(right, pd.Series):
        if not right.name:
            raise ValueError(
                "Unnamed Series are not supported for conditional_join."
            )
        right = right.to_frame()

    if isinstance(right.columns, pd.MultiIndex):
        raise ValueError(
            "MultiIndex columns are not supported for conditional joins."
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
        check("left_on", left_on, [str])
        check("right_on", right_on, [str])
        check("operator", op, [str])
        check_column(df, left_on)
        check_column(right, right_on)
        _check_operator(op)

    if all(
        (op == _JoinOperator.STRICTLY_EQUAL.value for *_, op in conditions)
    ):
        raise ValueError("Equality only joins are not supported.")

    check("how", how, [str])

    checker = {jointype.value for jointype in _JoinTypes}
    if how not in checker:
        raise ValueError(f"'how' should be one of {checker}.")

    check("sort_by_appearance", sort_by_appearance, [bool])

    return df, right, conditions, how, sort_by_appearance


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

    if (op in less_than_join_types.union(greater_than_join_types)) & (
        (is_string_dtype(left_column) | is_categorical_dtype(left_column))
    ):
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

    df.index = range(len(df))
    right.index = range(len(right))

    multiple_conditions = len(conditions) > 1

    if not multiple_conditions:
        left_on, right_on, op = conditions[0]

        result = _generic_func_cond_join(
            df[left_on], right[right_on], op, multiple_conditions
        )

        if result is None:
            return _create_conditional_join_empty_frame(df, right, how)

        return _create_conditional_join_frame(
            df, right, *result, how, sort_by_appearance
        )

    if eq_check:
        result = _multiple_conditional_join_eq(df, right, conditions)
    elif le_lt_check:
        result = _multiple_conditional_join_le_lt(df, right, conditions)
    else:
        result = _multiple_conditional_join_ne(df, right, conditions)

    if result is None:
        return _create_conditional_join_empty_frame(df, right, how)

    return _create_conditional_join_frame(
        df, right, *result, how, sort_by_appearance
    )


def _less_than_indices(
    left_c: pd.Series,
    right_c: pd.Series,
    strict: bool,
) -> tuple:
    """
    Use binary search to get indices where left_c
    is less than or equal to right_c.

    If strict is True, then only indices
    where `left_c` is less than
    (but not equal to) `right_c` are returned.

    A tuple of integer indexes
    for left_c and right_c is returned.
    """

    # no point going through all the hassle
    if left_c.min() > right_c.max():
        return None

    any_nulls = pd.isna(right_c)
    if any_nulls.any():
        right_c = right_c[~any_nulls]
    if right_c.empty:
        return None
    any_nulls = pd.isna(left_c)
    if any_nulls.any():
        left_c = left_c[~any_nulls]
    if left_c.empty:
        return None
    any_nulls = None

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

    # the idea here is that if there are any equal values
    # shift to the right to the immediate next position
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

        if not search_indices.size:
            return None

    right_c = [right_index[ind:len_right] for ind in search_indices]
    right_c = np.concatenate(right_c)
    left_c = np.repeat(left_index, len_right - search_indices)
    return left_c, right_c


def _greater_than_indices(
    left_c: pd.Series,
    right_c: pd.Series,
    strict: bool,
    multiple_conditions: bool,
) -> tuple:
    """
    Use binary search to get indices where left_c
    is greater than or equal to right_c.

    If strict is True, then only indices
    where `left_c` is greater than
    (but not equal to) `right_c` are returned.

    if multiple_conditions is False, a tuple of integer indexes
    for left_c and right_c is returned;
    else a tuple of the index for left_c, right_c, as well
    as the positions of left_c in right_c is returned.
    """

    # quick break, avoiding the hassle
    if left_c.max() < right_c.min():
        return None

    any_nulls = pd.isna(right_c)
    if any_nulls.any():
        right_c = right_c[~any_nulls]
    if right_c.empty:
        return None
    any_nulls = pd.isna(left_c)
    if any_nulls.any():
        left_c = left_c[~any_nulls]
    if left_c.empty:
        return None
    any_nulls = None

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

        if not search_indices.size:
            return None

    if multiple_conditions:
        return left_index, right_index, search_indices
    right_c = [right_index[:ind] for ind in search_indices]
    right_c = np.concatenate(right_c)
    left_c = np.repeat(left_index, search_indices)

    return left_c, right_c


def _not_equal_indices(left_c: pd.Series, right_c: pd.Series) -> tuple:
    """
    Use binary search to get indices where
    `left_c` is exactly  not equal to `right_c`.

    It is a combination of strictly less than
    and strictly greater than indices.

    A tuple of integer indexes for left_c and right_c
    is returned.
    """

    dummy = np.array([], dtype=int)

    # deal with nulls
    l1_nulls = dummy
    r1_nulls = dummy
    l2_nulls = dummy
    r2_nulls = dummy
    any_left_nulls = left_c.isna()
    any_right_nulls = right_c.isna()
    if any_left_nulls.any():
        l1_nulls = left_c.index[any_left_nulls.array]
        l1_nulls = l1_nulls.to_numpy(copy=False)
        r1_nulls = right_c.index
        # avoid NAN duplicates
        if any_right_nulls.any():
            r1_nulls = r1_nulls[~any_right_nulls.array]
        r1_nulls = r1_nulls.to_numpy(copy=False)
        nulls_count = l1_nulls.size
        # blow up nulls to match length of right
        l1_nulls = np.tile(l1_nulls, r1_nulls.size)
        # ensure length of right matches left
        if nulls_count > 1:
            r1_nulls = np.repeat(r1_nulls, nulls_count)
    if any_right_nulls.any():
        r2_nulls = right_c.index[any_right_nulls.array]
        r2_nulls = r2_nulls.to_numpy(copy=False)
        l2_nulls = left_c.index
        nulls_count = r2_nulls.size
        # blow up nulls to match length of left
        r2_nulls = np.tile(r2_nulls, l2_nulls.size)
        # ensure length of left matches right
        if nulls_count > 1:
            l2_nulls = np.repeat(l2_nulls, nulls_count)

    l1_nulls = np.concatenate([l1_nulls, l2_nulls])
    r1_nulls = np.concatenate([r1_nulls, r2_nulls])

    outcome = _less_than_indices(left_c, right_c, strict=True)

    if outcome is None:
        lt_left = dummy
        lt_right = dummy
    else:
        lt_left, lt_right = outcome

    outcome = _greater_than_indices(
        left_c, right_c, strict=True, multiple_conditions=False
    )

    if outcome is None:
        gt_left = dummy
        gt_right = dummy
    else:
        gt_left, gt_right = outcome

    left_c = np.concatenate([lt_left, gt_left, l1_nulls])
    right_c = np.concatenate([lt_right, gt_right, r1_nulls])

    if (not left_c.size) & (not right_c.size):
        return None

    return left_c, right_c


def _eq_indices(
    left_c: pd.Series,
    right_c: pd.Series,
) -> tuple:
    """
    Use binary search to get indices where left_c
    is equal to right_c.

    Returns a tuple of the left_index, right_index,
    lower_boundary and upper_boundary.
    """

    # no point going through all the hassle
    if left_c.min() > right_c.max():
        return None
    if left_c.max() < right_c.min():
        return None

    any_nulls = pd.isna(right_c)
    if any_nulls.any():
        right_c = right_c[~any_nulls]
    if right_c.empty:
        return None
    any_nulls = pd.isna(left_c)
    if any_nulls.any():
        left_c = left_c[~any_nulls]
    if left_c.empty:
        return None
    any_nulls = None

    if not right_c.is_monotonic_increasing:
        right_c = right_c.sort_values(kind="stable")

    left_index = left_c.index.to_numpy(dtype=int, copy=False)
    left_c = extract_array(left_c, extract_numpy=True)
    right_index = right_c.index.to_numpy(dtype=int, copy=False)
    right_c = extract_array(right_c, extract_numpy=True)

    lower_boundary = right_c.searchsorted(left_c, side="left")
    upper_boundary = right_c.searchsorted(left_c, side="right")
    keep_rows = lower_boundary < upper_boundary
    if not keep_rows.any():
        return None
    if not keep_rows.all():
        left_index = left_index[keep_rows]
        lower_boundary = lower_boundary[keep_rows]
        upper_boundary = upper_boundary[keep_rows]
    return left_index, right_index, lower_boundary, upper_boundary


def _generic_func_cond_join(
    left_c: pd.Series,
    right_c: pd.Series,
    op: str,
    multiple_conditions: bool,
) -> tuple:
    """
    Generic function to call any of the individual functions
    (_less_than_indices, _greater_than_indices,
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
        return _less_than_indices(left_c, right_c, strict)
    elif op in greater_than_join_types:
        return _greater_than_indices(
            left_c, right_c, strict, multiple_conditions
        )
    elif op == _JoinOperator.NOT_EQUAL.value:
        return _not_equal_indices(left_c, right_c)


def _generate_indices(
    left_index: np.ndarray, right_index: np.ndarray, conditions: list
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
        left_c, right_c, op = condition
        left_c = extract_array(left_c, extract_numpy=True)[left_index]
        right_c = extract_array(right_c, extract_numpy=True)[right_index]
        op = operator_map[op]
        mask = op(left_c, right_c)
        if not mask.any():
            return None
        if is_extension_array_dtype(mask):
            mask = mask.to_numpy(dtype=bool, na_value=False)
        if not mask.all():
            left_index = left_index[mask]
            right_index = right_index[mask]

    return left_index, right_index


def _multiple_conditional_join_ne(
    df: pd.DataFrame, right: pd.DataFrame, conditions: list
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

    # here we get indices for the first condition in conditions
    # then use those indices to get the final indices,
    # using _generate_indices
    first, *rest = conditions
    left_on, right_on, op = first

    # get indices from the first condition
    result = _generic_func_cond_join(
        df[left_on], right[right_on], op, multiple_conditions=False
    )

    if result is None:
        return None

    rest = (
        (df[left_on], right[right_on], op) for left_on, right_on, op in rest
    )

    return _generate_indices(*result, rest)


def _multiple_conditional_join_eq(
    df: pd.DataFrame, right: pd.DataFrame, conditions: list
) -> tuple:
    """
    Get indices for multiple conditions,
    if any of the conditions has an `==` operator.

    Returns a tuple of (df_index, right_index)
    """
    # TODO
    # this uses the idea in the `_range_indices` function
    # for less than and greater than;
    # I'd like to believe there is a smarter/more efficient way of doing this
    # where the filter occurs within the join, and avoids a blow-up
    # the current implementation uses
    # a list comprehension to find first matches
    # in a bid to reduce the blow up size ...
    # this applies only to integers/dates
    # and only offers advantages in scenarios
    # where the right is duplicated
    # for one to many joins,
    # or one to one or strings/category, use merge
    # as it is significantly faster than a binary search

    eqs = [
        (left_on, right_on)
        for left_on, right_on, op in conditions
        if op == _JoinOperator.STRICTLY_EQUAL.value
    ]

    left_on, right_on = zip(*eqs)
    left_on = [*left_on]
    right_on = [*right_on]

    strings_or_category = any(
        col
        for col in left_on
        if (is_string_dtype(df[col]) | is_categorical_dtype(df[col]))
    )

    if (
        strings_or_category
        | (not right.duplicated(subset=right_on).any(axis=None))
        | (not df.duplicated(subset=left_on).any(axis=None))
    ):
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
            copy=False,
        )._get_join_indexers()

        if not left_index.size:
            return None

        return _generate_indices(left_index, right_index, rest)

    left_on, right_on = eqs[0]
    outcome = _eq_indices(df[left_on], right[right_on])
    if not outcome:
        return None
    left_index, right_index, lower_boundary, upper_boundary = outcome
    eq_check = [condition for condition in conditions if condition != eqs[0]]

    rest = [
        (df.loc[left_index, left_on], right.loc[right_index, right_on], op)
        for left_on, right_on, op in eq_check
    ]
    rest = [
        (
            extract_array(left_c, extract_numpy=True),
            extract_array(right_c, extract_numpy=True),
            operator_map[op],
        )
        for left_c, right_c, op in rest
    ]

    def _extension_array_check(arr):
        """
        Convert boolean array to numpy array
        if it is an extension array.
        """
        if is_extension_array_dtype(arr):
            return arr.to_numpy(dtype=bool, na_value=False, copy=False)
        return arr

    pos = np.copy(upper_boundary)
    upper = np.copy(upper_boundary)
    counter = np.arange(left_index.size)

    # faster within C/Rust? better implemented within Pandas itself?
    # the idea here is that lower_boundary moves up by 1
    # till it gets to upper_boundary;
    # if we get all our matches before the end of the iteration, even better
    for _ in range((upper_boundary - lower_boundary).max()):
        if not counter.size:
            break
        if (lower_boundary == upper).any():
            keep_rows = lower_boundary < upper
            rest = [
                (left_c[keep_rows], right_c, op)
                for left_c, right_c, op in rest
            ]
            lower_boundary = lower_boundary[keep_rows]
            upper = upper[keep_rows]
            counter = counter[keep_rows]
        keep_rows = [
            op(left_c, right_c[lower_boundary]) for left_c, right_c, op in rest
        ]
        keep_rows = [_extension_array_check(arr) for arr in keep_rows]
        keep_rows = np.logical_and.reduce(keep_rows)
        if not keep_rows.any():
            lower_boundary += 1
            continue
        pos[counter[keep_rows]] = lower_boundary[keep_rows]
        counter = counter[~keep_rows]
        rest = [
            (left_c[~keep_rows], right_c, op) for left_c, right_c, op in rest
        ]
        upper = upper[~keep_rows]
        lower_boundary = lower_boundary[~keep_rows]
        lower_boundary += 1

    keep_rows = pos < upper_boundary

    if not keep_rows.any():
        return None

    if not keep_rows.all():
        left_index = left_index[keep_rows]
        pos = pos[keep_rows]
        upper_boundary = upper_boundary[keep_rows]

    repeater = upper_boundary - pos
    right_index = [
        right_index[start:end] for start, end in zip(pos, upper_boundary)
    ]

    right_index = np.concatenate(right_index)
    left_index = np.repeat(left_index, repeater)

    eq_check = [
        (df[left_on], right[right_on], op)
        for left_on, right_on, op in eq_check
    ]

    return _generate_indices(left_index, right_index, eq_check)


def _multiple_conditional_join_le_lt(
    df: pd.DataFrame, right: pd.DataFrame, conditions: list
) -> tuple:
    """
    Get indices for multiple conditions,
    where `>/>=` or `</<=` is present,
    and there is no `==` operator.

    Returns a tuple of (df_index, right_index)
    """

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
    # as far as I know, so a blowup of all the rows
    # is unavoidable.
    # future PR could use numba to improve performance, although it
    # still doesn't help that an optimisation path is not available
    # that I am aware of

    # first step is to get two conditions, if possible
    # where one has a less than operator
    # and the other has a greater than operator
    # get the indices from that
    # and then build the remaining indices,
    # using _generate_indices function
    # the aim of this for loop is to see if there is
    # the possiblity of a range join, and if there is
    # use the optimised path
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
        rest = [
            condition
            for condition in conditions
            if condition not in (ge_gt, le_lt)
        ]

        if rest:
            rest = (
                (df[left_on], right[right_on], op)
                for left_on, right_on, op in rest
            )
        else:
            rest = None

        return _range_indices(df, right, ge_gt, le_lt, rest)

    # no optimised path
    # blow up the rows and prune
    if le_lt:
        conditions = (
            condition for condition in conditions if condition != le_lt
        )

        conditions = (
            (df[left_on], right[right_on], op)
            for left_on, right_on, op in conditions
        )

        left_on, right_on, op = le_lt
        outcome = _generic_func_cond_join(
            df[left_on],
            right[right_on],
            op,
            multiple_conditions=False,
        )
        if outcome is None:
            return None

        return _generate_indices(*outcome, conditions)

    # no optimised path
    # blow up the rows and prune
    if ge_gt:
        conditions = (
            condition for condition in conditions if condition != ge_gt
        )

        conditions = (
            (df[left_on], right[right_on], op)
            for left_on, right_on, op in conditions
        )

        left_on, right_on, op = ge_gt
        outcome = _generic_func_cond_join(
            df[left_on],
            right[right_on],
            op,
            multiple_conditions=False,
        )
        if outcome is None:
            return None

        return _generate_indices(*outcome, conditions)


def _range_indices(
    df: pd.DataFrame,
    right: pd.DataFrame,
    first: tuple,
    second: tuple,
    rest: tuple = None,
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

    strict = False
    if op == _JoinOperator.GREATER_THAN.value:
        strict = True

    outcome = _greater_than_indices(
        df[left_on],
        right[right_on],
        strict,
        multiple_conditions=True,
    )

    if outcome is None:
        return None

    left_index, right_index, search_indices = outcome
    left_on, right_on, op = second
    right_c = right.loc[right_index, right_on]
    left_c = df.loc[left_index, left_on]
    left_c = extract_array(left_c, extract_numpy=True)
    op = operator_map[op]
    pos = np.copy(search_indices)
    counter = np.arange(left_index.size)
    ext_arr = is_extension_array_dtype(left_c)

    dupes = right_c.duplicated(keep="first")
    right_c = extract_array(right_c, extract_numpy=True)
    # use position, not label
    uniqs_index = np.arange(right_c.size)
    if dupes.any():
        uniqs_index = uniqs_index[~dupes]
        right_c = right_c[~dupes]

    for ind in range(uniqs_index.size):
        if not counter.size:
            break
        keep_rows = op(left_c, right_c[ind])
        if ext_arr:
            keep_rows = keep_rows.to_numpy(
                dtype=bool, na_value=False, copy=False
            )
        if not keep_rows.any():
            continue
        # get the index positions where left_c is </<= right_c
        # that minimum position combined with the equivalent position
        # from search_indices becomes our search space
        # for the equivalent left_c index
        pos[counter[keep_rows]] = uniqs_index[ind]
        counter = counter[~keep_rows]
        left_c = left_c[~keep_rows]

    dupes = None
    uniqs_index = None
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
    right_index = [
        right_index[start:end] for start, end in zip(pos, search_indices)
    ]

    # get indices and filter to get exact indices
    # that meet the condition
    right_index = np.concatenate(right_index)
    left_index = np.repeat(left_index, repeater)

    # here we search for actual positions
    # where left_c is </<= right_c
    # safe to index the arrays, since we are picking the positions
    # which are all in the original `df` and `right`
    # doing this allows some speed gains
    # while still ensuring correctness
    left_c = extract_array(df[left_on], extract_numpy=True)[left_index]
    right_c = extract_array(right[right_on], extract_numpy=True)[right_index]

    mask = op(left_c, right_c)

    if ext_arr:
        mask = mask.to_numpy(dtype=bool, na_value=False)

    if not mask.all():
        left_index = left_index[mask]
        right_index = right_index[mask]
    if not rest:
        return left_index, right_index

    return _generate_indices(left_index, right_index, rest)


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
        return pd.DataFrame(df, copy=False)

    if how == _JoinTypes.LEFT.value:
        right = right.dtypes.to_dict()
        right = {
            key: float if dtype.kind == "i" else dtype
            for key, dtype in right.items()
        }
        right = {
            key: pd.Series([], dtype=value) for key, value in right.items()
        }
        right = pd.DataFrame(right, copy=False)

    else:  # how == 'right'
        df = df.dtypes.to_dict()
        df = {
            key: float if dtype.kind == "i" else dtype
            for key, dtype in df.items()
        }
        df = {key: pd.Series([], dtype=value) for key, value in df.items()}
        df = pd.DataFrame(df, copy=False)
    df = pd.merge(
        df,
        right,
        left_index=True,
        right_index=True,
        how=how,
        copy=False,
        sort=False,
    )
    df.index = range(len(df))
    return df


def _create_conditional_join_frame(
    df: pd.DataFrame,
    right: pd.DataFrame,
    left_index: np.ndarray,
    right_index: np.ndarray,
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
        return pd.DataFrame({**df, **right}, copy=False)

    # dirty tests show slight speed gain when copy=False
    # which is achievable only within pd.merge
    if how == _JoinTypes.LEFT.value:
        right = right.loc[right_index]
        right.index = left_index
    else:
        df = df.loc[left_index]
        df.index = right_index
    df = pd.merge(
        df,
        right,
        left_index=True,
        right_index=True,
        how=how,
        copy=False,
        sort=False,
    )
    df.index = range(len(df))
    return df
