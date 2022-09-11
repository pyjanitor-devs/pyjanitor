from itertools import count
from pandas.core.common import apply_if_callable
from typing import Any, Optional
import pandas_flavor as pf
import pandas as pd
from pandas.core.dtypes.inference import is_array_like

from janitor.utils import check


@pf.register_dataframe_method
def case_when(
    df: pd.DataFrame, *args, *, default: Optional[Any] = 0, column_name: str
) -> pd.DataFrame:
    """
    Create a column based on a condition or multiple conditions.

    Example usage:

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame(
        ...     {
        ...         "a": [0, 0, 1, 2, "hi"],
        ...         "b": [0, 3, 4, 5, "bye"],
        ...         "c": [6, 7, 8, 9, "wait"],
        ...     }
        ... )
        >>> df
            a    b     c
        0   0    0     6
        1   0    3     7
        2   1    4     8
        3   2    5     9
        4  hi  bye  wait
        >>> df.case_when(
        ...     ((df.a == 0) & (df.b != 0)) | (df.c == "wait"), df.a,
        ...     (df.b == 0) & (df.a == 0), "x",
        ...     default = df.c,
        ...     column_name = "value",
        ... )
            a    b     c value
        0   0    0     6     x
        1   0    3     7     0
        2   1    4     8     8
        3   2    5     9     9
        4  hi  bye  wait    hi

    Similar to SQL and dplyr's case_when
    with inspiration from `pydatatable` if_else function.

    If your scenario requires direct replacement of values,
    pandas' `replace` method or `map` method should be better
    suited and more efficient; if the conditions check
    if a value is within a range of values, pandas' `cut` or `qcut`
    should be more efficient; `np.where/np.select` are also
    performant options.

    This function relies on `pd.Series.mask` method.

    When multiple conditions are satisfied, the first one is used.

    The variable `*args` parameters takes arguments of the form :
    `condition0`, `value0`, `condition1`, `value1`, ..., `default`.
    If `condition0` evaluates to `True`, then assign `value0` to
    `column_name`, if `condition1` evaluates to `True`, then
    assign `value1` to `column_name`, and so on. If none of the
    conditions evaluate to `True`, assign `default` to
    `column_name`.

    This function can be likened to SQL's `case_when`:

    ```sql
    CASE WHEN condition0 THEN value0
        WHEN condition1 THEN value1
        --- more conditions
        ELSE default
        END AS column_name
    ```

    compared to python's `if-elif-else`:

    ```python
    if condition0:
        value0
    elif condition1:
        value1
    # more elifs
    else:
        default
    ```

    :param df: A pandas DataFrame.
    :param args: Variable argument of conditions and expected values.
        Takes the form
        `condition0`, `value0`, `condition1`, `value1`, ... .
        `condition` can be a 1-D boolean array, a callable, or a string.
        If `condition` is a callable, it should evaluate
        to a 1-D boolean array. The array should have the same length
        as the DataFrame. If it is a string, it is computed on the dataframe,
        via `df.eval`, and should return a 1-D boolean array.
        `result` can be a scalar, a 1-D array, or a callable.
        If `result` is a callable, it should evaluate to a 1-D array.
        For a 1-D array, it should have the same length as the DataFrame.
    :param default: scalar, 1-D array or callable.
        If callable, it should evaluate to a 1-D array.
        The 1-D array should be the same length as the DataFrame.
        The element inserted in the output when all conditions
        evaluate to False. Default is 0.
    :param column_name: Name of column to assign results to. A new column
        is created, if it does not already exist in the DataFrame.
    :returns: A pandas DataFrame.
    """
    check("column_name", column_name, [str])
    default = if_else(df, args, default)
    return df.assign(**{column_name: default})


def if_else(df: pd.DataFrame, args, default: Optional[Any]) -> pd.Series:
    """
    Evaluates conditions against values;
    useful as a standalone for use within Pandas' assign.

    :param df: A pandas DataFrame.
    :param args: Variable argument of conditions and expected values.
        Takes the form
        `condition0`, `value0`, `condition1`, `value1`, ... .
        `condition` can be a 1-D boolean array, a callable, or a string.
        If `condition` is a callable, it should evaluate
        to a 1-D boolean array. The array should have the same length
        as the DataFrame. If it is a string, it is computed on the dataframe,
        via `df.eval`, and should return a 1-D boolean array.
        `result` can be a scalar, a 1-D array, or a callable.
        If `result` is a callable, it should evaluate to a 1-D array.
        For a 1-D array, it should have the same length as the DataFrame.
    :param default: scalar, 1-D array or callable.
        If callable, it should evaluate to a 1-D array.
        The 1-D array should be the same length as the DataFrame.
        The element inserted in the output when all conditions
        evaluate to False. Default is 0.
    :raises ValueError: If the condition fails to evaluate.
    :returns: A pandas Series.
    """

    booleans, replacements, default = _case_when_checks(df, args, default)
    # ensures value assignment is on a first come basis
    booleans = booleans[::-1]
    replacements = replacements[::-1]
    for condition, value, index in zip(booleans, replacements, count()):
        try:
            default = default.mask(condition, value)
        # error `feedoff` idea from SO
        # https://stackoverflow.com/a/46091127/7175713
        except Exception as e:
            raise ValueError(
                f"condition{index} and value{index} failed to evaluate. "
                f"Original error message: {e}"
            ) from e
    return default


def _case_when_checks(
    df: pd.DataFrame, args, default
) -> tuple[list, list, pd.Series]:
    """
    Preliminary checks on the case_when function.
    """
    if len(args) < 2:
        raise ValueError(
            "At least two arguments are required for the `args` parameter"
        )

    booleans = []
    replacements = []
    for index, value in enumerate(args):
        if index % 2:
            replacements.append(value)
        else:
            booleans.append(value)
    if len(booleans) != len(replacements):
        raise ValueError(
            "The number of conditions and values do not match. "
            f"There are {len(booleans)} conditions and {len(replacements)} "
            "values."
        )

    booleans = [
        apply_if_callable(condition, df)
        if callable(condition)
        else df.eval(condition)
        if isinstance(condition, str)
        else condition
        for condition in booleans
    ]

    replacements = [
        apply_if_callable(replacement, df)
        if callable(replacement)
        else replacement
        for replacement in replacements
    ]

    if callable(default):
        default = apply_if_callable(default, df)
    if pd.api.types.is_scalar(default):
        default = pd.Series([default]).repeat(len(df))
    if not is_array_like(default):
        raise TypeError(
            "The argument for the `default` parameter "
            "should evaluate to an array-like object, "
            f"instead got {type(default)!r}"
        )
    if isinstance(default, pd.Index):
        arr_ndim = default.nlevels
    else:
        arr_ndim = default.ndim
    if arr_ndim != 1:
        raise ValueError(
            f"The argument for the `default` parameter "
            "should evaluate to a 1-D array, "
            f"instead got dimension of length {arr_ndim}"
        )
    if len(default) != len(df):
        raise ValueError(
            f"The length of the argument for the `default` parameter "
            "is {len(default)}, "
            "which is different from the length of the dataframe, "
            f"{len(df)}"
        )
    if not isinstance(default, pd.Series):
        default = pd.Series(default)
    default.index = df.index

    return booleans, replacements, default
