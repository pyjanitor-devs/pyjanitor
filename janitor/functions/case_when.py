"""Implementation source for `case_when`."""

import warnings
from typing import Any

import pandas as pd
import pandas_flavor as pf
from pandas.api.types import is_scalar
from pandas.core.common import apply_if_callable

from janitor.utils import check, find_stack_level, refactored_function

warnings.simplefilter("always", DeprecationWarning)


@pf.register_dataframe_method
@refactored_function(
    message=(
        "This function will be deprecated in a 1.x release. "
        "Please use `pd.Series.case_when` instead."
    )
)
def case_when(
    df: pd.DataFrame, *args: Any, default: Any = None, column_name: str
) -> pd.DataFrame:
    """Create a column based on a condition or multiple conditions.

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

    Examples:
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

    !!! abstract "Version Changed"

        - 0.24.0
            - Added `default` parameter.

    Args:
        df: A pandas DataFrame.
        *args: Variable argument of conditions and expected values.
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
        default: This is the element inserted in the output
            when all conditions evaluate to False.
            Can be scalar, 1-D array or callable.
            If callable, it should evaluate to a 1-D array.
            The 1-D array should be the same length as the DataFrame.
        column_name: Name of column to assign results to. A new column
            is created if it does not already exist in the DataFrame.

    Raises:
        ValueError: If condition/value fails to evaluate.

    Returns:
        A pandas DataFrame.
    """  # noqa: E501
    # Preliminary checks on the case_when function.
    # The bare minimum checks are done; the remaining checks
    # are done within `pd.Series.mask`.
    check("column_name", column_name, [str])
    len_args = len(args)
    if len_args < 2:
        raise ValueError(
            "At least two arguments are required for the `args` parameter"
        )

    if len_args % 2:
        if default is None:
            warnings.warn(
                "The last argument in the variable arguments "
                "has been assigned as the default. "
                "Note however that this will be deprecated "
                "in a future release; use an even number "
                "of boolean conditions and values, "
                "and pass the default argument to the `default` "
                "parameter instead.",
                DeprecationWarning,
                stacklevel=find_stack_level(),
            )
            *args, default = args
        else:
            raise ValueError(
                "The number of conditions and values do not match. "
                f"There are {len_args - len_args//2} conditions "
                f"and {len_args//2} values."
            )

    booleans = []
    replacements = []

    for index, value in enumerate(args):
        if index % 2:
            if callable(value):
                value = apply_if_callable(value, df)
            replacements.append(value)
        else:
            if callable(value):
                value = apply_if_callable(value, df)
            elif isinstance(value, str):
                value = df.eval(value)
            booleans.append(value)

    if callable(default):
        default = apply_if_callable(default, df)
    if is_scalar(default):
        default = pd.Series([default]).repeat(len(df))
    if not hasattr(default, "shape"):
        default = pd.Series([*default])
    if isinstance(default, pd.Index):
        arr_ndim = default.nlevels
    else:
        arr_ndim = default.ndim
    if arr_ndim != 1:
        raise ValueError(
            "The argument for the `default` parameter "
            "should either be a 1-D array, a scalar, "
            "or a callable that can evaluate to a 1-D array."
        )
    if not isinstance(default, pd.Series):
        default = pd.Series(default)
    default.index = df.index
    # actual computation
    # ensures value assignment is on a first come basis
    booleans = booleans[::-1]
    replacements = replacements[::-1]
    for index, (condition, value) in enumerate(zip(booleans, replacements)):
        try:
            default = default.mask(condition, value)
        # error `feedoff` idea from SO
        # https://stackoverflow.com/a/46091127/7175713
        except Exception as error:
            raise ValueError(
                f"condition{index} and value{index} failed to evaluate. "
                f"Original error message: {error}"
            ) from error

    return df.assign(**{column_name: default})
