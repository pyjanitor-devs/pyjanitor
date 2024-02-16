from __future__ import annotations

from typing import Any, Hashable

import pandas as pd
import pandas_flavor as pf

from janitor.utils import deprecated_alias, refactored_function


@pf.register_dataframe_method
@refactored_function(
    message=(
        "This function will be deprecated in a 1.x release. "
        "Please use `pd.DataFrame.astype` instead."
    )
)
@deprecated_alias(column="column_name")
def change_type(
    df: pd.DataFrame,
    column_name: Hashable | list[Hashable] | pd.Index,
    dtype: type,
    ignore_exception: bool = False,
) -> pd.DataFrame:
    """Change the type of a column.

    This method does not mutate the original DataFrame.

    Exceptions that are raised can be ignored. For example, if one has a mixed
    dtype column that has non-integer strings and integers, and you want to
    coerce everything to integers, you can optionally ignore the non-integer
    strings and replace them with `NaN` or keep the original value.

    Intended to be the method-chaining alternative to:

    ```python
    df[col] = df[col].astype(dtype)
    ```

    !!!note

        This function will be deprecated in a 1.x release.
        Please use `pd.DataFrame.astype` instead.

    Examples:
        Change the type of a column.

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({"col1": range(3), "col2": ["m", 5, True]})
        >>> df
           col1  col2
        0     0     m
        1     1     5
        2     2  True
        >>> df.change_type(
        ...     "col1", dtype=str,
        ... ).change_type(
        ...     "col2", dtype=float, ignore_exception="fillna",
        ... )
          col1  col2
        0    0   NaN
        1    1   5.0
        2    2   1.0

        Change the type of multiple columns. To change the type of all columns,
        please use `DataFrame.astype` instead.

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({"col1": range(3), "col2": ["m", 5, True]})
        >>> df.change_type(['col1', 'col2'], str)
          col1  col2
        0    0     m
        1    1     5
        2    2  True

    Args:
        df: A pandas DataFrame.
        column_name: The column(s) in the dataframe.
        dtype: The datatype to convert to. Should be one of the standard
            Python types, or a numpy datatype.
        ignore_exception: One of `{False, "fillna", "keep_values"}`.

    Raises:
        ValueError: If unknown option provided for `ignore_exception`.

    Returns:
        A pandas DataFrame with changed column types.
    """  # noqa: E501

    df = df.copy()  # avoid mutating the original DataFrame
    if not ignore_exception:
        df[column_name] = df[column_name].astype(dtype)
    elif ignore_exception == "keep_values":
        df[column_name] = df[column_name].astype(dtype, errors="ignore")
    elif ignore_exception == "fillna":
        if isinstance(column_name, Hashable):
            column_name = [column_name]
        df[column_name] = df[column_name].map(_convert, dtype=dtype)
    else:
        raise ValueError("Unknown option for ignore_exception")

    return df


def _convert(x: Any, dtype: type) -> Any:
    """Casts item `x` to `dtype` or None if not possible."""

    try:
        return dtype(x)
    except ValueError:
        return None
