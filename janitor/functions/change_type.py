from typing import Hashable
import pandas as pd
import pandas_flavor as pf

from janitor.utils import deprecated_alias


@pf.register_dataframe_method
@deprecated_alias(column="column_name")
def change_type(
    df: pd.DataFrame,
    column_name: Hashable,
    dtype: type,
    ignore_exception: bool = False,
) -> pd.DataFrame:
    """Change the type of a column.

    This method mutates the original DataFrame.

    Exceptions that are raised can be ignored. For example, if one has a mixed
    dtype column that has non-integer strings and integers, and you want to
    coerce everything to integers, you can optionally ignore the non-integer
    strings and replace them with `NaN` or keep the original value.

    Intended to be the method-chaining alternative to:

    ```python
    df[col] = df[col].astype(dtype)
    ```

    Example:

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

    :param df: A pandas DataFrame.
    :param column_name: A column in the dataframe.
    :param dtype: The datatype to convert to. Should be one of the standard
        Python types, or a numpy datatype.
    :param ignore_exception: one of `{False, "fillna", "keep_values"}`.
    :returns: A pandas DataFrame with changed column types.
    :raises ValueError: If unknown option provided for
        `ignore_exception`.
    """
    if not ignore_exception:
        df[column_name] = df[column_name].astype(dtype)
    elif ignore_exception == "keep_values":
        df[column_name] = df[column_name].astype(dtype, errors="ignore")
    elif ignore_exception == "fillna":

        def convert(x, dtype):
            """Casts item `x` to `dtype` or None if not possible."""
            try:
                return dtype(x)
            except ValueError:
                return None

        df[column_name] = df[column_name].apply(lambda x: convert(x, dtype))
    else:
        raise ValueError("Unknown option for ignore_exception")
    return df
