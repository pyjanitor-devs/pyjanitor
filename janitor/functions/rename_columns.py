from typing import Callable, Dict, Union

import pandas as pd
import pandas_flavor as pf

from janitor.utils import check_column, deprecated_alias, refactored_function


@pf.register_dataframe_method
@refactored_function(
    message=(
        "This function will be deprecated in a 1.x release. "
        "Please use `pd.DataFrame.rename` instead."
    )
)
@deprecated_alias(old="old_column_name", new="new_column_name")
def rename_column(
    df: pd.DataFrame,
    old_column_name: str,
    new_column_name: str,
) -> pd.DataFrame:
    """Rename a column in place.

    This method does not mutate the original DataFrame.

    !!!note

        This function will be deprecated in a 1.x release.
        Please use `pd.DataFrame.rename` instead.

    This is just syntactic sugar/a convenience function for renaming one column at a time.
    If you are convinced that there are multiple columns in need of changing,
    then use the `pandas.DataFrame.rename` method.

    Examples:
        Change the name of column 'a' to 'a_new'.

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({"a": list(range(3)), "b": list("abc")})
        >>> df.rename_column(old_column_name='a', new_column_name='a_new')
           a_new  b
        0      0  a
        1      1  b
        2      2  c

    Args:
        df: The pandas DataFrame object.
        old_column_name: The old column name.
        new_column_name: The new column name.

    Returns:
        A pandas DataFrame with renamed columns.
    """  # noqa: E501

    check_column(df, [old_column_name])

    return df.rename(columns={old_column_name: new_column_name})


@pf.register_dataframe_method
@refactored_function(
    message=(
        "This function will be deprecated in a 1.x release. "
        "Please use `pd.DataFrame.rename` instead."
    )
)
def rename_columns(
    df: pd.DataFrame,
    new_column_names: Union[Dict, None] = None,
    function: Callable = None,
) -> pd.DataFrame:
    """Rename columns.

    This method does not mutate the original DataFrame.

    !!!note

        This function will be deprecated in a 1.x release.
        Please use `pd.DataFrame.rename` instead.

    One of the `new_column_names` or `function` are a required parameter.
    If both are provided, then `new_column_names` takes priority and `function`
    is never executed.

    Examples:
        Rename columns using a dictionary which maps old names to new names.

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({"a": list(range(3)), "b": list("xyz")})
        >>> df
           a  b
        0  0  x
        1  1  y
        2  2  z
        >>> df.rename_columns(new_column_names={"a": "a_new", "b": "b_new"})
           a_new b_new
        0      0     x
        1      1     y
        2      2     z

        Rename columns using a generic callable.

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({"a": list(range(3)), "b": list("xyz")})
        >>> df.rename_columns(function=str.upper)
           A  B
        0  0  x
        1  1  y
        2  2  z

    Args:
        df: The pandas DataFrame object.
        new_column_names: A dictionary of old and new column names.
        function: A function which should be applied to all the columns.

    Raises:
        ValueError: If both `new_column_names` and `function` are None.

    Returns:
        A pandas DataFrame with renamed columns.
    """  # noqa: E501

    if new_column_names is None and function is None:
        raise ValueError(
            "One of new_column_names or function must be provided"
        )

    if new_column_names is not None:
        check_column(df, new_column_names)
        return df.rename(columns=new_column_names)

    return df.rename(mapper=function, axis="columns")
