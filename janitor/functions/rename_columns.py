from typing import Callable, Dict, Union
import pandas_flavor as pf
import pandas as pd

from janitor.utils import check_column, deprecated_alias


@pf.register_dataframe_method
@deprecated_alias(old="old_column_name", new="new_column_name")
def rename_column(
    df: pd.DataFrame,
    old_column_name: str,
    new_column_name: str,
) -> pd.DataFrame:
    """Rename a column in place.

    This method does not mutate the original DataFrame.

    Example: Change the name of column 'a' to 'a_new'.

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({"a": list(range(3)), "b": list("abc")})
        >>> df.rename_column(old_column_name='a', new_column_name='a_new')
           a_new  b
        0      0  a
        1      1  b
        2      2  c

    This is just syntactic sugar/a convenience function for renaming one column at a time.
    If you are convinced that there are multiple columns in need of changing,
    then use the `pandas.DataFrame.rename` method.

    :param df: The pandas DataFrame object.
    :param old_column_name: The old column name.
    :param new_column_name: The new column name.
    :returns: A pandas DataFrame with renamed columns.
    """  # noqa: E501
    check_column(df, [old_column_name])

    return df.rename(columns={old_column_name: new_column_name})


@pf.register_dataframe_method
def rename_columns(
    df: pd.DataFrame,
    new_column_names: Union[Dict, None] = None,
    function: Callable = None,
) -> pd.DataFrame:
    """Rename columns.

    This method does not mutate the original DataFrame.

    Example: Rename columns using a dictionary which maps old names to new names.

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

    Example: Rename columns using a generic callable.

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({"a": list(range(3)), "b": list("xyz")})
        >>> df.rename_columns(function=str.upper)
           A  B
        0  0  x
        1  1  y
        2  2  z

    One of the `new_column_names` or `function` are a required parameter.
    If both are provided, then `new_column_names` takes priority and `function`
    is never executed.

    :param df: The pandas DataFrame object.
    :param new_column_names: A dictionary of old and new column names.
    :param function: A function which should be applied to all the columns.
    :returns: A pandas DataFrame with renamed columns.
    :raises ValueError: if both `new_column_names` and `function` are None.
    """  # noqa: E501

    if new_column_names is None and function is None:
        raise ValueError(
            "One of new_column_names or function must be provided"
        )

    if new_column_names is not None:
        check_column(df, new_column_names)
        return df.rename(columns=new_column_names)

    return df.rename(mapper=function, axis="columns")
