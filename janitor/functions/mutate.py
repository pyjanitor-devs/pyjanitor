"""Function for mutation of a column or columns."""
from typing import Optional, Union
import pandas as pd
import pandas_flavor as pf

from janitor.utils import check, check_column, deprecated_alias
from janitor.functions.utils import _select_index
from typing import Any, Union


@pf.register_dataframe_method
def mutate(
    df: pd.DataFrame,
    *args,
    by:Any=None,
    axis:Union[int, str]=0
) -> pd.DataFrame:
    """Coalesce two or more columns of data in order of column names provided.

    Given the variable arguments of column names,
    `coalesce` finds and returns the first non-missing value
    from these columns, for every row in the input dataframe.
    If all the column values are null for a particular row,
    then the `default_value` will be filled in.

    If `target_column_name` is not provided,
    then the first column is coalesced.

    This method does not mutate the original DataFrame.

    Example: Use `coalesce` with 3 columns, "a", "b" and "c".

        >>> import pandas as pd
        >>> import numpy as np
        >>> import janitor
        >>> df = pd.DataFrame({
        ...     "a": [np.nan, 1, np.nan],
        ...     "b": [2, 3, np.nan],
        ...     "c": [4, np.nan, np.nan],
        ... })
        >>> df.coalesce("a", "b", "c")
             a    b    c
        0  2.0  2.0  4.0
        1  1.0  3.0  NaN
        2  NaN  NaN  NaN

    Example: Provide a target_column_name.

        >>> df.coalesce("a", "b", "c", target_column_name="new_col")
             a    b    c  new_col
        0  NaN  2.0  4.0      2.0
        1  1.0  3.0  NaN      1.0
        2  NaN  NaN  NaN      NaN

    Example: Provide a default value.

        >>> import pandas as pd
        >>> import numpy as np
        >>> import janitor
        >>> df = pd.DataFrame({
        ...     "a": [1, np.nan, np.nan],
        ...     "b": [2, 3, np.nan],
        ... })
        >>> df.coalesce(
        ...     "a", "b",
        ...     target_column_name="new_col",
        ...     default_value=-1,
        ... )
             a    b  new_col
        0  1.0  2.0      1.0
        1  NaN  3.0      3.0
        2  NaN  NaN     -1.0

    This is more syntactic diabetes! For R users, this should look familiar to
    `dplyr`'s `coalesce` function; for Python users, the interface
    should be more intuitive than the `pandas.Series.combine_first`
    method.

    :param df: A pandas DataFrame.
    :param args: Either a dictionary or a tuple.
    :param by: Column(s) to group by.
    :returns: A pandas DataFrame with mutated columns.
    """

    if not args:
        return df

    for num, arg in enumerate(args):
        check(f"Argument {num} in the mutate function", arg, [dict])
        if isinstance(arg, dict):
            for col, func in arg.items():
                check(f"func for {col} in argument {num}", func, [str, callable, dict])
                if isinstance(func, dict):
                    for _, value in func.items():
                        check(f"func in nested dictionary for {col} in argument {num}", value, [str, callable])


    df = df.copy()

    for arg in args:
        if isinstance(arg, dict):
            for col, func in arg.items():
                if isinstance(func, dict):
                    for key, val in func.items():
                        df[key] = df[col].apply(val)
                else:
                    df[col] = df[col].apply(func)

    return df