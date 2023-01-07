"""Function for mutation of a column or columns."""
from typing import Any
import pandas as pd
import pandas_flavor as pf

from janitor.utils import check

from janitor.functions.utils import _select_index, SD
from pandas.core.common import apply_if_callable


@pf.register_dataframe_method
def mutate(
    df: pd.DataFrame,
    *args,
    by: Any = None,
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
    :raises ValueError: If a tuple is passed and the length is not 3.
    :returns: A pandas DataFrame with mutated columns.
    """

    if not args:
        return df

    for num, arg in enumerate(args):
        check(f"Argument {num} in the mutate function", arg, [dict, tuple])
        if isinstance(arg, dict):
            for col, func in arg.items():
                check(
                    f"The function for column {col} in argument {num}",
                    func,
                    [str, callable, dict],
                )
                if isinstance(func, dict):
                    for _, funcn in func.items():
                        check(
                            f"The function in the nested dictionary for "
                            f"column {col} in argument {num}",
                            funcn,
                            [str, callable],
                        )
        else:
            if len(arg) < 2:
                raise ValueError(
                    f"Argument {num} should have a minimum length of 2, "
                    f"instead got {len(arg)}"
                )
            if len(arg) > 3:
                raise ValueError(
                    f"Argument {num} should have a maximum length of 3, "
                    f"instead got {len(arg)}"
                )
            _, func, names = SD(*arg)
            check(
                f"The function (position 1 in the tuple) for argument {num} ",
                func,
                [str, callable, list, tuple],
            )
            if isinstance(func, (list, tuple)):
                for number, funcn in enumerate(func):
                    check(
                        f"Entry {number} in the function sequence "
                        f"for argument {num}",
                        funcn,
                        [str, callable],
                    )

            if names is not None:
                check(
                    f"The names (position 2 in the tuple) for argument {num} ",
                    names,
                    [str],
                )

    grp = None
    by_is_true = by is not None
    if by_is_true and isinstance(by, dict):
        grp = df.groupby(**by)
    elif by_is_true:
        grp = df.groupby(by)
    df = df.copy()

    for arg in args:
        if isinstance(arg, dict):
            for col, func in arg.items():
                if by_is_true:
                    val = grp[col]
                else:
                    val = df[col]
                if isinstance(func, str):
                    df[col] = val.transform(func)
                elif callable(func):
                    df[col] = apply_if_callable(func, val)
                elif isinstance(func, dict):
                    for key, funcn in func.items():
                        if isinstance(funcn, str):
                            df[key] = val.transform(funcn)
                        else:
                            df[key] = apply_if_callable(funcn, val)

        else:
            columns, func, names = SD(*arg)
            columns = _select_index(columns, df, axis="columns")
            columns = df.columns[columns]
            for col in columns:
                if by_is_true:
                    val = grp[col]
                else:
                    val = df[col]
                if isinstance(func, str):
                    if names is not None:
                        name = names.format(_col=col, _fn=func)
                    elif isinstance(func, str):
                        name = col
                    df[name] = val.transform(func)

    return df
