"""Function for mutation of a column or columns."""
from typing import Any
import pandas as pd
import pandas_flavor as pf

from janitor.utils import check
from pandas.core.common import apply_if_callable

from janitor.functions.utils import _select_index, SD
from collections import Counter


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

    by_is_true = by is not None

    for num, arg in enumerate(args):
        check(f"Argument {num} in the mutate function", arg, [dict, tuple])
        if isinstance(arg, dict):
            for col, func in arg.items():
                if isinstance(func, dict):
                    if not by_is_true:
                        raise ValueError(
                            "nested dictionary is supported only "
                            "if an argument is provided to the `by` parameter."
                        )
                    for _, funcn in func.items():
                        check(
                            f"The function in the nested dictionary for "
                            f"column {col} in argument {num}",
                            funcn,
                            [str, callable],
                        )
                elif by_is_true:
                    check(
                        f"The function for column {col} in argument {num}",
                        func,
                        [str, callable, dict],
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

            if names:
                check(
                    f"The names (position 2 in the tuple) for argument {num} ",
                    names,
                    [str],
                )

    grp = None
    if by_is_true and isinstance(by, dict):
        grp = df.groupby(**by)
    elif by_is_true:
        grp = df.groupby(by)
    df = df.copy()

    for arg in args:
        if isinstance(arg, dict):
            for col, func in arg.items():
                if not by_is_true:  # same as pd.DataFrame.assign
                    df[col] = apply_if_callable(func, df)
                elif isinstance(func, dict):
                    for key, funcn in func.items():
                        try:
                            df[key] = grp[col].transform(funcn)
                        except ValueError:
                            df[key] = funcn(grp[col])
                else:
                    try:
                        df[col] = grp[col].transform(func)
                    except ValueError:
                        df[col] = func(grp[col])

        else:
            columns, func, names = SD(*arg)
            columns = _select_index([columns], df, axis="columns")
            columns = df.columns[columns]
            if not isinstance(func, (list, tuple)):
                func = [func]
            func_names = [
                funcn.__name__ if callable(funcn) else funcn for funcn in func
            ]
            counts = None
            dupes = set()
            if len(func) > 1:
                counts = Counter(func_names)
                counts = {key: 0 for key, value in counts.items() if value > 1}
            # deal with duplicate function names
            if counts:
                func_list = []
                for funcn in func_names:
                    if funcn in counts:
                        if names:
                            name = f"{funcn}{counts[funcn]}"
                        else:
                            name = f"{counts[funcn]}"
                            dupes.add(name)
                        func_list.append(name)
                        counts[funcn] += 1
                    else:
                        func_list.append(funcn)
                func_names = func_list
            func_names = tuple(zip(func_names, func))
            for col in columns:
                for name, funcn in func_names:
                    if names:
                        name = names.format(_col=col, _fn=name)
                    elif name in dupes:
                        name = f"{col}{name}"
                    else:
                        name = col
                    if by_is_true:
                        try:
                            df[name] = grp[col].transform(funcn)
                        except ValueError:
                            df[name] = funcn(grp[col])
                    else:
                        try:
                            df[name] = df[col].transform(funcn)
                        except ValueError:
                            df[name] = funcn(df[col])

    return df
