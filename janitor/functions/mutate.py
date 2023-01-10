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
    """
    Transform columns or create new columns via a dictionary or a tuple.

    Similar to `pd.DataFrame.assign`,
    with added flexibility for multiple columns.

    The computation should return a 1-D array like object
    that is the same length as `df` or a scalar
    that can be broadcasted to the same length as `df`.

    Column transformation is also possible in the presence of `by`.

    A nested dictionary can be provided, in the presence of `by`,
    for passing new column names - it is not required,
    and not supported in the absence of `by`.
    Have a look at the examples below for usage
    in the presence and absence of `by`.

    If the variable argument is a tuple,
    it has to be of the form `(column_names, func, names_glue)`;
    the `names_glue` argument is optional.
    `column_names` can be selected with the
    [`select_columns`][janitor.functions.select.select_columns]
    syntax for flexibility.
    The function `func` should be a string
    (which is dispatched to `pd.transform`),
    or a callable, or a list/tuple of strings/callables.

    The `names_glue` argument allows for renaming, especially for
    multiple columns or multiple functions.
    The placeholders for `names_glue` are `_col`, which represents
    the column name, and `_fn` which represents the function name.
    Under the hood, it uses python's `str.format` method.

    `janitor.SD` offers a more explicit form
    of passing tuples to the `mutate` function.

    Example: Transformation with a dictionary:

        >>> import pandas as pd
        >>> import numpy as np
        >>> import janitor
        >>> pd.set_option("display.max_columns", None)
        >>> pd.set_option("display.expand_frame_repr", False)
        >>> pd.set_option("max_colwidth", None)
        >>> df = pd.DataFrame({
        ...     "col1": [5, 10, 15],
        ...     "col2": [3, 6, 9],
        ...     "col3": [10, 100, 1_000],
        ... })
        >>> df.mutate({"col4": df.col1.transform(np.log10)})
           col1  col2  col3      col4
        0     5     3    10  0.698970
        1    10     6   100  1.000000
        2    15     9  1000  1.176091

        >>> df.mutate(
        ...     {"col4": df.col1.transform(np.log10),
        ...      "col1": df.col1.transform(np.log10)}
        ...     )
               col1  col2  col3      col4
        0  0.698970     3    10  0.698970
        1  1.000000     6   100  1.000000
        2  1.176091     9  1000  1.176091

    Example: Transformation with a tuple:

        >>> df.mutate(("col1", np.log10))
               col1  col2  col3
        0  0.698970     3    10
        1  1.000000     6   100
        2  1.176091     9  1000

        >>> df.mutate(("col*", np.log10))
               col1      col2  col3
        0  0.698970  0.477121   1.0
        1  1.000000  0.778151   2.0
        2  1.176091  0.954243   3.0

    Example: Transform with a tuple and create new columns, using `names_glue`:

        >>> cols = SD(columns="col*", func=np.log10, names_glue="{_col}_log")
        >>> df.mutate(cols)
           col1  col2  col3  col1_log  col2_log  col3_log
        0     5     3    10  0.698970  0.477121       1.0
        1    10     6   100  1.000000  0.778151       2.0
        2    15     9  1000  1.176091  0.954243       3.0

        >>> df.mutate(("col*", np.log10, "{_col}_{_fn}"))
           col1  col2  col3  col1_log10  col2_log10  col3_log10
        0     5     3    10    0.698970    0.477121         1.0
        1    10     6   100    1.000000    0.778151         2.0
        2    15     9  1000    1.176091    0.954243         3.0

    Example: Transformation in the presence of a groupby:

        >>> data = {'avg_jump': [3, 4, 1, 2, 3, 4],
        ...         'avg_run': [3, 4, 1, 3, 2, 4],
        ...         'avg_swim': [2, 1, 2, 2, 3, 4],
        ...         'combine_id': [100200, 100200, 101200, 101200, 102201, 103202],
        ...         'category': ['heats', 'heats', 'finals', 'finals', 'heats', 'finals']}
        >>> df = pd.DataFrame(data)
        >>> df.mutate({"avg_run":"mean"}, by=['combine_id', 'category'])
           avg_jump  avg_run  avg_swim  combine_id category
        0         3      3.5         2      100200    heats
        1         4      3.5         1      100200    heats
        2         1      2.0         2      101200   finals
        3         2      2.0         2      101200   finals
        4         3      2.0         3      102201    heats
        5         4      4.0         4      103202   finals

        >>> df.mutate({"avg_run":{"avg_run_2":"mean"}}, by=['combine_id', 'category'])
           avg_jump  avg_run  avg_swim  combine_id category  avg_run_2
        0         3        3         2      100200    heats        3.5
        1         4        4         1      100200    heats        3.5
        2         1        1         2      101200   finals        2.0
        3         2        3         2      101200   finals        2.0
        4         3        2         3      102201    heats        2.0
        5         4        4         4      103202   finals        4.0

        >>> df.mutate(("avg*", "mean", "{_col}_2"), by=['combine_id', 'category'])
           avg_jump  avg_run  avg_swim  combine_id category  avg_jump_2  avg_run_2  avg_swim_2
        0         3        3         2      100200    heats         3.5        3.5         1.5
        1         4        4         1      100200    heats         3.5        3.5         1.5
        2         1        1         2      101200   finals         1.5        2.0         2.0
        3         2        3         2      101200   finals         1.5        2.0         2.0
        4         3        2         3      102201    heats         3.0        2.0         3.0
        5         4        4         4      103202   finals         4.0        4.0         4.0

    :param df: A pandas DataFrame.
    :param args: Either a dictionary or a tuple.
    :param by: Column(s) to group by.
    :raises ValueError: If a tuple is passed and the length is not 3.
    :returns: A pandas DataFrame with mutated columns.
    """  # noqa: E501

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
            counts = None
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
