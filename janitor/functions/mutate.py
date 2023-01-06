"""Function for mutation of a column or columns."""
from typing import Any
import pandas as pd
import pandas_flavor as pf

from janitor.utils import check

from janitor.functions.utils import SD, _process_SD
from itertools import product


@pf.register_dataframe_method
def mutate(
    df: pd.DataFrame,
    *args,
    by: Any = None,
) -> pd.DataFrame:
    """

    !!! info "New in version 0.25.0"

    !!!note

        Before reaching for `mutate`, try `pd.DataFrame.assign`.

    Transform columns via a tuple.

    The computation should return a 1-D array like object
    that is the same length as `df` or a scalar
    that can be broadcasted to the same length as `df`.

    The argument should be of the form `(columns, func, names_glue)`;
    the `names_glue` argument is optional.
    `columns` can be selected with the
    [`select_columns`][janitor.functions.select.select_columns]
    syntax for flexibility.
    The function `func` should be a string
    (which is dispatched to `pd.Series.transform`),
    or a callable, or a list/tuple of strings/callables.
    The function is called on each column in `columns`.

    The `names_glue` argument allows for renaming, especially for
    multiple columns or multiple functions.
    The placeholders for `names_glue` are `_col`, which represents
    the column name, and `_fn` which represents the function name.
    Under the hood, it uses python's `str.format` method.

    `janitor.SD` offers a more explicit form
    of passing tuples to the `mutate` function.

    `by` accepts a label, labels, mapping or function.
    Arguments supported in `pd.DataFrame.groupby`
    can also be passed to `by` via a dictionary.

    Example:

        >>> import pandas as pd
        >>> import numpy as np
        >>> import janitor as jn
        >>> data = {'avg_jump': [3, 4, 1, 2, 3, 4],
        ...         'avg_run': [3, 4, 1, 3, 2, 4],
        ...         'combine_id': [100200, 100200,
        ...                        101200, 101200,
        ...                        102201, 103202]}
        >>> df = pd.DataFrame(data)
        >>> df.mutate(("avg_run","mean"), by='combine_id')
           avg_jump  avg_run  combine_id
        0         3      3.5      100200
        1         4      3.5      100200
        2         1      2.0      101200
        3         2      2.0      101200
        4         3      2.0      102201
        5         4      4.0      103202

        >>> df.mutate(("avg_run","mean", "avg_run_2"), by='combine_id')
           avg_jump  avg_run  combine_id  avg_run_2
        0         3        3      100200        3.5
        1         4        4      100200        3.5
        2         1        1      101200        2.0
        3         2        3      101200        2.0
        4         3        2      102201        2.0
        5         4        4      103202        4.0

        >>> cols = jn.SD(columns="avg*", func="mean", names_glue="{_col}_{_fn}")
        >>> df.mutate(cols, by='combine_id')
           avg_jump  avg_run  combine_id  avg_jump_mean  avg_run_mean
        0         3        3      100200            3.5           3.5
        1         4        4      100200            3.5           3.5
        2         1        1      101200            1.5           2.0
        3         2        3      101200            1.5           2.0
        4         3        2      102201            3.0           2.0
        5         4        4      103202            4.0           4.0

    :param df: A pandas DataFrame.
    :param args: Either a dictionary or a tuple.
    :param by: Column(s) to group by.
    :raises ValueError: If a tuple is passed and the length is not 3.
    :returns: A pandas DataFrame with mutated columns.
    """  # noqa: E501

    if not args:
        return df

    args_to_process = []
    for num, arg in enumerate(args):
        check(f"Argument {num} in the mutate function", arg, [dict, tuple])
        if isinstance(arg, dict):
            for col, func in arg.items():
                check(
                    f"Entry {number} in the function sequence "
                    f"for argument {num}",
                    funcn,
                    [str, callable],
                )
                if isinstance(func, dict):
                    for _, funcn in func.items():
                        check(
                            f"func in nested dictionary for "
                            f"{col} in argument {num}",
                            funcn,
                            [str, callable],
                        )
        else:
            if len(arg) != 3:
                raise ValueError(
                    f"The tuple length of Argument {num} should be 3, "
                    f"instead got {len(arg)}"
                )

    by_is_true = by is not None
    grp = None
    if by_is_true and isinstance(by, dict):
        grp = df.groupby(**by)
    elif by_is_true:
        grp = df.groupby(by)
    df = df.copy()

    for arg in args_to_process:
        columns, names, func_names_and_func, dupes = _process_SD(df, arg)
        for col, (name, funcn) in product(columns, func_names_and_func):
            val = grp[col] if by_is_true else df[col]
            if names:
                name = names.format(_col=col, _fn=name)
            elif name in dupes:
                name = f"{col}{name}"
            else:
                name = col
            if isinstance(funcn, str):
                df[name] = val.transform(funcn)
            else:
                try:
                    df[name] = val.transform(funcn)
                except (ValueError, AttributeError):
                    df[name] = funcn(val)

    return df
