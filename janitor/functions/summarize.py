"""Alternative function to pd.agg for summarizing data."""
from typing import Any
import pandas as pd
import pandas_flavor as pf

from janitor.utils import check
from pandas.api.types import is_scalar

from janitor.functions.utils import _select_index, SD
from collections import Counter


@pf.register_dataframe_method
def summarize(
    df: pd.DataFrame,
    *args,
    by: Any = None,
) -> pd.DataFrame:
    """
    Reduction operation on columns via a dictionary or a tuple.

    It is a wrapper around `pd.DataFrame.agg`,
    with added flexibility for multiple columns.

    Before reaching for `summarize`, try `pd.DataFrame.agg`.

    The computation should return a single row
    for the entire dataframe,
    or a row per group, if `by` is present.

    If the variable argument is a tuple,
    it has to be of the form `(columns, func, names_glue)`;
    the `names_glue` argument is optional.
    `columns` can be selected with the
    [`select_columns`][janitor.functions.select.select_columns]
    syntax for flexibility.
    The function `func` should be a string
    (which is dispatched to `pd.Series.agg`),
    or a callable, or a list/tuple of strings/callables.

    The `names_glue` argument allows for renaming, especially for
    multiple columns or multiple functions.
    The placeholders for `names_glue` are `_col`, which represents
    the column name, and `_fn` which represents the function name.
    Under the hood, it uses python's `str.format` method.

    `janitor.SD` offers a more explicit form
    of passing tuples to the `summarize` function.


    Example - Summarize with a dictionary:

        >>> import pandas as pd
        >>> import numpy as np
        >>> import janitor as jn
        >>> pd.set_option("display.max_columns", None)
        >>> pd.set_option("display.expand_frame_repr", False)
        >>> pd.set_option("max_colwidth", None)
        >>> data = {'avg_jump': [3, 4, 1, 2, 3, 4],
        ...         'avg_run': [3, 4, 1, 3, 2, 4],
        ...         'avg_swim': [2, 1, 2, 2, 3, 4],
        ...         'combine_id': [100200, 100200, 101200, 101200, 102201, 103202],
        ...         'category': ['heats', 'heats', 'finals', 'finals', 'heats', 'finals']}
        >>> df = pd.DataFrame(data)
        >>> (df
        ... .summarize({"avg_run":"mean"}, by=['combine_id', 'category'])
        ... )
                             avg_run
        combine_id category
        100200     heats         3.5
        101200     finals        2.0
        102201     heats         2.0
        103202     finals        4.0

    Summarize with a new column name:

        >>> df.summarize({"avg_run_2":df.avg_run.mean()})
           avg_run_2
        0   2.833333
        >>> df.summarize({"avg_run_2":lambda f: f.avg_run.mean()}, by=['combine_id', 'category'])
                            avg_run_2
        combine_id category
        100200     heats         3.5
        101200     finals        2.0
        102201     heats         2.0
        103202     finals        4.0

    Summarize with a tuple:

        >>> cols = jn.SD(columns="avg*", func="mean", names_glue="{_col}_{_fn}")
        >>> df.summarize(cols)
           avg_jump_mean  avg_run_mean  avg_swim_mean
        0       2.833333      2.833333       2.333333
        >>> df.summarize(cols, by=['combine_id', 'category'])
                             avg_jump_mean  avg_run_mean  avg_swim_mean
        combine_id category
        100200     heats               3.5           3.5            1.5
        101200     finals              1.5           2.0            2.0
        102201     heats               3.0           2.0            3.0
        103202     finals              4.0           4.0            4.0

    :param df: A pandas DataFrame.
    :param args: Either a dictionary or a tuple.
    :param by: Column(s) to group by.
    :raises ValueError: If a tuple is passed and the length is not 3.
    :returns: A pandas DataFrame with summarized columns.
    """  # noqa: E501

    for num, arg in enumerate(args):
        check(f"Argument {num} in the summarize function", arg, [dict, tuple])
        if isinstance(arg, tuple):
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

    by_is_true = by is not None
    grp = None
    if by_is_true and isinstance(by, dict):
        grp = df.groupby(**by)
    elif by_is_true:
        grp = df.groupby(by)

    aggs = {}

    for arg in args:
        if isinstance(arg, dict):
            for col, func in arg.items():
                val = grp if by_is_true else df
                if isinstance(func, str):
                    outcome = val[col].agg(func)
                elif is_scalar(func):
                    outcome = func
                else:
                    try:
                        outcome = val.agg(func)
                    except (ValueError, AttributeError):
                        outcome = func(val)
                aggs[col] = outcome
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
                val = grp[col] if by_is_true else df[col]
                for name, funcn in func_names:
                    if names:
                        name = names.format(_col=col, _fn=name)
                    elif name in dupes:
                        name = f"{col}{name}"
                    else:
                        name = col
                    if isinstance(funcn, str):
                        outcome = val.agg(funcn)
                    else:
                        try:
                            outcome = val.agg(funcn)
                        except (ValueError, AttributeError):
                            outcome = funcn(val)
                    aggs[name] = outcome
    aggs = {
        col: [outcome] if is_scalar(outcome) else outcome
        for col, outcome in aggs.items()
    }
    return pd.DataFrame(aggs, copy=False)
