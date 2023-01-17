"""Alternative function to pd.agg for summarizing data."""
from typing import Any
import pandas as pd
import pandas_flavor as pf

from janitor.utils import check
from pandas.api.types import is_scalar

from janitor.functions.utils import SD, _process_SD
from itertools import product


@pf.register_dataframe_method
def summarize(
    df: pd.DataFrame,
    *args,
    by: Any = None,
) -> pd.DataFrame:
    """

    !!! info "New in version 0.25.0"

    !!!note

        Before reaching for `summarize`, try `pd.DataFrame.agg`.

    Reduction operation on columns via a tuple.

    It is a wrapper around `pd.DataFrame.agg`,
    with added flexibility for multiple columns.

    The argument should be of the form `(columns, func, names_glue)`;
    the `names_glue` argument is optional.
    `columns` can be selected with the
    [`select_columns`][janitor.functions.select.select_columns]
    syntax for flexibility.
    The function `func` should be a string
    (which is dispatched to `pd.Series.agg`),
    or a callable, or a list/tuple of strings/callables.
    The function is called on each column in `columns`.

    The `names_glue` argument allows for renaming, especially for
    multiple columns or multiple functions.
    The placeholders for `names_glue` are `_col`, which represents
    the column name, and `_fn` which represents the function name.
    Under the hood, it uses python's `str.format` method.

    `janitor.SD` offers a more explicit form
    of passing tuples to the `summarize` function.

    `by` accepts a label, labels, mapping or function.
    Arguments supported in `pd.DataFrame.groupby`
    can also be passed to `by` via a dictionary.


    Example:

        >>> import pandas as pd
        >>> import numpy as np
        >>> import janitor as jn
        >>> data = {'avg_jump': [3, 4, 1, 2, 3, 4],
        ...         'avg_run': [3, 4, 1, 3, 2, 4],
        ...         'avg_swim': [2, 1, 2, 2, 3, 4],
        ...         'combine_id': [100200, 100200,
        ...                        101200, 101200,
        ...                        102201, 103202],
        ...         'category': ['heats', 'heats',
        ...                      'finals', 'finals',
        ...                      'heats', 'finals']}
        >>> df = pd.DataFrame(data)
        >>> df.summarize(("avg_run", "mean"), by=['combine_id', 'category'])
                             avg_run
        combine_id category
        100200     heats         3.5
        101200     finals        2.0
        102201     heats         2.0
        103202     finals        4.0

    Summarize with a new column name:

        >>> df.summarize(("avg_run", "mean", "avg_run_2"))
           avg_run_2
        0   2.833333
        >>> df.summarize(("avg_run", "mean", "avg_run_2"), by=['combine_id', 'category'])
                            avg_run_2
        combine_id category
        100200     heats         3.5
        101200     finals        2.0
        102201     heats         2.0
        103202     finals        4.0

    Summarize with the placeholders in `names_glue`:

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
    :param args: A tuple.
    :param by: Column(s) to group by.
    :raises ValueError: If the tuple size is less than 2.
    :returns: A pandas DataFrame with summarized columns.
    """  # noqa: E501

    args_to_process = []
    for num, arg in enumerate(args):
        check(f"Argument {num} in the summarize function", arg, [tuple])
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
        entry = SD(*arg)
        func = entry.func
        names = entry.names_glue
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
        args_to_process.append(entry)

    by_is_true = by is not None
    grp = None
    if by_is_true and isinstance(by, dict):
        grp = df.groupby(**by)
    elif by_is_true:
        grp = df.groupby(by)

    aggs = {}

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
                outcome = val.agg(funcn)
            else:
                try:
                    outcome = val.agg(funcn)
                except (ValueError, AttributeError):
                    outcome = funcn(val)
            if isinstance(outcome, pd.DataFrame):
                outcome.columns = f"{name}_" + outcome.columns
                aggs.update(outcome)
            else:
                if is_scalar(outcome):
                    outcome = [outcome]
                aggs[name] = outcome
    return pd.DataFrame(aggs, copy=False)
