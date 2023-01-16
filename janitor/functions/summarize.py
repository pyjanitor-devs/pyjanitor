"""Alternative function to pd.agg for summarizing data."""
from typing import Any
import pandas as pd
import pandas_flavor as pf

from janitor.utils import check
from pandas.api.types import is_scalar

from janitor.functions.utils import col, _process_function, get_index_labels
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

    Reduction operation on columns via the `janitor.Column` class.

    It is a wrapper around `pd.DataFrame.agg`,
    with added flexibility for multiple columns.

    The `col` class allows for flexibility when aggregating.

    It has a `compute` method, for adding the functions that will
    be applied to the columns, and is of the form `.compute(*args, **kwargs)`.
    The variable `args` argument accepts the functions, while the
    keyword arguments `kwargs` accepts parameters to be passed to the
    functions.

    There is also a `rename` method, for renaming the columns after
    aggregation. It is a single string, which can also be used as
    a glue, with `_col` as a placeholder for column name,
    and `_fn` as a placeholder for function name.

    `by` accepts a label, labels, mapping, function, or `col` class.
    Arguments supported in `pd.DataFrame.groupby`
    can also be passed to `by` via a dictionary.


    Example:

        >>> import pandas as pd
        >>> import numpy as np
        >>> import janitor as jn
        >>> from janitor import col
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
        >>> arg = col("avg_run").compute("mean")
        >>> df.summarize(arg, by=['combine_id', 'category'])
                             avg_run
        combine_id category
        100200     heats         3.5
        101200     finals        2.0
        102201     heats         2.0
        103202     finals        4.0

    Summarize with a new column name:

        >>> arg = col("avg_run").compute("mean").rename("avg_run_2")
        >>> df.summarize(arg)
           avg_run_2
        0   2.833333
        >>> df.summarize(arg, by=['combine_id', 'category'])
                            avg_run_2
        combine_id category
        100200     heats         3.5
        101200     finals        2.0
        102201     heats         2.0
        103202     finals        4.0

    Summarize with the placeholders when renaming:

        >>> cols = col("avg*").compute("mean").rename("{_col}_{_fn}")
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
    :param args: instance(s) of the `janitor.col` class.
    :param by: Column(s) to group by.
    :raises ValueError: If a function is not provided for any of the arguments.
    :returns: A pandas DataFrame with summarized columns.
    """  # noqa: E501

    args_to_process = []
    for num, arg in enumerate(args):
        check(f"Argument {num} in the summarize function", arg, [col])
        if arg.func is None:
            raise ValueError(f"Kindly provide a function for Argument {num}")

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
    elif by_is_true and isinstance(by, col):
        if by.func:
            raise ValueError("Function assignment is not required within by")
        cols = get_index_labels([*by.cols], df, axis="columns")
        if by.remove_cols:
            exclude = get_index_labels([*by.remove_cols], df, axis="columns")
            cols = cols.difference(exclude, sort=False)
        grp = df.groupby(cols.tolist())
    elif by_is_true:
        grp = df.groupby(by)

    aggs = {}

    for arg in args:
        columns, names, func_names_and_func, dupes = _process_function(df, arg)
        for col_name, (name, (funcn, kwargs)) in product(
            columns, func_names_and_func
        ):
            val = grp[col_name] if by_is_true else df[col_name]
            if names:
                name = names.format(_col=col_name, _fn=name)
            elif name in dupes:
                name = f"{col_name}{name}"
            else:
                name = col_name
            if isinstance(funcn, str):
                outcome = val.agg(funcn, **kwargs)
            else:
                try:
                    outcome = val.agg(funcn, **kwargs)
                except (ValueError, AttributeError):
                    outcome = funcn(val, **kwargs)
            if isinstance(outcome, pd.DataFrame):
                outcome.columns = f"{name}_" + outcome.columns
                aggs.update(outcome)
            else:
                if is_scalar(outcome):
                    outcome = [outcome]
                aggs[name] = outcome
    return pd.DataFrame(aggs, copy=False)
