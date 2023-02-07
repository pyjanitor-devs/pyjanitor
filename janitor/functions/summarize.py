"""Alternative function to pd.agg for summarizing data."""
from typing import Any
import pandas as pd
import pandas_flavor as pf

from janitor.utils import check
from pandas.api.types import is_scalar

from janitor.functions.utils import col, get_index_labels
from itertools import product
from collections import defaultdict


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

    Reduction operation on columns via the `janitor.col` class.

    It is a wrapper around `pd.DataFrame.agg`,
    with added flexibility for multiple columns.

    The `col` class allows for flexibility when aggregating.

    It has a `compute` method, for adding the functions that will
    be applied to the columns, and is of the form `.compute(*args, **kwargs)`.
    The variable `args` argument accepts the function(s), while the
    keyword arguments `kwargs` accepts parameters to be passed to the
    functions.

    There is also a `rename` method, for renaming the columns after
    aggregation. It is a single string, which can also be used as
    a glue, with `_col` as a placeholder for column name,
    and `_fn` as a placeholder for function name.

    `by` accepts a label, labels, mapping, function, or `col` class.
    Arguments supported in `pd.DataFrame.groupby`
    can also be passed to `by` via a dictionary.

    Examples:

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

    Args:
        df: A pandas DataFrame.
        args: instance(s) of the `janitor.col` class.
        by: Column(s) to group by.

    Raises:
        ValueError: If a function is not provided.

    Returns:
        A pandas DataFrame with summarized columns.
    """  # noqa: E501

    for num, arg in enumerate(args):
        check(f"Argument {num} in the summarize function", arg, [col])
        if arg.func is None:
            raise ValueError(f"Kindly provide a function for Argument {num}")

    aggs = {}
    lambda_count = defaultdict(int)
    unique_funcs = set()
    tuples_exist = False
    for arg in args:
        columns = get_index_labels([*arg.cols], df, axis="columns")
        if arg.remove_cols:
            exclude = get_index_labels([*arg.remove_cols], df, axis="columns")
            columns = columns.difference(exclude, sort=False)
        funcns, kwargs = arg.func
        funcs = (
            funcn.__name__ if callable(funcn) else funcn for funcn in funcns
        )
        funcs = zip(funcns, funcs)
        for column, (fn, fn_name) in product(columns, funcs):
            if arg.names:
                col_name = arg.names.format(_col=column, _fn=fn_name)
                if col_name in aggs:
                    raise ValueError(
                        f"{col_name} already exists as a label "
                        "for an aggregated column"
                    )
                aggs[col_name] = column, fn, kwargs
            else:
                label = column, fn_name
                unique_funcs.add(fn_name)
                if (fn_name == "<lambda>") and (
                    lambda_count.get(label) or label in aggs
                ):
                    num = lambda_count[label] + 1
                    col_name = column, f"<lambda_{num}>"
                    lambda_count[label] += 1
                    if label in aggs:
                        aggs[(column, "<lambda_0>")] = aggs.pop(label)
                        unique_funcs.remove(fn_name)
                        unique_funcs.add("<lambda_0>")
                    aggs[col_name] = column, fn, kwargs
                    unique_funcs.add(f"<lambda_{num}>")
                elif fn_name == "<lambda>":
                    aggs[(column, fn_name)] = column, fn, kwargs
                elif fn_name != "<lambda>" and (label in aggs):
                    raise ValueError(
                        f"{label} already exists as a label "
                        "for an aggregated column"
                    )
                elif fn_name != "<lambda>":
                    aggs[label] = column, fn, kwargs
                tuples_exist = True
    if tuples_exist and (len(unique_funcs) == 1):
        aggs = {
            (key[0] if isinstance(key, tuple) else key): value
            for key, value in aggs.items()
        }
    elif tuples_exist:
        aggs = {
            (key if isinstance(key, tuple) else (key, "")): value
            for key, value in aggs.items()
        }

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
    aggregates = {}
    for col_name, (column, funcn, kwargs) in aggs.items():
        val = grp[column] if by_is_true else df[column]
        if isinstance(funcn, str):
            outcome = val.agg(funcn, **kwargs)
        else:
            try:
                outcome = val.agg(funcn, **kwargs)
            except (ValueError, AttributeError):
                outcome = funcn(val, **kwargs)

        if isinstance(outcome, pd.DataFrame):
            for label, value in outcome.items():
                name = (
                    (col_name[0], label)
                    if isinstance(col_name, tuple)
                    else label
                    if len(aggs) == 1
                    else (col_name, label)
                )
                if name in aggs:
                    raise ValueError(
                        f"{name} already exists as a label "
                        "for an aggregated column"
                    )
                aggregates[name] = value
        else:
            if col_name in aggregates:
                raise ValueError(
                    f"{col_name} already exists as a label "
                    "for an aggregated column"
                )
            if is_scalar(outcome):
                outcome = [outcome]
            aggregates[col_name] = outcome
    return pd.DataFrame(aggregates, copy=False)
