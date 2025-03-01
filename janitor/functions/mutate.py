"""Implementation of mutate."""

from functools import singledispatch
from typing import Any

import pandas as pd
import pandas_flavor as pf
from pandas.api.types import is_scalar
from pandas.core.common import apply_if_callable

from janitor.functions.select import get_index_labels


@pf.register_dataframe_method
def mutate(
    df: pd.DataFrame,
    *args,
    by: Any = None,
) -> pd.DataFrame:
    """

    !!! info "New in version 0.31.0"

    !!!note

        Before reaching for `mutate`, try `pd.DataFrame.assign`.

    mutate creates new columns that are functions of existing columns.
    It can also modify columns (if the name is the same as an existing column).

    The argument provided to *args* should be either a dictionary or a tuple.

    If the argument is a dictionary,
    the value should be either a string, a callable or a tuple.
    If it is a string or a callable, the key of the dictionary
    should be an existing column name.
    Note that if the value is a string,
    the string should be a pandas string function,
    e.g "sqrt", "mean", etc.
    If the value of the dictionary is a tuple,
    it should be of length 2, and of the form
    `(column_name, mutation_func)`,
    where column_name should exist in the DataFrame,
    and mutation_func should be either a string or a callable.
    Note that if mutation_func is a string,
    the string should be a pandas string function,
    e.g "sqrt", "mean", etc.
    The key in the dictionary can be a new column name,
    which will be created and added to the DataFrame.

    If the argument is a tuple, it should be of length 2,
    and of the form
    `(column_name, mutation_func)`,
    where column_name should exist in the DataFrame,
    and mutation_func should be either a string or a callable.
    Note that if mutation_func is a string,
    the string should be a pandas string function,
    e.g "sqrt", "mean", etc.
    Note that column_name can be anyting supported by
    the `jn.select` function; as such multiple columns
    can be processed here - they will be processed individually
    and appended to the DataFrame.

    `by` accepts anything supported by `pd.DataFrame.groupby`.
    Arguments supported in `pd.DataFrame.groupby`
    can also be passed to `by` via a dictionary.

    Example:

        >>> import pandas as pd
        >>> import numpy as np
        >>> import janitor
        >>> df = pd.DataFrame({
        ...     "col1": [5, 10, 15],
        ...     "col2": [3, 6, 9],
        ...     "col3": [10, 100, 1_000],
        ... })
        >>> df.mutate({"col4": ('col1', np.log10)})
           col1  col2  col3      col4
        0     5     3    10  0.698970
        1    10     6   100  1.000000
        2    15     9  1000  1.176091

        >>> df.mutate(
        ...     {"col4": ('col1',np.log10),
        ...      "col1": np.log10}
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

    Example: Transformation in the presence of a groupby:

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

        >>> df.mutate({"avg_run_2":("avg_run","mean")}, by='combine_id')
           avg_jump  avg_run  combine_id  avg_run_2
        0         3        3      100200        3.5
        1         4        4      100200        3.5
        2         1        1      101200        2.0
        3         2        3      101200        2.0
        4         3        2      102201        2.0
        5         4        4      103202        4.0

    :param df: A pandas DataFrame.
    :param args: Either a dictionary or a tuple.
    :param by: Column(s) to group by.
    :raises ValueError: If a tuple is passed and the length is not 2.
    :returns: A pandas DataFrame with mutated columns.
    """  # noqa: E501

    df = df.copy()
    if by is not None:
        if isinstance(by, dict):
            by = df.groupby(**by)
        else:
            if is_scalar(by):
                by = [by]
            by = df.groupby(by, sort=False, observed=True)
    for arg in args:
        df = _mutator(arg, df=df, by=by)
    return df


@singledispatch
def _mutator(arg, df, by):
    raise NotImplementedError(
        f"janitor.mutate is not supported for {type(arg)}"
    )


@_mutator.register(dict)
def _(arg, df, by):
    if by is None:
        val = df
    else:
        val = by
    for column_name, mutator in arg.items():
        if isinstance(mutator, tuple):
            column, func = mutator
            column = _process_within_dict(mutator=func, obj=val[column])
        else:
            column = _process_within_dict(
                mutator=mutator, obj=val[column_name]
            )
        df[column_name] = column
    return df


@_mutator.register(tuple)
def _(arg, df, by):
    if len(arg) != 2:
        raise ValueError("the tuple has to be a length of 2")
    column_names, mutator = arg
    column_names = get_index_labels(arg=[column_names], df=df, axis="columns")
    mapping = {column_name: mutator for column_name in column_names}
    return _mutator(mapping, df=df, by=by)


def _process_maybe_callable(func: callable, obj):
    try:
        column = obj.transform(func)
    except:  # noqa: E722
        column = apply_if_callable(maybe_callable=func, obj=obj)
    return column


def _process_maybe_string(func: str, obj):
    # treat as a pandas approved string function
    # https://pandas.pydata.org/docs/user_guide/groupby.html#built-in-aggregation-methods
    return obj.transform(func)


def _process_within_dict(mutator, obj):
    if isinstance(mutator, str):
        return _process_maybe_string(func=mutator, obj=obj)
    return _process_maybe_callable(func=mutator, obj=obj)
