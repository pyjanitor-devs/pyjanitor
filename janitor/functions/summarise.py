"""Implementation of summarise."""

from __future__ import annotations

from functools import singledispatch
from typing import Any

import pandas as pd
import pandas_flavor as pf
from pandas.api.types import is_scalar
from pandas.core.groupby.generic import DataFrameGroupBy

from janitor.functions.select import get_index_labels


@pf.register_dataframe_method
def summarise(
    df: pd.DataFrame,
    *args,
    by: Any = None,
) -> pd.DataFrame | pd.Series:
    """

    !!! info "New in version 0.31.0"

    !!!note

        Before reaching for `summarise`, try `pd.DataFrame.agg`.

    summarise creates a new dataframe;
    it returns one row for each combination of grouping columns.
    If there are no grouping variables,
    the output will have a single row
    summarising all observations in the input.

    The argument provided to *args* should be either a dictionary or a tuple.

    If the argument is a dictionary,
    the value should be either a string, a callable or a tuple.
    If it is a string or a callable, the key of the dictionary
    should be an existing column name.
    Note that if the value is a string,
    the string should be a pandas string function,
    e.g "sum", "mean", etc.
    If the value of the dictionary is a tuple,
    it should be of length 2, and of the form
    `(column_name, mutation_func)`,
    where column_name should exist in the DataFrame,
    and mutation_func should be either a string or a callable.
    Note that if mutation_func is a string,
    the string should be a pandas string function,
    e.g "sum", "mean", etc.
    The key in the dictionary can be a new column name.

    If the argument is a tuple, it should be of length 2,
    and of the form
    `(column_name, mutation_func)`,
    where column_name should exist in the DataFrame,
    and mutation_func should be either a string or a callable.
    Note that if mutation_func is a string,
    the string should be a pandas string function,
    e.g "sum", "mean", etc.
    Note that column_name can be anyting supported by
    the `jn.select` function; as such multiple columns
    can be processed here - they will be processed individually.

    `by` accepts anything supported by `pd.DataFrame.groupby`.
    `by` can be a DataFrameGroupBy object; it is assumed that
    `by` was created from `df`; the onus is on the user to
    ensure that, or the aggregations may yield incorrect results.
    Arguments supported in `pd.DataFrame.groupby`
    can also be passed to `by` via a dictionary.

    Example:

        >>> data = {'avg_jump': [3, 4, 1, 2, 3, 4],
        ...         'avg_run': [3, 4, 1, 3, 2, 4],
        ...         'combine_id': [100200, 100200,
        ...                        101200, 101200,
        ...                        102201, 103202]}
        >>> df = pd.DataFrame(data)
        >>> df.summarise(("avg_run","mean"), by='combine_id')
                    avg_run
        combine_id
        100200          3.5
        101200          2.0
        102201          2.0
        103202          4.0

        >>> df.summarise({"avg_run":"mean"}, by='combine_id')
                    avg_run
        combine_id
        100200          3.5
        101200          2.0
        102201          2.0
        103202          4.0

        >>> df.summarise({"avg_run_2":("avg_run","mean")}, by='combine_id')
                    avg_run_2
        combine_id
        100200            3.5
        101200            2.0
        102201            2.0
        103202            4.0

    :param df: A pandas DataFrame.
    :param args: Either a dictionary or a tuple.
    :param by: Column(s) to group by.
    :raises ValueError: If a tuple is passed and the length is not 2.
    :returns: A pandas DataFrame or Series with summarised columns.
    """  # noqa: E501

    df = df.copy()
    if by is not None:
        # it is assumed that by is created from df
        # onus is on user to ensure that
        if isinstance(by, DataFrameGroupBy):
            pass
        elif isinstance(by, dict):
            by = df.groupby(**by)
        else:
            if is_scalar(by):
                by = [by]
            by = df.groupby(by, sort=False, observed=True)
    dictionary = {}
    for arg in args:
        aggregate = _mutator(arg, df=df, by=by)
        dictionary.update(aggregate)
    values = map(is_scalar, dictionary.values())
    if all(values):
        return pd.Series(dictionary)
    values = (isinstance(obj, pd.Series) for obj in dictionary.values())
    if all(values):
        return pd.DataFrame(dictionary)
    return pd.concat(dictionary, axis=1, sort=False, copy=False)


@singledispatch
def _mutator(arg, df, by):
    raise NotImplementedError(
        f"janitor.summarise is not supported for {type(arg)}"
    )


@_mutator.register(dict)
def _(arg, df, by):
    if by is None:
        val = df
    else:
        val = by

    dictionary = {}
    for column_name, mutator in arg.items():
        if isinstance(mutator, tuple):
            column, func = mutator
            column = _process_within_dict(mutator=func, obj=val[column])
        else:
            column = _process_within_dict(
                mutator=mutator, obj=val[column_name]
            )
        dictionary[column_name] = column
    return dictionary


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
        column = obj.agg(func)
    except:  # noqa: E722
        column = func(obj)
    return column


def _process_maybe_string(func: str, obj):
    # treat as a pandas approved string function
    # https://pandas.pydata.org/docs/user_guide/groupby.html#built-in-aggregation-methods
    return obj.agg(func)


def _process_within_dict(mutator, obj):
    if isinstance(mutator, str):
        return _process_maybe_string(func=mutator, obj=obj)
    return _process_maybe_callable(func=mutator, obj=obj)
