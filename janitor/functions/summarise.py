"""Implementation of summarise."""

from __future__ import annotations

from functools import singledispatch
from typing import Any

import pandas as pd
import pandas_flavor as pf
from pandas.api.types import is_scalar
from pandas.core.common import apply_if_callable
from pandas.core.groupby.generic import DataFrameGroupBy

from janitor.functions.select import get_index_labels


@pf.register_dataframe_method
def summarise(
    df: pd.DataFrame,
    *args: tuple[dict | tuple],
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

    The argument provided to *args* should be either
    a dictionary, a callable or a tuple; however,
    anything can be passed, as long as it fits
    within pandas' aggregation semantics.

    - **dictionary argument**:
    If the argument is a dictionary,
    the value in the `{key:value}` pairing
    should be either a string, a callable or a tuple.

        - If the value in the dictionary
        is a string or a callable,
        the key of the dictionary
        should be an existing column name.

        !!!note

            - If the value is a string,
            the string should be a pandas string function,
            e.g "sum", "mean", etc.

        - If the value of the dictionary is a tuple,
        it should be of length 2, and of the form
        `(column_name, mutation_func)`,
        where `column_name` should exist in the DataFrame,
        and `mutation_func` should be either a string or a callable.

        !!!note

            - If `mutation_func` is a string,
            the string should be a pandas string function,
            e.g "sum", "mean", etc.

        The key in the dictionary can be a new column name.

    - **tuple argument**:
    If the argument is a tuple, it should be of length 2,
    and of the form
    `(column_name, mutation_func)`,
    where column_name should exist in the DataFrame,
    and `mutation_func` should be either a string or a callable.

        !!!note

            - if `mutation_func` is a string,
            the string should be a pandas string function,
            e.g "sum", "mean", etc.

        !!!note

            - `column_name` can be anything supported by the
            [`select`][janitor.functions.select.select] syntax;
            as such multiple columns can be processed here -
            they will be processed individually.

    - **callable argument**:
    If the argument is a callable, the callable is applied
    on the DataFrame or GroupBy object.
    The result from the callable should be a pandas Series
    or DataFrame.


    Aggregated columns cannot be reused in `summarise`.


    `by` can be a `DataFrameGroupBy` object; it is assumed that
    `by` was created from `df` - the onus is on the user to
    ensure that, or the aggregations may yield incorrect results.

    `by` accepts anything supported by `pd.DataFrame.groupby`.

    Arguments supported in `pd.DataFrame.groupby`
    can also be passed to `by` via a dictionary.

    Examples:
        >>> import pandas as pd
        >>> import janitor
        >>> data = {'avg_jump': [3, 4, 1, 2, 3, 4],
        ...         'avg_run': [3, 4, 1, 3, 2, 4],
        ...         'combine_id': [100200, 100200,
        ...                        101200, 101200,
        ...                        102201, 103202]}
        >>> df = pd.DataFrame(data)
        >>> df
           avg_jump  avg_run  combine_id
        0         3        3      100200
        1         4        4      100200
        2         1        1      101200
        3         2        3      101200
        4         3        2      102201
        5         4        4      103202

        Aggregation via a callable:
        >>> df.summarise(lambda df: df.sum(),by='combine_id')
                    avg_jump  avg_run
        combine_id
        100200             7        7
        101200             3        4
        102201             3        2
        103202             4        4

        Aggregation via a tuple:
        >>> df.summarise(("avg_run","mean"), by='combine_id')
                    avg_run
        combine_id
        100200          3.5
        101200          2.0
        102201          2.0
        103202          4.0

        Aggregation via a dictionary:
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

    Args:
        df: A pandas DataFrame.
        args: Either a dictionary or a tuple.
        by: Column(s) to group by.

    Raises:
        ValueError: If a tuple is passed and the length is not 2.

    Returns:
        A pandas DataFrame or Series with aggregated columns.

    """  # noqa: E501

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
    return pd.concat(dictionary, axis="columns", sort=False, copy=False)


@singledispatch
def _mutator(arg, df, by):
    if by is None:
        val = df
    else:
        val = by
    outcome = _process_maybe_callable(func=arg, obj=val)
    if isinstance(outcome, pd.Series):
        if not outcome.name:
            raise ValueError("Ensure the pandas Series object has a name")
        return {outcome.name: outcome}
    # assumption: a mapping - DataFrame/dictionary/...
    return {**outcome}


@_mutator.register(dict)
def _(arg, df, by):
    """Dispatch function for dictionary"""
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
    """Dispatch function for tuple"""
    if len(arg) != 2:
        raise ValueError("the tuple has to be a length of 2")
    column_names, mutator = arg
    column_names = get_index_labels(arg=[column_names], df=df, axis="columns")
    mapping = {column_name: mutator for column_name in column_names}
    return _mutator(mapping, df=df, by=by)


def _process_maybe_callable(func: callable, obj):
    """Function to handle callables"""
    try:
        column = obj.agg(func)
    except:  # noqa: E722
        column = apply_if_callable(maybe_callable=func, obj=obj)
    return column


def _process_maybe_string(func: str, obj):
    """Function to handle pandas string functions"""
    # treat as a pandas approved string function
    # https://pandas.pydata.org/docs/user_guide/groupby.html#built-in-aggregation-methods
    return obj.agg(func)


def _process_within_dict(mutator, obj):
    """Handle str/callables within a dictionary"""
    if isinstance(mutator, str):
        return _process_maybe_string(func=mutator, obj=obj)
    return _process_maybe_callable(func=mutator, obj=obj)
