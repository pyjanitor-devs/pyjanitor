"""Implementation of mutate."""

from __future__ import annotations

import copy
import warnings
from functools import singledispatch

import pandas as pd
import pandas_flavor as pf
from pandas.core.common import apply_if_callable
from pandas.core.groupby.generic import DataFrameGroupBy

from janitor.functions.select import get_index_labels
from janitor.utils import find_stack_level, refactored_function


@pf.register_dataframe_groupby_method
@refactored_function(
    message=("This function is deprecated. Please use `jn.get_columns` instead.")
)
def ungroup(
    df: DataFrameGroupBy,
) -> pd.DataFrame:
    """

    !!! info "New in version 0.32.0"

    Ungroups a GroupBy object into a DataFrame.

    Examples:
        >>> import pandas as pd
        >>> import janitor
        >>> data = {
        ...     "avg_jump": [3, 4, 1, 2, 3, 4],
        ...     "avg_run": [3, 4, 1, 3, 2, 4],
        ...     "combine_id": [100200, 100200, 101200, 101200, 102201, 103202],
        ... }
        >>> df = pd.DataFrame(data)
        >>> df
           avg_jump  avg_run  combine_id
        0         3        3      100200
        1         4        4      100200
        2         1        1      101200
        3         2        3      101200
        4         3        2      102201
        5         4        4      103202
        >>> df.groupby("combine_id").mutate("mean").ungroup()
           avg_jump  avg_run  combine_id
        0       3.5      3.5      100200
        1       3.5      3.5      100200
        2       1.5      2.0      101200
        3       1.5      2.0      101200
        4       3.0      2.0      102201
        5       4.0      4.0      103202

    Args:
        df: A pandas GroupBy object.

    Returns:
        A pandas DataFrame.
    """
    warnings.warn(
        "This function is deprecated. Kindly use `jn.get_columns` instead.",
        DeprecationWarning,
        stacklevel=find_stack_level(),
    )
    return df.obj


@pf.register_dataframe_groupby_method
@pf.register_dataframe_method
def mutate(
    df: pd.DataFrame | DataFrameGroupBy,
    *args: tuple[dict | tuple],
) -> pd.DataFrame | DataFrameGroupBy:
    """

    !!! info "New in version 0.31.0"

    !!!note

        Before reaching for `mutate`, try `pd.DataFrame.assign`.

    mutate creates new columns that are functions of existing columns.
    It can also modify columns (if the name is the same as an existing column).

    The argument provided to *args* should be either
    a dictionary, a callable or a tuple; however,
    anything can be passed, as long as it can
    be aligned with the original DataFrame.


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
        The key in the dictionary can be a new column name.

        !!!note

            - If `mutation_func` is a string,
            the string should be a pandas string function,
            e.g "sum", "mean", etc.



    - **tuple argument**:
    If the argument is a tuple, it should be of length 2,
    and of the form
    `(column_name, mutation_func)`,
    where `column_name` should exist in the DataFrame,
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
    on the DataFrame.
    The result from the callable should be a pandas Series
    or DataFrame.

    Mutation does not occur on the original DataFrame.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> import janitor
        >>> df = pd.DataFrame(
        ...     {
        ...         "col1": [5, 10, 15],
        ...         "col2": [3, 6, 9],
        ...         "col3": [10, 100, 1_000],
        ...     }
        ... )

        Transformation via a dictionary:
        >>> df.mutate({"col4": ("col1", np.log10), "col1": np.log10})
               col1  col2  col3      col4
        0  0.698970     3    10  0.698970
        1  1.000000     6   100  1.000000
        2  1.176091     9  1000  1.176091

        Transformation via a tuple:
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

        Transformation via a callable:
        >>> df.mutate(lambda df: df.sum(axis=1).rename("total"))
           col1  col2  col3  total
        0     5     3    10     18
        1    10     6   100    116
        2    15     9  1000   1024

    Args:
        df: A pandas DataFrame or GroupBy object.
        args: Either a dictionary or a tuple.

    Raises:
        ValueError: If a tuple is passed and the length is not 2.

    Returns:
        A pandas DataFrame, Series, or GroupBy object.
    """  # noqa: E501
    if isinstance(df, DataFrameGroupBy):
        df = copy.copy(df)
        df_ = df.obj.copy(deep=None)
        df.obj = df_
        for arg in args:
            df_ = _mutator(arg, df=df_, by=df)
        return df
    df = df.copy(deep=None)
    for arg in args:
        df = _mutator(arg, df=df, by=None)
    return df


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
        df[outcome.name] = outcome
        return df
    if isinstance(outcome, pd.DataFrame):
        for column in outcome:
            df[column] = outcome[column]
        return df
    raise TypeError(
        "The output from the mutation should be a named Series or a DataFrame"
    )


@_mutator.register(dict)
def _(arg, df, by):
    """Dispatch function for dictionary"""
    if by is None:
        val = df
    else:
        val = by
    for column_name, mutator in arg.items():
        if isinstance(mutator, tuple):
            column, func = mutator
            column = _apply_func_to_obj(mutator=func, obj=val[column])
        else:
            column = _apply_func_to_obj(mutator=mutator, obj=val[column_name])
        df[column_name] = column
    return df


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
        column = obj.transform(func)
    except:  # noqa: E722
        column = apply_if_callable(maybe_callable=func, obj=obj)
    return column


def _process_maybe_string(func: str, obj):
    """Function to handle pandas string functions"""
    # treat as a pandas approved string function
    # https://pandas.pydata.org/docs/user_guide/groupby.html#built-in-aggregation-methods
    return obj.transform(func)


def _apply_func_to_obj(mutator, obj):
    """Handle str/callables within a dictionary"""
    if isinstance(mutator, str):
        return _process_maybe_string(func=mutator, obj=obj)
    return _process_maybe_callable(func=mutator, obj=obj)
