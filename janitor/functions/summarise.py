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


@pf.register_dataframe_groupby_method
@pf.register_dataframe_method
def summarise(
    df: pd.DataFrame | DataFrameGroupBy,
    *args: tuple[dict | tuple],
) -> pd.DataFrame:
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
    should be either a string, a callable, or a tuple.

        - If the value in the dictionary
        is a string or a callable,
        the key of the dictionary
        should be an existing column name.

        The function is applied on the `df[column_name]` series.

        !!!note

            - If the value is a string,
            the string should be a pandas string function,
            e.g "sum", "mean", etc.

        - If the value of the dictionary is a tuple,
        it should be of length 2, and of the form
        `(column_name, aggfunc)`,
        where `column_name` should exist in the DataFrame,
        and `aggfunc` should be either a string or a callable.

        This option allows for custom renaming of the aggregation output,
        where the key in the dictionary can be a new column name.


    - **tuple argument**:
    If the argument is a tuple, it should be of length 2,
    and of the form
    `(column_name, aggfunc)`,
    where column_name should exist in the DataFrame,
    and `aggfunc` should be either a string or a callable.

        !!!note

            - if `aggfunc` is a string,
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


    Arguments supported in `pd.DataFrame.groupby`
    can also be passed to `by` via a dictionary.

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

        Aggregation on a DataFrame via a callable:
        >>> df.summarise(
        ...     lambda df: df.select("avg*").mean().rename("mean")
        ... )  # doctest: +NORMALIZE_WHITESPACE
                             mean
        avg_jump         2.833333
        avg_run          2.833333

        Aggregation on a DataFrame via a tuple:
        >>> df.summarise(("avg_*", "mean"))
              avg_jump   avg_run
        mean  2.833333  2.833333

        Aggregation on a DataFrame via a dictionary:
        >>> df.summarise({"avg_jump": "mean"})
              avg_jump
        mean  2.833333

        >>> df.summarise({"avg_run_2": ("avg_run", "mean")})
              avg_run_2
        mean   2.833333

        >>> grouped = df.groupby("combine_id")

        Aggregation on a grouped object via a callable:
        >>> grouped.summarise(lambda df: df.sum())  # doctest: +NORMALIZE_WHITESPACE
                 avg_jump  avg_run
        combine_id
        100200             7        7
        101200             3        4
        102201             3        2
        103202             4        4

        Aggregation on a grouped object via a tuple:
        >>> grouped.summarise(("avg_run", "mean"))  # doctest: +NORMALIZE_WHITESPACE
                    avg_run
        combine_id
        100200          3.5
        101200          2.0
        102201          2.0
        103202          4.0

        Aggregation on a grouped object via a dictionary:
        >>> grouped.summarise({"avg_run": "mean"})  # doctest: +NORMALIZE_WHITESPACE
                    avg_run
        combine_id
        100200          3.5
        101200          2.0
        102201          2.0
        103202          4.0
        >>> grouped.summarise(
        ...     {"avg_run_2": ("avg_run", "mean")}
        ... )  # doctest: +NORMALIZE_WHITESPACE
                    avg_run_2
        combine_id
        100200            3.5
        101200            2.0
        102201            2.0
        103202            4.0

    Args:
        df: A pandas DataFrame or DataFrameGroupBy object.
        args: Either a dictionary or a tuple.

    Raises:
        ValueError: If a tuple is passed and the length is not 2.

    Returns:
        A pandas DataFrame with aggregated columns.

    """  # noqa: E501
    if isinstance(df, DataFrameGroupBy):
        by = df
        df = df.obj
    else:
        by = None
    contents = []
    for arg in args:
        aggregate = _aggfunc(arg, df=df, by=by)
        contents.extend(aggregate)
    counts = 0
    for entry in contents:
        if isinstance(entry, pd.DataFrame):
            length = entry.columns.nlevels
        elif isinstance(entry.name, tuple):
            length = len(entry.name)
        else:
            length = 1
        counts = max(counts, length)
    contents_ = []
    for entry in contents:
        if isinstance(entry, pd.DataFrame):
            length_ = entry.columns.nlevels
            length = counts - length_
            if length:
                patch = [""] * length
                columns = [entry.columns.get_level_values(n) for n in range(length_)]
                columns.append(patch)
                names = [*entry.columns.names]
                names.extend([None] * length)
                columns = pd.MultiIndex.from_arrays(columns, names=names)
                entry.columns = columns
        elif is_scalar(entry.name):
            length = counts - 1
            if length:
                patch = [""] * length
                name = (entry.name, *patch)
                entry.name = name
        elif isinstance(entry.name, tuple):
            length = counts - len(entry.name)
            if length:
                patch = [""] * length
                name = (*entry.name, *patch)
                entry.name = name
        contents_.append(entry)
    return pd.concat(contents_, axis=1, copy=False, sort=False)


@singledispatch
def _aggfunc(arg, df, by):
    if by is None:
        val = df
    else:
        val = by
    outcome = apply_if_callable(maybe_callable=arg, obj=val)
    if isinstance(outcome, pd.Series):
        if not outcome.name:
            raise ValueError("Ensure the pandas Series object has a name")
        return [outcome]
    if isinstance(outcome, pd.DataFrame):
        return [outcome]
    raise TypeError(
        "The output from the aggregation should be a named Series or a DataFrame"
    )


@_aggfunc.register(tuple)
def _(arg, df, by):
    """Dispatch function for tuple"""
    if len(arg) != 2:
        raise ValueError("the tuple has to be a length of 2")
    column_name, aggfunc = arg
    column_names = get_index_labels(arg=[column_name], df=df, axis="columns")
    mapping = {column_name: aggfunc for column_name in column_names}
    return _aggfunc(mapping, df=df, by=by)


@_aggfunc.register(dict)
def _(arg, df, by):
    """Dispatch function for dictionary"""
    if by is None:
        val = df
    else:
        val = by

    contents = []
    for column_name, aggfunc in arg.items():
        if isinstance(aggfunc, tuple):
            if len(aggfunc) != 2:
                raise ValueError("the tuple has to be a length of 2")
            column, func = aggfunc
            column = val.agg({column: func})
            try:
                column = column.squeeze()
            except AttributeError:
                pass
            column = _convert_obj_to_named_series(
                obj=column,
                column_name=column_name,
                function=func,
            )
            if not isinstance(column, pd.Series):
                raise TypeError(
                    f"Expected a pandas Series object; instead got {type(column)}"
                )
        else:
            column = val.agg({column_name: aggfunc})
            try:
                column = column.squeeze()
            except AttributeError:
                pass
            column = _convert_obj_to_named_series(
                obj=column,
                column_name=column_name,
                function=aggfunc,
            )
        contents.append(column)
    return contents


def _convert_obj_to_named_series(obj, function: Any, column_name: Any):
    if isinstance(obj, pd.Series):
        obj.name = column_name
        return obj
    if not is_scalar(obj):
        return obj
    if isinstance(function, str):
        function_name = function
    else:
        function_name = function.__name__
    return pd.Series(data=obj, index=[function_name], name=column_name)
