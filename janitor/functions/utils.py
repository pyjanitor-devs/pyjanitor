"""Utility functions for all of the functions submodule."""
from itertools import chain
import fnmatch
import warnings

from collections.abc import Callable as dispatch_callable
import re
from typing import Hashable, Iterable, List, Optional, Pattern, Union

import pandas as pd
from janitor.utils import check, _expand_grid
from pandas.api.types import (
    union_categoricals,
    is_scalar,
    is_list_like,
)
import numpy as np
from multipledispatch import dispatch
from janitor.utils import check_column
import functools


def unionize_dataframe_categories(
    *dataframes, column_names: Optional[Iterable[pd.CategoricalDtype]] = None
) -> List[pd.DataFrame]:
    """
    Given a group of dataframes which contain some categorical columns, for
    each categorical column present, find all the possible categories across
    all the dataframes which have that column.
    Update each dataframes' corresponding column with a new categorical object
    that contains the original data
    but has labels for all the possible categories from all dataframes.
    This is useful when concatenating a list of dataframes which all have the
    same categorical columns into one dataframe.

    If, for a given categorical column, all input dataframes do not have at
    least one instance of all the possible categories,
    Pandas will change the output dtype of that column from `category` to
    `object`, losing out on dramatic speed gains you get from the former
    format.

    Usage example for concatenation of categorical column-containing
    dataframes:

    Instead of:

    ```python
    concatenated_df = pd.concat([df1, df2, df3], ignore_index=True)
    ```

    which in your case has resulted in `category` -> `object` conversion,
    use:

    ```python
    unionized_dataframes = unionize_dataframe_categories(df1, df2, df2)
    concatenated_df = pd.concat(unionized_dataframes, ignore_index=True)
    ```

    :param dataframes: The dataframes you wish to unionize the categorical
        objects for.
    :param column_names: If supplied, only unionize this subset of columns.
    :returns: A list of the category-unioned dataframes in the same order they
        were provided.
    :raises TypeError: If any of the inputs are not pandas DataFrames.
    """

    if any(not isinstance(df, pd.DataFrame) for df in dataframes):
        raise TypeError("Inputs must all be dataframes.")

    if column_names is None:
        # Find all columns across all dataframes that are categorical

        column_names = set()

        for dataframe in dataframes:
            column_names = column_names.union(
                [
                    column_name
                    for column_name in dataframe.columns
                    if isinstance(
                        dataframe[column_name].dtype, pd.CategoricalDtype
                    )
                ]
            )

    else:
        column_names = [column_names]
    # For each categorical column, find all possible values across the DFs

    category_unions = {
        column_name: union_categoricals(
            [df[column_name] for df in dataframes if column_name in df.columns]
        )
        for column_name in column_names
    }

    # Make a shallow copy of all DFs and modify the categorical columns
    # such that they can encode the union of all possible categories for each.

    refactored_dfs = []

    for df in dataframes:
        df = df.copy(deep=False)

        for column_name, categorical in category_unions.items():
            if column_name in df.columns:
                df[column_name] = pd.Categorical(
                    df[column_name], categories=categorical.categories
                )

        refactored_dfs.append(df)

    return refactored_dfs


def patterns(regex_pattern: Union[str, Pattern]) -> Pattern:
    """
    This function converts a string into a compiled regular expression;
    it can be used to select columns in the index or columns_names
    arguments of `pivot_longer` function.

    **Warning**:

        This function is deprecated. Kindly use `re.compile` instead.

    :param regex_pattern: string to be converted to compiled regular
        expression.
    :returns: A compile regular expression from provided
        `regex_pattern`.
    """
    warnings.warn(
        "This function is deprecated. Kindly use `re.compile` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    check("regular expression", regex_pattern, [str, Pattern])

    return re.compile(regex_pattern)


def _computations_expand_grid(others: dict) -> pd.DataFrame:
    """
    Creates a cartesian product of all the inputs in `others`.
    Uses numpy's `mgrid` to generate indices, which is used to
    `explode` all the inputs in `others`.

    There is a performance penalty for small entries
    in using this method, instead of `itertools.product`;
    however, there are significant performance benefits
    as the size of the data increases.

    Another benefit of this approach, in addition to the significant
    performance gains, is the preservation of data types.
    This is particularly relevant for pandas' extension arrays `dtypes`
    (categoricals, nullable integers, ...).

    A DataFrame of all possible combinations is returned.
    """

    for key in others:
        check("key", key, [Hashable])

    grid = {}

    for key, value in others.items():
        if is_scalar(value):
            value = np.asarray([value])
        elif is_list_like(value) and (not hasattr(value, "shape")):
            value = np.asarray([*value])
        if not value.size:
            raise ValueError(f"Kindly provide a non-empty array for {key}.")

        grid[key] = value

    others = None

    # slice obtained here is used in `np.mgrid`
    # to generate cartesian indices
    # which is then paired with grid.items()
    # to blow up each individual value
    # before creating the final DataFrame.
    grid = grid.items()
    grid_index = [slice(len(value)) for _, value in grid]
    grid_index = map(np.ravel, np.mgrid[grid_index])
    grid = zip(grid, grid_index)
    grid = ((*left, right) for left, right in grid)
    contents = {}
    for key, value, grid_index in grid:
        contents = {**contents, **_expand_grid(value, grid_index, key)}
    return pd.DataFrame(contents, copy=False)


@dispatch(pd.DataFrame, (list, tuple), str)
def _factorize(df, column_names, suffix, **kwargs):
    check_column(df, column_names=column_names, present=True)
    for col in column_names:
        df[f"{col}{suffix}"] = pd.factorize(df[col], **kwargs)[0]
    return df


@dispatch(pd.DataFrame, str, str)
def _factorize(df, column_name, suffix, **kwargs):  # noqa: F811
    check_column(df, column_names=column_name, present=True)
    df[f"{column_name}{suffix}"] = pd.factorize(df[column_name], **kwargs)[0]
    return df


@functools.singledispatch
def _select_column_names(columns_to_select, df):
    """
    base function for column selection.
    Returns a list of column names.
    """
    raise TypeError("This type is not supported in column selection.")


# hack to get it to recognize typing.Pattern
# functools.singledispatch does not natively
# recognize types from the typing module
# `type(re.compile(r"\d+"))` returns re.Pattern
# which is a type and functools.singledispatch
# accepts it without drama;
# however, the same type from typing.Pattern
# is not accepted.
@_select_column_names.register(type(re.compile(r"\d+")))  # noqa: F811
def _column_sel_dispatch(columns_to_select, df):  # noqa: F811
    """
    Base function for column selection.
    Applies only to regular expressions.
    `re.compile` is required for the regular expression.
    A list of column names is returned.
    """
    df_columns = df.columns
    filtered_columns = [
        column_name
        for column_name in df_columns
        if re.search(columns_to_select, column_name)
    ]

    if not filtered_columns:
        raise KeyError("No column name matched the regular expression.")
    df_columns = None

    return filtered_columns


@_select_column_names.register(tuple)  # noqa: F811
def _column_sel_dispatch(columns_to_select, df):  # noqa: F811
    """
    Base function for column selection.
    This caters to columns that are of tuple type.
    The tuple is returned as is, if it exists in the columns.
    """
    if columns_to_select not in df.columns:
        raise KeyError(f"No match was returned for {columns_to_select}")
    return columns_to_select


@_select_column_names.register(list)  # noqa: F811
def _column_sel_dispatch(columns_to_select, df):  # noqa: F811
    """
    Base function for column selection.
    Applies only to list type.
    It can take any of slice, str, callable, re.Pattern types,
    or a combination of these types.
    A list of column names is returned.
    """

    # takes care of boolean entries
    if all(map(pd.api.types.is_bool, columns_to_select)):
        if len(columns_to_select) != len(df.columns):
            raise ValueError(
                "The length of the list of booleans "
                "does not match the number of columns "
                "in the dataframe."
            )

        return [*df.columns[columns_to_select]]

    filtered_columns = []
    columns_to_select = (
        _select_column_names(entry, df) for entry in columns_to_select
    )

    # this is required,
    # to maintain `tuple` status
    # when combining all the entries into a single list
    columns_to_select = (
        [entry] if isinstance(entry, tuple) else entry
        for entry in columns_to_select
    )

    columns_to_select = chain.from_iterable(columns_to_select)

    # get rid of possible duplicates
    for column_name in columns_to_select:
        if column_name not in filtered_columns:
            filtered_columns.append(column_name)

    return filtered_columns


@_select_column_names.register(str)  # noqa: F811
def _column_sel_dispatch(columns_to_select, df):  # noqa: F811
    """
    Base function for column selection.
    Applies only to strings.
    It is also applicable to shell-like glob strings,
    specifically, the `*`.
    A list of column names is returned.
    """

    if "*" in columns_to_select:  # shell-style glob string (e.g., `*_thing_*`)
        filtered_columns = fnmatch.filter(df.columns, columns_to_select)
        if filtered_columns:
            return filtered_columns
    elif columns_to_select in df.columns:
        return [columns_to_select]
    raise KeyError(f"No match was returned for '{columns_to_select}'")


@_select_column_names.register(slice)  # noqa: F811
def _column_sel_dispatch(columns_to_select, df):  # noqa: F811
    """
    Base function for column selection.
    Applies only to slices.
    The start slice value must be a string or None;
    same goes for the stop slice value.
    The step slice value should be an integer or None.
    A slice, if passed correctly in a Multindex column,
    returns a list of tuples across all levels of the
    column.
    A list of column names is returned.
    """

    df_columns = df.columns
    filtered_columns = None
    start_check = None
    stop_check = None
    step_check = None

    if not df_columns.is_unique:
        raise ValueError(
            "The column labels are not unique. "
            "Kindly ensure the labels are unique "
            "to ensure the correct output."
        )

    start, stop, step = (
        columns_to_select.start,
        columns_to_select.stop,
        columns_to_select.step,
    )
    start_check = any((start is None, isinstance(start, str)))
    stop_check = any((stop is None, isinstance(stop, str)))
    step_check = any((step is None, isinstance(step, int)))
    if not start_check:
        raise ValueError(
            "The start value for the slice "
            "must either be a string or `None`."
        )
    if not stop_check:
        raise ValueError(
            "The stop value for the slice "
            "must either be a string or `None`."
        )
    if not step_check:
        raise ValueError(
            "The step value for the slice "
            "must either be an integer or `None`."
        )
    start_check = any((start is None, start in df_columns))
    stop_check = any((stop is None, stop in df_columns))
    if not start_check:
        raise ValueError(
            "The start value for the slice must either be `None` "
            "or exist in the dataframe's columns."
        )
    if not stop_check:
        raise ValueError(
            "The stop value for the slice must either be `None` "
            "or exist in the dataframe's columns."
        )

    if start is None:
        start = 0
    else:
        start = df_columns.get_loc(start)
    if stop is None:
        stop = len(df_columns) + 1
    else:
        stop = df_columns.get_loc(stop)

    if start > stop:
        filtered_columns = df_columns[slice(stop, start + 1, step)][::-1]
    else:
        filtered_columns = df_columns[slice(start, stop + 1, step)]
    df_columns = None
    return [*filtered_columns]


@_select_column_names.register(dispatch_callable)  # noqa: F811
def _column_sel_dispatch(columns_to_select, df):  # noqa: F811
    """
    Base function for column selection.
    Applies only to callables.
    The callable is applied to every column in the dataframe.
    Either True or False is expected per column.
    A list of column names is returned.
    """
    # the function will be applied per series.
    # this allows filtration based on the contents of the series
    # or based on the name of the series,
    # which happens to be a column name as well.
    # whatever the case may be,
    # the returned values should be a sequence of booleans,
    # with at least one True.

    filtered_columns = df.agg(columns_to_select)

    if not filtered_columns.any():
        raise ValueError("No match was returned for the provided callable.")

    return [*df.columns[filtered_columns]]
