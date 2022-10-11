"""Utility functions for all of the functions submodule."""
from itertools import chain
import fnmatch
import warnings
from collections.abc import Callable as dispatch_callable
import re
from typing import Hashable, Iterable, List, Optional, Pattern, Union, Any
from pandas.core.dtypes.generic import ABCPandasArray, ABCExtensionArray
from pandas.api.indexers import check_array_indexer
from dataclasses import dataclass


import pandas as pd
from janitor.utils import check, _expand_grid
from pandas.core.common import apply_if_callable
from pandas.api.types import (
    union_categoricals,
    is_scalar,
    is_list_like,
    is_datetime64_dtype,
    is_string_dtype,
    is_categorical_dtype,
    is_extension_array_dtype,
    is_bool_dtype,
    is_bool,
)
import numpy as np
from multipledispatch import dispatch
from janitor.utils import check_column
from functools import singledispatch


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
        contents.update(_expand_grid(value, grid_index, key))
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


def _is_str_or_cat(df_columns):
    """Check if the column/index is a string, or categorical with strings."""
    if is_categorical_dtype(df_columns):
        return is_string_dtype(df_columns.categories)
    return is_string_dtype(df_columns)


def _select_strings(df_columns, selection):
    """Generic function for string selection on rows/columns"""
    if _is_str_or_cat(df_columns):
        if selection in df_columns:
            return [selection]
        # fix for Github Issue 1160
        outcome = [
            fnmatch.fnmatchcase(column, selection) for column in df_columns
        ]
        if not any(outcome):
            raise KeyError(f"No match was returned for '{selection}'.")
        return df_columns[outcome]

    if is_datetime64_dtype(df_columns):
        try:
            timestamp = df_columns.get_loc(selection)
            if not isinstance(timestamp, int):
                return df_columns[timestamp]
            return [df_columns[timestamp]]
        except Exception as exc:
            raise KeyError(
                f"No match was returned for '{selection}'."
            ) from exc

    raise KeyError(f"No match was returned for '{selection}'.")


def _select_regex(df_columns, selection):
    """Generic function for selecting regex rows/columns."""
    if _is_str_or_cat(df_columns):
        bools = df_columns.str.contains(selection, na=False, regex=True)
        if not bools.any():
            raise KeyError(f"No match was returned for {selection}.")
        return df_columns[bools]
    raise KeyError(f"No match was returned for {selection}.")


def _select_slice(df_columns, selection, label="column"):
    """Generic function for selecting slice on rows/columns."""
    is_date_column = is_datetime64_dtype(df_columns)
    if not df_columns.is_monotonic_increasing:
        if not df_columns.is_unique:
            raise ValueError(
                f"Non-unique {label} labels should be monotonic increasing."
            )
        if is_date_column:
            raise ValueError(
                f"The {label} is a DatetimeIndex and should be "
                "monotonic increasing."
            )

    start, stop, step = (
        selection.start,
        selection.stop,
        selection.step,
    )

    step_check = None
    step_check = any((step is None, isinstance(step, int)))
    if not step_check:
        raise ValueError(
            "The step value for the slice "
            "must either be an integer or `None`."
        )
    if step and (not df_columns.is_unique):
        raise ValueError(
            "The step argument for slice is not applicable "
            "to non unique labels."
        )
    start_check = None
    stop_check = None
    if not is_date_column:
        start_check = any((start is None, start in df_columns))
        if not start_check:
            raise ValueError(
                "The start value for the slice must either be `None` "
                f"or exist in the dataframe's {label}."
            )
        stop_check = any((stop is None, stop in df_columns))
        if not stop_check:
            raise ValueError(
                "The stop value for the slice must either be `None` "
                f"or exist in the dataframe's {label}."
            )
    if start is None:
        start_ = 0
    else:
        start_ = df_columns.get_loc(start)
    if stop is None:
        stop_ = len(df_columns)
    else:
        stop_ = df_columns.get_loc(stop)
    start_check = isinstance(start_, slice)
    stop_check = isinstance(stop_, slice)

    if start_check:
        start = start_.start
    else:
        start = start_
    if stop_check:
        stop = stop_.stop - 1
    else:
        stop = stop_
    if start > stop:
        if start_check:
            start = start_.stop - 1
        if stop_check:
            stop = stop_.start

    if start > stop:
        return df_columns[slice(stop, start + 1, step)][::-1]
    return df_columns[slice(start, stop + 1, step)]


@dataclass
class IndexLabel:
    """
    Helper class for selecting on a Pandas MultiIndex.

    `label` can be a scalar, a slice, a sequence of labels
    - any argument that can be passed to
    `pd.MultiIndex.get_loc` or `pd.MultiIndex.get_locs`

    :param label: Value to be selected from the index.
    :param level: Determines hich level to select the labels from.
        If None, the labels are assumed to be selected
        from all levels. For multiple levels,
        the length of `label` should match the length of `level`.
    :returns: A dataclass.
    """

    label: Any
    level: Optional[Union[int, str]] = None


def _select_list(df, selection, func, label="columns"):
    """Generic function for list selection of rows/columns"""
    df_columns = getattr(df, label)
    if all(map(is_bool, selection)):
        if len(selection) != len(df_columns):
            raise ValueError(
                "The length of the list of booleans "
                f"({len(selection)}) does not match "
                f"the number of {label}({df_columns.size}) "
                "in the dataframe."
            )

        return df_columns[selection]

    selection = (func(entry, df) for entry in selection)

    selection = chain.from_iterable(selection)

    # get rid of possible duplicates
    return list(dict.fromkeys(selection))


@singledispatch
def _select_columns(columns_to_select, df):
    """
    base function for column selection.
    Returns a list of column names.
    """
    if columns_to_select in df.columns.tolist():
        return [columns_to_select]
    raise KeyError(f"No match was returned for {columns_to_select}.")


@_select_columns.register(str)  # noqa: F811
def _column_sel_dispatch(columns_to_select, df):  # noqa: F811
    """
    Base function for column selection.
    Applies only to strings.
    It is also applicable to shell-like glob strings,
    which are supported by `fnmatch`.
    A list/pandas Index of matching column names is returned.
    """
    return _select_strings(df.columns, columns_to_select)


@_select_columns.register(re.Pattern)  # noqa: F811
def _column_sel_dispatch(columns_to_select, df):  # noqa: F811
    """
    Base function for column selection.
    Applies only to regular expressions.
    `re.compile` is required for the regular expression.
    A pandas Index of matching column names is returned.
    """
    return _select_regex(df.columns, columns_to_select)


@_select_columns.register(slice)  # noqa: F811
def _column_sel_dispatch(columns_to_select, df):  # noqa: F811
    """
    Base function for column selection.
    Applies only to slices.

    The start slice value must be a string/tuple/None,
    or exist in the dataframe's columns;
    same goes for the stop slice value.
    The step slice value should be an integer or None.
    A slice, if passed correctly in a Multindex column,
    returns a list of tuples across all levels of the
    column.

    A pandas Index of matching column names is returned.
    """
    return _select_slice(df.columns, columns_to_select, label="column")


@_select_columns.register(dispatch_callable)  # noqa: F811
def _column_sel_dispatch(columns_to_select, df):  # noqa: F811
    """
    Base function for column selection.
    Applies only to callables.
    The callable is applied to every column in the dataframe.
    Either True or False is expected per column.

    A pandas Index of matching column names is returned.
    """
    # the function will be applied per series.
    # this allows filtration based on the contents of the series
    # or based on the name of the series,
    # which happens to be a column name as well.
    # whatever the case may be,
    # the returned values should be a sequence of booleans,
    # with at least one True.

    filtered_columns = df.apply(columns_to_select)

    if not is_bool_dtype(filtered_columns):
        raise TypeError(
            "The output of the applied callable should be a boolean array."
        )
    if not filtered_columns.any():
        raise KeyError(f"No match was returned for {columns_to_select}.")

    return df.columns[filtered_columns]


@_select_columns.register(IndexLabel)  # noqa: F811
def _column_sel_dispatch(columns_to_select, df):  # noqa: F811
    raise NotImplementedError(
        "`IndexLabel` cannot be combined " "with other selection options."
    )


@_select_columns.register(list)  # noqa: F811
def _column_sel_dispatch(columns_to_select, df):  # noqa: F811
    """
    Base function for column selection.
    Applies only to list type.
    It can take any of slice, str, callable, re.Pattern types, ...,
    or a combination of these types.
    A list of column names is returned.
    """
    return _select_list(
        df, columns_to_select, _select_columns, label="columns"
    )


@singledispatch
def _select_rows(rows, df):
    """
    base function for row selection.
    Returns a list of index names.
    """
    if rows in df.index:
        return [rows]
    raise KeyError(f"No match was returned for {rows}.")


@_select_rows.register(str)  # noqa: F811
def _row_sel_dispatch(rows, df):  # noqa: F811
    """
    Base function for row selection.
    Applies only to strings.
    It is also applicable to shell-like glob strings,
    which are supported by `fnmatch`.
    A list/pandas Index of matching rows is returned.
    """
    return _select_strings(df.index, rows)


@_select_rows.register(re.Pattern)  # noqa: F811
def _row_sel_dispatch(rows, df):  # noqa: F811
    """
    Base function for row selection.
    Applies only to regular expressions.
    `re.compile` is required for the regular expression.
    A pandas Index of matching rows is returned.
    """
    return _select_regex(df.index, rows)


@_select_rows.register(slice)  # noqa: F811
def _row_sel_dispatch(rows, df):  # noqa: F811
    """
    Base function for row selection.
    Applies only to slices.

    The start slice value must be a string/tuple/None,
    or exist in the dataframe's index;
    same goes for the stop slice value.
    The step slice value should be an integer or None.
    A slice, if passed correctly in a Multindex index,
    returns a list of tuples across all levels of the
    index.

    A pandas Index of matching rows is returned.
    """
    return _select_slice(df.index, rows, label="index")


@_select_rows.register(dispatch_callable)  # noqa: F811
def _row_sel_dispatch(rows, df):  # noqa: F811
    """
    Base function for row selection.
    Applies only to callables.
    The callable is applied to the dataframe.
    A valid list-like indexer is expected.

    A pandas Index of matching rows is returned.
    """
    rows = apply_if_callable(rows, df)

    _ = check_array_indexer(df.index, rows)

    return df.index[rows]


@_select_rows.register(IndexLabel)  # noqa: F811
def _row_sel_dispatch(rows, df):  # noqa: F811
    raise NotImplementedError(
        "`IndexLabel` cannot be combined " "with other selection options."
    )


@_select_rows.register(list)  # noqa: F811
def _row_sel_dispatch(rows, df):  # noqa: F811
    """
    Base function for row selection.
    Applies only to list type.
    It can take any of slice, str, callable, re.Pattern types, ...,
    or a combination of these types.
    A list of matching row labels is returned.
    """
    return _select_list(df, rows, _select_rows, label="index")


def _level_labels(
    index: pd.Index, label: Any, level: Optional[Union[list, int, str]] = None
):
    """
    Get labels from a pandas Index. It is meant to be used in level_labels,
    which should be called in any of the select functions
    (`select_columns`, `select_rows`, `select`),
    for selecting on multiple levels.

    `label` can be a scalar, a slice, a sequence of labels - any argument
    that can be passed to `pd.MultiIndex.get_loc` or `pd.MultiIndex.get_locs`.

    :param index: A Pandas Index.
    :param label: Value to select in Pandas index.
    :param level: Which level to select the labels from.
        If None, the labels are assumed to be selected
        from all levels.
        For multiple levels, the length of `arg` should
        match the length of `level`.
    :returns: A Pandas Index of matching labels.
    :raises TypeError: If `level` is a list and contains
        a mix of strings and integers.
    :raises ValueError: If `level` is a string and does not exist.
    :raises IndexError: If `level` is an integer and is not less than
        the number of levels of the Index.
    """
    if level is None:
        if is_scalar(label) or isinstance(label, tuple):
            arr = index.get_loc(label)
        else:
            arr = index.get_locs(label)

    else:

        check("level", level, [list, int, str])

        if isinstance(level, (str, int)):
            level = [level]

        all_str = (isinstance(entry, str) for entry in level)
        all_str = all(all_str)
        all_int = (isinstance(entry, int) for entry in level)
        all_int = all(all_int)
        if not all_str | all_int:
            raise TypeError(
                "All entries in the `level` parameter "
                "should be either strings or integers."
            )

        uniqs = set()
        level_numbers = []
        # check for duplicates
        # check if every value in `level` exists
        for lev in level:
            if isinstance(lev, str):
                if lev not in index.names:
                    raise ValueError(f"Level {lev} not found.")
                pos = index.names.index(lev)
                level_numbers.append(pos)
            else:
                # copied from pandas/indexes/multi.py
                n_levs = index.nlevels
                if lev < 0:
                    lev += n_levs
                if lev < 0:
                    orig_lev = lev - n_levs
                    raise IndexError(
                        f"Too many levels: Index has only {n_levs} levels, "
                        f"{orig_lev} is not a valid level number"
                    )
                elif lev >= n_levs:
                    raise IndexError(
                        f"Too many levels: Index has only {n_levs} levels, "
                        f"not {lev + 1}"
                    )
                level_numbers.append(lev)
            if lev in uniqs:
                raise ValueError(
                    f"Entries in `level` should be unique; "
                    f"{lev} exists multiple times."
                )
            uniqs.add(lev)  # noqa: PD005

        n_levs = len(level)
        n_levels = range(index.nlevels)
        ordered = list(n_levels[:n_levs]) == level_numbers
        if not ordered:
            tail = (num for num in n_levels if num not in level_numbers)
            level_numbers.extend(tail)
            index = index.reorder_levels(order=level_numbers)
        if is_scalar(label) or isinstance(label, tuple):
            arr = index.get_loc(label)
        else:
            arr = index.get_locs(label)
    if is_bool_dtype(arr):
        return arr.nonzero()[0]
    if isinstance(arr, slice):
        return np.r_[arr]
    if isinstance(arr, int):
        return [arr]
    return arr


def _select_index_labels(df, args, axis="index", invert=False):
    """
    Selection on rows/columns for a MultiIndex.
    """
    df_columns = getattr(df, axis)
    contents = [
        _level_labels(df_columns, arg.label, arg.level) for arg in args
    ]
    if len(contents) > 1:
        contents = np.concatenate(contents)
        # remove possible duplicates
        contents = pd.unique(contents)
    else:
        contents = contents[0]
    if invert:
        arr = np.ones(df_columns.size, dtype=np.bool8)
        arr[contents] = False
        return df.iloc(axis=axis)[arr]
    return df.iloc(axis=axis)[contents]


def _convert_to_numpy_array(
    left: np.ndarray, right: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert array to numpy array for use in numba
    """
    if is_extension_array_dtype(left):
        numpy_dtype = left.dtype.numpy_dtype
        left = left.to_numpy(dtype=numpy_dtype, copy=False)
        right = right.to_numpy(dtype=numpy_dtype, copy=False)
    elif isinstance(left, (ABCPandasArray, ABCExtensionArray)):
        left = left.to_numpy(copy=False)
        right = right.to_numpy(copy=False)
    return left, right
