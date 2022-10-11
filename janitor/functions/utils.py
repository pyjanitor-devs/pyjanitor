"""Utility functions for all of the functions submodule."""
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


# this section relates to column/row selection
# indices/slices/booleans are returned,
# instead of the actual labels
# as it is marginally faster when indexing the dataframe
def _is_str_or_cat(index):
    """
    Check if the column/index is a string,
    or categorical with strings.
    """
    if is_categorical_dtype(index):
        return is_string_dtype(index.categories)
    return is_string_dtype(index)


def _select_strings(index, selection):
    """
    Generic function for string selection on rows/columns.

    Returns a sequence of booleans, a slice, or an integer.
    """
    if _is_str_or_cat(index):
        if selection in index:
            return index.get_loc(selection)
        # fix for Github Issue 1160
        outcome = [fnmatch.fnmatchcase(column, selection) for column in index]
        if not any(outcome):
            raise KeyError(f"No match was returned for '{selection}'.")
        return outcome
    if is_datetime64_dtype(index):
        try:
            return index.get_loc(selection)
        except Exception as exc:
            raise KeyError(
                f"No match was returned for '{selection}'."
            ) from exc

    raise KeyError(f"No match was returned for '{selection}'.")


def _select_regex(index, selection):
    """
    Generic function for selecting regex rows/columns.

    Returns a sequence of booleans.
    """
    if _is_str_or_cat(index):
        bools = index.str.contains(selection, na=False, regex=True)
        if not bools.any():
            raise KeyError(f"No match was returned for {selection}.")
        return bools
    raise KeyError(f"No match was returned for {selection}.")


def _select_slice(index, selection, label="column"):
    """
    Generic function for selecting slice on rows/columns.

    Returns a slice object.
    """
    is_date_column = is_datetime64_dtype(index)
    if not index.is_monotonic_increasing:
        if not index.is_unique:
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
            f"The step value for the slice {selection} "
            "must either be an integer or None."
        )
    start_check = None
    stop_check = None
    if not is_date_column:
        start_check = any((start is None, start in index))
        if not start_check:
            raise ValueError(
                f"The start value for the slice {selection} "
                "must either be None "
                f"or exist in the dataframe's {label}."
            )
        stop_check = any((stop is None, stop in index))
        if not stop_check:
            raise ValueError(
                f"The stop value for the slice {selection} "
                "must either be None "
                f"or exist in the dataframe's {label}."
            )
    if start is None:
        start_ = 0
    else:
        start_ = index.get_loc(start)
    if stop is None:
        stop_ = len(index)
    else:
        stop_ = index.get_loc(stop)
    start_check = isinstance(start_, slice)
    stop_check = isinstance(stop_, slice)
    print(start_check, stop_check)

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
        index = range(index.size)
        slicer = slice(stop, start + 1, step)
        return index[slicer][::-1]
    return slice(start, stop + 1, step)


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
    """
    Generic function for list selection of rows/columns.

    Returns an array of integers.
    """
    index = getattr(df, label)
    if all(map(is_bool, selection)):
        if len(selection) != len(index):
            raise ValueError(
                "The length of the list of booleans "
                f"({len(selection)}) does not match "
                f"the number of {label}({index.size}) "
                "in the dataframe."
            )

        return np.asanyarray(selection).nonzero()[0]

    contents = []
    for entry in selection:
        arr = func(entry, df)
        if is_list_like(arr):
            arr = np.asanyarray(arr)
        if is_bool_dtype(arr):
            arr = arr.nonzero()[0]
        elif isinstance(arr, slice):
            arr = range(index.size)[arr]
        elif isinstance(arr, int):
            arr = [arr]
        contents.append(arr)
    if len(contents) > 1:
        contents = np.concatenate(contents)
        # remove possible duplicates
        return pd.unique(contents)
    return contents[0]


@singledispatch
def _select_columns(columns_to_select, df):
    """
    Base function for column selection.

    Returns either an integer, a slice,
    a sequence of booleans, or an array of integers.
    """
    if columns_to_select in df.columns.tolist():
        return df.columns.get_loc(columns_to_select)
    raise KeyError(f"No match was returned for {columns_to_select}.")


@_select_columns.register(str)  # noqa: F811
def _column_sel_dispatch(columns_to_select, df):  # noqa: F811
    """
    Base function for column selection.
    Applies only to strings.
    It is also applicable to shell-like glob strings,
    which are supported by `fnmatch`.

    Returns either a sequence of booleans, an integer,
    or a slice.
    """
    return _select_strings(df.columns, columns_to_select)


@_select_columns.register(re.Pattern)  # noqa: F811
def _column_sel_dispatch(columns_to_select, df):  # noqa: F811
    """
    Base function for column selection.
    Applies only to regular expressions.
    `re.compile` is required for the regular expression.

    Returns an array of booleans.
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

    Returns a slice object.
    """
    return _select_slice(df.columns, columns_to_select, label="column")


@_select_columns.register(dispatch_callable)  # noqa: F811
def _column_sel_dispatch(columns_to_select, df):  # noqa: F811
    """
    Base function for column selection.
    Applies only to callables.
    The callable is applied to every column in the dataframe.
    Either True or False is expected per column.

    Returns an array of booleans.
    """
    # the function will be applied per series.
    # this allows filtration based on the contents of the series
    # or based on the name of the series,
    # which happens to be a column name as well.
    # whatever the case may be,
    # the returned values should be a sequence of booleans,
    # with at least one True.

    bools = df.apply(columns_to_select)

    if not is_bool_dtype(bools):
        raise TypeError(
            "The output of the applied callable should be a boolean array."
        )
    if not bools.any():
        raise KeyError(f"No match was returned for {columns_to_select}.")

    return bools


@_select_columns.register(IndexLabel)  # noqa: F811
def _column_sel_dispatch(columns_to_select, df):  # noqa: F811
    return _level_labels(
        df.columns, columns_to_select.label, columns_to_select.level
    )


@_select_columns.register(list)  # noqa: F811
def _column_sel_dispatch(columns_to_select, df):  # noqa: F811
    """
    Base function for column selection.
    Applies only to list type.
    It can take any of slice, str, callable, re.Pattern types, ...,
    or a combination of these types.

    Returns an array of integers.
    """
    return _select_list(
        df, columns_to_select, _select_columns, label="columns"
    )


@singledispatch
def _select_rows(rows, df):
    """
    Base function for row selection.

    Returns a sequence of booleans, an integer, a slice,
    or an array of numbers.
    """
    if rows in df.index.tolist():
        return df.index.get_loc(rows)
    raise KeyError(f"No match was returned for {rows}.")


@_select_rows.register(str)  # noqa: F811
def _row_sel_dispatch(rows, df):  # noqa: F811
    """
    Base function for row selection.
    Applies only to strings.
    It is also applicable to shell-like glob strings,
    which are supported by `fnmatch`.

    Returns a sequence of booleans, an integer, or a slice.
    """
    return _select_strings(df.index, rows)


@_select_rows.register(re.Pattern)  # noqa: F811
def _row_sel_dispatch(rows, df):  # noqa: F811
    """
    Base function for row selection.
    Applies only to regular expressions.
    `re.compile` is required for the regular expression.

    Returns an array of booleans.
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

    Returns a slice object.
    """
    return _select_slice(df.index, rows, label="index")


@_select_rows.register(dispatch_callable)  # noqa: F811
def _row_sel_dispatch(rows, df):  # noqa: F811
    """
    Base function for row selection.
    Applies only to callables.
    The callable is applied to the dataframe.
    A valid list-like indexer is expected.

    Returns an array of booleans/integers.
    """
    rows = apply_if_callable(rows, df)

    _ = check_array_indexer(df.index, rows)

    return rows


@_select_rows.register(IndexLabel)  # noqa: F811
def _row_sel_dispatch(rows, df):  # noqa: F811
    return _level_labels(df.index, rows.label, rows.level)


@_select_rows.register(list)  # noqa: F811
def _row_sel_dispatch(rows, df):  # noqa: F811
    """
    Base function for row selection.
    Applies only to list type.
    It can take any of slice, str, callable, re.Pattern types, ...,
    or a combination of these types.

    Returns an array of integers.
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
    :returns: An array of integers.
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
    elif isinstance(arr, slice):
        index = range(len(index))
        return index[arr]
    elif isinstance(arr, int):
        return [arr]
    return arr


def _generic_select(
    df: pd.DataFrame, args: tuple, invert: bool, axis: str = "index"
) -> pd.DataFrame:
    """
    Index DataFrame on the index or columns.

    Returns a DataFrame.
    """
    indices = []
    func = {"index": _select_rows, "columns": _select_columns}
    for arg in args:
        if is_list_like(arg) and (not isinstance(arg, tuple)):
            indices.extend(arg)
        else:
            indices.append(arg)
    indices = func[axis](indices, df)
    if invert:
        index = getattr(df, axis)
        rev = np.ones(len(index), dtype=np.bool8)
        rev[indices] = False
        return df.iloc(axis=axis)[rev]
    return df.iloc(axis=axis)[indices]


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
