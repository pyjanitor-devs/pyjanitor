"""Utility functions for all of the functions submodule."""

from __future__ import annotations

import fnmatch
import inspect
import re
import unicodedata
import warnings
from collections.abc import Callable as dispatch_callable
from dataclasses import dataclass
from enum import Enum
from functools import singledispatch
from typing import (
    Any,
    Callable,
    Hashable,
    Iterable,
    List,
    Literal,
    Optional,
    Pattern,
    Union,
)

import numpy as np
import pandas as pd
import polars as pl
from multipledispatch import dispatch
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_dtype,
    is_list_like,
    is_scalar,
    is_string_dtype,
    union_categoricals,
)
from pandas.core.common import is_bool_indexer
from pandas.core.groupby.generic import DataFrameGroupBy, SeriesGroupBy

from janitor.errors import JanitorError
from janitor.utils import _expand_grid, check, check_column, find_stack_level

warnings.simplefilter("always", DeprecationWarning)


def unionize_dataframe_categories(
    *dataframes: Any,
    column_names: Optional[Iterable[pd.CategoricalDtype]] = None,
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

    Examples:
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

    Args:
        *dataframes: The dataframes you wish to unionize the categorical
            objects for.
        column_names: If supplied, only unionize this subset of columns.

    Raises:
        TypeError: If any of the inputs are not pandas DataFrames.

    Returns:
        A list of the category-unioned dataframes in the same order they
            were provided.
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
    """This function converts a string into a compiled regular expression.

    It can be used to select columns in the index or columns_names
    arguments of `pivot_longer` function.

    !!!warning

        This function is deprecated. Kindly use `re.compile` instead.

    Args:
        regex_pattern: String to be converted to compiled regular
            expression.

    Returns:
        A compile regular expression from provided `regex_pattern`.
    """
    warnings.warn(
        "This function is deprecated. Kindly use `re.compile` instead.",
        DeprecationWarning,
        stacklevel=find_stack_level(),
    )
    check("regular expression", regex_pattern, [str, Pattern])

    return re.compile(regex_pattern)


def _computations_expand_grid(others: dict) -> dict:
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

    A dictionary of all possible combinations is returned.
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
    # check length of keys and pad if necessary
    lengths = set(map(len, contents))
    if len(lengths) > 1:
        lengths = max(lengths)
        others = {}
        for key, value in contents.items():
            len_key = len(key)
            if len_key < lengths:
                padding = [""] * (lengths - len_key)
                key = (*key, *padding)
            others[key] = value
        return others
    return contents


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


def _is_str_or_cat(index):
    """
    Check if the column/index is a string,
    or categorical with strings.
    """
    if isinstance(index.dtype, pd.CategoricalDtype):
        return is_string_dtype(index.categories)
    return is_string_dtype(index)


def _select_regex(index, arg, source="regex"):
    "Process regex on a Pandas Index"
    assert source in ("fnmatch", "regex"), source
    try:
        if source == "fnmatch":
            arg, regex = arg
            bools = index.str.match(regex, na=False)
        else:
            bools = index.str.contains(arg, na=False, regex=True)
        if not bools.any():
            raise KeyError(f"No match was returned for '{arg}'")
        return bools
    except Exception as exc:
        raise KeyError(f"No match was returned for '{arg}'") from exc


def _select_callable(arg, func: Callable, axis=None):
    """
    Process a callable on a Pandas DataFrame/Index.
    """
    bools = func(arg)
    bools = np.asanyarray(bools)
    if not is_bool_dtype(bools):
        raise ValueError(
            "The output of the applied callable "
            "should be a 1-D boolean array."
        )
    if axis:
        arg = getattr(arg, axis)
    if len(bools) != len(arg):
        raise IndexError(
            f"The boolean array output from the callable {arg} "
            f"has wrong length: "
            f"{len(bools)} instead of {len(arg)}"
        )
    return bools


@dataclass
class DropLabel:
    """Helper class for removing labels within the `select` syntax.

    `label` can be any of the types supported in the `select`,
    `select_rows` and `select_columns` functions.
    An array of integers not matching the labels is returned.

    !!! info "New in version 0.24.0"

    Args:
        label: Label(s) to be dropped from the index.
    """

    label: Any


@singledispatch
def _select_index(arg, df, axis):
    """Base function for selection on a Pandas Index object.

    Returns either an integer, a slice,
    a sequence of booleans, or an array of integers,
    that match the exact location of the target.
    """
    try:
        return getattr(df, axis).get_loc(arg)
    except Exception as exc:
        raise KeyError(f"No match was returned for {arg}") from exc


@_select_index.register(str)  # noqa: F811
def _index_dispatch(arg, df, axis):  # noqa: F811
    """Base function for selection on a Pandas Index object.

    Applies only to strings.
    It is also applicable to shell-like glob strings,
    which are supported by `fnmatch`.

    Returns either a sequence of booleans, an integer,
    or a slice.
    """
    index = getattr(df, axis)
    if isinstance(index, pd.MultiIndex):
        index = index.get_level_values(0)
    if _is_str_or_cat(index) or is_datetime64_dtype(index):
        try:
            return index.get_loc(arg)
        except KeyError as exc:
            if _is_str_or_cat(index):
                if arg == "*":
                    return slice(None)
                # label selection should be case sensitive
                # fix for Github Issue 1160
                # translating to regex solves the case sensitivity
                # and also avoids the list comprehension
                # not that list comprehension is bad - i'd say it is efficient
                # however, the Pandas str.match method used in _select_regex
                # could offer more performance, especially if the
                # underlying array of the index is a PyArrow string array
                return _select_regex(
                    index, (arg, fnmatch.translate(arg)), source="fnmatch"
                )
            raise KeyError(f"No match was returned for '{arg}'") from exc
    raise KeyError(f"No match was returned for '{arg}'")


@_select_index.register(re.Pattern)  # noqa: F811
def _index_dispatch(arg, df, axis):  # noqa: F811
    """Base function for selection on a Pandas Index object.

    Applies only to regular expressions.
    `re.compile` is required for the regular expression.

    Returns an array of booleans.
    """
    index = getattr(df, axis)
    if isinstance(index, pd.MultiIndex):
        index = index.get_level_values(0)
    return _select_regex(index, arg)


@_select_index.register(range)  # noqa: F811
@_select_index.register(slice)  # noqa: F811
def _index_dispatch(arg, df, axis):  # noqa: F811
    """
    Base function for selection on a Pandas Index object.
    Applies only to slices.

    Returns a slice object.
    """
    index = getattr(df, axis)
    if not index.is_monotonic_increasing:
        if not index.is_unique:
            raise ValueError(
                "Non-unique Index labels should be monotonic increasing."
                "Kindly sort the index."
            )
        if is_datetime64_dtype(index):
            raise ValueError(
                "The DatetimeIndex should be monotonic increasing."
                "Kindly sort the index"
            )

    return index._convert_slice_indexer(arg, kind="loc")


@_select_index.register(dispatch_callable)  # noqa: F811
def _index_dispatch(arg, df, axis):  # noqa: F811
    """
    Base function for selection on a Pandas Index object.
    Applies only to callables.

    The callable is applied to the entire DataFrame.

    Returns an array of booleans.
    """
    # special case for selecting dtypes columnwise
    dtypes = (
        arg.__name__
        for _, arg in inspect.getmembers(pd.api.types, inspect.isfunction)
        if arg.__name__.startswith("is") and arg.__name__.endswith("type")
    )
    if (arg.__name__ in dtypes) and (axis == "columns"):
        bools = df.dtypes.map(arg)
        return np.asanyarray(bools)

    return _select_callable(df, arg, axis)


@_select_index.register(dict)  # noqa: F811
def _index_dispatch(arg, df, axis):  # noqa: F811
    """
    Base function for selection on a Pandas Index object.
    Applies only to a dictionary.

    Returns an array of integers.
    """
    level_label = {}
    index = getattr(df, axis)
    if not isinstance(index, pd.MultiIndex):
        return _select_index(list(arg), df, axis)
    all_str = (isinstance(entry, str) for entry in arg)
    all_str = all(all_str)
    all_int = (isinstance(entry, int) for entry in arg)
    all_int = all(all_int)
    if not all_str | all_int:
        raise TypeError(
            "The keys in the dictionary represent the levels "
            "in the MultiIndex, and should either be all "
            "strings or integers."
        )
    for key, value in arg.items():
        if isinstance(value, dispatch_callable):
            indexer = index.get_level_values(key)
            value = _select_callable(indexer, value)
        elif isinstance(value, re.Pattern):
            indexer = index.get_level_values(key)
            value = _select_regex(indexer, value)
        level_label[key] = value

    level_label = {
        index._get_level_number(level): label
        for level, label in level_label.items()
    }
    level_label = [
        level_label.get(num, slice(None)) for num in range(index.nlevels)
    ]
    return index.get_locs(level_label)


@_select_index.register(np.ndarray)  # noqa: F811
@_select_index.register(pd.api.extensions.ExtensionArray)  # noqa: F811
@_select_index.register(pd.Index)  # noqa: F811
@_select_index.register(pd.MultiIndex)  # noqa: F811
@_select_index.register(pd.Series)  # noqa: F811
def _index_dispatch(arg, df, axis):  # noqa: F811
    """
    Base function for selection on a Pandas Index object.
    Applies to pd.Series/pd.Index/pd.array/np.ndarray.

    Returns an array of integers.
    """
    index = getattr(df, axis)

    if is_bool_dtype(arg):
        if len(arg) != len(index):
            raise IndexError(
                f"{arg} is a boolean dtype and has wrong length: "
                f"{len(arg)} instead of {len(index)}"
            )
        return np.asanyarray(arg)
    try:
        if isinstance(arg, pd.Series):
            arr = arg.array
        else:
            arr = arg
        if isinstance(index, pd.MultiIndex) and not isinstance(
            arg, pd.MultiIndex
        ):
            return index.get_locs([arg])
        arr = index.get_indexer_for(arr)
        not_found = arr == -1
        if not_found.all():
            raise KeyError(
                f"No match was returned for any of the labels in {arg}"
            )
        elif not_found.any():
            not_found = set(arg).difference(index)
            raise KeyError(
                f"No match was returned for these labels in {arg} - "
                f"{*not_found,}"
            )
        return arr
    except Exception as exc:
        raise KeyError(f"No match was returned for {arg}") from exc


@_select_index.register(DropLabel)  # noqa: F811
def _column_sel_dispatch(cols, df, axis):  # noqa: F811
    """
    Base function for selection on a Pandas Index object.
    Returns the inverse of the passed label(s).

    Returns an array of integers.
    """
    arr = _select_index(cols.label, df, axis)
    index = np.arange(getattr(df, axis).size)
    arr = _index_converter(arr, index)
    return np.delete(index, arr)


@_select_index.register(set)
@_select_index.register(list)  # noqa: F811
def _index_dispatch(arg, df, axis):  # noqa: F811
    """
    Base function for selection on a Pandas Index object.
    Applies only to list type.
    It can take any of slice, str, callable, re.Pattern types, ...,
    or a combination of these types.

    Returns an array of integers.
    """
    index = getattr(df, axis)
    if is_bool_indexer(arg):
        if len(arg) != len(index):
            raise ValueError(
                "The length of the list of booleans "
                f"({len(arg)}) does not match "
                f"the length of the DataFrame's {axis}({index.size})."
            )

        return arg

    # shortcut for single unique dtype of scalars
    checks = (is_scalar(entry) for entry in arg)
    if all(checks):
        dtypes = {type(entry) for entry in arg}
        if len(dtypes) == 1:
            indices = index.get_indexer_for(list(arg))
            if (indices != -1).all():
                return indices
    # treat multiple DropLabel instances as a single unit
    checks = (isinstance(entry, DropLabel) for entry in arg)
    if sum(checks) > 1:
        drop_labels = (entry for entry in arg if isinstance(entry, DropLabel))
        drop_labels = [entry.label for entry in drop_labels]
        drop_labels = DropLabel(drop_labels)
        arg = [entry for entry in arg if not isinstance(entry, DropLabel)]
        arg.append(drop_labels)
    indices = [_select_index(entry, df, axis) for entry in arg]

    # single entry does not need to be combined
    # or materialized if possible;
    # this offers more performance
    if len(indices) == 1:
        if is_scalar(indices[0]):
            return indices
        indices = indices[0]
        if is_list_like(indices):
            indices = np.asanyarray(indices)
        return indices
    indices = [_index_converter(arr, index) for arr in indices]
    return np.concatenate(indices)


def _index_converter(arr, index):
    """Converts output from _select_index to an array_like"""
    if is_list_like(arr):
        arr = np.asanyarray(arr)
    if is_bool_dtype(arr):
        arr = arr.nonzero()[0]
    elif isinstance(arr, slice):
        arr = range(index.size)[arr]
    elif isinstance(arr, int):
        arr = [arr]
    return arr


def get_index_labels(
    arg, df: pd.DataFrame, axis: Literal["index", "columns"]
) -> pd.Index:
    """Convenience function to get actual labels from column/index

    !!! info "New in version 0.25.0"

    Args:
        arg: Valid inputs include: an exact column name to look for,
            a shell-style glob string (e.g. `*_thing_*`),
            a regular expression,
            a callable,
            or variable arguments of all the aforementioned.
            A sequence of booleans is also acceptable.
            A dictionary can be used for selection
            on a MultiIndex on different levels.
        df: The pandas DataFrame object.
        axis: Should be either `index` or `columns`.

    Returns:
        A pandas Index.
    """
    assert axis in {"index", "columns"}
    index = getattr(df, axis)
    return index[_select_index(arg, df, axis)]


def get_columns(group: Union[DataFrameGroupBy, SeriesGroupBy], label):
    """
    Helper function for selecting columns on a grouped object,
    using the
    [`select`][janitor.functions.select.select] syntax.

    !!! info "New in version 0.25.0"

    Args:
        group: A Pandas GroupBy object.
        label: column(s) to select.

    Returns:
        A pandas groupby object.
    """
    check("groupby object", group, [DataFrameGroupBy, SeriesGroupBy])
    label = get_index_labels(label, group.obj, axis="columns")
    label = label if is_scalar(label) else list(label)
    return group[label]


def _select(
    df: pd.DataFrame,
    invert: bool = False,
    rows=None,
    columns=None,
) -> pd.DataFrame:
    """
    Index DataFrame on the index or columns.

    Returns a DataFrame.
    """
    if rows is None:
        row_indexer = slice(None)
    else:
        outcome = _select_index([rows], df, axis="index")
        if invert:
            row_indexer = np.ones(df.index.size, dtype=np.bool_)
            row_indexer[outcome] = False
        else:
            row_indexer = outcome
    if columns is None:
        column_indexer = slice(None)
    else:
        outcome = _select_index([columns], df, axis="columns")
        if invert:
            column_indexer = np.ones(df.columns.size, dtype=np.bool_)
            column_indexer[outcome] = False
        else:
            column_indexer = outcome
    return df.iloc[row_indexer, column_indexer]


class _JoinOperator(Enum):
    """
    List of operators used in conditional_join.
    """

    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_THAN_OR_EQUAL = ">="
    LESS_THAN_OR_EQUAL = "<="
    STRICTLY_EQUAL = "=="
    NOT_EQUAL = "!="


less_than_join_types = {
    _JoinOperator.LESS_THAN.value,
    _JoinOperator.LESS_THAN_OR_EQUAL.value,
}
greater_than_join_types = {
    _JoinOperator.GREATER_THAN.value,
    _JoinOperator.GREATER_THAN_OR_EQUAL.value,
}


def _null_checks_cond_join(
    left: pd.Series, right: pd.Series
) -> Union[tuple, None]:
    """
    Checks for nulls in the arrays before conducting binary search.

    Relevant to _less_than_indices and _greater_than_indices
    """
    any_nulls = left.isna()
    if any_nulls.all():
        return None
    if any_nulls.any():
        left = left[~any_nulls]
    any_nulls = right.isna()
    if any_nulls.all():
        return None
    if any_nulls.any():
        right = right[~any_nulls]
    any_nulls = any_nulls.any()
    right_is_sorted = right.is_monotonic_increasing
    if not right_is_sorted:
        right = right.sort_values(kind="stable")

    left_index = left.index._values
    left = left._values
    right_index = right.index._values
    right = right._values

    return left, right, left_index, right_index, right_is_sorted, any_nulls


def _less_than_indices(
    left: pd.Series,
    right: pd.Series,
    strict: bool,
    multiple_conditions: bool,
    keep: str,
) -> tuple:
    """
    Use binary search to get indices where left
    is less than or equal to right.

    If strict is True, then only indices
    where `left` is less than
    (but not equal to) `right` are returned.

    A tuple of integer indexes
    for left and right is returned.
    """

    # no point going through all the hassle
    if left.min() > right.max():
        return None

    outcome = _null_checks_cond_join(left=left, right=right)
    if not outcome:
        return None
    left, right, left_index, right_index, right_is_sorted, any_nulls = outcome

    search_indices = right.searchsorted(left, side="left")

    # if any of the positions in `search_indices`
    # is equal to the length of `right_keys`
    # that means the respective position in `left`
    # has no values from `right` that are less than
    # or equal, and should therefore be discarded
    len_right = right.size
    rows_equal = search_indices == len_right

    if rows_equal.any():
        left = left[~rows_equal]
        left_index = left_index[~rows_equal]
        search_indices = search_indices[~rows_equal]

    # the idea here is that if there are any equal values
    # shift to the right to the immediate next position
    # that is not equal
    if strict:
        rows_equal = right[search_indices]
        rows_equal = left == rows_equal
        # replace positions where rows are equal
        # with positions from searchsorted('right')
        # positions from searchsorted('right') will never
        # be equal and will be the furthermost in terms of position
        # example : right -> [2, 2, 2, 3], and we need
        # positions where values are not equal for 2;
        # the furthermost will be 3, and searchsorted('right')
        # will return position 3.
        if rows_equal.any():
            replacements = right.searchsorted(left, side="right")
            # now we can safely replace values
            # with strictly less than positions
            search_indices = np.where(rows_equal, replacements, search_indices)
        # check again if any of the values
        # have become equal to length of right
        # and get rid of them
        rows_equal = search_indices == len_right

        if rows_equal.any():
            left = left[~rows_equal]
            left_index = left_index[~rows_equal]
            search_indices = search_indices[~rows_equal]

        if not search_indices.size:
            return None

    if multiple_conditions:
        return left_index, right_index, search_indices
    if right_is_sorted and (keep == "first"):
        if any_nulls:
            return left_index, right_index[search_indices]
        return left_index, search_indices
    right = [right_index[ind:len_right] for ind in search_indices]
    if keep == "first":
        right = [arr.min() for arr in right]
        return left_index, right
    if keep == "last":
        right = [arr.max() for arr in right]
        return left_index, right
    right = np.concatenate(right)
    left = left_index.repeat(len_right - search_indices)
    return left, right


def _greater_than_indices(
    left: pd.Series,
    right: pd.Series,
    strict: bool,
    multiple_conditions: bool,
    keep: str,
) -> tuple:
    """
    Use binary search to get indices where left
    is greater than or equal to right.

    If strict is True, then only indices
    where `left` is greater than
    (but not equal to) `right` are returned.

    if multiple_conditions is False, a tuple of integer indexes
    for left and right is returned;
    else a tuple of the index for left, right, as well
    as the positions of left in right is returned.
    """

    # quick break, avoiding the hassle
    if left.max() < right.min():
        return None

    outcome = _null_checks_cond_join(left=left, right=right)
    if not outcome:
        return None
    left, right, left_index, right_index, right_is_sorted, any_nulls = outcome

    search_indices = right.searchsorted(left, side="right")
    # if any of the positions in `search_indices`
    # is equal to 0 (less than 1), it implies that
    # left[position] is not greater than any value
    # in right
    rows_equal = search_indices < 1
    if rows_equal.any():
        left = left[~rows_equal]
        left_index = left_index[~rows_equal]
        search_indices = search_indices[~rows_equal]

    # the idea here is that if there are any equal values
    # shift downwards to the immediate next position
    # that is not equal
    if strict:
        rows_equal = right[search_indices - 1]
        rows_equal = left == rows_equal
        # replace positions where rows are equal with
        # searchsorted('left');
        # this works fine since we will be using the value
        # as the right side of a slice, which is not included
        # in the final computed value
        if rows_equal.any():
            replacements = right.searchsorted(left, side="left")
            # now we can safely replace values
            # with strictly greater than positions
            search_indices = np.where(rows_equal, replacements, search_indices)
        # any value less than 1 should be discarded
        # since the lowest value for binary search
        # with side='right' should be 1
        rows_equal = search_indices < 1
        if rows_equal.any():
            left = left[~rows_equal]
            left_index = left_index[~rows_equal]
            search_indices = search_indices[~rows_equal]

        if not search_indices.size:
            return None

    if multiple_conditions:
        return left_index, right_index, search_indices
    if right_is_sorted and (keep == "last"):
        if any_nulls:
            return left_index, right_index[search_indices - 1]
        return left_index, search_indices - 1
    right = [right_index[:ind] for ind in search_indices]
    if keep == "first":
        right = [arr.min() for arr in right]
        return left_index, right
    if keep == "last":
        right = [arr.max() for arr in right]
        return left_index, right
    right = np.concatenate(right)
    left = left_index.repeat(search_indices)
    return left, right


def _not_equal_indices(left: pd.Series, right: pd.Series, keep: str) -> tuple:
    """
    Use binary search to get indices where
    `left` is exactly  not equal to `right`.

    It is a combination of strictly less than
    and strictly greater than indices.

    A tuple of integer indexes for left and right
    is returned.
    """

    dummy = np.array([], dtype=int)

    # deal with nulls
    l1_nulls = dummy
    r1_nulls = dummy
    l2_nulls = dummy
    r2_nulls = dummy
    any_left_nulls = left.isna()
    any_right_nulls = right.isna()
    if any_left_nulls.any():
        l1_nulls = left.index[any_left_nulls.array]
        l1_nulls = l1_nulls.to_numpy(copy=False)
        r1_nulls = right.index
        # avoid NAN duplicates
        if any_right_nulls.any():
            r1_nulls = r1_nulls[~any_right_nulls.array]
        r1_nulls = r1_nulls.to_numpy(copy=False)
        nulls_count = l1_nulls.size
        # blow up nulls to match length of right
        l1_nulls = np.tile(l1_nulls, r1_nulls.size)
        # ensure length of right matches left
        if nulls_count > 1:
            r1_nulls = np.repeat(r1_nulls, nulls_count)
    if any_right_nulls.any():
        r2_nulls = right.index[any_right_nulls.array]
        r2_nulls = r2_nulls.to_numpy(copy=False)
        l2_nulls = left.index
        nulls_count = r2_nulls.size
        # blow up nulls to match length of left
        r2_nulls = np.tile(r2_nulls, l2_nulls.size)
        # ensure length of left matches right
        if nulls_count > 1:
            l2_nulls = np.repeat(l2_nulls, nulls_count)

    l1_nulls = np.concatenate([l1_nulls, l2_nulls])
    r1_nulls = np.concatenate([r1_nulls, r2_nulls])

    outcome = _less_than_indices(
        left, right, strict=True, multiple_conditions=False, keep=keep
    )

    if outcome is None:
        lt_left = dummy
        lt_right = dummy
    else:
        lt_left, lt_right = outcome

    outcome = _greater_than_indices(
        left, right, strict=True, multiple_conditions=False, keep=keep
    )

    if outcome is None:
        gt_left = dummy
        gt_right = dummy
    else:
        gt_left, gt_right = outcome

    left = np.concatenate([lt_left, gt_left, l1_nulls])
    right = np.concatenate([lt_right, gt_right, r1_nulls])

    if (not left.size) & (not right.size):
        return None
    return _keep_output(keep, left, right)


def _generic_func_cond_join(
    left: pd.Series,
    right: pd.Series,
    op: str,
    multiple_conditions: bool,
    keep: str,
) -> tuple:
    """
    Generic function to call any of the individual functions
    (_less_than_indices, _greater_than_indices,
    or _not_equal_indices).
    """
    strict = False

    if op in {
        _JoinOperator.GREATER_THAN.value,
        _JoinOperator.LESS_THAN.value,
        _JoinOperator.NOT_EQUAL.value,
    }:
        strict = True

    if op in less_than_join_types:
        return _less_than_indices(
            left=left,
            right=right,
            strict=strict,
            multiple_conditions=multiple_conditions,
            keep=keep,
        )
    if op in greater_than_join_types:
        return _greater_than_indices(
            left=left,
            right=right,
            strict=strict,
            multiple_conditions=multiple_conditions,
            keep=keep,
        )
    if op == _JoinOperator.NOT_EQUAL.value:
        return _not_equal_indices(left=left, right=right, keep=keep)


def _keep_output(keep: str, left: np.ndarray, right: np.ndarray):
    """return indices for left and right index based on the value of `keep`."""
    if keep == "all":
        return left, right
    grouped = pd.Series(right).groupby(left)
    if keep == "first":
        grouped = grouped.min()
        return grouped.index, grouped.array
    grouped = grouped.max()
    return grouped.index, grouped.array


class col:
    """Helper class for column selection within an expression.

    Args:
        column (Hashable): The name of the column to be selected.

    Raises:
        TypeError: If the `column` parameter is not hashable.

    !!! info "New in version 0.25.0"

    !!! warning

        `col` is currently considered experimental.
        The implementation and parts of the API
        may change without warning.

    """

    def __init__(self, column: Hashable):
        self.cols = column
        check("column", self.cols, [Hashable])
        self.join_args = None

    def __gt__(self, other):
        """Implements the greater-than comparison operator (`>`).

        Args:
            other (col): The other `col` object to compare to.

        Returns:
            col: The current `col` object.
        """
        self.join_args = (self.cols, other.cols, ">")
        return self

    def __ge__(self, other):
        """Implements the greater-than-or-equal-to comparison operator (`>=`).

        Args:
            other (col): The other `col` object to compare to.

        Returns:
            col: The current `col` object.
        """
        self.join_args = (self.cols, other.cols, ">=")
        return self

    def __lt__(self, other):
        """Implements the less-than comparison operator (`<`).

        Args:
            other (col): The other `col` object to compare to.

        Returns:
            col: The current `col` object.
        """
        self.join_args = (self.cols, other.cols, "<")
        return self

    def __le__(self, other):
        """Implements the less-than-or-equal-to comparison operator (`<=`).

        Args:
            other (col): The other `col` object to compare to.

        Returns:
            col: The current `col` object.
        """
        self.join_args = (self.cols, other.cols, "<=")
        return self

    def __ne__(self, other):
        """Implements the not-equal-to comparison operator (`!=`).

        Args:
            other (col): The other `col` object to compare to.

        Returns:
            col: The current `col` object.
        """
        self.join_args = (self.cols, other.cols, "!=")
        return self

    def __eq__(self, other):
        """Implements the equal-to comparison operator (`==`).

        Args:
            other (col): The other `col` object to compare to.

        Returns:
            col: The current `col` object.
        """
        self.join_args = (self.cols, other.cols, "==")
        return self


def _change_case(
    obj: Union[pd.Index, pd.Series, pl.Expr, list, str],
    case_type: str,
    object_type: str,
) -> str:
    """Change case of labels in obj."""
    case_types = {"preserve", "upper", "lower", "snake"}
    case_type = case_type.lower()
    if case_type not in case_types:
        raise JanitorError(f"type must be one of: {case_types}")

    if object_type == "pandas":
        if case_type == "preserve":
            return obj
        if case_type == "upper":
            return obj.str.upper()
        if case_type == "lower":
            return obj.str.lower()
        # Implementation taken from: https://gist.github.com/jaytaylor/3660565
        # by @jtaylor
        return (
            obj.str.replace(pat=r"(.)([A-Z][a-z]+)", repl=r"\1_\2", regex=True)
            .str.replace(pat=r"([a-z0-9])([A-Z])", repl=r"\1_\2", regex=True)
            .str.lower()
        )
    if object_type == "polars":
        if case_type == "preserve":
            return obj
        if case_type == "upper":
            return obj.str.to_uppercase()
        if case_type == "lower":
            return obj.str.to_lowercase()
        # Implementation taken from: https://gist.github.com/jaytaylor/3660565
        # by @jtaylor
        return (
            obj.str.replace_all(
                pattern=r"(.)([A-Z][a-z]+)", value=r"${1}_${2}", literal=False
            )
            .str.replace_all(
                pattern=r"([a-z0-9])([A-Z])", value=r"${1}_${2}", literal=False
            )
            .str.to_lowercase()
        )
    if object_type == "string":
        if case_type == "preserve":
            return obj
        if case_type == "upper":
            return obj.upper()
        if case_type == "lower":
            return obj.lower()
        # Implementation adapted from: https://gist.github.com/jaytaylor/3660565
        # by @jtaylor
        obj = re.sub(pattern=r"(.)([A-Z][a-z]+)", repl=r"\1_\2", string=obj)
        obj = re.sub(pattern=r"([a-z0-9])([A-Z])", repl=r"\1_\2", string=obj)
        return obj.lower()

    if case_type == "preserve":
        return obj
    if case_type == "upper":
        return [label.upper() for label in obj]
    if case_type == "lower":
        return [label.lower() for label in obj]
    # Implementation adapted from: https://gist.github.com/jaytaylor/3660565
    # by @jtaylor
    obj = [
        re.sub(pattern=r"(.)([A-Z][a-z]+)", repl=r"\1_\2", string=label)
        for label in obj
    ]
    obj = [
        re.sub(pattern=r"([a-z0-9])([A-Z])", repl=r"\1_\2", string=label)
        for label in obj
    ]
    obj = [label.lower() for label in obj]
    return obj


def _normalize_1(
    obj: Union[pd.Index, pd.Series, pl.Expr, list, str], object_type: str
) -> str:
    """Perform normalization of labels in obj."""
    FIXES = [(r"[ /:,?()\.-]", "_"), (r"['’]", ""), (r"[\xa0]", "_")]
    if object_type == "pandas":
        for search, replace in FIXES:
            obj = obj.str.replace(pat=search, repl=replace, regex=True)
    elif object_type == "polars":
        for search, replace in FIXES:
            obj = obj.str.replace_all(
                pattern=search, value=replace, literal=False
            )
    elif object_type == "string":
        for search, replace in FIXES:
            obj = re.sub(pattern=search, repl=replace, string=obj)
    else:
        for search, replace in FIXES:
            obj = [
                re.sub(pattern=search, repl=replace, string=label)
                for label in obj
            ]
    return obj


def _remove_special(
    object_type: str,
    obj: Union[pd.Index, pd.Series, pl.Expr, list, str] = None,
) -> str:
    """Remove special characters from obj."""
    if object_type == "pandas":
        return obj.str.replace(
            pat="[^A-Za-z_\\d]", repl="", regex=True
        ).str.strip()
    if object_type == "polars":
        return obj.str.replace_all(
            pattern="[^A-Za-z_\\d]", value="", literal=False
        ).str.strip_chars()
    elif object_type == "string":
        obj = [item for item in obj if item.isalnum() or (item == "_")]
        return "".join(obj)
    out = []
    for label in obj:
        word = [item for item in label if item.isalnum() or (item == "_")]
        word = "".join(word)
        out.append(word)
    return out


def _strip_accents(
    obj: Union[pd.Index, pd.Series, pl.Expr, list, str],
    object_type: str,
) -> str:
    """Remove accents from a label.

    Inspired from [StackOverflow][so].

    [so]: https://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-in-a-python-unicode-strin
    """  # noqa: E501
    if object_type == "pandas":
        return obj.map(
            lambda f: "".join(
                [
                    letter
                    for letter in unicodedata.normalize("NFD", str(f))
                    if not unicodedata.combining(letter)
                ]
            )
        )
    if object_type == "polars":
        return obj.map_elements(
            lambda word: [
                letter
                for letter in unicodedata.normalize("NFD", word)
                if not unicodedata.combining(letter)
            ],
            return_dtype=pl.List(pl.Utf8),
        ).list.join("")
    if object_type == "string":
        obj = [
            letter
            for letter in unicodedata.normalize("NFD", obj)
            if not unicodedata.combining(letter)
        ]
        return "".join(obj)
    out = []
    for label in obj:
        word = [
            letter
            for letter in unicodedata.normalize("NFD", label)
            if not unicodedata.combining(letter)
        ]
        word = "".join(word)
        out.append(word)
    return out


def _strip_underscores_func(
    obj: Union[pd.Index, pd.Series, pl.Expr, list, str],
    object_type: str,
    strip_underscores: Union[str, bool] = None,
) -> pd.DataFrame:
    """Strip underscores."""
    underscore_options = {None, "left", "right", "both", "l", "r", True}
    if strip_underscores not in underscore_options:
        raise JanitorError(
            f"strip_underscores must be one of: {underscore_options}"
        )
    if object_type == "pandas":
        if strip_underscores in {"left", "l"}:
            return obj.str.lstrip("_")
        if strip_underscores in {"right", "r"}:
            return obj.str.rstrip("_")
        if strip_underscores in {True, "both"}:
            return obj.str.strip("_")
        return obj

    if object_type == "polars":
        if strip_underscores in {"left", "l"}:
            return obj.str.strip_chars_start("_")
        if strip_underscores in {"right", "r"}:
            return obj.str.strip_chars_end("_")
        if strip_underscores in {True, "both"}:
            return obj.str.strip_chars("_")
        return obj

    if object_type == "string":
        if strip_underscores in {"left", "l"}:
            return obj.lstrip("_")
        if strip_underscores in {"right", "r"}:
            return obj.rstrip("_")
        if strip_underscores in {True, "both"}:
            return obj.strip("_")
        return obj

    if strip_underscores in {"left", "l"}:
        return [label.lstrip("_") for label in obj]
    if strip_underscores in {"right", "r"}:
        return [label.rstrip("_") for label in obj]
    if strip_underscores in {True, "both"}:
        return [label.strip("_") for label in obj]
    return obj


def make_clean_names(
    obj: Union[pd.Index, pd.Series, pl.Expr, list, str],
    strip_underscores: Optional[Union[str, bool]] = None,
    case_type: str = "lower",
    remove_special: bool = False,
    strip_accents: bool = False,
    enforce_string: bool = False,
    truncate_limit: int = None,
    object_type: str = "pandas",
) -> Union[pd.Index, pd.Series, pl.Expr, list]:
    """
    Generic function to clean labels in an object.
    It can be applied to a pandas Index/Series, a Polars Expression,
    or a python string/list.
    For pandas, there is a [`clean_names`][janitor.functions.clean_names.clean_names]
    method, which is a wrapper around the `make_clean_names` function.
    For polars, use this function via existing Polars functions. The examples below
    show how you can use this within polars.

    Examples:
        >>> import polars as pl
        >>> import janitor
        >>> df = pl.DataFrame(
        ...     {
        ...         "Aloha": range(3),
        ...         "Bell Chart": range(3),
        ...         "Animals@#$%^": range(3)
        ...     }
        ... )
        >>> df
        shape: (3, 3)
        ┌───────┬────────────┬──────────────┐
        │ Aloha ┆ Bell Chart ┆ Animals@#$%^ │
        │ ---   ┆ ---        ┆ ---          │
        │ i64   ┆ i64        ┆ i64          │
        ╞═══════╪════════════╪══════════════╡
        │ 0     ┆ 0          ┆ 0            │
        │ 1     ┆ 1          ┆ 1            │
        │ 2     ┆ 2          ┆ 2            │
        └───────┴────────────┴──────────────┘

        Clean the column names,
        via [rename](https://docs.pola.rs/py-polars/html/reference/dataframe/api/polars.DataFrame.rename.html#polars-dataframe-rename):
        >>> df.rename(
        ...     lambda objumn_name: make_clean_names(
        ...         obj=objumn_name, remove_special=True, object_type="string"
        ...     )
        ... )
        shape: (3, 3)
        ┌───────┬────────────┬─────────┐
        │ aloha ┆ bell_chart ┆ animals │
        │ ---   ┆ ---        ┆ ---     │
        │ i64   ┆ i64        ┆ i64     │
        ╞═══════╪════════════╪═════════╡
        │ 0     ┆ 0          ┆ 0       │
        │ 1     ┆ 1          ┆ 1       │
        │ 2     ┆ 2          ┆ 2       │
        └───────┴────────────┴─────────┘

        >>> df = pl.DataFrame({"raw": ["Abçdê fgí j"]})
        >>> df
        shape: (1, 1)
        ┌─────────────┐
        │ raw         │
        │ ---         │
        │ str         │
        ╞═════════════╡
        │ Abçdê fgí j │
        └─────────────┘

        Clean the column values,
        via [with_columns](https://docs.pola.rs/py-polars/html/reference/dataframe/api/polars.DataFrame.with_columns.html#polars-dataframe-with-columns):
        >>> df.with_columns(
        ...    pl.col("raw").pipe(
        ...        make_clean_names, object_type="polars", strip_accents=True
        ...    )
        ... )
        shape: (1, 1)
        ┌─────────────┐
        │ raw         │
        │ ---         │
        │ str         │
        ╞═════════════╡
        │ abcde_fgi_j │
        └─────────────┘

    !!! info "New in version 0.28.0"

    Args:
        obj: The object to clean. It can be a pandas Index,
            a pandas Series, a polars Expression, a python string,
            or a python list.
        strip_underscores: Removes the outer underscores from all
            labels. Default None keeps outer underscores. Values can be
            either 'left', 'right' or 'both' or the respective shorthand 'l',
            'r' and True.
        case_type: Whether to make the labels lower or uppercase.
            Current case may be preserved with 'preserve',
            while snake case conversion (from CamelCase or camelCase only)
            can be turned on using "snake".
            Default 'lower' makes all characters lowercase.
        remove_special: Remove special characters from the labels.
            Only letters, numbers and underscores are preserved.
        strip_accents: Whether or not to remove accents from
            the labels.
        enforce_string: Whether or not to convert the labels to string.
            Defaults to True, but can be turned off.
        truncate_limit: Truncates formatted labels to
            the specified length. Default None does not truncate.
        object_type: The type of object to clean. It should be either `pandas`,
            `polars`, a python `string`, or a python `list`.
    Returns:
        A pandas Index, pandas Series, polars Expression, a python string,
        or a python list.
    """  # noqa: E501
    if enforce_string and (object_type == "pandas"):
        if not (_is_str_or_cat(obj)):
            obj = obj.astype(str)
    elif enforce_string and (object_type == "list"):
        obj = [str(label) for label in obj]
    elif enforce_string and (object_type == "string"):
        obj = str(obj)
    elif enforce_string and (object_type == "polars"):
        obj = obj.cast(pl.Utf8)
    obj = _change_case(obj, case_type, object_type=object_type)
    obj = _normalize_1(obj, object_type=object_type)
    if remove_special:
        obj = _remove_special(object_type=object_type, obj=obj)
    if strip_accents:
        obj = _strip_accents(obj=obj, object_type=object_type)
    if object_type == "pandas":
        obj = obj.str.replace(pat="_+", repl="_", regex=True)
    elif object_type == "polars":
        obj = obj.str.replace(pattern="_+", value="_", literal=False)
    elif object_type == "string":
        obj = re.sub(pattern="_+", repl="_", string=obj)
    else:
        obj = [re.sub(pattern="_+", repl="_", string=label) for label in obj]
    obj = _strip_underscores_func(
        obj, strip_underscores=strip_underscores, object_type=object_type
    )
    if truncate_limit and (object_type == "pandas"):
        obj = obj.str[:truncate_limit]
    elif truncate_limit and (object_type == "polars"):
        obj = obj.str.slice(offset=0, length=truncate_limit)
    elif truncate_limit and (object_type == "string"):
        obj = obj[:truncate_limit]
    elif truncate_limit:
        obj = [label[:truncate_limit] for label in obj]
    return obj
