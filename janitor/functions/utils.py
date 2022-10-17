"""Utility functions for all of the functions submodule."""
import fnmatch
import warnings
from collections.abc import Callable as dispatch_callable
import re
from typing import (
    Hashable,
    Iterable,
    List,
    Optional,
    Pattern,
    Union,
    Callable,
)
from pandas.core.dtypes.generic import ABCPandasArray, ABCExtensionArray
from pandas.core.common import is_bool_indexer


import pandas as pd
from janitor.utils import check, _expand_grid
from pandas.api.types import (
    union_categoricals,
    is_scalar,
    is_list_like,
    is_datetime64_dtype,
    is_string_dtype,
    is_categorical_dtype,
    is_extension_array_dtype,
    is_bool_dtype,
)
import numpy as np
from multipledispatch import dispatch
from janitor.utils import check_column
from functools import singledispatch

warnings.simplefilter("always", DeprecationWarning)


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


def _is_str_or_cat(index):
    """
    Check if the column/index is a string,
    or categorical with strings.
    """
    if is_categorical_dtype(index):
        return is_string_dtype(index.categories)
    return is_string_dtype(index)


def _select_regex(index, arg, source="regex"):
    "Process regex on a Pandas Index"
    assert source in ("fnmatch", "regex"), source
    try:
        if source == "fnmatch":
            bools = index.str.match(arg, na=False)
        else:
            bools = index.str.contains(arg, na=False, regex=True)
        if not bools.any():
            raise KeyError(f"No match was returned for {arg}.")
        return bools
    except Exception as exc:
        raise KeyError(f"No match was returned for {arg}.") from exc


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


@singledispatch
def _select_index(arg, df, axis):
    """
    Base function for selection on a Pandas Index object.

    Returns either an integer, a slice,
    a sequence of booleans, or an array of integers,
    that match the exact location of the target.
    """
    try:
        return getattr(df, axis).get_loc(arg)
    except Exception as exc:
        raise KeyError(f"No match was returned for {arg}.") from exc


@_select_index.register(str)  # noqa: F811
def _index_dispatch(arg, df, axis):  # noqa: F811
    """
    Base function for selection on a Pandas Index object.
    Applies only to strings.
    It is also applicable to shell-like glob strings,
    which are supported by `fnmatch`.

    Returns either a sequence of booleans, an integer,
    or a slice.
    """
    index = getattr(df, axis)
    if _is_str_or_cat(index) or is_datetime64_dtype(index):
        try:
            return index.get_loc(arg)
        except KeyError as exc:
            if _is_str_or_cat(index):
                if isinstance(index, pd.MultiIndex):
                    index = index.get_level_values(0)
                # label selection should be case sensitive
                # fix for Github Issue 1160
                # translating to regex solves the case sensitivity
                # and also avoids the list comprehension
                # not that list comprehension is bad - i'd say it is efficient
                # however, the Pandas str.match method used in _select_regex
                # could offer more performance, especially if the
                # underlying array of the index is a PyArrow string array
                regex = fnmatch.translate(arg)
                return _select_regex(index, regex, source="fnmatch")
            raise KeyError(f"No match was returned for '{arg}'.") from exc
    raise KeyError(f"No match was returned for '{arg}'.")


@_select_index.register(re.Pattern)  # noqa: F811
def _index_dispatch(arg, df, axis):  # noqa: F811
    """
    Base function for selection on a Pandas Index object.
    Applies only to regular expressions.
    `re.compile` is required for the regular expression.

    Returns an array of booleans.
    """
    index = getattr(df, axis)
    if isinstance(index, pd.MultiIndex):
        index = index.get_level_values(0)
    return _select_regex(index, arg)


@_select_index.register(slice)  # noqa: F811
def _index_dispatch(arg, df, axis):  # noqa: F811
    """
    Base function for selection on a Pandas Index object.
    Applies only to slices.

    The start slice value must be a string/tuple/None,
    or exist in the dataframe's columns;
    same goes for the stop slice value.
    The step slice value should be an integer or None.

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

    return _select_callable(df, arg, axis)


@_select_index.register(dict)  # noqa: F811
def _index_dispatch(arg, df, axis):  # noqa: F811
    """
    Base function for selection on a Pandas Index object.
    Applies only to dictionary.

    Returns either a sequence of booleans, an integer,
    or a slice.
    """
    level_label = {}
    index = getattr(df, axis)
    if not isinstance(index, pd.MultiIndex):
        raise TypeError(
            "Index selection with a dictionary "
            "applies only to a MultiIndex."
        )
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
@_select_index.register(ABCPandasArray)  # noqa: F811
@_select_index.register(ABCExtensionArray)  # noqa: F811
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
        return arg
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
                f"No match was returned for any of the labels in {arg}."
            )
        elif not_found.any():
            not_found = set(arg).difference(index)
            raise KeyError(
                f"No match was returned for these labels in {arg} - "
                f"{*not_found,}"
            )
        return arr
    except Exception as exc:
        raise KeyError(f"No match was returned for '{arg}'.") from exc


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

    indices = [_select_index(entry, df, axis) for entry in arg]

    # single entry does not need to be combined
    # or materialized if possible;
    # this offers more performance
    if len(indices) == 1:
        if isinstance(indices[0], int):
            return indices
        if is_list_like(indices[0]):
            return np.asanyarray(indices[0])
        return indices[0]
    contents = []
    for arr in indices:
        if is_list_like(arr):
            arr = np.asanyarray(arr)
        if is_bool_dtype(arr):
            arr = arr.nonzero()[0]
        elif isinstance(arr, slice):
            arr = range(index.size)[arr]
        elif isinstance(arr, int):
            arr = [arr]
        contents.append(arr)
    contents = np.concatenate(contents)
    # remove possible duplicates
    return pd.unique(contents)


def _select(
    df: pd.DataFrame, args: tuple, invert: bool, axis: str
) -> pd.DataFrame:
    """
    Index DataFrame on the index or columns.

    Returns a DataFrame.
    """
    indices = _select_index(list(args), df, axis)
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
