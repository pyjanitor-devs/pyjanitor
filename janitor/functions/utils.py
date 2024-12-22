"""Utility functions for all of the functions submodule."""

from __future__ import annotations

import re
import unicodedata
import warnings
from typing import (
    Any,
    Hashable,
    Iterable,
    List,
    Optional,
    Pattern,
    Union,
)

import numpy as np
import pandas as pd
from multipledispatch import dispatch
from pandas.api.types import (
    is_list_like,
    is_scalar,
    is_string_dtype,
    union_categoricals,
)

from janitor.errors import JanitorError
from janitor.utils import (
    _expand_grid,
    check,
    check_column,
    find_stack_level,
)

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


def _change_case(
    obj: str,
    case_type: str,
) -> str:
    """Change case of obj."""
    case_types = {"preserve", "upper", "lower", "snake"}
    case_type = case_type.lower()
    if case_type not in case_types:
        raise JanitorError(f"type must be one of: {case_types}")

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


def _normalize_1(obj: str) -> str:
    """Perform normalization of obj."""
    FIXES = [(r"[ /:,?()\.-]", "_"), (r"['’]", ""), (r"[\xa0]", "_")]
    for search, replace in FIXES:
        obj = re.sub(pattern=search, repl=replace, string=obj)

    return obj


def _remove_special(
    obj: str,
) -> str:
    """Remove special characters from obj."""
    obj = [item for item in obj if item.isalnum() or (item == "_")]
    return "".join(obj)


def _strip_accents(
    obj: str,
) -> str:
    """Remove accents from obj.

    Inspired from [StackOverflow][so].

    [so]: https://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-in-a-python-unicode-strin
    """  # noqa: E501

    obj = [
        letter
        for letter in unicodedata.normalize("NFD", obj)
        if not unicodedata.combining(letter)
    ]
    return "".join(obj)


def _strip_underscores_func(
    obj: str,
    strip_underscores: Union[str, bool] = None,
) -> str:
    """Strip underscores from obj."""
    underscore_options = {None, "left", "right", "both", "l", "r", True}
    if strip_underscores not in underscore_options:
        raise JanitorError(
            f"strip_underscores must be one of: {underscore_options}"
        )

    if strip_underscores in {"left", "l"}:
        return obj.lstrip("_")
    if strip_underscores in {"right", "r"}:
        return obj.rstrip("_")
    if strip_underscores in {True, "both"}:
        return obj.strip("_")
    return obj


def _is_str_or_cat(index):
    """
    Check if the column/index is a string,
    or categorical with strings.
    """
    if isinstance(index.dtype, pd.CategoricalDtype):
        return is_string_dtype(index.categories)
    return is_string_dtype(index)
