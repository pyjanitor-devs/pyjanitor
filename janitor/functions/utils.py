"""Utility functions for all of the functions submodule."""

import re
from enum import Enum
from typing import Hashable, Iterable, List, Optional, Pattern, Union

import pandas as pd
from janitor.utils import check, _expand_grid
from pandas.api.types import (
    union_categoricals,
    is_scalar,
    is_extension_array_dtype,
    is_list_like,
)
import numpy as np
from multipledispatch import dispatch
from janitor.utils import check_column


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

    :param regex_pattern: string to be converted to compiled regular
        expression.
    :returns: A compile regular expression from provided
        `regex_pattern`.
    """
    check("regular expression", regex_pattern, [str, Pattern])

    return re.compile(regex_pattern)


class FILLTYPE(Enum):
    """List of fill types for fill_direction."""

    UP = ("bfill",)
    DOWN = ("ffill",)
    UPDOWN = "bfill", "ffill"
    DOWNUP = "ffill", "bfill"


def _computations_expand_grid(others: dict) -> pd.DataFrame:
    """
    Creates a cartesian product of all the inputs in `others`.
    Combines NumPy's `mgrid`, with the `take` method in NumPy/pandas
    to expand each input to the length of the cumulative product of
    all inputs in `others`.

    There is a performance penalty for small entries
    (lenght less than 10)
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
            value = pd.DataFrame([value])
        elif (not isinstance(value, pd.Series)) and is_extension_array_dtype(
            value
        ):
            value = pd.DataFrame(value)
        elif is_list_like(value) and (not hasattr(value, "shape")):
            value = np.asarray([*value])

        grid[key] = value

    others = None

    # slice obtained here is used in `np.mgrid`
    # to generate cartesian indices
    # which is then paired with grid.items()
    # to blow up each individual value
    # before finally recombining, via pd.concat,
    # to create a dataframe.
    grid_index = [slice(len(value)) for _, value in grid.items()]
    grid_index = np.mgrid[grid_index]
    grid_index = map(np.ravel, grid_index)
    grid = zip(grid.items(), grid_index)
    grid = ((*left, right) for left, right in grid)
    grid = {
        key: _expand_grid(value, grid_index) for key, value, grid_index in grid
    }

    # creates a MultiIndex with the keys
    # since grid is a dictionary
    return pd.concat(grid, axis="columns", sort=False, copy=False)


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
