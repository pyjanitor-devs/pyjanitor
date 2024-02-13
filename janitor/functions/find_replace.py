"""Implementation for find_replace."""

from typing import Any, Dict

import pandas as pd
import pandas_flavor as pf

from janitor.utils import refactored_function


@pf.register_dataframe_method
@refactored_function(
    message=(
        "This function will be deprecated in a 1.x release. "
        "Please use `pd.DataFrame.replace` instead."
    )
)
def find_replace(
    df: pd.DataFrame, match: str = "exact", **mappings: Any
) -> pd.DataFrame:
    """Perform a find-and-replace action on provided columns.

    !!!note

        This function will be deprecated in a 1.x release.
        Please use `pd.DataFrame.replace` instead.

    Depending on use case, users can choose either exact, full-value matching,
    or regular-expression-based fuzzy matching
    (hence allowing substring matching in the latter case).
    For strings, the matching is always case sensitive.

    Examples:
        For instance, given a DataFrame containing orders at a coffee shop:

        >>> df = pd.DataFrame({
        ...     "customer": ["Mary", "Tom", "Lila"],
        ...     "order": ["ice coffee", "lemonade", "regular coffee"]
        ... })
        >>> df
          customer           order
        0     Mary      ice coffee
        1      Tom        lemonade
        2     Lila  regular coffee

        Our task is to replace values `ice coffee` and `regular coffee`
        of the `order` column into `latte`.

        Example 1 - exact matching (functional usage):

        >>> df = find_replace(
        ...     df,
        ...     match="exact",
        ...     order={"ice coffee": "latte", "regular coffee": "latte"},
        ... )
        >>> df
          customer     order
        0     Mary     latte
        1      Tom  lemonade
        2     Lila     latte

        Example 1 - exact matching (method chaining):

        >>> df = df.find_replace(
        ...     match="exact",
        ...     order={"ice coffee": "latte", "regular coffee": "latte"},
        ... )
        >>> df
          customer     order
        0     Mary     latte
        1      Tom  lemonade
        2     Lila     latte

        Example 2 - Regular-expression-based matching (functional usage):

        >>> df = find_replace(
        ...     df,
        ...     match='regex',
        ...     order={'coffee$': 'latte'},
        ... )
        >>> df
          customer     order
        0     Mary     latte
        1      Tom  lemonade
        2     Lila     latte

        Example 2 - Regular-expression-based matching (method chaining usage):

        >>> df = df.find_replace(
        ...     match='regex',
        ...     order={'coffee$': 'latte'},
        ... )
        >>> df
          customer     order
        0     Mary     latte
        1      Tom  lemonade
        2     Lila     latte

    To perform a find and replace on the entire DataFrame,
    pandas' `df.replace()` function provides the appropriate functionality.
    You can find more detail on the [replace] docs.

    [replace]: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html

    This function only works with column names that have no spaces
    or punctuation in them.
    For example, a column name `item_name` would work with `find_replace`,
    because it is a contiguous string that can be parsed correctly,
    but `item name` would not be parsed correctly by the Python interpreter.

    If you have column names that might not be compatible,
    we recommend calling on [`clean_names()`][janitor.functions.clean_names.clean_names]
    as the first method call. If, for whatever reason, that is not possible,
    then `_find_replace` is available as a function
    that you can do a pandas [pipe](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.pipe.html) call on.

    Args:
        df: A pandas DataFrame.
        match: Whether or not to perform an exact match or not.
            Valid values are "exact" or "regex".
        **mappings: keyword arguments corresponding to column names
            that have dictionaries passed in indicating what to find (keys)
            and what to replace with (values).

    Returns:
        A pandas DataFrame with replaced values.
    """  # noqa: E501
    for column_name, mapper in mappings.items():
        df = _find_replace(df, column_name, mapper, match=match)
    return df


def _find_replace(
    df: pd.DataFrame, column_name: str, mapper: Dict, match: str = "exact"
) -> pd.DataFrame:
    """Utility function for `find_replace`.

    The code in here was the original implementation of `find_replace`,
    but we decided to change out the front-facing API to accept
    kwargs + dictionaries for readability,
    and instead dispatch underneath to this function.
    This implementation was kept
    because it has a number of validations that are quite useful.

    Args:
        df: A pandas DataFrame.
        column_name: The column on which the find/replace action is to be
            made. Must be a string.
        mapper: A dictionary that maps
            `thing to find` -> `thing to replace`.
            Note: Does not support null-value replacement.
        match: A string that dictates whether exact match or
            regular-expression-based fuzzy match will be used for finding
            patterns. Default to `exact`. Can only be `exact` or `regex`.

    Raises:
        ValueError: Is trying to use null replacement. Kindly use
            `.fillna()` instead.
        ValueError: If `match` is not one of `exact` or `regex`.

    Returns:
        A pandas DataFrame.
    """
    if any(map(pd.isna, mapper.keys())):
        raise ValueError(
            "find_replace() does not support null replacement. "
            "Use DataFrame.fillna() instead."
        )
    if match.lower() not in ("exact", "regex"):
        raise ValueError("`match` can only be 'exact' or 'regex'.")

    if match.lower() == "exact":
        df[column_name] = df[column_name].apply(lambda x: mapper.get(x, x))
    if match.lower() == "regex":
        for k, v in mapper.items():
            condition = df[column_name].str.contains(k, regex=True)
            df.loc[condition, column_name] = v
    return df
