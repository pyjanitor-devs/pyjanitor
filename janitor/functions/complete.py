from typing import Optional, Union, List, Tuple, Dict, Any
from pandas.core.common import apply_if_callable
import pandas_flavor as pf
import pandas as pd
import functools
from pandas.api.types import is_list_like, is_scalar

from janitor.utils import check, check_column

from janitor.functions.utils import _computations_expand_grid


@pf.register_dataframe_method
def complete(
    df: pd.DataFrame,
    *columns,
    sort: bool = True,
    by: Optional[Union[list, str]] = None,
    fill_value: Optional[Union[Dict, Any]] = None,
    explicit: bool = True,
) -> pd.DataFrame:
    """
    It is modeled after tidyr's `complete` function, and is a wrapper around
    [`expand_grid`][janitor.functions.expand_grid.expand_grid], `pd.merge`
    and `pd.fillna`.

    Combinations of column names or a list/tuple of column names, or even a
    dictionary of column names and new values are possible.

    MultiIndex columns are not supported.

    Example:

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame(
        ...     {
        ...         "Year": [1999, 2000, 2004, 1999, 2004],
        ...         "Taxon": [
        ...             "Saccharina",
        ...             "Saccharina",
        ...             "Saccharina",
        ...             "Agarum",
        ...             "Agarum",
        ...         ],
        ...         "Abundance": [4, 5, 2, 1, 8],
        ...     }
        ... )
        >>> df
           Year       Taxon  Abundance
        0  1999  Saccharina          4
        1  2000  Saccharina          5
        2  2004  Saccharina          2
        3  1999      Agarum          1
        4  2004      Agarum          8

    Expose missing pairings of `Year` and `Taxon`:

        >>> df.complete("Year", "Taxon")
           Year       Taxon  Abundance
        0  1999      Agarum        1.0
        1  1999  Saccharina        4.0
        2  2000      Agarum        NaN
        3  2000  Saccharina        5.0
        4  2004      Agarum        8.0
        5  2004  Saccharina        2.0

    Expose missing years from 1999 to 2004 :

        >>> df.complete(
        ...     {"Year": range(df.Year.min(), df.Year.max() + 1)},
        ...     "Taxon",
        ... )
            Year       Taxon  Abundance
        0   1999      Agarum        1.0
        1   1999  Saccharina        4.0
        2   2000      Agarum        NaN
        3   2000  Saccharina        5.0
        4   2001      Agarum        NaN
        5   2001  Saccharina        NaN
        6   2002      Agarum        NaN
        7   2002  Saccharina        NaN
        8   2003      Agarum        NaN
        9   2003  Saccharina        NaN
        10  2004      Agarum        8.0
        11  2004  Saccharina        2.0

    :param df: A pandas DataFrame.
    :param *columns: This refers to the columns to be
        completed. It could be column labels (string type),
        a list/tuple of column labels, or a dictionary that pairs
        column labels with new values.
    :param sort: Sort DataFrame based on *columns. Default is `True`.
    :param by: label or list of labels to group by.
        The explicit missing rows are returned per group.
    :param fill_value: Scalar value to use instead of NaN
        for missing combinations. A dictionary, mapping columns names
        to a scalar value is also accepted.
    :param explicit: Determines if only implicitly missing values
        should be filled (`False`), or all nulls existing in the dataframe
        (`True`). Default is `True`.
    :returns: A pandas DataFrame with explicit missing rows, if any.
    """

    if not columns:
        return df

    df = df.copy()

    return _computations_complete(df, columns, sort, by, fill_value, explicit)


def _computations_complete(
    df: pd.DataFrame,
    columns: List[Union[List, Tuple, Dict, str]],
    sort: bool,
    by: Optional[Union[list, str]],
    fill_value: Optional[Union[Dict, Any]],
    explicit: bool,
) -> pd.DataFrame:
    """
    This function computes the final output for the `complete` function.

    If `by` is present, then `groupby().apply()` is used.

    A DataFrame, with rows of missing values, if any, is returned.
    """

    (
        columns,
        column_checker,
        sort,
        by,
        fill_value,
        explicit,
    ) = _data_checks_complete(df, columns, sort, by, fill_value, explicit)

    all_strings = True
    for column in columns:
        if not isinstance(column, str):
            all_strings = False
            break

    # nothing to 'complete' here
    if all_strings and len(columns) == 1:
        return df

    # under the right conditions, stack/unstack can be faster
    # plus it always returns a sorted DataFrame
    # which does help in viewing the missing rows
    # however, using a merge keeps things simple
    # with a stack/unstack,
    # the relevant columns combination should be unique
    # and there should be no nulls
    # trade-off for the simplicity of merge is not so bad
    # of course there could be a better way ...
    if by is None:
        uniques = _generic_complete(df, columns, all_strings)
    else:
        uniques = df.groupby(by)
        uniques = uniques.apply(_generic_complete, columns, all_strings)
        uniques = uniques.droplevel(-1)
        column_checker = by + column_checker

    if fill_value is None:
        return df.merge(
            uniques, on=column_checker, how="outer", sort=sort, copy=False
        )
    # if fill_value is present
    if is_scalar(fill_value):
        # faster when fillna operates on a Series basis
        fill_value = {col: fill_value for col in df}
    if explicit:
        return df.merge(
            uniques, on=column_checker, how="outer", sort=sort, copy=False
        ).fillna(fill_value, downcast="infer")
    # keep only columns that are not part of column_checker
    # IOW, we are excluding columns that were not used
    # to generate the combinations
    fill_value = {
        col: value
        for col, value in fill_value.items()
        if col not in column_checker
    }
    if not fill_value:  # there is nothing to fill
        return df.merge(
            uniques, on=column_checker, how="outer", sort=sort, copy=False
        )

    # when explicit is False
    # filter out rows from `unique` that already exist in the parent dataframe
    # fill the null values in the trimmed `unique`
    # and merge back to the main dataframe

    # to get a name that does not exist in the columns
    indicator = "".join(column_checker)
    trimmed = df.loc(axis=1)[column_checker]
    uniques = uniques.merge(
        trimmed, how="left", sort=False, copy=False, indicator=indicator
    )
    trimmed = uniques.iloc(axis=1)[-1] == "left_only"
    uniques = uniques.loc[trimmed, column_checker]
    trimmed = None
    indicator = None
    # iteration used here, instead of assign (which is also a for loop),
    # to cater for scenarios where the column_name is not a string
    # assign only works with keys that are strings
    for column_name, value in fill_value.items():
        uniques[column_name] = value
    df = pd.concat([df, uniques], sort=False, copy=False, axis="index")
    if sort:
        df = df.sort_values(column_checker)
    df.index = range(len(df))
    return df


def _generic_complete(df: pd.DataFrame, columns: list, all_strings: bool):
    """
    Generate cartesian product for `_computations_complete`.

    Returns a Series or DataFrame, with no duplicates.
    """
    if all_strings:
        uniques = {col: df[col].unique() for col in columns}
        uniques = _computations_expand_grid(uniques)
        uniques = uniques.droplevel(level=-1, axis="columns")
        return uniques

    uniques = {}
    for index, column in enumerate(columns):
        if isinstance(column, dict):
            column = _complete_column(column, df)
            uniques = {**uniques, **column}
        else:
            uniques[index] = _complete_column(column, df)

    if len(uniques) == 1:
        _, uniques = uniques.popitem()
        return uniques.to_frame()

    uniques = _computations_expand_grid(uniques)
    return uniques.droplevel(level=0, axis="columns")


@functools.singledispatch
def _complete_column(column, df):
    """
    Args:
        column : str/list/dict
        df: Pandas DataFrame

    A Pandas Series/DataFrame with no duplicates,
    or a list of unique Pandas Series is returned.
    """
    raise TypeError("This type is not supported in the `complete` function.")


@_complete_column.register(str)  # noqa: F811
def _sub_complete_column(column, df):  # noqa: F811
    """
    Args:
        column : str
        df: Pandas DataFrame

    Returns:
        Pandas Series
    """

    column = df[column]

    if not column.is_unique:
        return column.drop_duplicates()
    return column


@_complete_column.register(list)  # noqa: F811
def _sub_complete_column(column, df):  # noqa: F811
    """
    Args:
        column : list
        df: Pandas DataFrame

    Returns:
        Pandas DataFrame
    """

    column = df.loc[:, column]

    if column.duplicated().any(axis=None):
        return column.drop_duplicates()

    return column


@_complete_column.register(dict)  # noqa: F811
def _sub_complete_column(column, df):  # noqa: F811
    """
    Args:
        column : dictionary
        df: Pandas DataFrame

    Returns:
        A dictionary of unique pandas Series.
    """

    collection = {}
    for key, value in column.items():
        arr = apply_if_callable(value, df[key])
        if not is_list_like(arr):
            raise ValueError(f"value for {key} should be a 1-D array.")
        if not hasattr(arr, "shape"):
            arr = pd.Series([*arr], name=key)

        if not arr.size > 0:
            raise ValueError(
                f"Kindly ensure the provided array for {key} "
                "has at least one value."
            )

        if isinstance(arr, pd.Index):
            arr_ndim = arr.nlevels
        else:
            arr_ndim = arr.ndim

        if arr_ndim != 1:
            raise ValueError(f"Kindly provide a 1-D array for {key}.")

        if not isinstance(arr, pd.Series):
            arr = pd.Series(arr)

        if not arr.is_unique:
            arr = arr.drop_duplicates()

        arr.name = key

        collection[key] = arr

    return collection


def _data_checks_complete(
    df: pd.DataFrame,
    columns: List[Union[List, Tuple, Dict, str]],
    sort: Optional[bool],
    by: Optional[Union[list, str]],
    fill_value: Optional[Union[Dict, Any]],
    explicit: bool,
):
    """
    Function to check parameters in the `complete` function.
    Checks the type of the `columns` parameter, as well as the
    types within the `columns` parameter.

    Check is conducted to ensure that column names are not repeated.
    Also checks that the names in `columns` actually exist in `df`.

    Returns `df`, `columns`, `column_checker`, `by`, `fill_value`,
    and `explicit` if all checks pass.
    """
    # TODO: get `complete` to work on MultiIndex columns,
    # if there is sufficient interest with use cases
    if isinstance(df.columns, pd.MultiIndex):
        raise ValueError("`complete` does not support MultiIndex columns.")

    columns = [
        [*grouping] if isinstance(grouping, tuple) else grouping
        for grouping in columns
    ]
    column_checker = []
    for grouping in columns:
        check("grouping", grouping, [list, dict, str])
        if not grouping:
            raise ValueError("grouping cannot be empty")
        if isinstance(grouping, str):
            column_checker.append(grouping)
        else:
            column_checker.extend(grouping)

    # columns should not be duplicated across groups
    column_checker_no_duplicates = set()
    for column in column_checker:
        if column in column_checker_no_duplicates:
            raise ValueError(f"{column} column should be in only one group.")
        column_checker_no_duplicates.add(column)  # noqa: PD005

    check_column(df, column_checker)
    column_checker_no_duplicates = None

    check("sort", sort, [bool])

    if by is not None:
        if isinstance(by, str):
            by = [by]
        check("by", by, [list])
        check_column(df, by)

    check("explicit", explicit, [bool])

    fill_value_check = is_scalar(fill_value), isinstance(fill_value, dict)
    if not any(fill_value_check):
        raise TypeError(
            "`fill_value` should either be a dictionary or a scalar value."
        )
    if fill_value_check[-1]:
        check_column(df, fill_value)
        for column_name, value in fill_value.items():
            if not is_scalar(value):
                raise ValueError(
                    f"The value for {column_name} should be a scalar."
                )

    return columns, column_checker, sort, by, fill_value, explicit
