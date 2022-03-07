from typing import Optional, Union, List, Tuple, Dict, Any
from pandas.core.common import apply_if_callable
from pandas.core.construction import extract_array
import pandas_flavor as pf
import pandas as pd
import functools
from pandas.api.types import is_list_like, is_scalar, is_categorical_dtype

from janitor.utils import check, check_column

from janitor.functions.utils import _computations_expand_grid


@pf.register_dataframe_method
def complete(
    df: pd.DataFrame,
    *columns,
    sort: bool = False,
    by: Optional[Union[list, str]] = None,
    fill_value: Optional[Union[Dict, Any]] = None,
    explicit: bool = True,
) -> pd.DataFrame:
    """
    It is modeled after tidyr's `complete` function, and is a wrapper around
    [`expand_grid`][janitor.functions.expand_grid.expand_grid], `pd.merge`
    and `pd.fillna`. In a way, it is the inverse of `pd.dropna`, as it exposes
    implicitly missing rows.

    Combinations of column names or a list/tuple of column names, or even a
    dictionary of column names and new values are possible.

    MultiIndex columns are not supported.

    Example:

        >>> import pandas as pd
        >>> import janitor
        >>> import numpy as np
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

        >>> df.complete("Year", "Taxon", sort=True)
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
        ...     sort=True
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

    Fill missing values:

        >>> df = pd.DataFrame(
        ...     dict(
        ...         group=(1, 2, 1, 2),
        ...         item_id=(1, 2, 2, 3),
        ...         item_name=("a", "a", "b", "b"),
        ...         value1=(1, np.nan, 3, 4),
        ...         value2=range(4, 8),
        ...     )
        ... )
        >>> df
           group  item_id item_name  value1  value2
        0      1        1         a     1.0       4
        1      2        2         a     NaN       5
        2      1        2         b     3.0       6
        3      2        3         b     4.0       7
        >>> df.complete(
        ...     "group",
        ...     ("item_id", "item_name"),
        ...     fill_value={"value1": 0, "value2": 99},
        ...     sort=True
        ... )
           group  item_id item_name  value1  value2
        0      1        1         a       1       4
        1      1        2         a       0      99
        2      1        2         b       3       6
        3      1        3         b       0      99
        4      2        1         a       0      99
        5      2        2         a       0       5
        6      2        2         b       0      99
        7      2        3         b       4       7

    Limit the fill to only implicit missing values
    by setting explicit to `False`:

        >>> df.complete(
        ...     "group",
        ...     ("item_id", "item_name"),
        ...     fill_value={"value1": 0, "value2": 99},
        ...     explicit=False,
        ...     sort=True
        ... )
           group  item_id item_name  value1  value2
        0      1        1         a     1.0     4.0
        1      1        2         a     0.0    99.0
        2      1        2         b     3.0     6.0
        3      1        3         b     0.0    99.0
        4      2        1         a     0.0    99.0
        5      2        2         a     NaN     5.0
        6      2        2         b     0.0    99.0
        7      2        3         b     4.0     7.0

    :param df: A pandas DataFrame.
    :param *columns: This refers to the columns to be
        completed. It could be column labels (string type),
        a list/tuple of column labels, or a dictionary that pairs
        column labels with new values.
    :param sort: Sort DataFrame based on *columns. Default is `False`.
    :param by: label or list of labels to group by.
        The explicit missing rows are returned per group.
    :param fill_value: Scalar value to use instead of NaN
        for missing combinations. A dictionary, mapping columns names
        to a scalar value is also accepted.
    :param explicit: Determines if only implicitly missing values
        should be filled (`False`), or all nulls existing in the dataframe
        (`True`). Default is `True`. `explicit` is applicable only
        if `fill_value` is not `None`.
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
    if (all_strings and len(columns) == 1) or df.empty:
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
        uniques = _generic_complete(df, columns, all_strings, sort)
    else:
        uniques = df.groupby(by)
        uniques = uniques.apply(_generic_complete, columns, all_strings, sort)
        uniques = uniques.droplevel(-1)
        column_checker = by + column_checker

    columns = df.columns
    indicator = False
    if fill_value is not None and not explicit:
        # to get a name that does not exist in the columns
        indicator = "".join(columns)
    df = pd.merge(
        uniques,
        df,
        how="outer",
        on=column_checker,
        copy=False,
        sort=False,
        indicator=indicator,
    )

    if fill_value is not None:
        if is_scalar(fill_value):
            # faster when fillna operates on a Series basis
            fill_value = {
                col: fill_value for col in columns if df[col].hasnans
            }
        if explicit:
            df = df.fillna(fill_value, downcast="infer")
        else:
            # keep only columns that are not part of column_checker
            # IOW, we are excluding columns that were not used
            # to generate the combinations
            fill_value = {
                col: value
                for col, value in fill_value.items()
                if col not in column_checker
            }
            if fill_value:
                # when explicit is False
                # use the indicator parameter to identify rows
                # for `left_only`, and fill the relevant columns in fill_value
                # with the associated value.
                boolean_filter = df.loc[:, indicator] == "left_only"
                df = df.drop(columns=indicator)
                # iteration used here,
                # instead of assign (which is also a for loop),
                # to cater for scenarios where the column_name is not a string
                # assign only works with keys that are strings
                # Also, the output wil be floats (for numeric types),
                # even if all the columns could be integers
                # user can always convert to int if required
                for column_name, value in fill_value.items():
                    # for categorical dtypes, set the categories first
                    if is_categorical_dtype(df[column_name]):
                        df[column_name] = df[column_name].cat.add_categories(
                            [value]
                        )
                    df.loc[boolean_filter, column_name] = value

    if not df.columns.equals(columns):
        return df.reindex(columns=columns)
    return df


def _generic_complete(
    df: pd.DataFrame, columns: list, all_strings: bool, sort: bool
):
    """
    Generate cartesian product for `_computations_complete`.

    Returns a DataFrame, with no duplicates.
    """
    if all_strings:
        if sort:
            uniques = {}
            for col in columns:
                column = extract_array(df[col], extract_numpy=True)
                _, column = pd.factorize(column, sort=sort)
                uniques[col] = column
        else:
            uniques = {col: df[col].unique() for col in columns}
        uniques = _computations_expand_grid(uniques)
        uniques.columns = columns
        return uniques

    uniques = {}
    df_columns = []
    for index, column in enumerate(columns):
        if not isinstance(column, str):
            df_columns.extend(column)
        else:
            df_columns.append(column)
        if isinstance(column, dict):
            column = _complete_column(column, df, sort)
            uniques = {**uniques, **column}
        else:
            uniques[index] = _complete_column(column, df, sort)

    if len(uniques) == 1:
        _, uniques = uniques.popitem()
        return uniques.to_frame()

    uniques = _computations_expand_grid(uniques)
    uniques.columns = df_columns
    return uniques


@functools.singledispatch
def _complete_column(column: str, df, sort):
    """
    Args:
        column : str/list/dict
        df: Pandas DataFrame
        sort: whether or not to sort the Series.

    A Pandas Series/DataFrame with no duplicates,
    or a dictionary of unique Pandas Series is returned.
    """
    # the cost of checking uniqueness is expensive,
    # especially for large data
    # dirty tests also show that drop_duplicates
    # is faster than pd.unique for fairly large data

    column = df[column]
    dupes = column.duplicated()

    if dupes.any():
        column = column[~dupes]

    if sort and not column.is_monotonic_increasing:
        column = column.sort_values()

    return column


@_complete_column.register(list)  # noqa: F811
def _sub_complete_column(column, df, sort):  # noqa: F811
    """
    Args:
        column : list
        df: Pandas DataFrame
        sort: whether or not to sort the DataFrame.

    Returns:
        Pandas DataFrame
    """

    outcome = df.loc[:, column]
    dupes = outcome.duplicated()

    if dupes.any():
        outcome = outcome.loc[~dupes]

    if sort:
        outcome = outcome.sort_values(by=column)

    return outcome


@_complete_column.register(dict)  # noqa: F811
def _sub_complete_column(column, df, sort):  # noqa: F811
    """
    Args:
        column : dictionary
        df: Pandas DataFrame
        sort: whether or not to sort the Series.

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

        dupes = arr.duplicated()

        if dupes.any():
            arr = arr[~dupes]

        if sort and not arr.is_monotonic_increasing:
            arr = arr.sort_values()

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
