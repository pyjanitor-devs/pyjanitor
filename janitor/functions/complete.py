from __future__ import annotations

import warnings
from typing import Any

import pandas as pd
import pandas_flavor as pf
from pandas.api.types import is_list_like, is_scalar
from pandas.core.common import apply_if_callable

from janitor.functions.utils import _computations_expand_grid
from janitor.utils import check, check_column, find_stack_level

warnings.simplefilter("always", UserWarning)


@pf.register_dataframe_method
def complete(
    df: pd.DataFrame,
    *columns: Any,
    sort: bool = False,
    by: str | list = None,
    fill_value: dict | Any = None,
    explicit: bool = True,
) -> pd.DataFrame:
    """
    Complete a data frame with missing combinations of data.

    It is modeled after tidyr's `complete` function, and is a wrapper around
    [`expand_grid`][janitor.functions.expand_grid.expand_grid], `pd.merge`
    and `pd.fillna`. In a way, it is the inverse of `pd.dropna`, as it exposes
    implicitly missing rows.

    The variable `columns` parameter can be a combination
    of column names or a list/tuple of column names,
    or a pandas Index, Series, or DataFrame.
    If a pandas Index, Series, or DataFrame is passed, it should
    have a name or names that exist in `df`.
    A callable can also be passed - the callable should evaluate
    to a pandas Index, Series, or DataFrame.
    User should ensure that the pandas object is unique - no checks are done
    to ensure uniqueness.

    Examples:
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

        Expose missing years from 1999 to 2004:
        >>> index = pd.Index(range(1999,2005),name='Year')
        >>> df.complete(
        ...     index,
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
        0      1        1         a     1.0     4.0
        1      1        2         a     0.0    99.0
        2      1        2         b     3.0     6.0
        3      1        3         b     0.0    99.0
        4      2        1         a     0.0    99.0
        5      2        2         a     0.0     5.0
        6      2        2         b     0.0    99.0
        7      2        3         b     4.0     7.0

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

        Expose missing rows per group, using a callable:
        >>> df = pd.DataFrame(
        ...     {
        ...         "state": ["CA", "CA", "HI", "HI", "HI", "NY", "NY"],
        ...         "year": [2010, 2013, 2010, 2012, 2016, 2009, 2013],
        ...         "value": [1, 3, 1, 2, 3, 2, 5],
        ...     }
        ... )
        >>> df
          state  year  value
        0    CA  2010      1
        1    CA  2013      3
        2    HI  2010      1
        3    HI  2012      2
        4    HI  2016      3
        5    NY  2009      2
        6    NY  2013      5

        >>> def new_year_values(df):
        ...     values = range(df.year.min(), df.year.max() + 1)
        ...     values = pd.Series(values, name="year")
        ...     return values

        >>> df.complete(new_year_values, by='state',sort=True)
            state  year  value
        0     CA  2010    1.0
        1     CA  2011    NaN
        2     CA  2012    NaN
        3     CA  2013    3.0
        4     HI  2010    1.0
        5     HI  2011    NaN
        6     HI  2012    2.0
        7     HI  2013    NaN
        8     HI  2014    NaN
        9     HI  2015    NaN
        10    HI  2016    3.0
        11    NY  2009    2.0
        12    NY  2010    NaN
        13    NY  2011    NaN
        14    NY  2012    NaN
        15    NY  2013    5.0

    Args:
        df: A pandas DataFrame.
        *columns: This refers to the columns to be completed.
            It could be column labels (string type),
            a list/tuple of column labels,
            or a pandas Index, Series, or DataFrame.
            It can also be a callable that gets evaluated
            to a padnas Index, Series, or DataFrame.
        sort: Sort DataFrame based on *columns.
        by: Label or list of labels to group by.
            The explicit missing rows are returned per group.
        fill_value: Scalar value to use instead of NaN
            for missing combinations. A dictionary, mapping columns names
            to a scalar value is also accepted.
        explicit: Determines if only implicitly missing values
            should be filled (`False`), or all nulls existing in the dataframe
            (`True`). `explicit` is applicable only
            if `fill_value` is not `None`.

    Returns:
        A pandas DataFrame with explicit missing rows, if any.
    """  # noqa: E501

    if not columns:
        return df

    # no copy is made of the original dataframe
    # since pd.merge (computed some lines below)
    # makes a new object - essentially a copy
    return _computations_complete(df, columns, sort, by, fill_value, explicit)


def _create_cartesian_dataframe(df, columns, column_checker, sort):
    """
    Create a DataFrame from the
    combination of all pandas objects
    """
    objects = _create_pandas_object(df, columns=columns, sort=sort)
    objects = dict(zip(range(len(objects)), objects))
    objects = _computations_expand_grid(objects)
    objects = dict(zip(column_checker, objects.values()))
    objects = pd.DataFrame(objects, copy=False)
    return objects


def _computations_complete(
    df: pd.DataFrame,
    columns: list | tuple | dict | str,
    sort: bool,
    by: list | str,
    fill_value: dict | Any,
    explicit: bool,
) -> pd.DataFrame:
    """
    This function computes the final output for the `complete` function.

    If `by` is present, then `groupby()` is used.

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

    if by is None:
        uniques = _create_cartesian_dataframe(
            df=df, column_checker=column_checker, columns=columns, sort=sort
        )
    else:
        grouped = df.groupby(by, sort=False)
        uniques = {}
        for group_name, frame in grouped:
            _object = _create_cartesian_dataframe(
                df=frame,
                column_checker=column_checker,
                columns=columns,
                sort=sort,
            )
            uniques[group_name] = _object
        column_checker = by + column_checker
        by.append("".join(column_checker))
        uniques = pd.concat(uniques, names=by, copy=False, sort=False)
        uniques = uniques.droplevel(axis=0, level=-1)
    columns = df.columns
    indicator = False
    if (fill_value is not None) and not explicit:
        # to get a name that does not exist in the columns
        indicator = "".join(columns)
    out = pd.merge(
        uniques,
        df,
        how="outer",
        on=column_checker,
        copy=False,
        sort=False,
        indicator=indicator,
    )
    if indicator:
        indicator = out.pop(indicator)
    if not out.columns.equals(columns):
        out = out.loc[:, columns]
    if fill_value is None:
        return out
    # keep only columns that are not part of column_checker
    # IOW, we are excluding columns that were not used
    # to generate the combinations
    null_columns = [
        col for col in out if out[col].hasnans and col not in column_checker
    ]
    if not null_columns:
        return out
    if is_scalar(fill_value):
        # faster when fillna operates on a Series basis
        fill_value = {col: fill_value for col in null_columns}
    else:
        fill_value = {
            col: _fill_value
            for col, _fill_value in fill_value.items()
            if col in null_columns
        }
    if not fill_value:
        return out
    if explicit:
        return out.fillna(fill_value)
    # when explicit is False
    # use the indicator parameter to identify rows
    # for `left_only`, and fill the relevant columns
    # in fill_value with the associated value.
    boolean_filter = indicator == "left_only"
    # iteration used here,
    # instead of assign (which is also a for loop),
    # to cater for scenarios where the column_name
    # is not a string
    # assign only works with keys that are strings
    # Also, the output wil be floats (for numeric types),
    # even if all the columns could be integers
    # user can always convert to int if required
    for column_name, value in fill_value.items():
        # for categorical dtypes, set the categories first
        if isinstance(out[column_name].dtype, pd.CategoricalDtype):
            out[column_name] = out[column_name].cat.add_categories([value])
        out.loc[boolean_filter, column_name] = value

    return out


def _data_checks_complete(
    df: pd.DataFrame,
    columns: list | tuple | dict | str,
    sort: bool,
    by: list | str,
    fill_value: dict | Any,
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

    if by:
        if not isinstance(by, list):
            by = [by]
        check_column(df, column_names=by, present=True)

    columns = [
        [*grouping] if isinstance(grouping, tuple) else grouping
        for grouping in columns
    ]

    def _check_pandas_object(grouping, column_checker):
        """
        Check if object is a pandas object.
        """
        if isinstance(grouping, pd.DataFrame):
            column_checker.extend(grouping.columns)
        elif isinstance(grouping, pd.MultiIndex):
            if None in grouping.names:
                raise ValueError(
                    "Ensure all labels in the MultiIndex are named."
                )
            column_checker.extend(grouping.names)
        elif isinstance(grouping, (pd.Series, pd.Index)):
            if not grouping.name:
                name_of_type = type(grouping).__name__
                raise ValueError(f"Ensure the {name_of_type} has a name.")
            column_checker.append(grouping.name)
        else:
            grouping = None
        return grouping, column_checker

    column_checker = []
    for grouping in columns:
        if is_scalar(grouping):
            column_checker.append(grouping)
        elif isinstance(grouping, list):
            if not grouping:
                raise ValueError("entry in columns argument cannot be empty")
            column_checker.extend(grouping)
        elif isinstance(grouping, dict):
            warnings.warn(
                "A dictionary argument is no longer supported, "
                "and will be deprecated in the next pyjanitor release. "
                "Instead, pass a pandas Index, a Series or a DataFrame.",
                DeprecationWarning,
                stacklevel=find_stack_level(),
            )
            if not grouping:
                raise ValueError("entry in columns argument cannot be empty")
            column_checker.extend(grouping)
        elif callable(grouping):
            grouping = apply_if_callable(
                maybe_callable=grouping,
                obj=df.iloc[:1],
            )
            _grouping, column_checker = _check_pandas_object(
                grouping=grouping, column_checker=column_checker
            )
            if _grouping is None:
                raise TypeError(
                    "The callable should evaluate to either "
                    "a pandas DataFrame, Index, or Series; "
                    f"instead got {type(grouping)}."
                )
        elif isinstance(grouping, (pd.DataFrame, pd.Index, pd.Series)):
            grouping, column_checker = _check_pandas_object(
                grouping=grouping, column_checker=column_checker
            )
        else:
            raise TypeError(
                "The complete function expects a scalar, a list/tuple, "
                "a pandas Index, Series, DataFrame, "
                "or a callable that returns "
                "a pandas Index, Series, or DataFrame"
                f"instead, got {type(grouping)}"
            )

    # columns should not be duplicated across groups
    # nor should it exist in `by`
    column_checker_no_duplicates = set()
    for column in column_checker:
        if column is None:
            raise ValueError("label in the columns argument cannot be None.")
        if column in column_checker_no_duplicates:
            raise ValueError(f"{column} should be in only one group.")
        if by and (column in by):
            raise ValueError(
                f"{column} already exists as a label in the `by` argument."
            )
        column_checker_no_duplicates.add(column)  # noqa: PD005

    check_column(df, column_names=column_checker, present=True)

    check("explicit", explicit, [bool])

    column_checker_no_duplicates = None

    check("sort", sort, [bool])

    fill_value_check = is_scalar(fill_value), isinstance(fill_value, dict)
    if not any(fill_value_check):
        raise TypeError(
            "fill_value should either be a dictionary or a scalar value."
        )
    if fill_value_check[-1]:
        check_column(df, fill_value)
        for column_name, value in fill_value.items():
            if not is_scalar(value):
                raise ValueError(
                    f"The value for {column_name} should be a scalar."
                )

    return columns, column_checker, sort, by, fill_value, explicit


def _create_pandas_objects_from_dict(df, column, sort):
    """
    Create pandas object if column is a dictionary
    """
    collection = []
    for key, value in column.items():
        arr = apply_if_callable(value, df)
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

        if sort:
            _, arr = pd.factorize(arr, sort=sort)

        if isinstance(key, tuple):  # handle a MultiIndex column
            arr = pd.DataFrame(arr, columns=pd.MultiIndex.from_tuples([key]))

        else:
            arr = pd.Series(arr, name=key)

        collection.append(arr)

    return collection


def _create_pandas_object(df, columns, sort):
    """
    Create pandas objects before building the cartesian DataFrame.
    """
    objects = []
    for column in columns:
        if is_scalar(column):
            _object = df[column].drop_duplicates()
            if sort:
                _object = _object.sort_values()
            objects.append(_object)
        elif isinstance(column, list):
            _object = df.loc[:, column].drop_duplicates()
            if sort:
                _object = _object.sort_values(column)
            objects.append(_object)
        elif isinstance(column, dict):
            _object = _create_pandas_objects_from_dict(
                df=df, column=column, sort=sort
            )
            objects.extend(_object)
        else:
            _object = apply_if_callable(maybe_callable=column, obj=df)
            objects.append(_object)
    return objects
