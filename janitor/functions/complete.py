from __future__ import annotations

from typing import Any

import pandas as pd
import pandas_flavor as pf
from pandas.api.types import is_scalar

from janitor.utils import check, check_column


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

    It is modeled after tidyr's `complete` function.
    In a way, it is the inverse of `pd.dropna`, as it exposes
    implicitly missing rows.

    The variable `columns` parameter can be a column name,
    a list of column names,
    or a pandas Index, Series, or DataFrame.
    If a pandas Index, Series, or DataFrame is passed, it should
    have a name or names that exist in `df`.

    A callable can also be passed - the callable should evaluate
    to a pandas Index, Series, or DataFrame,
    and the names of the pandas object should exist in `df`.

    A dictionary can also be passed -
    the values of the dictionary should be
    either be a 1D array
    or a callable that evaluates to a
    1D array,
    while the keys of the dictionary
    should exist in `df`.

    User should ensure that the pandas object is unique and/or sorted
    - no checks are done to ensure uniqueness and/or sortedness.

    If `by` is present, the DataFrame is *completed* per group.
    `by` should be a column name, or a list of column names.

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
        >>> df.complete(index, "Taxon", sort=True)
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

        A dictionary can be used as well:
        >>> dictionary = {'Year':range(1999,2005)}
        >>> df.complete(dictionary, "Taxon", sort=True)
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
        ...     ["item_id", "item_name"],
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
        ...     ["item_id", "item_name"],
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
        ...     return pd.RangeIndex(start=df.year.min(), stop=df.year.max() + 1, name='year')
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
            It could be a column name,
            a list of column names,
            or a pandas Index, Series, or DataFrame.

            It can also be a callable that gets evaluated
            to a pandas Index, Series, or DataFrame.

            It can also be a dictionary,
            where the values are either a 1D array
            or a callable that evaluates to a
            1D array,
            while the keys of the dictionary
            should exist in `df`.
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
    return _computations_complete(df, columns, sort, by, fill_value, explicit)


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

    A DataFrame, with rows of missing values, if any, is returned.
    """
    check("explicit", explicit, [bool])

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

    uniques = df.expand(*columns, by=by, sort=sort)
    if by is None:
        merge_columns = uniques.columns.tolist()
    else:
        merge_columns = [*uniques.index.names]
        merge_columns.extend(uniques.columns.tolist())

    columns = df.columns
    if (fill_value is not None) and not explicit:
        # to get a name that does not exist in the columns
        indicator = "".join(columns)
    else:
        indicator = False
    out = pd.merge(
        uniques,
        df,
        how="outer",
        on=merge_columns,
        copy=False,
        sort=False,
        indicator=indicator,
    )
    if indicator:
        indicator = out.pop(indicator)
    if not out.columns.equals(columns):
        out = out.reindex(columns=columns, copy=False)
    if fill_value is None:
        return out
    # keep only columns that are not part of column_checker
    # IOW, we are excluding columns that were not used
    # to generate the combinations
    null_columns = out.columns.difference(merge_columns)
    null_columns = [col for col in null_columns if out[col].hasnans]
    if not null_columns:
        return out
    if is_scalar(fill_value):
        # faster when fillna operates on a Series basis
        fill_value = {col: fill_value for col in null_columns}
    else:
        fill_value = {col: fill_value[col] for col in null_columns}

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
