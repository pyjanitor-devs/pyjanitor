from typing import Optional, Union, List, Tuple, Dict, Any
from pandas.core.common import apply_if_callable
import numpy as np
import pandas_flavor as pf
import pandas as pd
import functools
from pandas.api.types import is_list_like, is_scalar

from janitor.utils import check, check_column

from janitor.functions.utils import _computations_expand_grid


@pf.register_dataframe_method
def complete(
    df: pd.DataFrame,
    *columns: Any,
    sort: bool = False,
    by: Optional[Union[list, str]] = None,
    fill_value: Optional[Union[Dict, Any]] = None,
    explicit: bool = True,
) -> pd.DataFrame:
    """Complete a data frame with missing combinations of data.

    It is modeled after tidyr's `complete` function, and is a wrapper around
    [`expand_grid`][janitor.functions.expand_grid.expand_grid], `pd.merge`
    and `pd.fillna`. In a way, it is the inverse of `pd.dropna`, as it exposes
    implicitly missing rows.

    Combinations of column names or a list/tuple of column names, or even a
    dictionary of column names and new values are possible.
    If a dictionary is passed,
    the user is required to ensure that the values are unique 1-D arrays.
    The keys in a dictionary must be present in the dataframe.

    `complete` can also be executed on the names of the index -
    the index should have names
    (`df.index.names` or `df.index.name` should not have any None).
    Groupby is not applicable if `completing` on the index.
    When `completing` on the index, only the passed levels are returned.



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

        Expose missing values via the index:
        >>> index = pd.date_range('1/1/2000', periods=4, freq='T')
        >>> series = pd.Series([0.0, None, 2.0, 3.0], index=index)
        >>> DT = pd.DataFrame({'s': series})
        >>> DT.index.name = 'dates'
        >>> DT
                               s
        dates
        2000-01-01 00:00:00  0.0
        2000-01-01 00:01:00  NaN
        2000-01-01 00:02:00  2.0
        2000-01-01 00:03:00  3.0
        >>> dates = {'dates':lambda f: pd.date_range(f.min(), f.max(), freq='30S')}
        >>> DT.complete(dates)
                               s
        dates
        2000-01-01 00:00:00  0.0
        2000-01-01 00:00:30  NaN
        2000-01-01 00:01:00  NaN
        2000-01-01 00:01:30  NaN
        2000-01-01 00:02:00  2.0
        2000-01-01 00:02:30  NaN
        2000-01-01 00:03:00  3.0

        The above can be solved in a simpler way with `pd.DataFrame.asfreq`:

        >>> DT.asfreq(freq='30S')
                               s
        dates
        2000-01-01 00:00:00  0.0
        2000-01-01 00:00:30  NaN
        2000-01-01 00:01:00  NaN
        2000-01-01 00:01:30  NaN
        2000-01-01 00:02:00  2.0
        2000-01-01 00:02:30  NaN
        2000-01-01 00:03:00  3.0

    Args:
        df: A pandas DataFrame.
        *columns: This refers to the columns to be
            completed. It can also refer to the names of the index.
            It could be column labels (string type),
            a list/tuple of column labels, or a dictionary that pairs
            column labels with new values.
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

    # no copy made of the original dataframe
    # since pd.merge (computed some lines below)
    # makes a new object - essentially a copy
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

    If `by` is present, then `groupby()` is used.

    A DataFrame, with rows of missing values, if any, is returned.
    """
    (
        columns,
        column_checker,
        index,
        sort,
        by,
        fill_value,
        explicit,
    ) = _data_checks_complete(df, columns, sort, by, fill_value, explicit)

    all_scalars = all(map(is_scalar, columns))

    # nothing to 'complete' here
    if (all_scalars and len(columns) == 1) or df.empty:
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
        uniques = _generic_complete(
            df=df,
            columns=columns,
            all_scalars=all_scalars,
            index=index,
            sort=sort,
        )
    else:
        column_checker = by + column_checker
        uniques = df.groupby(by, group_keys=True)
        # apply is basically a for loop
        # for scenarios where Pandas does not have
        # a vectorized option
        uniques = {
            key: _generic_complete(
                df=value,
                columns=columns,
                all_scalars=all_scalars,
                index=index,
                sort=sort,
            )
            for key, value in uniques
        }
        uniques = pd.concat(uniques, names=column_checker, copy=False)
        if not index:
            uniques = uniques.droplevel(-1, axis=0)
    columns = df.columns
    index_labels = df.index.names
    indicator = False
    if fill_value is not None and not explicit:
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

    if fill_value is not None:
        if is_scalar(fill_value):
            # faster when fillna operates on a Series basis
            fill_value = {
                col: fill_value for col in columns if out[col].hasnans
            }
        if explicit:
            out = out.fillna(fill_value, downcast="infer")
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
                boolean_filter = out.loc[:, indicator] == "left_only"
                out = out.drop(columns=indicator)
                # iteration used here,
                # instead of assign (which is also a for loop),
                # to cater for scenarios where the column_name is not a string
                # assign only works with keys that are strings
                # Also, the output wil be floats (for numeric types),
                # even if all the columns could be integers
                # user can always convert to int if required
                for column_name, value in fill_value.items():
                    # for categorical dtypes, set the categories first
                    if isinstance(out[column_name].dtype, pd.CategoricalDtype):
                        out[column_name] = out[column_name].cat.add_categories(
                            [value]
                        )
                    out.loc[boolean_filter, column_name] = value

    if index and (index_labels != out.index.names):
        labels = [label for label in index_labels if label in out.index.names]
        return out.reorder_levels(order=labels, axis="index")
    if not out.columns.equals(columns):
        return out.reindex(columns=columns)
    return out


def _generic_complete(
    df: pd.DataFrame, columns: list, all_scalars: bool, index: bool, sort: bool
):
    """Generate cartesian product for `_computations_complete`.

    Returns a DataFrame, with no duplicates.
    """
    if all_scalars:
        if index:
            uniques = {
                col: pd.factorize(df.index.get_level_values(col), sort=sort)[
                    -1
                ]
                for col in columns
            }
        else:
            uniques = {
                col: pd.factorize(df[col], sort=sort)[-1] for col in columns
            }
        uniques = _computations_expand_grid(uniques)
        if index:
            # it is assured that scalars cannot be single
            # hence a MultiIndex
            uniques = {key[0]: value for key, value in uniques.items()}
            data = list(uniques.values())
            uniques = pd.MultiIndex.from_arrays(data, names=uniques)
            if df.columns.nlevels == 1:
                columns = pd.Index([], name=df.columns.name)
            else:
                length = len(df.columns.names)
                columns = pd.MultiIndex.from_arrays(
                    [[]] * length, names=df.columns.names
                )
            return pd.DataFrame([], index=uniques, columns=columns, copy=False)
        uniques = pd.DataFrame(uniques, copy=False)
        uniques.columns = columns
        return uniques

    uniques = {}
    for ind, column in enumerate(columns):
        if isinstance(column, dict):
            len_columns = len(columns)
            column = _complete_column(column, df=df, index=index, sort=sort)
            # iteration here avoids any potential index collision
            column = {
                ind + len_columns + key: value for key, value in column.items()
            }
            uniques.update(column)
        else:
            uniques[ind] = _complete_column(
                column, df=df, index=index, sort=sort
            )
    uniques = _computations_expand_grid(uniques)
    if index:
        uniques = {key[-1]: value for key, value in uniques.items()}
        if len(uniques) > 1:
            data = list(uniques.values())
            uniques = pd.MultiIndex.from_arrays(data, names=uniques)
        else:
            key = next(iter(uniques))
            data = uniques[key]
            uniques = pd.Index(data, name=key)
        if df.columns.nlevels == 1:
            columns = pd.Index([], name=df.columns.name)
        else:
            length = len(df.columns.names)
            columns = pd.MultiIndex.from_arrays(
                [[]] * length, names=df.columns.names
            )
        return pd.DataFrame([], index=uniques, columns=columns, copy=False)

    uniques = pd.DataFrame(uniques, copy=False)
    uniques.columns = uniques.columns.droplevel(0)
    return uniques


@functools.singledispatch
def _complete_column(column, df, index, sort):
    """
    Args:
        column: scalar/list/dict
        df: Pandas DataFrame
        sort: whether or not to sort the Series.

    Returns:
        A Pandas Series/DataFrame with no duplicates,
        or a dictionary of unique Pandas Series.
    """

    if index:
        _, arr = pd.factorize(df.index.get_level_values(column), sort=sort)
    else:
        _, arr = pd.factorize(df.loc(axis=1)[column], sort=sort)
    return pd.Series(arr, name=column)


@_complete_column.register(list)  # noqa: F811
def _sub_complete_column(column, df, index, sort):  # noqa: F811
    """
    Args:
        column: list
        df: Pandas DataFrame
        sort: whether or not to sort the DataFrame.

    Returns:
        Pandas DataFrame
    """

    if index:
        outcome = df.index
        exclude = [label for label in outcome.names if label not in column]
        if exclude:
            outcome = outcome.droplevel(level=exclude)
        # ideally, there shouldn't be nulls in the index
        exclude = [outcome.get_level_values(label).isna() for label in column]
        exclude = np.column_stack(exclude).all(axis=1)
        if exclude.any():
            outcome = outcome[~exclude]
        _, outcome = pd.factorize(outcome, sort=sort)
        outcome.names = column

    else:
        outcome = df.loc(axis=1)[column]

        exclude = outcome.isna().all(axis=1)

        if exclude.any(axis=None):
            outcome = outcome.loc[~exclude]

        exclude = outcome.duplicated()

        if exclude.any():
            outcome = outcome.loc[~exclude]

        if sort:
            outcome = outcome.sort_values(by=column)

    return outcome


@_complete_column.register(dict)  # noqa: F811
def _sub_complete_column(column, df, index, sort):  # noqa: F811
    """
    Args:
        column: dictionary
        df: Pandas DataFrame
        sort: whether or not to sort the Series.

    Returns:
        A dictionary of unique pandas Series.
    """

    collection = {}
    for ind, (key, value) in enumerate(column.items()):
        if index:
            arr = apply_if_callable(value, df.index.get_level_values(key))
        else:
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

        if sort:
            _, arr = pd.factorize(arr, sort=sort)

        if isinstance(key, tuple):  # handle a MultiIndex column
            arr = pd.DataFrame(arr, columns=pd.MultiIndex.from_tuples([key]))

        else:
            arr = pd.Series(arr, name=key)

        collection[ind] = arr

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

    Returns `df`, `columns`, `column_checker`, `index`, `by`, `fill_value`,
    and `explicit` if all checks pass.
    """

    df_columns = df.columns
    index_labels = (
        df.index.names
        if isinstance(df.index, pd.MultiIndex)
        else [df.index.name]
    )
    if by:
        if not isinstance(by, list):
            by = [by]
        for label in by:
            if label in df_columns:
                continue
            elif label in index_labels:
                index = True
                continue
            else:
                raise ValueError(
                    f"{label} in by is neither in the dataframe's columns, "
                    "nor is it a label in the dataframe's index names."
                )

    columns = [
        [*grouping] if isinstance(grouping, tuple) else grouping
        for grouping in columns
    ]
    column_checker = []
    for grouping in columns:
        if is_scalar(grouping):
            column_checker.append(grouping)
        else:
            check("grouping", grouping, [list, dict])
            if not grouping:
                raise ValueError("entry in columns argument cannot be empty")
            column_checker.extend(grouping)

    # columns should not be duplicated across groups
    column_checker_no_duplicates = set()
    for column in column_checker:
        if column is None:
            raise ValueError("label in the columns argument cannot be None.")
        if column in column_checker_no_duplicates:
            raise ValueError(f"{column} should be in only one group.")
        column_checker_no_duplicates.add(column)  # noqa: PD005

    # columns should either be all in columns
    # or all in index.names/index.name
    # ideally there shouldn't be None in either index names or columns
    index = False
    for label in column_checker:
        if by and (label in by):
            raise ValueError(f"{label} already exists in by.")
        if label in df_columns:
            continue
        elif label in index_labels:
            index = True
            continue
        else:
            raise ValueError(
                f"{label} is neither in the dataframe's columns, "
                "nor is it a label in the dataframe's index names."
            )

    if index:
        for label in column_checker:
            if label not in index_labels:
                raise ValueError(
                    f"{label} not found in the dataframe's index names."
                )

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

    return columns, column_checker, index, sort, by, fill_value, explicit
