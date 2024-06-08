"""Implementation source for `expand_grid`."""

from __future__ import annotations

from collections import defaultdict

import pandas as pd
import pandas_flavor as pf
from pandas.api.types import is_scalar
from pandas.core.common import apply_if_callable
from pandas.core.dtypes.concat import concat_compat

from janitor.functions.utils import _computations_expand_grid
from janitor.utils import check, check_column, deprecated_kwargs

msg = "df_key is deprecated. The column names "
msg += "of the DataFrame will be used instead."


@pf.register_dataframe_method
@deprecated_kwargs(
    "df_key",
    message=msg,
)
def expand_grid(
    df: pd.DataFrame = None,
    df_key: str = None,
    *,
    others: dict,
) -> pd.DataFrame:
    """Creates a DataFrame from a cartesian combination of all inputs.

    It is not restricted to a pandas DataFrame;
    it can work with any list-like structure
    that is 1 or 2 dimensional.

    Data types are preserved in this function,
    including pandas' extension array dtypes.

    If a pandas Series/DataFrame is passed, and has a labeled index, or
    a MultiIndex index, the index is discarded; the final DataFrame
    will have a RangeIndex.

    Examples:

        >>> import pandas as pd
        >>> from janitor.functions.expand_grid import expand_grid
        >>> df = pd.DataFrame({"x": [1, 2], "y": [2, 1]})
        >>> data = {"z": [1, 2, 3]}
        >>> df.expand_grid(others=data)
           x  y  z
        0  1  2  1
        1  1  2  2
        2  1  2  3
        3  2  1  1
        4  2  1  2
        5  2  1  3

        `expand_grid` works with non-pandas objects:

        >>> data = {"x": [1, 2, 3], "y": [1, 2]}
        >>> expand_grid(others=data)
           x  y
        0  1  1
        1  1  2
        2  2  1
        3  2  2
        4  3  1
        5  3  2

    Args:
        df: A pandas DataFrame.
        df_key: Name of key for the dataframe.
            It becomes part of the column names of the dataframe.
            !!!warning "Deprecated in 0.28.0"
        others: A dictionary that contains the data
            to be combined with the dataframe.
            If no dataframe exists, all inputs
            in `others` will be combined to create a DataFrame.

    Returns:
        A pandas DataFrame.
    """
    check("others", others, [dict])

    if df is not None:
        key = tuple(df.columns)
        df = {key: df}
        others = {**df, **others}
    others = _computations_expand_grid(others)
    return pd.DataFrame(others, copy=False)


@pf.register_dataframe_method
def expand(
    df: pd.DataFrame, *columns: tuple, by: str | list = None
) -> pd.DataFrame:
    """
    Creates a DataFrame from a cartesian combination of all inputs.

    expand() is often useful with `pd.merge` to convert implicit
    missing values to explicit missing values - similar to
    [`complete`][janitor.functions.complete.complete].

    It can also be used to figure out which combinations are missing
    (e.g identify gaps in your DataFrame).

    The variable `columns` parameter can be a combination
    of column names or a list/tuple of column names.
    Use a tuple if the columns is a pandas MultiIndex;
    if it is a 2D array use a list; otherwise pass a hashable.

    A dictionary can also be passed -
    the values of the dictionary should be
    either be a 1D or 2D array
    or a callable that evaluates to a
    1D or 2D array. The array should be unique;
    no check is done to verify this.

    If `by` is present, the DataFrame is *expanded* per group.
    `by` should be a column name, or a list of column names.

    Examples:
        >>> import pandas as pd
        >>> import janitor
        >>> data = [{'type': 'apple', 'year': 2010, 'size': 'XS'},
        ...         {'type': 'orange', 'year': 2010, 'size': 'S'},
        ...         {'type': 'apple', 'year': 2012, 'size': 'M'},
        ...         {'type': 'orange', 'year': 2010, 'size': 'S'},
        ...         {'type': 'orange', 'year': 2011, 'size': 'S'},
        ...         {'type': 'orange', 'year': 2012, 'size': 'M'}]
        >>> df = pd.DataFrame(data)
        >>> df
             type  year size
        0   apple  2010   XS
        1  orange  2010    S
        2   apple  2012    M
        3  orange  2010    S
        4  orange  2011    S
        5  orange  2012    M

        Get unique observations:
        >>> df.expand('type')
             type
        0   apple
        1  orange
        >>> df.expand('size')
          size
        0   XS
        1    S
        2    M
        >>> df.expand('type', 'size')
             type size
        0   apple   XS
        1   apple    S
        2   apple    M
        3  orange   XS
        4  orange    S
        5  orange    M
        >>> df.expand('type','size','year')
              type size  year
        0    apple   XS  2010
        1    apple   XS  2012
        2    apple   XS  2011
        3    apple    S  2010
        4    apple    S  2012
        5    apple    S  2011
        6    apple    M  2010
        7    apple    M  2012
        8    apple    M  2011
        9   orange   XS  2010
        10  orange   XS  2012
        11  orange   XS  2011
        12  orange    S  2010
        13  orange    S  2012
        14  orange    S  2011
        15  orange    M  2010
        16  orange    M  2012
        17  orange    M  2011

        Get observations that only occur in the data:
        >>> df.expand(['type','size'])
             type size
        0   apple   XS
        1  orange    S
        2   apple    M
        3  orange    M
        >>> df.expand(['type','size','year'])
             type size  year
        0   apple   XS  2010
        1  orange    S  2010
        2   apple    M  2012
        3  orange    S  2011
        4  orange    M  2012

        Expand the DataFrame to include new observations:
        >>> df.expand('type','size',{'new_year':range(2010,2014)})
              type size  new_year
        0    apple   XS      2010
        1    apple   XS      2011
        2    apple   XS      2012
        3    apple   XS      2013
        4    apple    S      2010
        5    apple    S      2011
        6    apple    S      2012
        7    apple    S      2013
        8    apple    M      2010
        9    apple    M      2011
        10   apple    M      2012
        11   apple    M      2013
        12  orange   XS      2010
        13  orange   XS      2011
        14  orange   XS      2012
        15  orange   XS      2013
        16  orange    S      2010
        17  orange    S      2011
        18  orange    S      2012
        19  orange    S      2013
        20  orange    M      2010
        21  orange    M      2011
        22  orange    M      2012
        23  orange    M      2013

        Filter for missing observations:
        >>> combo = df.expand('type','size','year')
        >>> anti_join = df.merge(combo, how='right', indicator=True)
        >>> anti_join.query("_merge=='right_only").drop(columns="_merge")
              type  year size
        1    apple  2012   XS
        2    apple  2011   XS
        3    apple  2010    S
        4    apple  2012    S
        5    apple  2011    S
        6    apple  2010    M
        8    apple  2011    M
        9   orange  2010   XS
        10  orange  2012   XS
        11  orange  2011   XS
        14  orange  2012    S
        16  orange  2010    M
        18  orange  2011    M

        Expand within each group, using `by`:
        >>> df.expand('year','size',by='type')
                year size
        type
        apple   2010   XS
        apple   2010    M
        apple   2012   XS
        apple   2012    M
        orange  2010    S
        orange  2010    M
        orange  2011    S
        orange  2011    M
        orange  2012    S
        orange  2012    M

    Args:
        df: A pandas DataFrame.
        columns: Specification of columns to expand.
            It could be column labels,
            or a list/tuple of column labels.
            It can also be a dictionay,
            where the values are either a 1D/2D array
            or a callable that evaluates to a
            1D/2D array.
            The array should be unique;
            no check is done to verify this.
        by: Label or list of labels to group by.

    Returns:
        A pandas DataFrame.
    """

    if by is None:
        others = _build_dict_for_expand(df=df, columns=columns)
        others = _computations_expand_grid(others=others)
        return pd.DataFrame(others, copy=False)
    if not is_scalar(by) and not isinstance(by, list):
        raise TypeError(
            "The argument to the by parameter "
            "should be a scalar or a list; "
            f"instead got {type(by).__name__}"
        )
    check_column(df, column_names=by, present=True)
    grouped = df.groupby(by=by, sort=False)
    index = grouped._grouper.result_index
    others = defaultdict(list)
    lengths = []
    for _, frame in grouped:
        _others = _build_dict_for_expand(df=frame, columns=columns)
        _others = _computations_expand_grid(others=_others)
        length = _others[next(iter(_others))].size
        lengths.append(length)
        for k, v in _others.items():
            others[k].append(v)
    others = {key: concat_compat(value) for key, value in others.items()}
    index = index.repeat(lengths)
    others = pd.DataFrame(data=others, index=index, copy=False)
    return others


def _build_dict_for_expand(df: pd.DataFrame, columns: tuple) -> dict:
    """
    Build dictionary for expand()
    """
    others = {}
    for column in columns:
        if is_scalar(column) or isinstance(column, tuple):
            arr = df[column].drop_duplicates()
            others[column] = arr
        elif isinstance(column, list):
            arr = df.loc[:, column].drop_duplicates()
            others[tuple(column)] = arr
        elif isinstance(column, dict):
            for label, _arr in column.items():
                _arr = apply_if_callable(maybe_callable=_arr, obj=df)
                others[label] = _arr
        else:
            raise TypeError(
                "The arguments to the variable columns parameter "
                "should either be a scalar, a list, a tuple or a dictionary; "
                f"instead got {type(column).__name__}"
            )
    return others
