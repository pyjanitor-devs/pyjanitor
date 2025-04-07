"""Implementation source for `expand_grid`."""

from __future__ import annotations

from collections import defaultdict
from functools import singledispatch
from typing import Optional, Union

import numpy as np
import pandas as pd
import pandas_flavor as pf
from pandas.api.types import is_scalar
from pandas.core.common import apply_if_callable
from pandas.core.dtypes.concat import concat_compat

from janitor.functions.utils import _computations_expand_grid
from janitor.utils import check, check_column, refactored_function


@pf.register_dataframe_method
@refactored_function(
    message=(
        "This function will be deprecated in a 1.x release. "
        "Please use `janitor.cartesian_product` instead."
    )
)
def expand_grid(
    df: Optional[pd.DataFrame] = None,
    df_key: Optional[str] = None,
    *,
    others: Optional[dict] = None,
) -> Union[pd.DataFrame, None]:
    """
    Creates a DataFrame from a cartesian combination of all inputs.

    !!!note

        This function will be deprecated in a 1.x release;
        use [`cartesian_product`][janitor.functions.expand_grid.cartesian_product]
        instead.

    It is not restricted to a pandas DataFrame;
    it can work with any list-like structure
    that is 1 or 2 dimensional.

    If method-chaining to a DataFrame, a string argument
    to `df_key` parameter must be provided.

    Data types are preserved in this function,
    including pandas' extension array dtypes.

    The output will always be a DataFrame, usually with a MultiIndex column,
    with the keys of the `others` dictionary serving as the top level columns.

    If a pandas Series/DataFrame is passed, and has a labeled index, or
    a MultiIndex index, the index is discarded; the final DataFrame
    will have a RangeIndex.

    The MultiIndexed DataFrame can be flattened using pyjanitor's
    [`collapse_levels`][janitor.functions.collapse_levels.collapse_levels]
    method; the user can also decide to drop any of the levels, via pandas'
    `droplevel` method.

    Examples:
        >>> import pandas as pd
        >>> import janitor as jn
        >>> df = pd.DataFrame({"x": [1, 2], "y": [2, 1]})
        >>> data = {"z": [1, 2, 3]}
        >>> df.expand_grid(df_key="df", others=data)
          df     z
           x  y  0
        0  1  2  1
        1  1  2  2
        2  1  2  3
        3  2  1  1
        4  2  1  2
        5  2  1  3

        `expand_grid` works with non-pandas objects:

        >>> data = {"x": [1, 2, 3], "y": [1, 2]}
        >>> jn.expand_grid(others=data)
           x  y
           0  0
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
        others: A dictionary that contains the data
            to be combined with the dataframe.
            If no dataframe exists, all inputs
            in `others` will be combined to create a DataFrame.

    Raises:
        KeyError: If there is a DataFrame and `df_key` is not provided.

    Returns:
        A pandas DataFrame of the cartesian product.
        If `df` is not provided, and `others` is not provided,
        None is returned.
    """  # noqa: E501

    if df is not None:
        check("df", df, [pd.DataFrame])
        if not df_key:
            raise KeyError(
                "Using `expand_grid` as part of a "
                "DataFrame method chain requires that "
                "a string argument be provided for "
                "the `df_key` parameter. "
            )

        check("df_key", df_key, [str])

    if not others and (df is not None):
        return df

    if not others:
        return None

    check("others", others, [dict])

    for key in others:
        check("key", key, [str])

    if df is not None:
        others = {**{df_key: df}, **others}

    others = _computations_expand_grid(others)
    return pd.DataFrame(others, copy=False)


@pf.register_dataframe_method
def expand(
    df: pd.DataFrame,
    *columns: tuple,
    sort: bool = False,
    by: str | list = None,
) -> pd.DataFrame:
    """
    Creates a DataFrame from a cartesian combination of all inputs.

    Inspiration is from tidyr's expand() function.

    expand() is often useful with
    [pd.merge](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html)
    to convert implicit
    missing values to explicit missing values - similar to
    [`complete`][janitor.functions.complete.complete].

    It can also be used to figure out which combinations are missing
    (e.g identify gaps in your DataFrame).

    The variable `columns` parameter can be a column name,
    a list of column names, a pandas Index/Series/DataFrame,
    or a callable, which when applied to the DataFrame,
    evaluates to a pandas Index/Series/DataFrame.

    A dictionary can also be passed
    to the variable `columns` parameter -
    the values of the dictionary should be
    either be a 1D array
    or a callable that evaluates to a
    1D array. The array should be unique;
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
        >>> anti_join.query("_merge=='right_only'").drop(columns="_merge")
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
             a list/tuple of column labels,
             or a pandas Index/Series/DataFrame.

            It can also be a callable;
            the callable will be applied to the
            entire DataFrame. The callable should
            return a pandas Series/Index/DataFrame.

            It can also be a dictionary,
            where the values are either a 1D array
            or a callable that evaluates to a
            1D array.
            The array should be unique;
            no check is done to verify this.
        sort: If True, sort the DataFrame.
        by: Label or list of labels to group by.

    Returns:
        A pandas DataFrame.
    """  # noqa: E501
    if by is None:
        contents = _build_pandas_objects_for_expand(df=df, columns=columns)
        return cartesian_product(*contents, sort=sort)
    if not is_scalar(by) and not isinstance(by, list):
        raise TypeError(
            "The argument to the by parameter "
            "should be a scalar or a list; "
            f"instead got {type(by).__name__}"
        )
    check_column(df, column_names=by, present=True)
    grouped = df.groupby(by=by, sort=False, dropna=False, observed=True)
    index = grouped._grouper.result_index
    dictionary = defaultdict(list)
    lengths = []
    for _, frame in grouped:
        objects = _build_pandas_objects_for_expand(df=frame, columns=columns)
        objects = _compute_cartesian_product(inputs=objects, sort=False)
        length = objects[next(iter(objects))].size
        lengths.append(length)
        for k, v in objects.items():
            dictionary[k].append(v)
    dictionary = {
        key: concat_compat(value) for key, value in dictionary.items()
    }
    index = index.repeat(lengths)
    out = pd.DataFrame(data=dictionary, index=index, copy=False)
    if sort:
        headers = out.columns.tolist()
        return out.sort_values(headers)
    return out


def _build_pandas_objects_for_expand(df: pd.DataFrame, columns: tuple) -> list:
    """
    Build pandas_objects for expand().
    These will be passed to _cartesian_product
    """
    contents = []
    for position, column in enumerate(columns):
        if is_scalar(column) or isinstance(column, tuple):
            arr = df[column].drop_duplicates()
            contents.append(arr)
        elif isinstance(column, list):
            arr = df.loc[:, column].drop_duplicates()
            contents.append(arr)
        elif isinstance(column, dict):
            for label, arr in column.items():
                arr = apply_if_callable(maybe_callable=arr, obj=df)
                arr = pd.Series(arr, name=label)
                contents.append(arr)
        elif isinstance(column, (pd.Series, pd.Index, pd.DataFrame)):
            contents.append(column)
        elif callable(column):
            arr = apply_if_callable(maybe_callable=column, obj=df)
            contents.append(arr)
        else:
            raise TypeError(
                "The arguments to the variable columns parameter "
                "should either be a column name, a list of column names, "
                "a pandas Index/Series/DataFrame, "
                "a callable that evaluates to a "
                "pandas Index/Series/DataFrame, "
                "or a dictionary, "
                "where the value is a 1D array; "
                f"instead got type {type(column).__name__} "
                f"at position {position}"
            )
    return contents


def cartesian_product(*inputs: tuple, sort: bool = False) -> pd.DataFrame:
    """Creates a DataFrame from a cartesian combination of all inputs.

    Inspiration is from tidyr's expand_grid() function.

    The input argument should be a pandas Index/Series/DataFrame,
    or a dictionary - the values of the dictionary should be
    a 1D array.

    Examples:
        >>> import pandas as pd
        >>> import janitor as jn
        >>> df = pd.DataFrame({"x": [1, 2], "y": [2, 1]})
        >>> data = pd.Series([1, 2, 3], name='z')
        >>> jn.cartesian_product(df, data)
           x  y  z
        0  1  2  1
        1  1  2  2
        2  1  2  3
        3  2  1  1
        4  2  1  2
        5  2  1  3

        `cartesian_product` also works with non-pandas objects:

        >>> data = {"x": [1, 2, 3], "y": [1, 2]}
        >>> cartesian_product(data)
           x  y
        0  1  1
        1  1  2
        2  2  1
        3  2  2
        4  3  1
        5  3  2

    Args:
        *inputs: Variable arguments. The arguments should be
            a pandas Index/Series/DataFrame, or a dictionary,
            where the values in the dictionary is a 1D array.
        sort: If True, sort the output DataFrame.

    Returns:
        A pandas DataFrame.
    """
    contents = []
    for entry in inputs:
        if isinstance(entry, dict):
            for label, value in entry.items():
                arr = pd.Series(value, name=label)
                contents.append(arr)
        else:
            contents.append(entry)
    outcome = _compute_cartesian_product(inputs=contents, sort=sort)
    # the values in the outcome dictionary are copies,
    # based on numpy indexing semantics;
    # as such, it is safe to pass copy=False
    return pd.DataFrame(data=outcome, copy=False)


def _compute_cartesian_product(inputs: tuple, sort: bool) -> dict:
    """
    Compute the cartesian product of pandas objects.
    """
    unique_names = set()
    for position, entry in enumerate(inputs):
        unique_names = _validate_pandas_object(
            entry,
            position=position,
            unique_names=unique_names,
        )

    length_of_objects = [len(pandas_object) for pandas_object in inputs]
    cartesian_lengths = np.indices(length_of_objects)
    cartesian_lengths = cartesian_lengths.reshape((len(inputs), -1))
    zipped = zip(inputs, cartesian_lengths)
    contents = {}
    for pandas_object, indexer in zipped:
        if isinstance(pandas_object, pd.DataFrame):
            if sort:
                headers = pandas_object.columns.tolist()
                pandas_object = pandas_object.sort_values(headers)
            for label, array in pandas_object.items():
                contents[label] = array._values[indexer]
        elif isinstance(pandas_object, pd.MultiIndex):
            if sort:
                pandas_object, _ = pandas_object.sortlevel()
            for label in pandas_object.names:
                array = pandas_object.get_level_values(label)._values[indexer]
                contents[label] = array
        else:
            if sort:
                pandas_object = pandas_object.sort_values()
            array = pandas_object._values[indexer]
            contents[pandas_object.name] = array

    if all(map(is_scalar, contents)):
        return contents

    lengths = (len(key) for key in contents if isinstance(key, tuple))
    lengths = max(lengths)
    others = {}
    # manage differing tuple lengths
    # or a mix of tuples and scalars
    for key, value in contents.items():
        if is_scalar(key):
            key = (key, *([""] * (lengths - 1)))
            others[key] = value
        elif len(key) == lengths:
            others[key] = value
        else:
            key = (*key, *([""] * (lengths - len(key))))
            others[key] = value
    return others


@singledispatch
def _validate_pandas_object(pandas_object, position, unique_names) -> tuple:
    """
    Validate pandas object, and ensure the names of the pandas_object
    are not duplicated.

    Args:
        pandas_object: object to be validated.
        position: position of pandas_object in sequence.
        unique_names: python set of names.

    Raises:
        TypeError: If object is not a pandas Index/Series/DataFrame.

    Returns:
        A tuple (pandas_object, unique_names, length_of_objects)
    """
    raise TypeError(
        "input should be either a Pandas DataFrame, "
        "a pandas Series, or a pandas Index; "
        f"instead the object at position {position} "
        f"is of type {type(pandas_object).__name__}"
    )


@_validate_pandas_object.register(pd.DataFrame)
def _validate_pandas_object_(  # noqa: F811
    pandas_object, position, unique_names
) -> tuple:
    """
    Validate a pandas DataFrame
    """
    for label in pandas_object:
        if label in unique_names:
            raise ValueError(
                f"Label {label} in the DataFrame at "
                f"position {position} is duplicated."
            )
        unique_names.add(label)
    return unique_names


@_validate_pandas_object.register(pd.MultiIndex)
def _validate_pandas_object_(  # noqa: F811
    pandas_object, position, unique_names
) -> tuple:
    """
    Validate a pandas MultiIndex
    """
    labels = pandas_object.names
    if None in labels:
        raise ValueError(
            f"Kindly ensure all levels in the MultiIndex "
            f"at position {position} is labeled."
        )
    for label in labels:
        if label in unique_names:
            raise ValueError(
                f"Label {label} in the MultiIndex "
                f"at position {position} is duplicated."
            )
        unique_names.add(label)
    return unique_names


@_validate_pandas_object.register(pd.Index)
@_validate_pandas_object.register(pd.Series)
def _validate_pandas_object_(  # noqa: F811
    pandas_object, position, unique_names
) -> tuple:
    """
    Validate a pandas MultiIndex
    """
    label = pandas_object.name
    if not label:
        raise ValueError(
            f"Kindly ensure the {type(pandas_object).__name__} "
            f"at position {position} has a name."
        )
    if label in unique_names:
        raise ValueError(
            f"Label {label} in the {type(pandas_object).__name__} "
            f"at position {position} is duplicated."
        )
    unique_names.add(label)
    return unique_names
