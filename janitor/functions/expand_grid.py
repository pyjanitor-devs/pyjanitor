"""Implementation source for `expand_grid`."""

from functools import singledispatch
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import pandas_flavor as pf
from pandas.api.types import is_scalar

from janitor.functions.utils import _computations_expand_grid
from janitor.utils import check, refactored_function


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
    others: Optional[Dict] = None,
) -> Union[pd.DataFrame, None]:
    """Creates a DataFrame from a cartesian combination of all inputs.

        !!!note

        This function will be deprecated in a 1.x release.
        Please use [`cartesian_product`][janitor.functions.expand_grid.cartesian_product]
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
        >>> from janitor.functions.expand_grid import expand_grid
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
        >>> expand_grid(others=data)
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


def cartesian_product(*inputs: tuple) -> pd.DataFrame:
    """Creates a DataFrame from a cartesian combination of all inputs.

    Inspiration is from tidyr's expand_grid() function.

    The input argument should be a pandas Index/Series/DataFrame.

    Examples:

        >>> import pandas as pd
        >>> from janitor import cartesian_product
        >>> df = pd.DataFrame({"x": [1, 2], "y": [2, 1]})
        >>> data = pd.Series([1, 2, 3], name='z')
        >>> cartesian_product(df, data)
           x  y  z
        0  1  2  1
        1  1  2  2
        2  1  2  3
        3  2  1  1
        4  2  1  2
        5  2  1  3

    Args:
        *inputs: Variable arguments. The arguments should be
            a pandas Index/Series/DataFrame.

    Returns:
        A pandas DataFrame.
    """
    outcome = _compute_cartesian_product(inputs=inputs)
    return pd.DataFrame(data=outcome, copy=False)


def _compute_cartesian_product(inputs: tuple) -> dict:
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
            for label, array in pandas_object.items():
                contents[label] = array._values[indexer]
        elif isinstance(pandas_object, pd.MultiIndex):
            for label in pandas_object.names:
                array = pandas_object.get_level_values(label)._values[indexer]
                contents[label] = array
        else:
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
