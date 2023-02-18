"""Implementation source for `expand_grid`."""
from typing import Dict, Optional, Union
import pandas_flavor as pf
import pandas as pd

from janitor.utils import check

from janitor.functions.utils import _computations_expand_grid


@pf.register_dataframe_method
def expand_grid(
    df: Optional[pd.DataFrame] = None,
    df_key: Optional[str] = None,
    *,
    others: Optional[Dict] = None,
) -> Union[pd.DataFrame, None]:
    """Creates a DataFrame from a cartesian combination of all inputs.

    It is not restricted to DataFrame;
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
    """

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
