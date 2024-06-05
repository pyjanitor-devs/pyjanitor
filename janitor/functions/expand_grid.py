"""Implementation source for `expand_grid`."""

from __future__ import annotations

import pandas as pd
import pandas_flavor as pf

from janitor.functions.utils import _computations_expand_grid
from janitor.utils import check, deprecated_kwargs

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

    It is not restricted to DataFrame;
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
