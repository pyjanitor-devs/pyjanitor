"""Implementation of the `toset` function."""
from typing import Set
import pandas_flavor as pf
import pandas as pd


@pf.register_series_method
def toset(series: pd.Series) -> Set:
    """Return a set of the values.

    These are each a scalar type, which is a Python scalar
    (for str, int, float) or a pandas scalar
    (for Timestamp/Timedelta/Interval/Period)

    Example:

        >>> import pandas as pd
        >>> import janitor
        >>> s = pd.Series([1, 2, 3, 5, 5], index=["a", "b", "c", "d", "e"])
        >>> s
        a    1
        b    2
        c    3
        d    5
        e    5
        dtype: int64
        >>> s.toset()
        {1, 2, 3, 5}

    :param series: A pandas series.
    :returns: A set of values.
    """

    return set(series.tolist())
