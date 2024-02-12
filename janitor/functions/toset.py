"""Implementation of the `toset` function."""

from typing import Set

import pandas as pd
import pandas_flavor as pf

from janitor.utils import refactored_function


@pf.register_series_method
@refactored_function(
    message=(
        "This function will be deprecated in a 1.x release. "
        "Please use `set(df[column])` instead."
    )
)
def toset(series: pd.Series) -> Set:
    """Return a set of the values.

    !!!note

        This function will be deprecated in a 1.x release.
        Please use `set(df[column])` instead.

    These are each a scalar type, which is a Python scalar
    (for str, int, float) or a pandas scalar
    (for Timestamp/Timedelta/Interval/Period)

    Examples:
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

    Args:
        series: A pandas series.

    Returns:
        A set of values.
    """

    return set(series.tolist())
