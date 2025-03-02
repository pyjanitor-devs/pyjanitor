"""Implementation of the `toset` function."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pandas_flavor as pf


@pf.register_series_method
def alias(series: pd.Series, alias: Any = None) -> pd.Series:
    """Return a Series with a new name. Accepts either a scalar or a callable.


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

    if alias is None:
        return series
    if callable(alias):
        alias = alias(series.name)
    series.name = alias
    return series
