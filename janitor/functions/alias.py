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
        >>> s = pd.Series([1, 2, 3], name='series')
        >>> s
        0    1
        1    2
        2    3
        Name: series, dtype: int64
        >>> s.alias('series_new')
        0    1
        1    2
        2    3
        Name: series_new, dtype: int64
        >>> s.alias(str.upper)
        0    1
        1    2
        2    3
        Name: SERIES, dtype: int64

    Args:
        series: A pandas Series.
        alias: scalar or callable to create a new name for the pandas Series.

    Returns:
        A new pandas Series.
    """
    series = series[:]
    if alias is None:
        return series
    if callable(alias):
        alias = alias(series.name)
    series.name = alias
    return series
