"""Miscellaneous mathematical operators."""

import warnings
from typing import TYPE_CHECKING, Tuple

import pandas_flavor as pf

if TYPE_CHECKING:
    from numpy import ndarray
    from pandas import Series


@pf.register_series_method
def log(s: "Series", error: str = "warn") -> "Series":
    """
    Take natural logarithm of the Series.

    Each value in the series should be positive. Use `error` to control the
    behavior if there are nonpositive entries in the series.

    Examples:
        >>> import pandas as pd
        >>> import janitor
        >>> s = pd.Series([0, 1, 3], name="numbers")
        >>> s.log(error="ignore")
        0         NaN
        1    0.000000
        2    1.098612
        Name: numbers, dtype: float64

    Args:
        s: Input Series.
        error: Determines behavior when taking the log of nonpositive
            entries. If `'warn'` then a `RuntimeWarning` is thrown. If
            `'raise'`, then a `RuntimeError` is thrown. Otherwise, nothing
            is thrown and log of nonpositive values is `np.nan`.

    Raises:
        RuntimeError: Raised when there are nonpositive values in the
            Series and `error='raise'`.

    Returns:
        Transformed Series.
    """
    import numpy as np

    s = s.copy()
    nonpositive = s <= 0
    if (nonpositive).any():
        msg = f"Log taken on {nonpositive.sum()} nonpositive value(s)"
        if error.lower() == "warn":
            warnings.warn(msg, RuntimeWarning)
        if error.lower() == "raise":
            raise RuntimeError(msg)
        else:
            pass
    s[nonpositive] = np.nan
    return np.log(s)


@pf.register_series_method
def exp(s: "Series") -> "Series":
    """Take the exponential transform of the series.

    Examples:
        >>> import pandas as pd
        >>> import janitor
        >>> s = pd.Series([0, 1, 3], name="numbers")
        >>> s.exp()
        0     1.000000
        1     2.718282
        2    20.085537
        Name: numbers, dtype: float64

    Args:
        s: Input Series.

    Returns:
        Transformed Series.
    """
    import numpy as np

    return np.exp(s)


@pf.register_series_method
def sigmoid(s: "Series") -> "Series":
    """Take the sigmoid transform of the series.

    The sigmoid function is defined:

    ```python
    sigmoid(x) = 1 / (1 + exp(-x))
    ```

    Examples:
        >>> import pandas as pd
        >>> import janitor
        >>> s = pd.Series([-1, 0, 4], name="numbers")
        >>> s.sigmoid()
        0    0.268941
        1    0.500000
        2    0.982014
        Name: numbers, dtype: float64

    Args:
        s: Input Series.

    Returns:
        Transformed Series.
    """
    import scipy

    return scipy.special.expit(s)


@pf.register_series_method
def softmax(s: "Series") -> "Series":
    """Take the softmax transform of the series.

    The softmax function transforms each element of a collection by
    computing the exponential of each element divided by the sum of the
    exponentials of all the elements.

    That is, if x is a one-dimensional numpy array or pandas Series:

    ```python
    softmax(x) = exp(x)/sum(exp(x))
    ```

    Examples:
        >>> import pandas as pd
        >>> import janitor
        >>> s = pd.Series([0, 1, 3], name="numbers")
        >>> s.softmax()
        0    0.042010
        1    0.114195
        2    0.843795
        Name: numbers, dtype: float64

    Args:
        s: Input Series.

    Returns:
        Transformed Series.
    """
    import pandas as pd
    import scipy

    return pd.Series(scipy.special.softmax(s), index=s.index, name=s.name)


@pf.register_series_method
def logit(s: "Series", error: str = "warn") -> "Series":
    """Take logit transform of the Series.

    The logit transform is defined:

    ```python
    logit(p) = log(p/(1-p))
    ```

    Each value in the series should be between 0 and 1. Use `error` to
    control the behavior if any series entries are outside of (0, 1).

    Examples:
        >>> import pandas as pd
        >>> import janitor
        >>> s = pd.Series([0.1, 0.5, 0.9], name="numbers")
        >>> s.logit()
        0   -2.197225
        1    0.000000
        2    2.197225
        Name: numbers, dtype: float64

    Args:
        s: Input Series.
        error: Determines behavior when `s` is outside of `(0, 1)`.
            If `'warn'` then a `RuntimeWarning` is thrown. If `'raise'`, then a
            `RuntimeError` is thrown. Otherwise, nothing is thrown and `np.nan`
            is returned for the problematic entries; defaults to `'warn'`.

    Raises:
        RuntimeError: If `error` is set to `'raise'`.

    Returns:
        Transformed Series.
    """
    import numpy as np
    import scipy

    s = s.copy()
    outside_support = (s <= 0) | (s >= 1)
    if (outside_support).any():
        msg = f"{outside_support.sum()} value(s) are outside of (0, 1)"
        if error.lower() == "warn":
            warnings.warn(msg, RuntimeWarning)
        if error.lower() == "raise":
            raise RuntimeError(msg)
        else:
            pass
    s[outside_support] = np.nan
    return scipy.special.logit(s)


@pf.register_series_method
def normal_cdf(s: "Series") -> "Series":
    """Transforms the Series via the CDF of the Normal distribution.

    Examples:
        >>> import pandas as pd
        >>> import janitor
        >>> s = pd.Series([-1, 0, 3], name="numbers")
        >>> s.normal_cdf()
        0    0.158655
        1    0.500000
        2    0.998650
        dtype: float64

    Args:
        s: Input Series.

    Returns:
        Transformed Series.
    """
    import pandas as pd
    import scipy

    return pd.Series(scipy.stats.norm.cdf(s), index=s.index)


@pf.register_series_method
def probit(s: "Series", error: str = "warn") -> "Series":
    """Transforms the Series via the inverse CDF of the Normal distribution.

    Each value in the series should be between 0 and 1. Use `error` to
    control the behavior if any series entries are outside of (0, 1).

    Examples:
        >>> import pandas as pd
        >>> import janitor
        >>> s = pd.Series([0.1, 0.5, 0.8], name="numbers")
        >>> s.probit()
        0   -1.281552
        1    0.000000
        2    0.841621
        dtype: float64

    Args:
        s: Input Series.
        error: Determines behavior when `s` is outside of `(0, 1)`.
            If `'warn'` then a `RuntimeWarning` is thrown. If `'raise'`, then
            a `RuntimeError` is thrown. Otherwise, nothing is thrown and
            `np.nan` is returned for the problematic entries.

    Raises:
        RuntimeError: When there are problematic values
            in the Series and `error='raise'`.

    Returns:
        Transformed Series
    """
    import numpy as np
    import pandas as pd
    import scipy

    s = s.copy()
    outside_support = (s <= 0) | (s >= 1)
    if (outside_support).any():
        msg = f"{outside_support.sum()} value(s) are outside of (0, 1)"
        if error.lower() == "warn":
            warnings.warn(msg, RuntimeWarning)
        if error.lower() == "raise":
            raise RuntimeError(msg)
        else:
            pass
    s[outside_support] = np.nan
    with np.errstate(all="ignore"):
        out = pd.Series(scipy.stats.norm.ppf(s), index=s.index)
    return out


@pf.register_series_method
def z_score(
    s: "Series",
    moments_dict: dict = None,
    keys: Tuple[str, str] = ("mean", "std"),
) -> "Series":
    """Transforms the Series into z-scores.

    The z-score is defined:

    ```python
    z = (s - s.mean()) / s.std()
    ```

    Examples:
        >>> import pandas as pd
        >>> import janitor
        >>> s = pd.Series([0, 1, 3], name="numbers")
        >>> s.z_score()
        0   -0.872872
        1   -0.218218
        2    1.091089
        Name: numbers, dtype: float64

    Args:
        s: Input Series.
        moments_dict: If not `None`, then the mean and standard
            deviation used to compute the z-score transformation is
            saved as entries in `moments_dict` with keys determined by
            the `keys` argument; defaults to `None`.
        keys: Determines the keys saved in `moments_dict`
            if moments are saved; defaults to (`'mean'`, `'std'`).

    Returns:
        Transformed Series.
    """
    mean = s.mean()
    std = s.std()
    if std == 0:
        return 0
    if moments_dict is not None:
        moments_dict[keys[0]] = mean
        moments_dict[keys[1]] = std
    return (s - mean) / std


@pf.register_series_method
def ecdf(s: "Series") -> Tuple["ndarray", "ndarray"]:
    """Return cumulative distribution of values in a series.

    Null values must be dropped from the series,
    otherwise a `ValueError` is raised.

    Also, if the `dtype` of the series is not numeric,
    a `TypeError` is raised.

    Examples:
        >>> import pandas as pd
        >>> import janitor
        >>> s = pd.Series([0, 4, 0, 1, 2, 1, 1, 3])
        >>> x, y = s.ecdf()
        >>> x  # doctest: +SKIP
        array([0, 0, 1, 1, 1, 2, 3, 4])
        >>> y  # doctest: +SKIP
        array([0.125, 0.25 , 0.375, 0.5  , 0.625, 0.75 , 0.875, 1.   ])

        You can then plot the ECDF values, for example:

        >>> from matplotlib import pyplot as plt
        >>> plt.scatter(x, y)  # doctest: +SKIP

    Args:
        s: A pandas series. `dtype` should be numeric.

    Raises:
        TypeError: If series is not numeric.
        ValueError: If series contains nulls.

    Returns:
        x: Sorted array of values.
        y: Cumulative fraction of data points with value `x` or lower.
    """
    import numpy as np
    import pandas.api.types as pdtypes

    if not pdtypes.is_numeric_dtype(s):
        raise TypeError(f"series {s.name} must be numeric!")
    if not s.isna().sum() == 0:
        raise ValueError(f"series {s.name} contains nulls. Please drop them.")

    n = len(s)
    x = np.sort(s)
    y = np.arange(1, n + 1) / n

    return x, y
