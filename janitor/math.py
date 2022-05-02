""" Miscellaneous mathematical operators. """

import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import pandas_flavor as pf
from pandas.api.types import is_numeric_dtype
from scipy.special import expit
from scipy.special import logit as scipy_logit
from scipy.special import softmax as scipy_softmax
from scipy.stats import norm


@pf.register_series_method
def log(s: pd.Series, error: str = "warn") -> pd.Series:
    """
    Take natural logarithm of the Series.

    Each value in the series should be positive. Use `error` to control the
    behavior if there are nonpositive entries in the series.

        >>> import pandas as pd
        >>> import janitor
        >>> s = pd.Series([0, 1, 3], name="numbers")
        >>> s.log(error="ignore")
        0         NaN
        1    0.000000
        2    1.098612
        Name: numbers, dtype: float64

    :param s: Input Series.
    :param error: Determines behavior when taking the log of nonpositive
        entries. If `'warn'` then a `RuntimeWarning` is thrown. If `'raise'`,
        then a `RuntimeError` is thrown. Otherwise, nothing is thrown and
        log of nonpositive values is `np.nan`; defaults to `'warn'`.
    :raises RuntimeError: Raised when there are nonpositive values in the
        Series and `error='raise'`.
    :return: Transformed Series.
    """
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
def exp(s: pd.Series) -> pd.Series:
    """
    Take the exponential transform of the series.

        >>> import pandas as pd
        >>> import janitor
        >>> s = pd.Series([0, 1, 3], name="numbers")
        >>> s.exp()
        0     1.000000
        1     2.718282
        2    20.085537
        Name: numbers, dtype: float64

    :param s: Input Series.
    :return: Transformed Series.
    """
    return np.exp(s)


@pf.register_series_method
def sigmoid(s: pd.Series) -> pd.Series:
    """
    Take the sigmoid transform of the series where:

    ```python
    sigmoid(x) = 1 / (1 + exp(-x))
    ```

        >>> import pandas as pd
        >>> import janitor
        >>> s = pd.Series([-1, 0, 4], name="numbers")
        >>> s.sigmoid()
        0    0.268941
        1    0.500000
        2    0.982014
        Name: numbers, dtype: float64

    :param s: Input Series.
    :return: Transformed Series.
    """
    return expit(s)


@pf.register_series_method
def softmax(s: pd.Series) -> pd.Series:
    """
    Take the softmax transform of the series.

    The softmax function transforms each element of a collection by
    computing the exponential of each element divided by the sum of the
    exponentials of all the elements.

    That is, if x is a one-dimensional numpy array or pandas Series:

    ```python
    softmax(x) = exp(x)/sum(exp(x))
    ```

        >>> import pandas as pd
        >>> import janitor
        >>> s = pd.Series([0, 1, 3], name="numbers")
        >>> s.softmax()
        0    0.042010
        1    0.114195
        2    0.843795
        Name: numbers, dtype: float64

    :param s: Input Series.
    :return: Transformed Series.
    """
    return scipy_softmax(s)


@pf.register_series_method
def logit(s: pd.Series, error: str = "warn") -> pd.Series:
    """
    Take logit transform of the Series where:

    ```python
    logit(p) = log(p/(1-p))
    ```

    Each value in the series should be between 0 and 1. Use `error` to
    control the behavior if any series entries are outside of (0, 1).

        >>> import pandas as pd
        >>> import janitor
        >>> s = pd.Series([0.1, 0.5, 0.9], name="numbers")
        >>> s.logit()
        0   -2.197225
        1    0.000000
        2    2.197225
        Name: numbers, dtype: float64

    :param s: Input Series.
    :param error: Determines behavior when `s` is outside of `(0, 1)`.
        If `'warn'` then a `RuntimeWarning` is thrown. If `'raise'`, then a
        `RuntimeError` is thrown. Otherwise, nothing is thrown and `np.nan`
        is returned for the problematic entries; defaults to `'warn'`.
    :return: Transformed Series.
    :raises RuntimeError: if `error` is set to `'raise'`.
    """
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
    return scipy_logit(s)


@pf.register_series_method
def normal_cdf(s: pd.Series) -> pd.Series:
    """
    Transforms the Series via the CDF of the Normal distribution.

        >>> import pandas as pd
        >>> import janitor
        >>> s = pd.Series([-1, 0, 3], name="numbers")
        >>> s.normal_cdf()
        0    0.158655
        1    0.500000
        2    0.998650
        dtype: float64

    :param s: Input Series.
    :return: Transformed Series.
    """
    return pd.Series(norm.cdf(s), index=s.index)


@pf.register_series_method
def probit(s: pd.Series, error: str = "warn") -> pd.Series:
    """
    Transforms the Series via the inverse CDF of the Normal distribution.

    Each value in the series should be between 0 and 1. Use `error` to
    control the behavior if any series entries are outside of (0, 1).

        >>> import pandas as pd
        >>> import janitor
        >>> s = pd.Series([0.1, 0.5, 0.8], name="numbers")
        >>> s.probit()
        0   -1.281552
        1    0.000000
        2    0.841621
        dtype: float64

    :param s: Input Series.
    :param error: Determines behavior when `s` is outside of `(0, 1)`.
        If `'warn'` then a `RuntimeWarning` is thrown. If `'raise'`, then
        a `RuntimeError` is thrown. Otherwise, nothing is thrown and `np.nan`
        is returned for the problematic entries; defaults to `'warn'`.
    :raises RuntimeError: Raised when there are problematic values
        in the Series and `error='raise'`.
    :return: Transformed Series
    """
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
        out = pd.Series(norm.ppf(s), index=s.index)
    return out


@pf.register_series_method
def z_score(
    s: pd.Series,
    moments_dict: dict = None,
    keys: Tuple[str, str] = ("mean", "std"),
) -> pd.Series:
    """
    Transforms the Series into z-scores where:

    ```python
    z = (s - s.mean()) / s.std()
    ```

        >>> import pandas as pd
        >>> import janitor
        >>> s = pd.Series([0, 1, 3], name="numbers")
        >>> s.z_score()
        0   -0.872872
        1   -0.218218
        2    1.091089
        Name: numbers, dtype: float64

    :param s: Input Series.
    :param moments_dict: If not `None`, then the mean and standard
        deviation used to compute the z-score transformation is
        saved as entries in `moments_dict` with keys determined by
        the `keys` argument; defaults to `None`.
    :param keys: Determines the keys saved in `moments_dict`
        if moments are saved; defaults to (`'mean'`, `'std'`).
    :return: Transformed Series.
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
def ecdf(s: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return cumulative distribution of values in a series.

    Intended to be used with the following pattern:

    ```python
    df = pd.DataFrame(...)

    # Obtain ECDF values to be plotted
    x, y = df["column_name"].ecdf()

    # Plot ECDF values
    plt.scatter(x, y)
    ```

    Null values must be dropped from the series,
    otherwise a `ValueError` is raised.

    Also, if the `dtype` of the series is not numeric,
    a `TypeError` is raised.

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({"numbers": [0, 4, 0, 1, 2, 1, 1, 3]})
        >>> x, y = df["numbers"].ecdf()
        >>> x
        array([0, 0, 1, 1, 1, 2, 3, 4])
        >>> y
        array([0.125, 0.25 , 0.375, 0.5  , 0.625, 0.75 , 0.875, 1.   ])

    :param s: A pandas series. `dtype` should be numeric.
    :returns: `(x, y)`.
        `x`: sorted array of values.
        `y`: cumulative fraction of data points with value `x` or lower.
    :raises TypeError: if series is not numeric.
    :raises ValueError: if series contains nulls.
    """
    if not is_numeric_dtype(s):
        raise TypeError(f"series {s.name} must be numeric!")
    if not s.isna().sum() == 0:
        raise ValueError(f"series {s.name} contains nulls. Please drop them.")

    n = len(s)
    x = np.sort(s)
    y = np.arange(1, n + 1) / n

    return x, y
