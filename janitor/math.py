""" Miscellaneous mathematical operators. """

import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import pandas_flavor as pf
from pandas.api.types import is_numeric_dtype
from scipy.special import expit
from scipy.special import softmax as scipy_softmax
from scipy.stats import norm


@pf.register_series_method
def log(s: pd.Series, error: str = "warn") -> pd.Series:
    """
    Take natural logarithm of the Series.

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

    :param s: Input Series.
    :param error: Determines behavior when `s / (1-s)` is outside of `(0, 1)`.
        If `'warn'` then a `RuntimeWarning` is thrown. If `'raise'`, then a
        `RuntimeError` is thrown. Otherwise, nothing is thrown and `np.nan`
        is returned for the problematic entries; defaults to `'warn'`.
    :return: Transformed Series.
    :raises RuntimeError: if `error` is set to `'raise'`.
    """
    s = s.copy()
    odds_ratio = s / (1 - s)
    outside_support = (odds_ratio <= 0) | (odds_ratio >= 1)
    if (outside_support).any():
        msg = f"Odds ratio for {outside_support.sum()} value(s) \
are outside of (0, 1)"
        if error.lower() == "warn":
            warnings.warn(msg, RuntimeWarning)
        if error.lower() == "raise":
            raise RuntimeError(msg)
        else:
            pass
    odds_ratio[outside_support] = np.nan
    return odds_ratio.log(error="ignore")


@pf.register_series_method
def normal_cdf(s: pd.Series) -> pd.Series:
    """
    Transforms the Series via the CDF of the Normal distribution.

    :param s: Input Series.
    :return: Transformed Series.
    """
    return pd.Series(norm.cdf(s), index=s.index)


@pf.register_series_method
def probit(s: pd.Series, error: str = "warn") -> pd.Series:
    """
    Transforms the Series via the inverse CDF of the Normal distribution.

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
