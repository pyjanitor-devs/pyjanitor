""" Miscellaneous mathematical operators. """

import datetime as dt
import functools
import os
import re
import sys
import unicodedata
import warnings
from fnmatch import translate
from functools import partial, reduce
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union

import numpy as np
import pandas as pd
import pandas_flavor as pf
from scipy.special import expit
from scipy.stats import mode, norm
from sklearn.preprocessing import LabelEncoder

from .errors import JanitorError
from .utils import (
    _clean_accounting_column,
    _currency_column_to_numeric,
    _replace_empty_string_with_none,
    _replace_original_empty_string_with_none,
    _strip_underscores,
    check,
    check_column,
    deprecated_alias,
)


@pf.register_series_method
def log(s: pd.Series, error: str = "warn") -> pd.Series:
    """
    Take natural logarithm of the Series

    Parameters
    ----------
    s : pd.Series
        Input Series
    error : str, optional
        Determines behavior when taking the log of nonpositive entries. If
        "warn" then a RuntimeWarning is thrown. If "raise", then a RuntimeError
        is thrown. Otherwise, nothing is thrown and log of nonpositive values
        is np.nan, by default "warn"

    Returns
    -------
    pd.Series
        Transformed Series

    Raises
    ------
    RuntimeError
        Raised when there are nonpositive values in the Series and error="raise"
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
    """Take the exponential transform of the series"""
    return np.exp(s)


@pf.register_series_method
def sigmoid(s: pd.Series) -> pd.Series:
    """
    Take the sigmoid transform of the series where sigmoid(x) = 1 / (1 + exp(-x))
    """
    return expit(s)


@pf.register_series_method
def logit(s: pd.Series, error: str = "warn") -> pd.Series:
    """
    Take logit transform of the Series
    where logit(p) = log(p/(1-p))

    Parameters
    ----------
    s : pd.Series
        Input Series
    error : str, optional
        Determines behavior when s / (1-s) is outside of (0, 1). If
        "warn" then a RuntimeWarning is thrown. If "raise", then a RuntimeError
        is thrown. Otherwise, nothing is thrown and np.nan is returned
        for the problematic entries, by default "warn"

    Returns
    -------
    pd.Series
        Transformed Series

    Raises
    ------
    RuntimeError
        Raised when there are problematic values in the Series and error="raise"
    """
    s = s.copy()
    odds_ratio = s / (1 - s)
    outside_support = (odds_ratio <= 0) | (odds_ratio >= 1)
    if (outside_support).any():
        msg = f"Odds ratio for {outside_support.sum()} value(s) are outside of (0, 1)"
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
    """Transforms the Series via the CDF of the Normal distribution"""
    return pd.Series(norm.cdf(s), index=s.index)


@pf.register_series_method
def probit(s: pd.Series, error: str = "warn") -> pd.Series:
    """
    Transforms the Series via the inverse CDF of the Normal distribution

    Parameters
    ----------
    s : pd.Series
        Input Series
    error : str, optional
        Determines behavior when s is outside of (0, 1). If
        "warn" then a RuntimeWarning is thrown. If "raise", then a RuntimeError
        is thrown. Otherwise, nothing is thrown and np.nan is returned
        for the problematic entries, by default "warn"

    Returns
    -------
    pd.Series
        Transformed Series

    Raises
    ------
    RuntimeError
        Raised when there are problematic values in the Series and error="raise"
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
    s: pd.Series, moments_dict: dict = None, keys: Tuple[str] = ("mean", "std")
) -> pd.Series:
    """
    Transforms the Series into z-scores

    Parameters
    ----------
    s : pd.Series
        Input Series
    moments_dict : dict, optional
        If not None, then the mean and standard deviation used to compute
        the z-score transformation is saved as entries in
        moments_dict with keys determined by the keys argument, by default None
    keys: tuple of str, optional
        Determines the keys saved in moments_dict if moments are saved

    Returns
    -------
    pd.Series
        Transformed Series
    """
    mean = s.mean()
    std = s.std()
    if std == 0:
        return 0
    if moments_dict is not None:
        moments_dict[keys[0]] = mean
        moments_dict[keys[1]] = std
    return (s - mean) / std
