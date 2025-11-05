# helper functions for conditional_join.py

from enum import Enum
from typing import Sequence

import numpy as np
import pandas as pd


class _JoinOperator(Enum):
    """
    List of operators used in conditional_join.
    """

    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_THAN_OR_EQUAL = ">="
    LESS_THAN_OR_EQUAL = "<="
    STRICTLY_EQUAL = "=="
    NOT_EQUAL = "!="


less_than_join_types = {
    _JoinOperator.LESS_THAN.value,
    _JoinOperator.LESS_THAN_OR_EQUAL.value,
}
greater_than_join_types = {
    _JoinOperator.GREATER_THAN.value,
    _JoinOperator.GREATER_THAN_OR_EQUAL.value,
}


def _maybe_remove_nulls_from_dataframe(
    df: pd.DataFrame, columns: Sequence, return_bools: bool = False
):
    """
    Remove nulls if op is not !=;
    """
    any_nulls = df.loc[:, [*columns]].isna().any(axis=1)
    if any_nulls.all():
        return None
    if return_bools:
        any_nulls = ~any_nulls
        return any_nulls
    if any_nulls.any():
        df = df.loc[~any_nulls]
    return df


def _null_checks_cond_join(series: pd.Series) -> tuple | None:
    """
    Checks for nulls in the pandas series before conducting binary search.
    """
    any_nulls = series.isna()
    if any_nulls.all():
        return None
    if any_nulls.any():
        series = series[~any_nulls]
    return series, any_nulls.any()


def _sort_if_not_monotonic(series: pd.Series) -> pd.Series | None:
    """
    Sort the pandas `series` if it is not monotonic increasing
    """

    is_sorted = series.is_monotonic_increasing
    if not is_sorted:
        series = series.sort_values(kind="stable")
    return series, is_sorted


def _keep_output(keep: str, left: np.ndarray, right: np.ndarray):
    """return indices for left and right index based on the value of `keep`."""
    if keep == "all":
        return left, right
    grouped = pd.Series(right).groupby(left, sort=False)
    if keep == "first":
        grouped = grouped.min()
        return grouped.index, grouped._values
    grouped = grouped.max()
    return grouped.index, grouped._values
