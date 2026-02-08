"""Statistical test helpers for tabyl frequency tables."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pandas_flavor as pf
from scipy import stats

from janitor.utils import check


@dataclass(frozen=True)
class ChiSqTestResult:
    """Result container for chi-squared tests on tabyls."""

    statistic: float
    pvalue: float
    dof: int
    expected: pd.DataFrame | np.ndarray
    observed: pd.DataFrame | np.ndarray
    residuals: pd.DataFrame | np.ndarray
    stdres: pd.DataFrame | np.ndarray


@dataclass(frozen=True)
class FisherTestResult:
    """Result container for Fisher's exact test on tabyls."""

    statistic: float
    pvalue: float


def _identify_totals(df: pd.DataFrame) -> list[str]:
    removed: list[str] = []
    if "Total" in df.columns:
        removed.append("column")
    first_col = df.columns[0]
    if df[first_col].eq("Total").any():
        removed.append("row")
    return removed


def _drop_totals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Total" in df.columns:
        df = df.drop(columns=["Total"])
    first_col = df.columns[0]
    total_mask = df[first_col].eq("Total")
    if total_mask.any():
        df = df.loc[~total_mask].reset_index(drop=True)
    return df


def _validate_two_way_tabyl(df: pd.DataFrame, func_name: str) -> None:
    if df.shape[1] < 3 or df.shape[0] < 2:
        raise ValueError(
            f"`{func_name}` requires a 2-way tabyl with at least two rows and "
            "two columns of counts."
        )
    if {"n", "percent"}.issubset(set(df.columns)):
        raise ValueError(
            f"`{func_name}` only supports 2-way tabyls; 1-way tabyls include "
            "'n' and 'percent' columns."
        )

    count_cols = df.columns[1:]
    non_numeric = [
        col for col in count_cols if not pd.api.types.is_numeric_dtype(df[col])
    ]
    if non_numeric:
        raise TypeError(
            f"`{func_name}` requires numeric counts in columns: {non_numeric}."
        )
    if df[count_cols].isna().any().any():
        raise ValueError(f"`{func_name}` requires non-null counts.")
    if (df[count_cols] < 0).any().any():
        raise ValueError(f"`{func_name}` requires non-negative counts.")


def _as_tabyl_frame(
    values: np.ndarray,
    row_name: object,
    row_labels: list[object],
    col_labels: pd.Index,
) -> pd.DataFrame:
    result = pd.DataFrame(values, columns=col_labels)
    result.insert(0, row_name, row_labels)
    return result


def _warn_if_totals(
    df: pd.DataFrame, func_name: str, counts_df: pd.DataFrame | None = None
) -> None:
    removed = _identify_totals(df)
    if counts_df is not None and not removed:
        if counts_df.shape[0] < df.shape[0]:
            removed.append("row")
        if counts_df.shape[1] < df.shape[1]:
            removed.append("column")
    if not removed:
        return
    removed_text = " and ".join(removed)
    warnings.warn(
        f"Totals {removed_text} detected in tabyl; removing before running "
        f"{func_name}.",
        UserWarning,
        stacklevel=2,
    )


@pf.register_dataframe_method
def chisq_test(df: pd.DataFrame, tabyl_results: bool = True) -> ChiSqTestResult:
    """Apply chi-squared test to a two-way tabyl.

    This method expects a two-way frequency table created via ``tabyl``.
    If totals are present (via ``adorn_totals``), they are removed with a
    warning before the test is run.

    Examples:
        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({"gear": [3, 3, 4, 4], "cyl": [4, 6, 4, 6]})
        >>> result = df.tabyl("gear", "cyl").chisq_test()
        >>> result.statistic  # doctest: +SKIP
        0.0

    Args:
        df: A two-way tabyl DataFrame.
        tabyl_results: If True, return observed/expected/residual tables as
            tabyl-formatted DataFrames. If False, return NumPy arrays.

    Returns:
        A ChiSqTestResult with the test statistic, p-value, degrees of
        freedom, and observed/expected/residual tables.
    """
    check("tabyl_results", tabyl_results, [bool])

    counts_df = df.attrs.get("_original_counts", df)
    counts_df = counts_df.copy()
    _warn_if_totals(df, "chisq_test", counts_df)
    counts_df = _drop_totals(counts_df)
    _validate_two_way_tabyl(counts_df, "chisq_test")

    row_name = counts_df.columns[0]
    col_labels = counts_df.columns[1:]
    row_labels = counts_df[row_name].tolist()

    observed_values = counts_df[col_labels].to_numpy(dtype=float)
    total = observed_values.sum()
    if total == 0:
        raise ValueError("`chisq_test` requires counts that sum to a positive value.")
    result = stats.chi2_contingency(observed_values)
    expected = getattr(result, "expected_freq", None)
    if expected is None:
        expected = result.expected

    with np.errstate(divide="ignore", invalid="ignore"):
        residuals = (observed_values - expected) / np.sqrt(expected)
        row_probs = observed_values.sum(axis=1, keepdims=True) / total
        col_probs = observed_values.sum(axis=0, keepdims=True) / total
        stdres_denom = np.sqrt(expected * (1 - row_probs) * (1 - col_probs))
        stdres = np.divide(
            observed_values - expected,
            stdres_denom,
            out=np.full_like(expected, np.nan, dtype=float),
            where=stdres_denom != 0,
        )

    if tabyl_results:
        observed = counts_df.copy()
        expected = _as_tabyl_frame(expected, row_name, row_labels, col_labels)
        residuals = _as_tabyl_frame(residuals, row_name, row_labels, col_labels)
        stdres = _as_tabyl_frame(stdres, row_name, row_labels, col_labels)
    else:
        observed = observed_values

    return ChiSqTestResult(
        statistic=result.statistic,
        pvalue=result.pvalue,
        dof=result.dof,
        expected=expected,
        observed=observed,
        residuals=residuals,
        stdres=stdres,
    )


@pf.register_dataframe_method
def fisher_test(df: pd.DataFrame) -> FisherTestResult:
    """Apply Fisher's exact test to a two-way tabyl.

    This method expects a 2x2 frequency table created via ``tabyl``.
    If totals are present (via ``adorn_totals``), they are removed with a
    warning before the test is run.

    Examples:
        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({"gear": [3, 3, 4, 4], "cyl": [4, 6, 4, 6]})
        >>> result = df.tabyl("gear", "cyl").fisher_test()
        >>> result.pvalue  # doctest: +SKIP
        1.0

    Args:
        df: A two-way tabyl DataFrame.

    Returns:
        A FisherTestResult with the odds ratio and p-value.
    """
    counts_df = df.attrs.get("_original_counts", df)
    counts_df = counts_df.copy()
    _warn_if_totals(df, "fisher_test", counts_df)
    counts_df = _drop_totals(counts_df)
    _validate_two_way_tabyl(counts_df, "fisher_test")

    if counts_df.shape[0] != 2 or (counts_df.shape[1] - 1) != 2:
        raise ValueError("`fisher_test` requires a 2x2 tabyl.")

    observed_values = counts_df.iloc[:, 1:].to_numpy()
    if observed_values.sum() == 0:
        raise ValueError("`fisher_test` requires counts that sum to a positive value.")
    result = stats.fisher_exact(observed_values)
    statistic = result.statistic if hasattr(result, "statistic") else result[0]
    pvalue = result.pvalue if hasattr(result, "pvalue") else result[1]
    return FisherTestResult(statistic=statistic, pvalue=pvalue)
