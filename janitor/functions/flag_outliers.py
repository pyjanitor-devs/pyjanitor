"""Implementation source for `flag_outliers`."""

from __future__ import annotations

import math
from typing import Hashable, Literal

import pandas as pd
import pandas_flavor as pf

from janitor.utils import check, check_column


@pf.register_dataframe_method
def flag_outliers(
    df: pd.DataFrame,
    column_name: Hashable,
    method: Literal["iqr", "zscore"] = "iqr",
    threshold: float | None = None,
    flag_column_name: Hashable | None = None,
) -> pd.DataFrame:
    """Flag outliers in one numeric column.

    Adds a boolean column. ``True`` marks an outlier. The original
    DataFrame is not modified.

    ``method="iqr"`` flags values strictly below ``Q1 - threshold * IQR``
    or strictly above ``Q3 + threshold * IQR``. ``method="zscore"`` flags
    values whose absolute sample Z-score is strictly greater than
    ``threshold``. The sample standard deviation uses ``ddof=1``.

    ``threshold=None`` selects the usual cutoff for the method: ``1.5``
    for IQR and ``3`` for Z-score. One extreme value in a small sample
    can stay inside the Z-score cutoff because it increases the sample
    standard deviation.

    Missing values (``NaN`` and ``pd.NA``) are ignored when the quartiles,
    mean, and standard deviation are computed, and they are never flagged.
    A column whose observed values are all equal has no outliers. If the
    IQR is zero and some values still differ, those values are flagged.
    Statistics that are undefined, such as the sample standard deviation
    of fewer than two observations, flag nothing.

    ``column_name`` selects one column. Call the method again to flag
    another column.

    Examples:
        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({"values": [10, 12, 11, 13, 100, 9, 11]})
        >>> df.flag_outliers(column_name="values")
           values  values_outlier_flag
        0      10                False
        1      12                False
        2      11                False
        3      13                False
        4     100                 True
        5       9                False
        6      11                False
        >>> wide = pd.DataFrame(
        ...     {"values": [10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 50]}
        ... )
        >>> wide.flag_outliers(column_name="values", method="zscore")
            values  values_outlier_flag
        0       10                False
        1       11                False
        2       10                False
        3       11                False
        4       10                False
        5       11                False
        6       10                False
        7       11                False
        8       10                False
        9       11                False
        10      10                False
        11      50                 True

    Args:
        df: Input pandas DataFrame.
        column_name: Name of the numeric column to check.
        method: ``"iqr"`` (default) or ``"zscore"``.
        threshold: Positive cutoff. ``None`` uses ``1.5`` for IQR and
            ``3`` for Z-score.
        flag_column_name: Name of the boolean column to add. Defaults to
            ``"<column_name>_outlier_flag"``.

    Raises:
        ValueError: If ``column_name`` is not in the DataFrame, or if
            ``flag_column_name`` already is.
        ValueError: If ``method`` is not ``"iqr"`` or ``"zscore"``.
        TypeError: If ``threshold`` is not a real number.
        ValueError: If ``threshold`` is not finite and greater than zero.
        TypeError: If ``column_name`` is not numeric.

    Returns:
        A copy of ``df`` with a boolean outlier column appended.
    """
    check_column(df, [column_name])
    if flag_column_name is None:
        flag_column_name = f"{column_name}_outlier_flag"
    check_column(df, [flag_column_name], present=False)

    if method not in ("iqr", "zscore"):
        raise ValueError(f"Invalid method {method!r}. Choose either 'iqr' or 'zscore'.")

    if threshold is None:
        threshold = 3.0 if method == "zscore" else 1.5
    elif isinstance(threshold, bool):
        raise TypeError(
            f"threshold must be a positive number, got {type(threshold).__name__}."
        )
    else:
        check("threshold", threshold, [int, float])
        if not math.isfinite(float(threshold)) or threshold <= 0:
            raise ValueError(f"threshold must be a positive number, got {threshold}.")

    series = df[column_name]
    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError(f"Column {column_name!r} must be numeric to detect outliers.")

    if method == "iqr":
        quartile_1 = series.quantile(0.25)
        quartile_3 = series.quantile(0.75)
        iqr = quartile_3 - quartile_1
        if pd.isna(iqr):
            outlier = pd.Series(False, index=df.index)
        else:
            lower = quartile_1 - threshold * iqr
            upper = quartile_3 + threshold * iqr
            outlier = (series < lower) | (series > upper)
    else:
        scale = series.std(ddof=1, skipna=True)
        if pd.isna(scale) or scale == 0:
            outlier = pd.Series(False, index=df.index)
        else:
            center = series.mean(skipna=True)
            outlier = ((series - center) / scale).abs() > threshold

    result = df.copy()
    result[flag_column_name] = outlier.fillna(False).astype(bool).to_numpy()
    return result
