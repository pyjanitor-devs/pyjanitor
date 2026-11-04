"""Implementation source for `flag_outliers`."""

from __future__ import annotations

import math
from numbers import Real
from typing import Hashable, Literal

import numpy as np
import pandas as pd
import pandas_flavor as pf

from janitor.utils import check_column


@pf.register_dataframe_method
def flag_outliers(
    df: pd.DataFrame,
    column_name: Hashable,
    method: Literal["iqr", "zscore"] = "iqr",
    threshold: Real | None = None,
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
    Positive and negative infinity are outliers. They are left out of
    those statistics, so the cutoff stays finite and the calculation does
    not warn. A column whose observed finite values are all equal has no
    finite outliers. If the IQR is zero and some finite values still
    differ, those values are flagged. Statistics that are undefined, such
    as the sample standard deviation of fewer than two finite observations,
    flag nothing among the finite values.

    Quartiles are computed in float64, including for integer columns.
    Boolean and complex columns are rejected. ``threshold`` may be any
    real number, including a NumPy scalar. Booleans are rejected.

    ``column_name`` selects one column. Call the method again to flag
    another column. When the columns are a MultiIndex, the default flag
    name keeps the upper levels and appends ``_outlier_flag`` to the last
    level: ``("group", "values")`` becomes
    ``("group", "values_outlier_flag")``.

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
        threshold: Positive real cutoff, including NumPy scalars.
            ``None`` uses ``1.5`` for IQR and ``3`` for Z-score.
        flag_column_name: Name of the boolean column to add. Defaults to
            ``"<column_name>_outlier_flag"``, or that suffix on the last
            MultiIndex level.

    Raises:
        ValueError: If ``column_name`` is not in the DataFrame, matches
            more than one column, or if ``flag_column_name`` already is.
        ValueError: If ``method`` is not ``"iqr"`` or ``"zscore"``.
        TypeError: If ``threshold`` is not a real number.
        ValueError: If ``threshold`` is not finite and greater than zero.
        TypeError: If ``column_name`` is boolean, complex, or otherwise
            not a real numeric dtype.

    Returns:
        A copy of ``df`` with a boolean outlier column appended.
    """
    check_column(df, [column_name])
    if flag_column_name is None:
        if isinstance(df.columns, pd.MultiIndex) and isinstance(column_name, tuple):
            flag_column_name = column_name[:-1] + (f"{column_name[-1]}_outlier_flag",)
        else:
            flag_column_name = f"{column_name}_outlier_flag"
    check_column(df, [flag_column_name], present=False)

    if method not in ("iqr", "zscore"):
        raise ValueError(f"Invalid method {method!r}. Choose either 'iqr' or 'zscore'.")

    if threshold is None:
        threshold = 3.0 if method == "zscore" else 1.5
    elif isinstance(threshold, bool) or not isinstance(threshold, Real):
        raise TypeError(
            f"threshold must be a positive number, got {type(threshold).__name__}."
        )
    else:
        try:
            threshold = float(threshold)
        except OverflowError as error:
            raise ValueError(
                "threshold must be a positive number, "
                "got a value too large to convert to float."
            ) from error
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError(f"threshold must be a positive number, got {threshold}.")

    series = df[column_name]
    if isinstance(series, pd.DataFrame):
        raise ValueError(
            f"Column {column_name!r} matches more than one column. "
            "Pass a single column."
        )
    if (
        pd.api.types.is_bool_dtype(series)
        or pd.api.types.is_complex_dtype(series)
        or not pd.api.types.is_numeric_dtype(series)
    ):
        raise TypeError(f"Column {column_name!r} must be numeric to detect outliers.")

    # Float64 avoids int64 overflow in the quartile arithmetic. Infinities
    # are outliers on their own and are removed before the cutoff is built.
    # errstate covers overflow near the float64 range and 0/0 on a constant
    # column, whose Z-scores are NaN and therefore not flags.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        numeric = series.astype("float64")
        infinite = np.isinf(numeric.to_numpy())
        finite = numeric.mask(infinite)
        if method == "iqr":
            quartile_1 = finite.quantile(0.25)
            quartile_3 = finite.quantile(0.75)
            iqr = quartile_3 - quartile_1
            if pd.isna(iqr):
                outlier = np.zeros(len(df), dtype=bool)
            else:
                lower = quartile_1 - threshold * iqr
                upper = quartile_3 + threshold * iqr
                outlier = ((finite < lower) | (finite > upper)).to_numpy()
        else:
            scale = finite.std(ddof=1, skipna=True)
            if pd.isna(scale):
                outlier = np.zeros(len(df), dtype=bool)
            else:
                center = finite.mean(skipna=True)
                outlier = (((finite - center) / scale).abs() > threshold).fillna(False)
                outlier = outlier.to_numpy()

    result = df.copy()
    result[flag_column_name] = np.logical_or(outlier, infinite)
    return result
