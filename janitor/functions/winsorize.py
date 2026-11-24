"""Implementation of the `winsorize` function."""

from __future__ import annotations

from numbers import Real
from typing import Any, Literal

import pandas as pd
import pandas_flavor as pf
from pandas.api.types import (
    is_bool_dtype,
    is_extension_array_dtype,
    is_integer_dtype,
    is_numeric_dtype,
)

from janitor.functions.select import get_index_labels
from janitor.utils import check, check_column

WINSORIZE_METHODS = {"quantile", "iqr"}


@pf.register_dataframe_method
def winsorize(
    df: pd.DataFrame,
    column_names: Any = None,
    limits: tuple[float | None, float | None] = (0.05, 0.95),
    method: Literal["quantile", "iqr"] = "quantile",
    iqr_factor: float = 1.5,
    by: Any = None,
    suffix: str | None = None,
) -> pd.DataFrame:
    """Cap extreme values in numeric columns (winsorization).

    Values below a lower bound are set to the lower bound, and values above
    an upper bound are set to the upper bound. Unlike dropping outliers,
    winsorizing keeps every row, which makes it a common cleaning step
    before aggregation, reporting or model training.

    Two ways of computing the bounds are supported:

    - `method="quantile"` (default): the bounds are the `limits` quantiles
      of each column. `limits=(0.05, 0.95)` caps the bottom and top 5%.
    - `method="iqr"`: the bounds are Tukey's fences,
      `Q1 - iqr_factor * IQR` and `Q3 + iqr_factor * IQR`, where
      `IQR = Q3 - Q1`.

    Bounds are computed per column. If `by` is given, they are computed
    within each group instead, so that each group is capped against its
    own distribution. Rows whose group key is missing are left unchanged.

    Missing values are ignored when computing bounds and stay missing.
    Integer columns are returned as floats, since the bounds are usually
    not whole numbers.

    This method does not mutate the original DataFrame.

    Examples:
        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame(
        ...     {"revenue": [10, 12, 11, 13, 500], "units": [1, 2, 3, 4, 5]}
        ... )
        >>> df.winsorize("revenue", limits=(0.1, 0.9))
           revenue  units
        0     10.4      1
        1     12.0      2
        2     11.0      3
        3     13.0      4
        4    305.2      5

        Use Tukey's fences instead of quantiles.

        >>> df.winsorize("revenue", method="iqr")
           revenue  units
        0     10.0      1
        1     12.0      2
        2     11.0      3
        3     13.0      4
        4     16.0      5

        Cap only the upper tail, and keep the original column.

        >>> df.winsorize("revenue", limits=(None, 0.8), suffix="_capped")
           revenue  units  revenue_capped
        0       10      1            10.0
        1       12      2            12.0
        2       11      3            11.0
        3       13      4            13.0
        4      500      5           110.4

        Cap values within each group.

        >>> sales = pd.DataFrame(
        ...     {
        ...         "store": ["a", "a", "a", "b", "b", "b"],
        ...         "amount": [1, 2, 30, 100, 200, 3000],
        ...     }
        ... )
        >>> sales.winsorize("amount", limits=(0, 0.5), by="store")
          store  amount
        0     a     1.0
        1     a     2.0
        2     a     2.0
        3     b   100.0
        4     b   200.0
        5     b   200.0

    Args:
        df: A pandas DataFrame.
        column_names: The columns to winsorize. Accepts any selection that
            [`select`][janitor.functions.select.select] accepts, such as a
            column name, a list of names, a glob string or a callable.
            Defaults to every numeric (non-boolean) column that is not
            in `by`.
        limits: The `(lower, upper)` quantiles used as bounds when
            `method="quantile"`. Each must be between 0 and 1. Use `None`
            on one side to leave that tail untouched. Ignored when
            `method="iqr"`.
        method: How to compute the bounds, either `"quantile"` or `"iqr"`.
        iqr_factor: How many IQRs below Q1 and above Q3 the bounds lie when
            `method="iqr"`. Must be non-negative. Ignored when
            `method="quantile"`.
        by: Optional column(s) to group by. Bounds are computed separately
            for each group.
        suffix: If given, the winsorized values are written to new columns
            named `<column><suffix>`, and the original columns are kept.
            Otherwise, the original columns are overwritten.

    Raises:
        TypeError: If `method`, `iqr_factor` or `suffix` has the wrong type.
        TypeError: If a selected column is not numeric.
        ValueError: If `method` is not `"quantile"` or `"iqr"`.
        ValueError: If `limits` is not a pair of quantiles between 0 and 1
            with the lower one smaller than the upper one.
        ValueError: If `iqr_factor` is negative.
        ValueError: If a column in `by` is also selected for winsorizing.
        ValueError: If a column created by `suffix` already exists.

    Returns:
        A pandas DataFrame with the extreme values capped.
    """
    check("df", df, [pd.DataFrame])
    check("method", method, [str])
    if method not in WINSORIZE_METHODS:
        raise ValueError(f"`method` must be 'quantile' or 'iqr', not {method!r}.")
    if method == "quantile":
        lower_q, upper_q = _validate_limits(limits)
    else:
        if not _is_real_number(iqr_factor):
            raise TypeError("`iqr_factor` must be an int or a float.")
        if iqr_factor < 0:
            raise ValueError("`iqr_factor` must be non-negative.")
    if suffix is not None:
        check("suffix", suffix, [str])
        if not suffix:
            raise ValueError("`suffix` must be a non-empty string.")

    by_columns = [] if by is None else list(get_index_labels([by], df, axis="columns"))
    if column_names is None:
        numeric = df.select_dtypes(include="number").columns
        selected = [col for col in numeric if col not in by_columns]
    else:
        selected = list(get_index_labels([column_names], df, axis="columns"))
        overlap = [col for col in selected if col in by_columns]
        if overlap:
            raise ValueError(
                f"Columns {overlap} are used in `by` and cannot be winsorized."
            )
    for col in selected:
        if not is_numeric_dtype(df[col]) or is_bool_dtype(df[col]):
            raise TypeError(
                f"Column {col!r} must be numeric to be winsorized; "
                f"got dtype {df[col].dtype}."
            )
    if suffix is not None:
        check_column(df, [f"{col}{suffix}" for col in selected], present=False)

    result = df.copy()
    # Group keys are passed as arrays so they are matched by position,
    # which also works when the index has duplicate labels.
    keys = [result[col].array for col in by_columns] or None
    for col in selected:
        values = _as_float(result[col])
        if method == "quantile":
            lower = _quantile(values, lower_q, keys)
            upper = _quantile(values, upper_q, keys)
        else:
            q1 = _quantile(values, 0.25, keys)
            q3 = _quantile(values, 0.75, keys)
            spread = iqr_factor * (q3 - q1)
            lower, upper = q1 - spread, q3 + spread
        target = col if suffix is None else f"{col}{suffix}"
        result[target] = values.clip(lower=lower, upper=upper)

    return result


def _is_real_number(value: Any) -> bool:
    """Return True for ints and floats, but not booleans."""
    return isinstance(value, Real) and not isinstance(value, bool)


def _validate_limits(limits: Any) -> tuple[float | None, float | None]:
    """Check that `limits` is a valid `(lower, upper)` pair of quantiles."""
    if not isinstance(limits, (tuple, list)) or len(limits) != 2:
        raise ValueError("`limits` must be a (lower, upper) pair.")
    lower, upper = limits
    if lower is None and upper is None:
        raise ValueError("At least one of the `limits` must not be None.")
    for name, value in (("lower", lower), ("upper", upper)):
        if value is None:
            continue
        if not _is_real_number(value) or not 0 <= value <= 1:
            raise ValueError(
                f"The {name} limit must be a number between 0 and 1, not {value!r}."
            )
    if lower is not None and upper is not None and lower >= upper:
        raise ValueError("The lower limit must be smaller than the upper limit.")
    return lower, upper


def _as_float(series: pd.Series) -> pd.Series:
    """Convert integer columns to float so they can hold fractional bounds."""
    if not is_integer_dtype(series):
        return series
    if is_extension_array_dtype(series.dtype):
        return series.astype("Float64")
    return series.astype("float64")


def _quantile(
    series: pd.Series,
    q: float | None,
    keys: list | None,
) -> float | pd.Series | None:
    """Return the `q` quantile of `series`, per group if `keys` are given.

    A grouped quantile is returned as a Series aligned with `series`, so it
    can be passed straight to `Series.clip`. Rows whose group key is missing
    get NaN, which `Series.clip` treats as no bound.
    """
    if q is None:
        return None
    if keys is None:
        return series.quantile(q)
    return series.groupby(keys, sort=False).transform("quantile", q)
