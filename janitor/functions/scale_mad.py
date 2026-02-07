"""Implementation of the `scale_mad` function."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Hashable, Literal

import pandas as pd
import pandas_flavor as pf
from pandas.api.types import is_numeric_dtype

from janitor.utils import check, check_column

ZeroMadStrategy = Literal["skip", "center", "raise", "one"]

MAD_SCALE_FACTOR = 1.4826
ZERO_MAD_STRATEGIES: set[str] = {"skip", "center", "raise", "one"}


def _median_absolute_deviation(series: pd.Series) -> float:
    """Return the median absolute deviation for a pandas Series."""
    median = series.median(skipna=True)
    return series.sub(median).abs().median(skipna=True)


def _normalize_columns(
    df: pd.DataFrame,
    columns: Iterable[Hashable] | Callable[[pd.DataFrame], Iterable[Hashable]] | None,
) -> list[Hashable]:
    """Normalize columns input into a list of column labels."""
    if columns is None:
        return list(df.select_dtypes(include=["number"]).columns)

    if callable(columns):
        columns = columns(df)
        if columns is None:
            raise TypeError("`columns` callable must return column names.")

    if isinstance(columns, pd.Index):
        columns = list(columns)
    elif isinstance(columns, tuple) and columns in df.columns:
        columns = [columns]
    elif isinstance(columns, str) or not isinstance(columns, Iterable):
        columns = [columns]
    else:
        columns = list(columns)

    return list(dict.fromkeys(columns))


@pf.register_dataframe_method
def scale_mad(
    df: pd.DataFrame,
    columns: Iterable[Hashable]
    | Callable[[pd.DataFrame], Iterable[Hashable]]
    | None = None,
    clip: float | None = None,
    zero_mad: ZeroMadStrategy = "skip",
    suffix: str | None = None,
) -> pd.DataFrame:
    """Robustly scale numeric columns using median and MAD.

    The median absolute deviation (MAD) is scaled by 1.4826 to make it
    comparable to the standard deviation under a normal distribution.

    This method does not mutate the original DataFrame.

    Examples:
        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [10, 10, 10, 10]})
        >>> df.scale_mad().round(3)
               x   y
        0 -1.012  10
        1 -0.337  10
        2  0.337  10
        3  1.012  10

        Create a new column and clip the results.

        >>> df.scale_mad(columns=["x"], suffix="_mad", clip=3).round(3)
           x   y  x_mad
        0  1  10 -1.012
        1  2  10 -0.337
        2  3  10  0.337
        3  4  10  1.012

    Args:
        df: A pandas DataFrame.
        columns: Columns to scale. Can be an iterable of column names or a
            callable that takes the DataFrame and returns column names.
            Defaults to all numeric columns.
        clip: Optional symmetric clipping threshold. If provided, values are
            clipped to ``[-clip, clip]``.
        zero_mad: Strategy for columns with zero (or missing) MAD.
            "skip" leaves the column unchanged, "center" subtracts the median,
            and "raise" raises a ValueError. "one" is accepted as an alias for
            "center".
        suffix: Optional suffix to append to scaled column names. If provided,
            new columns are created; otherwise, original columns are
            overwritten.

    Raises:
        TypeError: If `columns` or `clip` has an invalid type.
        ValueError: If `clip` is negative or `zero_mad` is invalid.
        ValueError: If `zero_mad` is "raise" and MAD is zero.

    Returns:
        A pandas DataFrame with scaled values.
    """
    check("df", df, [pd.DataFrame])

    if clip is not None:
        check("clip", clip, [int, float])
        if clip < 0:
            raise ValueError("`clip` must be non-negative.")

    check("zero_mad", zero_mad, [str])
    zero_mad = zero_mad.lower()
    if zero_mad not in ZERO_MAD_STRATEGIES:
        raise ValueError("zero_mad must be one of 'skip', 'center', 'one', or 'raise'.")
    if zero_mad == "one":
        zero_mad = "center"

    if suffix is not None:
        check("suffix", suffix, [str])
        if suffix == "":
            suffix = None

    selected_columns = _normalize_columns(df, columns)
    if selected_columns:
        check_column(df, selected_columns)

    result = df.copy()
    for col in selected_columns:
        series = result[col]
        if not is_numeric_dtype(series):
            continue
        median = series.median(skipna=True)
        mad = _median_absolute_deviation(series)
        if mad == 0 or pd.isna(mad):
            if zero_mad == "skip":
                scaled = series
            elif zero_mad == "center":
                scaled = series - median
            else:
                raise ValueError(f"MAD is zero for column '{col}'.")
        else:
            scaled = (series - median) / (mad * MAD_SCALE_FACTOR)
        if clip is not None:
            scaled = scaled.clip(lower=-clip, upper=clip)
        target = f"{col}{suffix}" if suffix is not None else col
        if suffix is not None:
            check_column(result, target, present=False)
        result[target] = scaled

    return result
