"""Implementation source for `flag_outliers`."""

from typing import Hashable, Iterable, Literal, Optional, Union

import pandas as pd
import pandas_flavor as pf


def _check_column(
    df: pd.DataFrame,
    column_names: Union[Iterable, str],
    present: bool = True,
):
    """Check presence or absence of columns in a DataFrame."""
    if isinstance(column_names, str) or not isinstance(column_names, Iterable):
        column_names = [column_names]
    for column_name in column_names:
        if present and column_name not in df.columns:
            raise ValueError(f"{column_name} not present in dataframe columns!")
        elif not present and column_name in df.columns:
            raise ValueError(f"{column_name} already present in dataframe columns!")


@pf.register_dataframe_method
def flag_outliers(
    df: pd.DataFrame,
    column_name: str,
    method: Literal["iqr", "zscore"] = "iqr",
    threshold: float = 1.5,
    flag_column_name: Optional[Hashable] = None,
) -> pd.DataFrame:
    """Creates a new boolean column flagging outlier values in a numeric column.

    Supports two detection methods:

    - ``iqr``: Flags values below ``Q1 - threshold * IQR`` or above
      ``Q3 + threshold * IQR``.
    - ``zscore``: Flags values whose absolute Z-score exceeds ``threshold``
      (default threshold should be set to 3.0 for Z-score method).

    This method does not mutate the original DataFrame.

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

    Args:
        df: Input pandas DataFrame.
        column_name: Name of the numeric column to check for outliers.
        method: Outlier detection method. Either ``"iqr"`` (default) or
            ``"zscore"``.
        threshold: Multiplier for IQR method (default ``1.5``) or the
            Z-score cutoff (commonly ``3.0``). Must be a positive number.
        flag_column_name: Name for the output boolean flag column. Defaults
            to ``"<column_name>_outlier_flag"`` if not provided.

    Raises:
        ValueError: If ``column_name`` is not present in the DataFrame.
        ValueError: If ``flag_column_name`` is already present in the
            DataFrame.
        ValueError: If ``method`` is not one of ``"iqr"`` or ``"zscore"``.
        ValueError: If ``threshold`` is not a positive number.
        TypeError: If the specified column is not numeric.

    Returns:
        Input DataFrame with a new boolean outlier flag column appended.

    <!--
    # noqa: DAR402
    -->
    """
    _check_column(df, [column_name])

    if flag_column_name is None:
        flag_column_name = f"{column_name}_outlier_flag"

    _check_column(df, [flag_column_name], present=False)

    if method not in ("iqr", "zscore"):
        raise ValueError(
            f"Invalid method '{method}'. Choose either 'iqr' or 'zscore'."
        )

    if threshold <= 0:
        raise ValueError(
            f"threshold must be a positive number, got {threshold}."
        )

    if not pd.api.types.is_numeric_dtype(df[column_name]):
        raise TypeError(
            f"Column '{column_name}' must be numeric to detect outliers."
        )

    series = df[column_name]

    if method == "iqr":
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        outlier_mask = (series < lower) | (series > upper)
    else:
        mean = series.mean()
        std = series.std()
        if std == 0:
            outlier_mask = pd.Series([False] * len(df), index=df.index)
        else:
            outlier_mask = ((series - mean) / std).abs() > threshold

    df = df.copy()
    df[flag_column_name] = outlier_mask
    return df
