from typing import Hashable, Iterable, Optional, Union
import pandas_flavor as pf
import pandas as pd
import numpy as np

from janitor.utils import check_column


@pf.register_dataframe_method
def flag_nulls(
    df: pd.DataFrame,
    column_name: Optional[Hashable] = "null_flag",
    columns: Optional[Union[str, Iterable[str], Hashable]] = None,
) -> pd.DataFrame:
    """Creates a new column to indicate whether you have null values in a given
    row. If the columns parameter is not set, looks across the entire
    DataFrame, otherwise will look only in the columns you set.

    This method does not mutate the original DataFrame.

    Example:

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({
        ...     "a": ["w", "x", None, "z"], "b": [5, None, 7, 8],
        ... })
        >>> df.flag_nulls()
              a    b  null_flag
        0     w  5.0          0
        1     x  NaN          1
        2  None  7.0          1
        3     z  8.0          0
        >>> df.flag_nulls(columns="b")
              a    b  null_flag
        0     w  5.0          0
        1     x  NaN          1
        2  None  7.0          0
        3     z  8.0          0

    :param df: Input pandas DataFrame.
    :param column_name: Name for the output column.
    :param columns: List of columns to look at for finding null values. If you
        only want to look at one column, you can simply give its name. If set
        to None (default), all DataFrame columns are used.
    :returns: Input dataframe with the null flag column.
    :raises ValueError: if `column_name` is already present in the
        DataFrame.
    :raises ValueError: if any column within `columns` is not present in
        the DataFrame.

    <!--
    # noqa: DAR402
    -->
    """
    # Sort out columns input
    if isinstance(columns, str):
        columns = [columns]
    elif columns is None:
        columns = df.columns
    elif not isinstance(columns, Iterable):
        # catches other hashable types
        columns = [columns]

    # Input sanitation checks
    check_column(df, columns)
    check_column(df, [column_name], present=False)

    # This algorithm works best for n_rows >> n_cols. See issue #501
    null_array = np.zeros(len(df))
    for col in columns:
        null_array = np.logical_or(null_array, pd.isna(df[col]))

    df = df.copy()
    df[column_name] = null_array.astype(int)
    return df
