"""Implementation for `drop_duplicate_columns`."""

from typing import Hashable

import pandas as pd
import pandas_flavor as pf


@pf.register_dataframe_method
def drop_duplicate_columns(
    df: pd.DataFrame, column_name: Hashable, nth_index: int = 0
) -> pd.DataFrame:
    """Remove a duplicated column specified by `column_name`.

    Specifying `nth_index=0` will remove the first column,
    `nth_index=1` will remove the second column,
    and so on and so forth.

    The corresponding tidyverse R's library is:
    `select(-<column_name>_<nth_index + 1>)`

    Examples:
        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({
        ...     "a": range(2, 5),
        ...     "b": range(3, 6),
        ...     "A": range(4, 7),
        ...     "a*": range(6, 9),
        ... }).clean_names(remove_special=True)
        >>> df
           a  b  a  a
        0  2  3  4  6
        1  3  4  5  7
        2  4  5  6  8
        >>> df.drop_duplicate_columns(column_name="a", nth_index=1)
           a  b  a
        0  2  3  6
        1  3  4  7
        2  4  5  8

    Args:
        df: A pandas DataFrame
        column_name: Name of duplicated columns.
        nth_index: Among the duplicated columns,
            select the nth column to drop.

    Returns:
        A pandas DataFrame
    """
    col_indexes = [
        col_idx
        for col_idx, col_name in enumerate(df.columns)
        if col_name == column_name
    ]

    # Select the column to remove based on nth_index.
    removed_col_idx = col_indexes[nth_index]
    # Filter out columns except for the one to be removed.
    filtered_cols = [
        c_i for c_i, _ in enumerate(df.columns) if c_i != removed_col_idx
    ]

    return df.iloc[:, filtered_cols]
