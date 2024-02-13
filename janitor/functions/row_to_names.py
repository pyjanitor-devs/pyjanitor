"""Implementation of the `row_to_names` function."""

import warnings

import numpy as np
import pandas as pd
import pandas_flavor as pf

from janitor.utils import check, deprecated_alias


@pf.register_dataframe_method
@deprecated_alias(row_number="row_numbers", remove_row="remove_rows")
def row_to_names(
    df: pd.DataFrame,
    row_numbers: int = 0,
    remove_rows: bool = False,
    remove_rows_above: bool = False,
    reset_index: bool = False,
) -> pd.DataFrame:
    """Elevates a row, or rows, to be the column names of a DataFrame.

    This method does not mutate the original DataFrame.

    Contains options to remove the elevated row from the DataFrame along with
    removing the rows above the selected row.

    Examples:
        Replace column names with the first row and reset the index.

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({
        ...     "a": ["nums", 6, 9],
        ...     "b": ["chars", "x", "y"],
        ... })
        >>> df
              a      b
        0  nums  chars
        1     6      x
        2     9      y
        >>> df.row_to_names(0, remove_rows=True, reset_index=True)
          nums chars
        0    6     x
        1    9     y
        >>> df.row_to_names([0,1], remove_rows=True, reset_index=True)
          nums chars
          6    x
        0    9     y

        Remove rows above the elevated row and the elevated row itself.

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({
        ...     "a": ["bla1", "nums", 6, 9],
        ...     "b": ["bla2", "chars", "x", "y"],
        ... })
        >>> df
              a      b
        0  bla1   bla2
        1  nums  chars
        2     6      x
        3     9      y
        >>> df.row_to_names(1, remove_rows=True, remove_rows_above=True, reset_index=True)
          nums chars
        0    6     x
        1    9     y

    Args:
        df: A pandas DataFrame.
        row_numbers: Position of the row(s) containing the variable names.
            Note that indexing starts from 0. It can also be a list,
            in which case, a MultiIndex column is created.
            Defaults to 0 (first row).
        remove_row: Whether the row(s) should be removed from the DataFrame.
        remove_rows_above: Whether the row(s) above the selected row should
            be removed from the DataFrame.
        reset_index: Whether the index should be reset on the returning DataFrame.

    Returns:
        A pandas DataFrame with set column names.
    """  # noqa: E501
    if not pd.options.mode.copy_on_write:
        df = df.copy()

    check("row_number", row_numbers, [int, list])
    if isinstance(row_numbers, list):
        for entry in row_numbers:
            check("entry in the row_number argument", entry, [int])

    warnings.warn(
        "The function row_to_names will, in the official 1.0 release, "
        "change its behaviour to reset the dataframe's index by default. "
        "You can prepare for this change right now by explicitly setting "
        "`reset_index=True` when calling on `row_to_names`."
    )
    # should raise if positional indexers are missing
    # IndexError: positional indexers are out-of-bounds
    headers = df.iloc[row_numbers]
    if isinstance(headers, pd.DataFrame) and (len(headers) == 1):
        headers = headers.squeeze()
    if isinstance(headers, pd.Series):
        headers = pd.Index(headers)
    else:
        headers = [entry.array for _, entry in headers.items()]
        headers = pd.MultiIndex.from_tuples(headers)

    df.columns = headers
    df.columns.name = None

    df_index = df.index
    if remove_rows_above:
        if isinstance(row_numbers, list):
            if not (np.diff(row_numbers) == 1).all():
                raise ValueError(
                    "The remove_rows_above argument is applicable "
                    "only if the row_numbers argument is an integer, "
                    "or the integers in a list are consecutive increasing, "
                    "with a difference of 1."
                )
            tail = row_numbers[0]
        else:
            tail = row_numbers
        df = df.iloc[tail:]
    if remove_rows:
        if isinstance(row_numbers, int):
            row_numbers = [row_numbers]
        df_index = df.index.symmetric_difference(df_index[row_numbers])
        df = df.loc[df_index]
    if reset_index:
        df.index = range(len(df))
    return df
