"""Implementation of the `row_to_names` function."""
import warnings
import pandas_flavor as pf
import pandas as pd

from janitor.utils import check


@pf.register_dataframe_method
def row_to_names(
    df: pd.DataFrame,
    row_number: int = 0,
    remove_row: bool = False,
    remove_rows_above: bool = False,
    reset_index: bool = False,
) -> pd.DataFrame:
    """Elevates a row to be the column names of a DataFrame.

    This method mutates the original DataFrame.

    Contains options to remove the elevated row from the DataFrame along with
    removing the rows above the selected row.

    Example: Replace column names with the first row and reset the index.

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
        >>> df.row_to_names(0, remove_row=True, reset_index=True)
          nums chars
        0    6     x
        1    9     y

    Example: Remove rows above the elevated row and the elevated row itself.

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
        >>> df.row_to_names(1, remove_row=True, remove_rows_above=True, reset_index=True)
          nums chars
        0    6     x
        1    9     y

    :param df: A pandas DataFrame.
    :param row_number: Position of the row containing the variable names.
        Note that indexing starts from 0. Defaults to 0 (first row).
    :param remove_row: Whether the row should be removed from the DataFrame.
        Defaults to False.
    :param remove_rows_above: Whether the rows above the selected row should
        be removed from the DataFrame. Defaults to False.
    :param reset_index: Whether the index should be reset on the returning
        DataFrame. Defaults to False.
    :returns: A pandas DataFrame with set column names.
    """  # noqa: E501

    check("row_number", row_number, [int])

    warnings.warn(
        "The function row_to_names will, in the official 1.0 release, "
        "change its behaviour to reset the dataframe's index by default. "
        "You can prepare for this change right now by explicitly setting "
        "`reset_index=True` when calling on `row_to_names`."
    )

    df.columns = df.iloc[row_number, :]
    df.columns.name = None

    if remove_row:
        df = df.drop(df.index[row_number])

    if remove_rows_above:
        df = df.drop(df.index[range(row_number)])

    if reset_index:
        df = df.reset_index(drop=["index"])

    return df
