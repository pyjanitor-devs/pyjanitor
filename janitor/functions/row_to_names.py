"""Implementation of the `row_to_names` function."""

from __future__ import annotations

from functools import singledispatch

import numpy as np
import pandas as pd
import pandas_flavor as pf

from janitor.utils import check, deprecated_alias


@pf.register_dataframe_method
@deprecated_alias(row_number="row_numbers", remove_row="remove_rows")
def row_to_names(
    df: pd.DataFrame,
    row_numbers: int | list | slice = 0,
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
             6     x
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
            It can be an integer, a list or a slice.
            Defaults to 0 (first row).
        remove_rows: Whether the row(s) should be removed from the DataFrame.
        remove_rows_above: Whether the row(s) above the selected row should
            be removed from the DataFrame.
        reset_index: Whether the index should be reset on the returning DataFrame.

    Returns:
        A pandas DataFrame with set column names.
    """  # noqa: E501

    return _row_to_names(
        row_numbers,
        df=df,
        remove_rows=remove_rows,
        remove_rows_above=remove_rows_above,
        reset_index=reset_index,
    )


@singledispatch
def _row_to_names(
    row_numbers, df, remove_rows, remove_rows_above, reset_index
) -> pd.DataFrame:
    """
    Base function for row_to_names.
    """
    raise TypeError(
        "row_numbers should be either an integer, "
        "a slice or a list; "
        f"instead got type {type(row_numbers).__name__}"
    )


@_row_to_names.register(int)  # noqa: F811
def _row_to_names_dispatch(  # noqa: F811
    row_numbers, df, remove_rows, remove_rows_above, reset_index
):
    df_ = df[:]
    headers = df_.iloc[row_numbers]
    df_.columns = headers
    df_.columns.name = None
    if not remove_rows and not remove_rows_above and not reset_index:
        return df_
    if not remove_rows and not remove_rows_above and reset_index:
        return df_.reset_index(drop=True)

    len_df = len(df_)
    arrays = [arr._values for _, arr in df_.items()]
    if remove_rows_above and remove_rows:
        indexer = np.arange(row_numbers + 1, len_df)
    elif remove_rows_above:
        indexer = np.arange(row_numbers, len_df)
    elif remove_rows:
        indexer = np.arange(len_df)
        mask = np.ones(len_df, dtype=np.bool_)
        mask[row_numbers] = False
        indexer = indexer[mask]
    arrays = {num: arr[indexer] for num, arr in enumerate(arrays)}
    if reset_index:
        df_index = pd.RangeIndex(start=0, stop=indexer.size)
    else:
        df_index = df_.index[indexer]
    _df = pd.DataFrame(data=arrays, index=df_index, copy=False)
    _df.columns = df_.columns
    return _df


@_row_to_names.register(slice)  # noqa: F811
def _row_to_names_dispatch(  # noqa: F811
    row_numbers, df, remove_rows, remove_rows_above, reset_index
):
    if row_numbers.step is not None:
        raise ValueError(
            "The step argument for slice is not supported in row_to_names."
        )
    df_ = df[:]
    headers = df_.iloc[row_numbers]
    if isinstance(headers, pd.DataFrame) and (len(headers) == 1):
        headers = headers.squeeze()
        df_.columns = headers
        df_.columns.name = None
    else:
        headers = [array._values for _, array in headers.items()]
        headers = pd.MultiIndex.from_tuples(headers)
        df_.columns = headers
    if not remove_rows and not remove_rows_above and not reset_index:
        return df_
    if not remove_rows and not remove_rows_above and reset_index:
        return df_.reset_index(drop=True)
    len_df = len(df_)
    arrays = [arr._values for _, arr in df_.items()]
    if remove_rows_above and remove_rows:
        indexer = np.arange(row_numbers.stop, len_df)
    elif remove_rows_above:
        indexer = np.arange(row_numbers.start, len_df)
    elif remove_rows:
        indexer = np.arange(len_df)
        mask = np.ones(len_df, dtype=np.bool_)
        mask[row_numbers] = False
        indexer = indexer[mask]
    arrays = {num: arr[indexer] for num, arr in enumerate(arrays)}
    if reset_index:
        df_index = pd.RangeIndex(start=0, stop=indexer.size)
    else:
        df_index = df_.index[indexer]
    _df = pd.DataFrame(data=arrays, index=df_index, copy=False)
    _df.columns = df_.columns
    return _df


@_row_to_names.register(list)  # noqa: F811
def _row_to_names_dispatch(  # noqa: F811
    row_numbers, df, remove_rows, remove_rows_above, reset_index
):
    if remove_rows_above:
        raise ValueError(
            "The remove_rows_above argument is applicable "
            "only if the row_numbers argument is an integer "
            "or a slice."
        )

    for entry in row_numbers:
        check("entry in the row_numbers argument", entry, [int])

    df_ = df[:]
    headers = df_.iloc[row_numbers]
    if isinstance(headers, pd.DataFrame) and (len(headers) == 1):
        headers = headers.squeeze()
        df_.columns = headers
        df_.columns.name = None
    else:
        headers = [array._values for _, array in headers.items()]
        headers = pd.MultiIndex.from_tuples(headers)
        df_.columns = headers

    if not remove_rows and reset_index:
        return df_.reset_index(drop=True)
    if not remove_rows and not reset_index:
        return df_

    len_df = len(df_)
    arrays = [arr._values for _, arr in df_.items()]
    indexer = np.arange(len_df)
    mask = np.ones(len_df, dtype=np.bool_)
    mask[row_numbers] = False
    indexer = indexer[mask]

    arrays = {num: arr[indexer] for num, arr in enumerate(arrays)}
    if reset_index:
        df_index = pd.RangeIndex(start=0, stop=indexer.size)
    else:
        df_index = df_.index[indexer]
    _df = pd.DataFrame(data=arrays, index=df_index, copy=False)
    _df.columns = df_.columns
    return _df
