"""Implementation of deconcatenating columns."""
from typing import Hashable, List, Optional, Tuple, Union
import pandas_flavor as pf
import pandas as pd
from janitor.errors import JanitorError

from janitor.utils import deprecated_alias


@pf.register_dataframe_method
@deprecated_alias(column="column_name")
def deconcatenate_column(
    df: pd.DataFrame,
    column_name: Hashable,
    sep: Optional[str] = None,
    new_column_names: Optional[Union[List[str], Tuple[str]]] = None,
    autoname: str = None,
    preserve_position: bool = False,
) -> pd.DataFrame:
    """De-concatenates a single column into multiple columns.

    The column to de-concatenate can be either a collection (list, tuple, ...)
    which can be separated out with `pd.Series.tolist()`,
    or a string to slice based on `sep`.

    To determine this behaviour automatically,
    the first element in the column specified is inspected.

    If it is a string, then `sep` must be specified.
    Else, the function assumes that it is an iterable type
    (e.g. `list` or `tuple`),
    and will attempt to deconcatenate by splitting the list.

    Given a column with string values, this is the inverse of the
    [`concatenate_columns`][janitor.functions.concatenate_columns.concatenate_columns]
    function.

    Used to quickly split columns out of a single column.

    Example:

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({"m": ["1-x", "2-y", "3-z"]})
        >>> df
             m
        0  1-x
        1  2-y
        2  3-z
        >>> df.deconcatenate_column("m", sep="-", autoname="col")
             m col1 col2
        0  1-x    1    x
        1  2-y    2    y
        2  3-z    3    z

    The keyword argument `preserve_position`
    takes `True` or `False` boolean
    that controls whether the `new_column_names`
    will take the original position
    of the to-be-deconcatenated `column_name`:

    - When `preserve_position=False` (default), `df.columns` change from
      `[..., column_name, ...]` to `[..., column_name, ..., new_column_names]`.
      In other words, the deconcatenated new columns are appended to the right
      of the original dataframe and the original `column_name` is NOT dropped.
    - When `preserve_position=True`, `df.column` change from
      `[..., column_name, ...]` to `[..., new_column_names, ...]`.
      In other words, the deconcatenated new column will REPLACE the original
      `column_name` at its original position, and `column_name` itself
      is dropped.

    The keyword argument `autoname` accepts a base string
    and then automatically creates numbered column names
    based off the base string.
    For example, if `col` is passed in as the argument to `autoname`,
    and 4 columns are created, then the resulting columns will be named
    `col1, col2, col3, col4`.
    Numbering is always 1-indexed, not 0-indexed,
    in order to make the column names human-friendly.

    This method does not mutate the original DataFrame.

    :param df: A pandas DataFrame.
    :param column_name: The column to split.
    :param sep: The separator delimiting the column's data.
    :param new_column_names: A list of new column names post-splitting.
    :param autoname: A base name for automatically naming the new columns.
        Takes precedence over `new_column_names` if both are provided.
    :param preserve_position: Boolean for whether or not to preserve original
        position of the column upon de-concatenation.
    :returns: A pandas DataFrame with a deconcatenated column.
    :raises ValueError: If `column_name` is not present in the
        DataFrame.
    :raises ValueError: If `sep` is not provided and the column values
        are of type `str`.
    :raises ValueError: If either `new_column_names` or `autoname`
        is not supplied.
    :raises JanitorError: If incorrect number of names is provided
        within `new_column_names`.
    """  # noqa: E501

    if column_name not in df.columns:
        raise ValueError(f"column name {column_name} not present in DataFrame")

    if isinstance(df[column_name].iloc[0], str):
        if sep is None:
            raise ValueError(
                "`sep` must be specified if the column values "
                "are of type `str`."
            )
        df_deconcat = df[column_name].str.split(sep, expand=True)
    else:
        df_deconcat = pd.DataFrame(
            df[column_name].to_list(), columns=new_column_names, index=df.index
        )

    if new_column_names is None and autoname is None:
        raise ValueError(
            "One of `new_column_names` or `autoname` must be supplied."
        )

    if autoname:
        new_column_names = [
            f"{autoname}{i}" for i in range(1, df_deconcat.shape[1] + 1)
        ]

    if not len(new_column_names) == df_deconcat.shape[1]:
        raise JanitorError(
            f"You need to provide {len(df_deconcat.shape[1])} names "
            "to `new_column_names`"
        )

    df_deconcat.columns = new_column_names
    df_new = pd.concat([df, df_deconcat], axis=1)

    if preserve_position:
        df_original = df.copy()
        cols = list(df_original.columns)
        index_original = cols.index(column_name)

        for i, col_new in enumerate(new_column_names):
            cols.insert(index_original + i, col_new)

        df_new = df_new.select_columns(cols).drop(columns=column_name)

    return df_new
