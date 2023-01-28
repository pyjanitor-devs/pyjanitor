"""Implementation of move."""
import pandas_flavor as pf
import pandas as pd
import numpy as np

from typing import Any
from janitor.functions.utils import _select_index, _index_converter


@pf.register_dataframe_method
def move(
    df: pd.DataFrame,
    source: Any,
    target: Any = None,
    position: str = "before",
    axis: int = 0,
) -> pd.DataFrame:
    """
    Changes rows or columns positions in the dataframe. It uses the
    [`select_columns`][janitor.functions.select.select_columns] or
    [`select_rows`][janitor.functions.select.select_rows] syntax,
    making it easy to move blocks of rows or columns at once.

    This operation does not reset the index of the dataframe. User must
    explicitly do so.

    This function does not apply to multilevel dataframes, and the dataframe
    must have unique column names or indices.

    Example: Moving a row

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({"a": [2, 4, 6, 8], "b": list("wxyz")})
        >>> df
           a  b
        0  2  w
        1  4  x
        2  6  y
        3  8  z
        >>> df.move(source=0, target=3, position="before", axis=0)
           a  b
        1  4  x
        2  6  y
        0  2  w
        3  8  z

    Example: Moving a column

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({"a": [2, 4, 6], "b": [1, 3, 5], "c": [7, 8, 9]})
        >>> df
           a  b  c
        0  2  1  7
        1  4  3  8
        2  6  5  9
        >>> df.move(source="a", target="c", position="after", axis=1)
           b  c  a
        0  1  7  2
        1  3  8  4
        2  5  9  6
        >>> df.move(source = 'c', target=None, position='before', axis=1)
           c  a  b
        0  7  2  1
        1  8  4  3
        2  9  6  5
        >>> df.move(source = 'b', target=None, position='after', axis=1)
           a  c  b
        0  2  7  1
        1  4  8  3
        2  6  9  5

    :param df: The pandas DataFrame object.
    :param source: Columns or rows to move.
    :param target: Columns or rows to move adjacent to.
        If `None` and `position == 'before'`, `source`
        is moved to the beginning; if `position == 'after'`,
        `source` is moved to the end.
    :param position: Specifies the destination of the columns/rows.
        Values can be either `before` or `after`; defaults to `before`.
    :param axis: Axis along which the function is applied. 0 to move a
        row, 1 to move a column.
    :returns: The dataframe with the Series moved.
    :raises ValueError: If `axis` is not `0` or `1`.
    :raises ValueError: If `position` is not `before` or `after`.
    """
    if axis not in [0, 1]:
        raise ValueError(f"Invalid axis '{axis}'. Can only be 0 or 1.")

    if position not in ["before", "after"]:
        raise ValueError(
            f"Invalid position '{position}'. Can only be 'before' or 'after'."
        )

    mapping = {0: "index", 1: "columns"}
    names = getattr(df, mapping[axis])

    assert names.is_unique
    assert not isinstance(names, pd.MultiIndex)

    index = np.arange(names.size)
    source = _select_index([source], df, mapping[axis])
    source = _index_converter(source, index)
    if target is None:
        if position == "after":
            target = np.array([names.size])
        else:
            target = np.array([0])
    else:
        target = _select_index([target], df, mapping[axis])
        target = _index_converter(target, index)
    index = np.delete(index, source)

    if position == "before":
        position = index.searchsorted(target[0])
    else:
        position = index.searchsorted(target[-1]) + 1
    start = index[:position]
    end = index[position:]
    position = np.concatenate([start, source, end])

    return df.iloc(axis=axis)[position]
