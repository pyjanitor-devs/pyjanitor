"""Implementation of move."""

from typing import Any

import numpy as np
import pandas as pd
import pandas_flavor as pf

from janitor.functions.select import _index_converter, _select_index


@pf.register_dataframe_method
def move(
    df: pd.DataFrame,
    source: Any,
    target: Any = None,
    position: str = "before",
    axis: int = 0,
) -> pd.DataFrame:
    """Changes rows or columns positions in the dataframe.

    It uses the
    [`select`][janitor.functions.select.select] syntax,
    making it easy to move blocks of rows or columns at once.

    This operation does not reset the index of the dataframe. User must
    explicitly do so.

    The dataframe must have unique column names or indices.

    Examples:
        Move a row:
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

        Move a column:
        >>> import pandas as pd
        >>> import janitor
        >>> data = [{"a": 1, "b": 1, "c": 1,
        ...          "d": "a", "e": "a","f": "a"}]
        >>> df = pd.DataFrame(data)
        >>> df
           a  b  c  d  e  f
        0  1  1  1  a  a  a
        >>> df.move(source="a", target="c", position="after", axis=1)
           b  c  a  d  e  f
        0  1  1  1  a  a  a
        >>> df.move(source="f", target="b", position="before", axis=1)
           a  f  b  c  d  e
        0  1  a  1  1  a  a
        >>> df.move(source="a", target=None, position="after", axis=1)
           b  c  d  e  f  a
        0  1  1  a  a  a  1

        Move columns:
        >>> from pandas.api.types import is_numeric_dtype, is_string_dtype
        >>> df.move(source=is_string_dtype, target=None, position="before", axis=1)
           d  e  f  a  b  c
        0  a  a  a  1  1  1
        >>> df.move(source=is_numeric_dtype, target=None, position="after", axis=1)
           d  e  f  a  b  c
        0  a  a  a  1  1  1
        >>> df.move(source = ["d", "f"], target=is_numeric_dtype, position="before", axis=1)
           d  f  a  b  c  e
        0  a  a  1  1  1  a

    Args:
        df: The pandas DataFrame object.
        source: Columns or rows to move.
        target: Columns or rows to move adjacent to.
            If `None` and `position == 'before'`, `source`
            is moved to the beginning; if `position == 'after'`,
            `source` is moved to the end.
        position: Specifies the destination of the columns/rows.
            Values can be either `before` or `after`; defaults to `before`.
        axis: Axis along which the function is applied. 0 to move along
            the index, 1 to move along the columns.

    Raises:
        ValueError: If `axis` is not `0` or `1`.
        ValueError: If `position` is not `before` or `after`.

    Returns:
        The dataframe with the Series moved.
    """  # noqa: E501
    if axis not in [0, 1]:
        raise ValueError(f"Invalid axis '{axis}'. Can only be 0 or 1.")

    if position not in ["before", "after"]:
        raise ValueError(
            f"Invalid position '{position}'. Can only be 'before' or 'after'."
        )

    mapping = {0: "index", 1: "columns"}
    names = getattr(df, mapping[axis])

    assert names.is_unique

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
