"""Implementation of move."""
import pandas_flavor as pf
import pandas as pd

from typing import Union


@pf.register_dataframe_method
def move(
    df: pd.DataFrame,
    source: Union[int, str],
    target: Union[int, str],
    position: str = "before",
    axis: int = 0,
) -> pd.DataFrame:
    """
    Moves a column or row to a position adjacent to another column or row in
    the dataframe.

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

    :param df: The pandas DataFrame object.
    :param source: Column or row to move.
    :param target: Column or row to move adjacent to.
    :param position: Specifies whether the Series is moved to before or
        after the adjacent Series. Values can be either `before` or `after`;
        defaults to `before`.
    :param axis: Axis along which the function is applied. 0 to move a
        row, 1 to move a column.
    :returns: The dataframe with the Series moved.
    :raises ValueError: If `axis` is not `0` or `1`.
    :raises ValueError: If `position` is not `before` or `after`.
    :raises ValueError: If  `source` row or column is not in dataframe.
    :raises ValueError: If `target` row or column is not in dataframe.
    """
    if axis not in [0, 1]:
        raise ValueError(f"Invalid axis '{axis}'. Can only be 0 or 1.")

    if position not in ["before", "after"]:
        raise ValueError(
            f"Invalid position '{position}'. Can only be 'before' or 'after'."
        )

    df = df.copy()
    if axis == 0:
        names = list(df.index)

        if source not in names:
            raise ValueError(f"Source row '{source}' not in dataframe.")

        if target not in names:
            raise ValueError(f"Target row '{target}' not in dataframe.")

        names.remove(source)
        pos = names.index(target)

        if position == "after":
            pos += 1
        names.insert(pos, source)

        df = df.loc[names, :]
    else:
        names = list(df.columns)

        if source not in names:
            raise ValueError(f"Source column '{source}' not in dataframe.")

        if target not in names:
            raise ValueError(f"Target column '{target}' not in dataframe.")

        names.remove(source)
        pos = names.index(target)

        if position == "after":
            pos += 1
        names.insert(pos, source)

        df = df.loc[:, names]

    return df
