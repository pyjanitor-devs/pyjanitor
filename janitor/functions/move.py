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
    Move column or row to a position adjacent to another column or row in
    dataframe. Must have unique column names or indices.

    This operation does not reset the index of the dataframe. User must
    explicitly do so.

    Does not apply to multilevel dataframes.

    Functional usage syntax:

    ```python
    df = move(df, source=3, target=15, position='after', axis=0)
    ```

    Method chaining syntax:

    ```python
    import pandas as pd
    import janitor
    df = (
        pd.DataFrame(...)
        .move(source=3, target=15, position='after', axis=0)
    )
    ```

    :param df: The pandas Dataframe object.
    :param source: column or row to move
    :param target: column or row to move adjacent to
    :param position: Specifies whether the Series is moved to before or
        after the adjacent Series. Values can be either `before` or `after`;
        defaults to `before`.
    :param axis: Axis along which the function is applied. 0 to move a
        row, 1 to move a column.
    :returns: The dataframe with the Series moved.
    :raises ValueError: if `axis` is not `0` or `1``.
    :raises ValueError: if `position` is not `before` or `after``.
    :raises ValueError: if  `source` row or column is not in dataframe.
    :raises ValueError: if `target` row or column is not in dataframe.
    """
    df = df.copy()
    if axis not in [0, 1]:
        raise ValueError(f"Invalid axis '{axis}'. Can only be 0 or 1.")

    if position not in ["before", "after"]:
        raise ValueError(
            f"Invalid position '{position}'. Can only be 'before' or 'after'."
        )

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
