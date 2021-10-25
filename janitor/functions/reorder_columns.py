from typing import Hashable, Iterable, Union
import pandas_flavor as pf
import pandas as pd

from janitor.utils import check


@pf.register_dataframe_method
def reorder_columns(
    df: pd.DataFrame, column_order: Union[Iterable[str], pd.Index, Hashable]
) -> pd.DataFrame:
    """Reorder DataFrame columns by specifying desired order as list of col names.

    Columns not specified retain their order and follow after specified cols.

    Validates column_order to ensure columns are all present in DataFrame.

    This method does not mutate the original DataFrame.

    Functional usage syntax:

    Given `DataFrame` with column names `col1`, `col2`, `col3`:

    ```python
    df = reorder_columns(df, ['col2', 'col3'])
    ```

    Method chaining syntax:

    ```python
    import pandas as pd
    import janitor
    df = pd.DataFrame(...).reorder_columns(['col2', 'col3'])
    ```

    The column order of `df` is now `col2`, `col3`, `col1`.

    Internally, this function uses `DataFrame.reindex` with `copy=False`
    to avoid unnecessary data duplication.

    :param df: `DataFrame` to reorder
    :param column_order: A list of column names or Pandas `Index`
        specifying their order in the returned `DataFrame`.
    :returns: A pandas DataFrame with reordered columns.
    :raises IndexError: if a column within `column_order` is not found
        within the DataFrame.
    """
    check("column_order", column_order, [list, tuple, pd.Index])

    if any(col not in df.columns for col in column_order):
        raise IndexError(
            "A column in `column_order` was not found in the DataFrame."
        )

    # if column_order is a Pandas index, needs conversion to list:
    column_order = list(column_order)

    return df.reindex(
        columns=(
            column_order
            + [col for col in df.columns if col not in column_order]
        ),
        copy=False,
    )
