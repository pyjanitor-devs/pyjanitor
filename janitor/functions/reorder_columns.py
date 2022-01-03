from typing import Hashable, Iterable, Union
import pandas_flavor as pf
import pandas as pd

from janitor.utils import check


@pf.register_dataframe_method
def reorder_columns(
    df: pd.DataFrame, column_order: Union[Iterable[str], pd.Index, Hashable]
) -> pd.DataFrame:
    """Reorder DataFrame columns by specifying desired order as list of col names.

    Columns not specified retain their order and follow after the columns specified
    in `column_order`.

    All columns specified within the `column_order` list must be present within `df`.

    This method does not mutate the original DataFrame.

    Example:

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({"col1": [1, 1, 1], "col2": [2, 2, 2], "col3": [3, 3, 3]})
        >>> df
           col1  col2  col3
        0     1     2     3
        1     1     2     3
        2     1     2     3
        >>> df.reorder_columns(['col3', 'col1'])
           col3  col1  col2
        0     3     1     2
        1     3     1     2
        2     3     1     2

    Notice that the column order of `df` is now `col3`, `col1`, `col2`.

    Internally, this function uses `DataFrame.reindex` with `copy=False`
    to avoid unnecessary data duplication.

    :param df: `DataFrame` to reorder
    :param column_order: A list of column names or Pandas `Index`
        specifying their order in the returned `DataFrame`.
    :returns: A pandas DataFrame with reordered columns.
    :raises IndexError: if a column within `column_order` is not found
        within the DataFrame.
    """  # noqa: E501
    check("column_order", column_order, [list, tuple, pd.Index])

    if any(col not in df.columns for col in column_order):
        raise IndexError(
            "One or more columns in `column_order` were not found in the "
            "DataFrame."
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
