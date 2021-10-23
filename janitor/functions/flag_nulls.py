from typing import Hashable, Iterable, Optional, Union
import pandas_flavor as pf
import pandas as pd
import numpy as np

from janitor.utils import check_column


@pf.register_dataframe_method
def flag_nulls(
    df: pd.DataFrame,
    column_name: Optional[Hashable] = "null_flag",
    columns: Optional[Union[str, Iterable[str], Hashable]] = None,
) -> pd.DataFrame:
    """Creates a new column to indicate whether you have null values in a given
    row. If the columns parameter is not set, looks across the entire
    DataFrame, otherwise will look only in the columns you set.

    ```python
    import pandas as pd
    import janitor as jn

    df = pd.DataFrame(
        {'a': [1, 2, None, 4],
            'b': [5.0, None, 7.0, 8.0]})

    df.flag_nulls()

    jn.functions.flag_nulls(df)

    df.flag_nulls(columns=['b'])
    ```

    :param df: Input Pandas dataframe.
    :param column_name: Name for the output column. Defaults to 'null_flag'.
    :param columns: List of columns to look at for finding null values. If you
        only want to look at one column, you can simply give its name. If set
        to None (default), all DataFrame columns are used.
    :returns: Input dataframe with the null flag column.
    :raises ValueError: if `column_name` is already present in the
        DataFrame.
    :raises ValueError: if a column within `columns` is no present in
        the DataFrame.

    .. # noqa: DAR402
    """
    # Sort out columns input
    if isinstance(columns, str):
        columns = [columns]
    elif columns is None:
        columns = df.columns
    elif not isinstance(columns, Iterable):
        # catches other hashable types
        columns = [columns]

    # Input sanitation checks
    check_column(df, columns)
    check_column(df, [column_name], present=False)

    # This algorithm works best for n_rows >> n_cols. See issue #501
    null_array = np.zeros(len(df))
    for col in columns:
        null_array = np.logical_or(null_array, pd.isna(df[col]))

    df = df.copy()
    df[column_name] = null_array.astype(int)
    return df
