from typing import Hashable
import pandas_flavor as pf
import pandas as pd

from janitor.utils import deprecated_alias


@pf.register_dataframe_method
@deprecated_alias(column="column_name")
def dropnotnull(df: pd.DataFrame, column_name: Hashable) -> pd.DataFrame:
    """
    Drop rows that do not have null values in the given column.

    This method does not mutate the original DataFrame.

    Example usage:

    ```python
        df = pd.DataFrame(...).dropnotnull('column3')
    ```

    :param df: A pandas DataFrame.
    :param column_name: The column name to drop rows from.
    :returns: A pandas DataFrame with dropped rows.
    """
    return df[pd.isna(df[column_name])]
