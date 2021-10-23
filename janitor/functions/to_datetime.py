from typing import Hashable
import pandas_flavor as pf
import pandas as pd

from janitor.utils import deprecated_alias


@pf.register_dataframe_method
@deprecated_alias(column="column_name")
def to_datetime(
    df: pd.DataFrame, column_name: Hashable, **kwargs
) -> pd.DataFrame:
    """
    Method-chainable `pd.to_datetime`.

    This method mutates the original DataFrame.

    Functional usage syntax:

    ```python
        df = to_datetime(df, 'col1', format='%Y%m%d')
    ```

    Method chaining syntax:

    ```python
        import pandas as pd
        import janitor
        df = pd.DataFrame(...).to_datetime('col1', format='%Y%m%d')
    ```

    :param df: A pandas DataFrame.
    :param column_name: Column name.
    :param kwargs: provide any kwargs that `pd.to_datetime` can take.
    :returns: A pandas DataFrame with updated datetime data.
    """
    df[column_name] = pd.to_datetime(df[column_name], **kwargs)

    return df
