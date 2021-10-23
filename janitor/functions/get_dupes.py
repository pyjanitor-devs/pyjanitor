from typing import Hashable, Iterable, Optional, Union
import pandas_flavor as pf
import pandas as pd

from janitor.utils import deprecated_alias


@pf.register_dataframe_method
@deprecated_alias(columns="column_names")
def get_dupes(
    df: pd.DataFrame,
    column_names: Optional[Union[str, Iterable[str], Hashable]] = None,
) -> pd.DataFrame:
    """
    Return all duplicate rows.

    This method does not mutate the original DataFrame.

    Functional usage syntax:

    ```python
    df = pd.DataFrame(...)
    df = get_dupes(df)
    ```

    Method chaining syntax:

    ```python
    import pandas as pd
    import janitor
    df = pd.DataFrame(...).get_dupes()
    ```

    :param df: The pandas DataFrame object.
    :param column_names: (optional) A column name or an iterable
        (list or tuple) of column names. Following pandas API, this only
        considers certain columns for identifying duplicates. Defaults to using
        all columns.
    :returns: The duplicate rows, as a pandas DataFrame.
    """
    dupes = df.duplicated(subset=column_names, keep=False)
    return df[dupes == True]  # noqa: E712
