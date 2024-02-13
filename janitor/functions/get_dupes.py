"""Implementation of the `get_dupes` function"""

from typing import Hashable, Iterable, Optional, Union

import pandas as pd
import pandas_flavor as pf

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

    Examples:
        Method chaining syntax:

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({
        ...     "item": ["shoe", "shoe", "bag", "shoe", "bag"],
        ...     "quantity": [100, 100, 75, 200, 75],
        ... })
        >>> df
           item  quantity
        0  shoe       100
        1  shoe       100
        2   bag        75
        3  shoe       200
        4   bag        75
        >>> df.get_dupes()
           item  quantity
        0  shoe       100
        1  shoe       100
        2   bag        75
        4   bag        75

        Optional `column_names` usage:

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({
        ...     "item": ["shoe", "shoe", "bag", "shoe", "bag"],
        ...     "quantity": [100, 100, 75, 200, 75],
        ... })
        >>> df
           item  quantity
        0  shoe       100
        1  shoe       100
        2   bag        75
        3  shoe       200
        4   bag        75
        >>> df.get_dupes(column_names=["item"])
           item  quantity
        0  shoe       100
        1  shoe       100
        2   bag        75
        3  shoe       200
        4   bag        75
        >>> df.get_dupes(column_names=["quantity"])
           item  quantity
        0  shoe       100
        1  shoe       100
        2   bag        75
        4   bag        75

    Args:
        df: The pandas DataFrame object.
        column_names: A column name or an iterable
            (list or tuple) of column names. Following pandas API, this only
            considers certain columns for identifying duplicates. Defaults
            to using all columns.

    Returns:
        The duplicate rows, as a pandas DataFrame.
    """
    return df.loc[df.duplicated(subset=column_names, keep=False)]
