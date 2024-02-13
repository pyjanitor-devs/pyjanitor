"""Implementation source for `dropnotnull`."""

from typing import Hashable

import pandas as pd
import pandas_flavor as pf

from janitor.utils import deprecated_alias


@pf.register_dataframe_method
@deprecated_alias(column="column_name")
def dropnotnull(df: pd.DataFrame, column_name: Hashable) -> pd.DataFrame:
    """Drop rows that do *not* have null values in the given column.

    This method does not mutate the original DataFrame.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({"a": [1., np.NaN, 3.], "b": [None, "y", "z"]})
        >>> df
             a     b
        0  1.0  None
        1  NaN     y
        2  3.0     z
        >>> df.dropnotnull("a")
            a  b
        1 NaN  y
        >>> df.dropnotnull("b")
             a     b
        0  1.0  None

    Args:
        df: A pandas DataFrame.
        column_name: The column name to drop rows from.

    Returns:
        A pandas DataFrame with dropped rows.
    """
    return df[pd.isna(df[column_name])]
