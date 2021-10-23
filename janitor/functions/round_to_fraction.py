from typing import Hashable
import pandas_flavor as pf
import pandas as pd

from janitor.utils import check, deprecated_alias
import numpy as np


@pf.register_dataframe_method
@deprecated_alias(col_name="column_name")
def round_to_fraction(
    df: pd.DataFrame,
    column_name: Hashable = None,
    denominator: float = None,
    digits: float = np.inf,
) -> pd.DataFrame:
    """
    Round all values in a column to a fraction.

    This method mutates the original DataFrame.

    Taken from [Source](https://github.com/sfirke/janitor/issues/235).

    Also, optionally round to a specified number of digits.

    Method-chaining usage:

    ```python
        # Round to two decimal places
        df = pd.DataFrame(...).round_to_fraction('a', 2)
    ```

    :param df: A pandas DataFrame.
    :param column_name: Name of column to round to fraction.
    :param denominator: The denominator of the fraction for rounding
    :param digits: The number of digits for rounding after rounding to the
        fraction. Default is np.inf (i.e. no subsequent rounding)
    :returns: A pandas DataFrame with a column's values rounded.
    """
    if denominator:
        check("denominator", denominator, [float, int])

    if digits:
        check("digits", digits, [float, int])

    df[column_name] = round(df[column_name] * denominator, 0) / denominator
    if not np.isinf(digits):
        df[column_name] = round(df[column_name], digits)

    return df
