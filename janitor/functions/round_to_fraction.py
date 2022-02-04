"""Implementation of `round_to_fraction`"""
from typing import Hashable

import numpy as np
import pandas as pd
import pandas_flavor as pf
from janitor.utils import check, check_column, deprecated_alias


@pf.register_dataframe_method
@deprecated_alias(col_name="column_name")
def round_to_fraction(
    df: pd.DataFrame,
    column_name: Hashable,
    denominator: float,
    digits: float = np.inf,
) -> pd.DataFrame:
    """Round all values in a column to a fraction.

    This method mutates the original DataFrame.

    Taken from [the R package](https://github.com/sfirke/janitor/issues/235).

    Also, optionally round to a specified number of digits.

    Example: Round numeric column to the nearest 1/4 value.

        >>> import numpy as np
        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({
        ...     "a1": [1.263, 2.499, np.nan],
        ...     "a2": ["x", "y", "z"],
        ... })
        >>> df
              a1 a2
        0  1.263  x
        1  2.499  y
        2    NaN  z
        >>> df.round_to_fraction("a1", denominator=4)
             a1 a2
        0  1.25  x
        1  2.50  y
        2   NaN  z

    :param df: A pandas DataFrame.
    :param column_name: Name of column to round to fraction.
    :param denominator: The denominator of the fraction for rounding. Must be
        a positive number.
    :param digits: The number of digits for rounding after rounding to the
        fraction. Default is np.inf (i.e. no subsequent rounding).
    :returns: A pandas DataFrame with a column's values rounded.
    :raises ValueError: If `denominator` is not a positive number.
    """
    check_column(df, column_name)
    check("denominator", denominator, [float, int])
    check("digits", digits, [float, int])

    if denominator <= 0:
        raise ValueError("denominator is expected to be a positive number.")

    df[column_name] = round(df[column_name] * denominator, 0) / denominator
    if not np.isinf(digits):
        df[column_name] = round(df[column_name], digits)

    return df
