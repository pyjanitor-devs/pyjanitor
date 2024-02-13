"""Implementation of remove_empty."""

import pandas as pd
import pandas_flavor as pf


@pf.register_dataframe_method
def remove_empty(df: pd.DataFrame, reset_index: bool = True) -> pd.DataFrame:
    """Drop all rows and columns that are completely null.

    This method does not mutate the original DataFrame.

    Implementation is inspired from [StackOverflow][so].

    [so]: https://stackoverflow.com/questions/38884538/python-pandas-find-all-rows-where-all-values-are-nan

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({
        ...     "a": [1, np.nan, 2],
        ...     "b": [3, np.nan, 4],
        ...     "c": [np.nan, np.nan, np.nan],
        ... })
        >>> df
             a    b   c
        0  1.0  3.0 NaN
        1  NaN  NaN NaN
        2  2.0  4.0 NaN
        >>> df.remove_empty()
             a    b
        0  1.0  3.0
        1  2.0  4.0

    Args:
        df: The pandas DataFrame object.
        reset_index: Determines if the index is reset.

    Returns:
        A pandas DataFrame.
    """  # noqa: E501
    outcome = df.isna()
    outcome = df.loc[~outcome.all(axis=1), ~outcome.all(axis=0)]
    if reset_index:
        return outcome.reset_index(drop=True)
    return outcome
