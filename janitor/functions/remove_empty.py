"""Implementation of remove_empty."""
import pandas_flavor as pf
import pandas as pd


@pf.register_dataframe_method
def remove_empty(df: pd.DataFrame) -> pd.DataFrame:
    """Drop all rows and columns that are completely null.

    This method also resets the index (by default) since it doesn't make sense
    to preserve the index of a completely empty row.

    This method mutates the original DataFrame.

    Implementation is inspired from [StackOverflow][so].

    [so]: https://stackoverflow.com/questions/38884538/python-pandas-find-all-rows-where-all-values-are-nan

    Example:

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

    :param df: The pandas DataFrame object.
    :returns: A pandas DataFrame.
    """  # noqa: E501
    nanrows = df.index[df.isna().all(axis=1)]
    df = df.drop(index=nanrows).reset_index(drop=True)

    nancols = df.columns[df.isna().all(axis=0)]
    df = df.drop(columns=nancols)

    return df
