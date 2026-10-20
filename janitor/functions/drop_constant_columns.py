"""Implementation of drop_constant_columns."""

import pandas as pd
import pandas_flavor as pf


@pf.register_dataframe_method
def drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Finds and drops the constant columns from a Pandas DataFrame.

    Missing values count as a single distinct value, even across different
    missing sentinels: a column with one non-null value and missing entries
    is kept, while an all-missing column is dropped.

    Examples:
        >>> import pandas as pd
        >>> import janitor
        >>> data_dict = {
        ...     "a": [1, 1, 1],
        ...     "b": [1, 2, 3],
        ...     "c": [1, 1, 1],
        ...     "d": ["rabbit", "leopard", "lion"],
        ...     "e": ["Cambridge", "Shanghai", "Basel"],
        ... }
        >>> df = pd.DataFrame(data_dict)
        >>> df
           a  b  c        d          e
        0  1  1  1   rabbit  Cambridge
        1  1  2  1  leopard   Shanghai
        2  1  3  1     lion      Basel
        >>> df.drop_constant_columns()
           b        d          e
        0  1   rabbit  Cambridge
        1  2  leopard   Shanghai
        2  3     lion      Basel

    Args:
        df: Input Pandas DataFrame

    Returns:
        The Pandas DataFrame with the constant columns dropped.
    """
    # Count missing sentinels as a single distinct level, even when a column
    # mixes None, NaN, and pandas.NA in object dtype.
    distinct_levels = df.nunique(dropna=True) + df.isna().any(axis=0).astype(int)
    return df.loc[:, distinct_levels.ne(1)]
