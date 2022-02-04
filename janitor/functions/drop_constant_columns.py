"""Implementation of drop_constant_columns."""
import pandas_flavor as pf
import pandas as pd


@pf.register_dataframe_method
def drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Finds and drops the constant columns from a Pandas DataFrame.

    Example:

        >>> import pandas as pd
        >>> import janitor
        >>> data_dict = {
        ...     "a": [1, 1, 1],
        ...     "b": [1, 2, 3],
        ...     "c": [1, 1, 1],
        ...     "d": ["rabbit", "leopard", "lion"],
        ...     "e": ["Cambridge", "Shanghai", "Basel"]
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

    :param df: Input Pandas DataFrame
    :returns: The Pandas DataFrame with the constant columns dropped.
    """
    # Find the constant columns
    constant_columns = []
    for col in df.columns:
        if len(df[col].unique()) == 1:
            constant_columns.append(col)

    # Drop constant columns from df and return it
    return df.drop(labels=constant_columns, axis=1)
