"""Implementation of `shuffle` functions."""
import pandas_flavor as pf
import pandas as pd


@pf.register_dataframe_method
def shuffle(
    df: pd.DataFrame, random_state=None, reset_index: bool = True
) -> pd.DataFrame:
    """Shuffle the rows of the DataFrame.

    This method does not mutate the original DataFrame.

    Super-sugary syntax! Underneath the hood, we use `df.sample(frac=1)`,
    with the option to set the random state.

    Example:

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({
        ...     "col1": range(5),
        ...     "col2": list("abcde"),
        ... })
        >>> df
           col1 col2
        0     0    a
        1     1    b
        2     2    c
        3     3    d
        4     4    e
        >>> df.shuffle(random_state=42)
           col1 col2
        0     1    b
        1     4    e
        2     2    c
        3     0    a
        4     3    d

    :param df: A pandas DataFrame.
    :param random_state: If provided, set a seed for the random number
        generator.
    :param reset_index: If True, reset the dataframe index to the default
        RangeIndex.
    :returns: A shuffled pandas DataFrame.
    """
    result = df.sample(frac=1, random_state=random_state)
    if reset_index:
        result = result.reset_index(drop=True)
    return result
