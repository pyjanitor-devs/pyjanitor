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

    Example usage:
    ```python
        df = pd.DataFrame(...).shuffle()
    ```

    :param df: A pandas DataFrame
    :param random_state: (optional) A seed for the random number generator.
    :param reset_index: (optional) Resets index to default integers
    :returns: A shuffled pandas DataFrame.
    """
    result = df.sample(frac=1, random_state=random_state)
    if reset_index:
        result = result.reset_index(drop=True)
    return result
