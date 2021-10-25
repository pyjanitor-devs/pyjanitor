"""Implementation source for chainable function `also`."""
from typing import Callable
import pandas_flavor as pf
import pandas as pd


@pf.register_dataframe_method
def also(df: pd.DataFrame, func: Callable, *args, **kwargs) -> pd.DataFrame:
    """
    Run a function with side effects.

    THis function allows you to run an arbitrary function
    in the `pyjanitor` method chain.
    Doing so will let you do things like save the dataframe to disk midway
    while continuing to modify the dataframe afterwards.

    Example usage:

    ```python
    df = (
        pd.DataFrame(...)
        .query(...)
        .also(lambda df: print(f"DataFrame shape is: {df.shape}"))
        .transform_column(...)
        .also(lambda df: df.to_csv("midpoint.csv"))
        .also(
            lambda df: print(
                f"Column col_name has these values: {set(df['col_name'].unique())}"
            )
        )
        .group_add(...)
    )
    ```

    :param df: A pandas dataframe.
    :param func: A function you would like to run in the method chain.
        It should take one DataFrame object as a parameter and have no return.
        If there is a return, it will be ignored.
    :param args: Optional arguments for `func`.
    :param kwargs: Optional keyword arguments for `func`.
    :returns: The input pandas DataFrame.
    """  # noqa: E501
    func(df.copy(), *args, **kwargs)
    return df
