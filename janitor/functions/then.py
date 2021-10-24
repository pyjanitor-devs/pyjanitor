from typing import Callable
import pandas_flavor as pf
import pandas as pd


@pf.register_dataframe_method
def then(df: pd.DataFrame, func: Callable) -> pd.DataFrame:
    """
    Add an arbitrary function to run in the `pyjanitor` method chain.

    This method does not mutate the original DataFrame.

    :param df: A pandas dataframe.
    :param func: A function you would like to run in the method chain.
        It should take one parameter and return one parameter, each being the
        DataFrame object. After that, do whatever you want in the middle.
        Go crazy.
    :returns: A pandas DataFrame.
    """
    df = func(df)
    return df
