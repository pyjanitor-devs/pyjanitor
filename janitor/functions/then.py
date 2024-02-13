"""Implementation source for `then`."""

from typing import Callable

import pandas as pd
import pandas_flavor as pf

from janitor.utils import refactored_function


@pf.register_dataframe_method
@refactored_function(
    message="This function will be deprecated in a 1.x release. "
    "Kindly use `pd.DataFrame.pipe` instead."
)
def then(df: pd.DataFrame, func: Callable) -> pd.DataFrame:
    """Add an arbitrary function to run in the `pyjanitor` method chain.

    This method does not mutate the original DataFrame.

    !!!note

        This function will be deprecated in a 1.x release.
        Please use `pd.DataFrame.pipe` instead.

    Examples:
        A trivial example using a lambda `func`.

        >>> import pandas as pd
        >>> import janitor
        >>> (pd.DataFrame({"a": [1, 2, 3], "b": [7, 8, 9]})
        ...  .then(lambda df: df * 2))
           a   b
        0  2  14
        1  4  16
        2  6  18

    Args:
        df: A pandas DataFrame.
        func: A function you would like to run in the method chain.
            It should take one parameter and return one parameter, each being
            the DataFrame object. After that, do whatever you want in the
            middle. Go crazy.

    Returns:
        A pandas DataFrame.
    """
    df = func(df)
    return df
