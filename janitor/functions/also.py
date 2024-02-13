"""Implementation source for chainable function `also`."""

from typing import Any, Callable

import pandas as pd
import pandas_flavor as pf


@pf.register_dataframe_method
def also(
    df: pd.DataFrame, func: Callable, *args: Any, **kwargs: Any
) -> pd.DataFrame:
    """Run a function with side effects.

    This function allows you to run an arbitrary function
    in the `pyjanitor` method chain.
    Doing so will let you do things like save the dataframe to disk midway
    while continuing to modify the dataframe afterwards.

    Examples:
        >>> import pandas as pd
        >>> import janitor
        >>> df = (
        ...     pd.DataFrame({"a": [1, 2, 3], "b": list("abc")})
        ...     .query("a > 1")
        ...     .also(lambda df: print(f"DataFrame shape is: {df.shape}"))
        ...     .rename_column(old_column_name="a", new_column_name="a_new")
        ...     .also(lambda df: df.to_csv("midpoint.csv"))
        ...     .also(
        ...         lambda df: print(f"Columns: {df.columns}")
        ...     )
        ... )
        DataFrame shape is: (2, 2)
        Columns: Index(['a_new', 'b'], dtype='object')

    Args:
        df: A pandas DataFrame.
        func: A function you would like to run in the method chain.
            It should take one DataFrame object as a parameter and have no return.
            If there is a return, it will be ignored.
        *args: Optional arguments for `func`.
        **kwargs: Optional keyword arguments for `func`.

    Returns:
        The input pandas DataFrame, unmodified.
    """  # noqa: E501
    func(df.copy(), *args, **kwargs)
    return df
