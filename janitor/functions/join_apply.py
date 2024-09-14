"""Implementation of the `join_apply` function"""

from typing import Callable

import pandas as pd
import pandas_flavor as pf


@pf.register_dataframe_method
def join_apply(
    df: pd.DataFrame,
    func: Callable,
    new_column_name: str,
) -> pd.DataFrame:
    """Join the result of applying a function across dataframe rows.

    This method does not mutate the original DataFrame.

    This is a convenience function that allows us to apply arbitrary functions
    that take any combination of information from any of the columns. The only
    requirement is that the function signature takes in a row from the
    DataFrame.

    Examples:
        Sum the result of two columns into a new column.

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({"a":[1, 2, 3], "b": [2, 3, 4]})
        >>> df
           a  b
        0  1  2
        1  2  3
        2  3  4
        >>> df.join_apply(
        ...     func=lambda x: 2 * x["a"] + x["b"],
        ...     new_column_name="2a+b",
        ... )
           a  b  2a+b
        0  1  2     4
        1  2  3     7
        2  3  4    10

        Incorporating conditionals in `func`.

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({"a": [1, 2, 3], "b": [20, 30, 40]})
        >>> df
           a   b
        0  1  20
        1  2  30
        2  3  40
        >>> def take_a_if_even(x):
        ...     if x["a"] % 2 == 0:
        ...         return x["a"]
        ...     else:
        ...         return x["b"]
        >>> df.join_apply(take_a_if_even, "a_if_even")
           a   b  a_if_even
        0  1  20         20
        1  2  30          2
        2  3  40         40

    Args:
        df: A pandas DataFrame.
        func: A function that is applied elementwise across all rows of the
            DataFrame.
        new_column_name: Name of the resulting column.

    Returns:
        A pandas DataFrame with new column appended.
    """  # noqa: E501
    df = df.copy().join(df.apply(func, axis=1).rename(new_column_name))
    return df
