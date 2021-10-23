from typing import Callable
import pandas_flavor as pf
import pandas as pd


@pf.register_dataframe_method
def join_apply(
    df: pd.DataFrame, func: Callable, new_column_name: str
) -> pd.DataFrame:
    """
    Join the result of applying a function across dataframe rows.

    This method does not mutate the original DataFrame.

    This is a convenience function that allows us to apply arbitrary functions
    that take any combination of information from any of the columns. The only
    requirement is that the function signature takes in a row from the
    DataFrame.

    The example below shows us how to sum the result of two columns into a new
    column.

    ```python
        df = (
            pd.DataFrame({'a':[1, 2, 3], 'b': [2, 3, 4]})
            .join_apply(lambda x: 2 * x['a'] + x['b'], new_column_name="2a+b")
        )
    ```

    This following example shows us how to use conditionals in the same
    function.

    ```python
        def take_a_if_even(x):
            if x['a'] % 2:
                return x['a']
            else:
                return x['b']

        df = (
            pd.DataFrame({'a': [1, 2, 3], 'b': [2, 3, 4]})
            .join_apply(take_a_if_even, 'a_if_even')
        )
    ```

    :param df: A pandas DataFrame
    :param func: A function that is applied elementwise across all rows of the
        DataFrame.
    :param new_column_name: New column name.
    :returns: A pandas DataFrame with new column appended.
    """
    df = df.copy().join(df.apply(func, axis=1).rename(new_column_name))
    return df
