from typing import Hashable, Iterable, Union
import pandas_flavor as pf
import pandas as pd

from janitor.functions.utils import _factorize


@pf.register_dataframe_method
def factorize_columns(
    df: pd.DataFrame,
    column_names: Union[str, Iterable[str], Hashable],
    suffix: str = "_enc",
    **kwargs,
) -> pd.DataFrame:
    """
    Converts labels into numerical data

    This method will create a new column with the string `_enc` appended
    after the original column's name.
    This can be overriden with the suffix parameter.

    Internally this method uses pandas `factorize` method.
    It takes in an optional suffix and keyword arguments also.
    An empty string as suffix will override the existing column.

    This method mutates the original DataFrame.

    Functional usage syntax:

    ```python
    df = factorize_columns(
        df,
        column_names="my_categorical_column",
        suffix="_enc"
    )  # one way
    ```

    Method chaining syntax:

    ```python
    import pandas as pd
    import janitor
    categorical_cols = ['col1', 'col2', 'col4']
    df = (
        pd.DataFrame(...)
        .factorize_columns(
            column_names=categorical_cols,
            suffix="_enc"
        )
    )
    ```

    :param df: The pandas DataFrame object.
    :param column_names: A column name or an iterable (list
        or tuple) of column names.
    :param suffix: Suffix to be used for the new column. Default value is _enc.
        An empty string suffix means, it will override the existing column
    :param **kwargs: Keyword arguments. It takes any of the keyword arguments,
        which the pandas factorize method takes like sort,na_sentinel,size_hint

    :returns: A pandas DataFrame.
    """
    df = _factorize(df, column_names, suffix, **kwargs)
    return df
