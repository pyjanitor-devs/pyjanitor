from typing import Hashable, Iterable, Union
import warnings
import pandas_flavor as pf
import pandas as pd

from janitor.utils import deprecated_alias
from janitor.functions.utils import _factorize


@pf.register_dataframe_method
@deprecated_alias(columns="column_names")
def label_encode(
    df: pd.DataFrame, column_names: Union[str, Iterable[str], Hashable]
) -> pd.DataFrame:
    """
    Convert labels into numerical data.

    This method will create a new column with the string `_enc` appended
    after the original column's name. Consider this to be syntactic sugar.

    This method behaves differently from `encode_categorical`. This method
    creates a new column of numeric data. `encode_categorical` replaces the
    dtype of the original column with a *categorical* dtype.

    This method mutates the original DataFrame.

    Functional usage syntax:

    ```python
    df = label_encode(df, column_names="my_categorical_column")  # one way
    ```

    Method chaining syntax:

    ```python
    import pandas as pd
    import janitor
    categorical_cols = ['col1', 'col2', 'col4']
    df = pd.DataFrame(...).label_encode(column_names=categorical_cols)
    ```

    :param df: The pandas DataFrame object.
    :param column_names: A column name or an iterable (list
        or tuple) of column names.
    :returns: A pandas DataFrame.
    """
    warnings.warn(
        "label_encode will be deprecated in a 1.x release. \
        Please use factorize_columns instead"
    )
    df = _factorize(df, column_names, "_enc")
    return df
