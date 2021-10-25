from typing import Hashable, Iterable, Union
import pandas_flavor as pf
import pandas as pd

from janitor.utils import deprecated_alias


@pf.register_dataframe_method
@deprecated_alias(columns="column_names")
def remove_columns(
    df: pd.DataFrame, column_names: Union[str, Iterable[str], Hashable]
) -> pd.DataFrame:
    """Remove the set of columns specified in `column_names`.

    This method does not mutate the original DataFrame.

    Intended to be the method-chaining alternative to `del df[col]`.

    Method chaining syntax:

        df = pd.DataFrame(...).remove_columns(column_names=['col1', 'col2'])

    :param df: A pandas DataFrame
    :param column_names: The columns to remove.
    :returns: A pandas DataFrame.
    """
    return df.drop(columns=column_names)
