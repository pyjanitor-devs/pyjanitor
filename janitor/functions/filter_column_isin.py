from typing import Hashable, Iterable
import pandas_flavor as pf
import pandas as pd

from janitor.utils import deprecated_alias


@pf.register_dataframe_method
@deprecated_alias(column="column_name")
def filter_column_isin(
    df: pd.DataFrame,
    column_name: Hashable,
    iterable: Iterable,
    complement: bool = False,
) -> pd.DataFrame:
    """
    Filter a dataframe for values in a column that exist in another iterable.

    This method does not mutate the original DataFrame.

    Assumes exact matching; fuzzy matching not implemented.

    The below example syntax will filter the DataFrame such that we only get
    rows for which the `names` are exactly `James` and `John`.

    ```python
        df = (
            pd.DataFrame(...)
            .clean_names()
            .filter_column_isin(column_name="names", iterable=["James", "John"]
            )
        )
    ```

    This is the method chaining alternative to:

    ```python
        df = df[df['names'].isin(['James', 'John'])]
    ```

    If `complement` is `True`, then we will only get rows for which the names
    are not `James` or `John`.

    :param df: A pandas DataFrame
    :param column_name: The column on which to filter.
    :param iterable: An iterable. Could be a list, tuple, another pandas
        Series.
    :param complement: Whether to return the complement of the selection or
        not.
    :returns: A filtered pandas DataFrame.
    :raises ValueError: if `iterable` does not have a length of `1`
        or greater.
    """
    if len(iterable) == 0:
        raise ValueError(
            "`iterable` kwarg must be given an iterable of length 1 or greater"
        )
    criteria = df[column_name].isin(iterable)

    if complement:
        return df[~criteria]
    return df[criteria]
