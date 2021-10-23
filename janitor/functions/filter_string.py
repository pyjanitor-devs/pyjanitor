from typing import Hashable
import pandas_flavor as pf
import pandas as pd

from janitor.utils import deprecated_alias


@pf.register_dataframe_method
@deprecated_alias(column="column_name")
def filter_string(
    df: pd.DataFrame,
    column_name: Hashable,
    search_string: str,
    complement: bool = False,
) -> pd.DataFrame:
    """
    Filter a string-based column according to whether it contains a substring.

    This is super sugary syntax that builds on top of
    `pandas.Series.str.contains`.

    Because this uses internally `pandas.Series.str.contains`, which allows a
    regex string to be passed into it, thus `search_string` can also be a regex
    pattern.

    This method does not mutate the original DataFrame.

    This function allows us to method chain filtering operations:

    ```python
        df = (pd.DataFrame(...)
              .filter_string('column', search_string='pattern', complement=False)
              ...)  # chain on more data preprocessing.
    ```

    This stands in contrast to the in-place syntax that is usually used:

    ```python
        df = pd.DataFrame(...)
        df = df[df['column'].str.contains('pattern')]]
    ```

    As can be seen here, the API design allows for a more seamless flow in
    expressing the filtering operations.

    Functional usage syntax:

    ```python
        df = filter_string(df,
                           column_name='column',
                           search_string='pattern',
                           complement=False)
    ```

    Method chaining syntax:

    ```python
        df = (pd.DataFrame(...)
              .filter_string(column_name='column',
                             search_string='pattern',
                             complement=False)
              ...)
    ```

    :param df: A pandas DataFrame.
    :param column_name: The column to filter. The column should contain strings.
    :param search_string: A regex pattern or a (sub-)string to search.
    :param complement: Whether to return the complement of the filter or not.
    :returns: A filtered pandas DataFrame.
    """  # noqa: E501
    criteria = df[column_name].str.contains(search_string)
    if complement:
        return df[~criteria]
    return df[criteria]
