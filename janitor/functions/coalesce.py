from typing import Optional, Union
import pandas as pd
import pandas_flavor as pf

from janitor.utils import check, deprecated_alias
from janitor.functions.utils import _select_column_names


@pf.register_dataframe_method
@deprecated_alias(columns="column_names", new_column_name="target_column_name")
def coalesce(
    df: pd.DataFrame,
    *column_names,
    target_column_name: Optional[str] = None,
    default_value: Optional[Union[int, float, str]] = None,
) -> pd.DataFrame:
    """
    Coalesce two or more columns of data in order of column names provided.

    This finds the first non-missing value at each position.

    This method does not mutate the original DataFrame.

    TODO: Turn the example in this docstring into a Jupyter notebook.

    Example:

    ```python
        import pandas as pd
        import janitor as jn

        df = pd.DataFrame({"A": [1, 2, np.nan],
                           "B": [np.nan, 10, np.nan],
                           "C": [5, 10, 7]})

             A     B   C
        0  1.0   NaN   5
        1  2.0  10.0  10
        2  NaN   NaN   7

        df.coalesce('A', 'B', 'C',
                    target_column_name = 'D')

            A     B   C    D
        0  1.0   NaN   5  1.0
        1  2.0  10.0  10  2.0
        2  NaN   NaN   7  7.0
    ```

    If no target column is provided, then the first column is updated,
    with the null values removed:

    ```python
        df.coalesce('A', 'B', 'C')

            A     B   C
        0  1.0   NaN   5
        1  2.0  10.0  10
        2  7.0   NaN   7
    ```

    If nulls remain, you can fill it with the `default_value`:

    ```python
        df = pd.DataFrame({'s1':[np.nan,np.nan,6,9,9],
                           's2':[np.nan,8,7,9,9]})

            s1   s2
        0  NaN  NaN
        1  NaN  8.0
        2  6.0  7.0
        3  9.0  9.0
        4  9.0  9.0

        df.coalesce('s1', 's2',
                    target_column_name = 's3',
                    default_value = 0)

            s1   s2   s3
        0  NaN  NaN  0.0
        1  NaN  8.0  8.0
        2  6.0  7.0  6.0
        3  9.0  9.0  9.0
        4  9.0  9.0  9.0
    ```


    Functional usage syntax:

    ```python
        df = coalesce(df, 'col1', 'col2', target_column_name ='col3')
    ```

    Method chaining syntax:

    ```python
        import pandas as pd
        import janitor
        df = pd.DataFrame(...).coalesce('col1', 'col2')
    ```

    The first example will create a new column called `col3` with values from
    `col2` inserted where values from `col1` are `NaN`.
    The second example will update the values of `col1`,
    since it is the first column in `column_names`.

    This is more syntactic diabetes! For R users, this should look familiar to
    `dplyr`'s `coalesce` function; for Python users, the interface
    should be more intuitive than the `pandas.Series.combine_first`
    method.

    :param df: A pandas DataFrame.
    :param column_names: A list of column names.
    :param target_column_name: The new column name after combining.
        If `None`, then the first column in `column_names` is updated,
        with the Null values replaced.
    :param default_value: A scalar to replace any remaining nulls
        after coalescing.
    :returns: A pandas DataFrame with coalesced columns.
    :raises ValueError: if length of `column_names` is less than 2.
    """

    if not column_names:
        return df

    if len(column_names) < 2:
        raise ValueError(
            """
            The number of columns to coalesce
            should be a minimum of 2.
            """
        )

    column_names = [*column_names]

    column_names = _select_column_names(column_names, df)
    if target_column_name:
        check("target_column_name", target_column_name, [str])
    if default_value:
        check("default_value", default_value, [int, float, str])

    if target_column_name is None:
        target_column_name = column_names[0]
    # bfill/ffill combo is faster than combine_first
    outcome = (
        df.filter(column_names)
        .bfill(axis="columns")
        .ffill(axis="columns")
        .iloc[:, 0]
    )
    if outcome.hasnans and (default_value is not None):
        outcome = outcome.fillna(default_value)
    return df.assign(**{target_column_name: outcome})
