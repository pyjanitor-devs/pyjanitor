from typing import Callable, List, Union
import pandas_flavor as pf
import pandas as pd

from janitor.utils import deprecated_alias


@pf.register_dataframe_method
@deprecated_alias(new_column="new_column_name", agg_column="agg_column_name")
def groupby_agg(
    df: pd.DataFrame,
    by: Union[List, str],
    new_column_name: str,
    agg_column_name: str,
    agg: Union[Callable, str],
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Shortcut for assigning a groupby-transform to a new column.

    This method does not mutate the original DataFrame.

    Without this function, we would have to write a verbose line:

    ```python
        df = df.assign(...=df.groupby(...)[...].transform(...))
    ```

    Now, this function can be method-chained:

    ```python
        import pandas as pd
        import janitor
        df = pd.DataFrame(...).groupby_agg(by='group',
                                           agg='mean',
                                           agg_column_name="col1"
                                           new_column_name='col1_mean_by_group',
                                           dropna = True/False)
    ```

    Functional usage syntax:

    ```python

        import pandas as pd
        import janitor as jn

        jn.groupby_agg(
            df,
            by= column name/list of column names,
            agg=aggregation function,
            agg_column_name = col,
            new_column_name= new column name,
            dropna = True/False)
    ```

    Method chaining usage syntax:

    ```python

        df.groupby_agg(
            by= column name/list of column names,
            agg=aggregation function,
            agg_column_name = col,
            new_column_name= new column name,
            dropna = True/False)
    ```


    :param df: A pandas DataFrame.
    :param by: Column(s) to groupby on, either a `str` or
               a `list` of `str`
    :param new_column_name: Name of the aggregation output column.
    :param agg_column_name: Name of the column to aggregate over.
    :param agg: How to aggregate.
    :param dropna: Whether or not to include null values,
        if present in the `by` column(s). Default is True.
    :returns: A pandas DataFrame.
    """

    return df.assign(
        **{
            new_column_name: df.groupby(by, dropna=dropna)[
                agg_column_name
            ].transform(agg)
        }
    )
