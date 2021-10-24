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



        df = df.assign(...=df.groupby(...)[...].transform(...))

    Now, this function can be method-chained:



        import pandas as pd
        import janitor
        df = pd.DataFrame(...).groupby_agg(by='group',
                                           agg='mean',
                                           agg_column_name="col1"
                                           new_column_name='col1_mean_by_group',
                                           dropna = True/False)

    Examples::

        import pandas as pd
        import janitor as jn

            group  var1
        0      1     1
        1      1     1
        2      1     1
        3      1     1
        4      1     2
        5      2     1
        6      2     2
        7      2     2
        8      2     2
        9      2     3

    Let's get the count per `group` and `var1`::

        df.groupby_agg(
            by = ['group', 'var1'],
            agg = 'size',
            agg_column_name = 'var1',
            new_column_name = 'count'
        )

            group  var1  size
        0      1     1     4
        1      1     1     4
        2      1     1     4
        3      1     1     4
        4      1     2     1
        5      2     1     1
        6      2     2     3
        7      2     2     3
        8      2     2     3
        9      2     3     1

    If the data has null values,
    you can include the null values by passing `False` to `dropna`;
    this feature was introduced in Pandas 1.1::

            name   type  num  nulls
        0  black  chair    4    1.0
        1  black  chair    5    1.0
        2  black   sofa   12    NaN
        3    red   sofa    4    NaN
        4    red  plate    3    3.0

    Let's get the count, including the null values,
    grouping on `nulls` column::

        df.groupby_agg(
            by="nulls",
            new_column_name="num_count",
            agg_column_name="num",
            agg="size",
            dropna=False,
        )

            name   type  num  nulls  num_count
        0  black  chair    4    1.0          2
        1  black  chair    5    1.0          2
        2  black   sofa   12    NaN          2
        3    red   sofa    4    NaN          2
        4    red  plate    3    3.0          1

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
    df = df.copy()

    df[new_column_name] = df.groupby(by, dropna=dropna)[
        agg_column_name
    ].transform(agg)
    return df
