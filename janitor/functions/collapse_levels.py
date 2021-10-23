import pandas as pd
import pandas_flavor as pf

from janitor.utils import check


@pf.register_dataframe_method
def collapse_levels(df: pd.DataFrame, sep: str = "_") -> pd.DataFrame:
    """
    Flatten multi-level column dataframe to a single level.

    This method mutates the original DataFrame.

    Given a `DataFrame` containing multi-level columns, flatten to single-
    level by string-joining the column labels in each level.

    After a `groupby` / `aggregate` operation where `.agg()` is passed a
    list of multiple aggregation functions, a multi-level DataFrame is
    returned with the name of the function applied in the second level.

    It is sometimes convenient for later indexing to flatten out this
    multi-level configuration back into a single level. This function does
    this through a simple string-joining of all the names across different
    levels in a single column.

    Method chaining syntax given two value columns `[max_speed, type]`:

    ```python
        data = {"class": ["bird", "bird", "bird", "mammal", "mammal"],
                "max_speed": [389, 389, 24, 80, 21],
                "type": ["falcon", "falcon", "parrot", "Lion", "Monkey"]}

        df = (
            pd.DataFrame(data)
                .groupby('class')
                .agg(['mean', 'median'])
                .collapse_levels(sep='_')
        )
    ```

    Before applying `.collapse_levels`, the `.agg` operation returns a
    multi-level column DataFrame whose columns are `(level 1, level 2)`:


        [('class', ''), ('max_speed', 'mean'), ('max_speed', 'median'),
        ('type', 'mean'), ('type', 'median')]

    `.collapse_levels` then flattens the column names to:


        ['class', 'max_speed_mean', 'max_speed_median',
        'type_mean', 'type_median']

    :param df: A pandas DataFrame.
    :param sep: String separator used to join the column level names
    :returns: A flattened pandas DataFrame.
    """
    check("sep", sep, [str])

    # if already single-level, just return the DataFrame
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    df.columns = [
        sep.join(str(el) for el in tup if str(el) != "")
        for tup in df  # noqa: PD011
    ]

    return df
