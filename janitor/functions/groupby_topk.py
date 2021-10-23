from typing import Dict, Hashable
import pandas_flavor as pf
import pandas as pd

from janitor.utils import check_column


@pf.register_dataframe_method
def groupby_topk(
    df: pd.DataFrame,
    groupby_column_name: Hashable,
    sort_column_name: Hashable,
    k: int,
    sort_values_kwargs: Dict = None,
) -> pd.DataFrame:
    """
    Return top `k` rows from a groupby of a set of columns.

    Returns a DataFrame that has the top `k` values grouped by `groupby_column_name`
    and sorted by `sort_column_name`.
    Additional parameters to the sorting (such as `ascending=True`)
    can be passed using `sort_values_kwargs`.

    List of all sort_values() parameters can be found
    [here](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_values.html).


    ```python
        import pandas as pd
        import janitor as jn

           age  ID result
        0   20   1   pass
        1   22   2   fail
        2   24   3   pass
        3   23   4   pass
        4   21   5   fail
        5   22   6   pass
    ```

    Ascending top 3:

    ```python
        df.groupby_topk('result', 'age', 3)

                    age  ID result
        result
        fail    4   21   5   fail
                1   22   2   fail
        pass    0   20   1   pass
                5   22   6   pass
                3   23   4   pass
    ```

    Descending top 2:

    ```python
        df.groupby_topk('result', 'age', 2, {'ascending':False})

                    age  ID result
        result
        fail    1   22   2   fail
                4   21   5   fail
        pass    2   24   3   pass
                3   23   4   pass
    ```

    Functional usage syntax:

    ```python
        import pandas as pd
        import janitor as jn

        df = pd.DataFrame(...)
        df = jn.groupby_topk(
            df = df,
            groupby_column_name = 'groupby_column',
            sort_column_name = 'sort_column',
            k = 5
            )
    ```

    Method-chaining usage syntax:

    ```python
        import pandas as pd
        import janitor as jn

        df = (
            pd.DataFrame(...)
            .groupby_topk(
            df = df,
            groupby_column_name = 'groupby_column',
            sort_column_name = 'sort_column',
            k = 5
            )
        )
    ```

    :param df: A pandas DataFrame.
    :param groupby_column_name: Column name to group input DataFrame `df` by.
    :param sort_column_name: Name of the column to sort along the
        input DataFrame `df`.
    :param k: Number of top rows to return from each group after sorting.
    :param sort_values_kwargs: Arguments to be passed to sort_values function.
    :returns: A pandas DataFrame with top `k` rows that are grouped by
        `groupby_column_name` column with each group sorted along the
        column `sort_column_name`.
    :raises ValueError: if `k` is less than 1.
    :raises ValueError: if `groupby_column_name` not in DataFrame `df`.
    :raises ValueError: if `sort_column_name` not in DataFrame `df`.
    :raises KeyError: if `inplace:True` is present in `sort_values_kwargs`.
    """  # noqa: E501

    # Convert the default sort_values_kwargs from None to empty Dict
    sort_values_kwargs = sort_values_kwargs or {}

    # Check if groupby_column_name and sort_column_name exists in the DataFrame
    check_column(df, [groupby_column_name, sort_column_name])

    # Check if k is greater than 0.
    if k < 1:
        raise ValueError(
            "Numbers of rows per group to be returned must be greater than 0."
        )

    # Check if inplace:True in sort values kwargs because it returns None
    if (
        "inplace" in sort_values_kwargs.keys()
        and sort_values_kwargs["inplace"]
    ):
        raise KeyError("Cannot use `inplace=True` in `sort_values_kwargs`.")

    return df.groupby(groupby_column_name).apply(
        lambda d: d.sort_values(sort_column_name, **sort_values_kwargs).head(k)
    )
