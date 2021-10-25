import pandas_flavor as pf
import pandas as pd
from natsort import index_natsorted


@pf.register_dataframe_method
def sort_naturally(
    df: pd.DataFrame, column_name: str, **natsorted_kwargs
) -> pd.DataFrame:
    """Sort a DataFrame by a column using *natural* sorting.

    Natural sorting is distinct from
    the default lexiographical sorting provided by `pandas`.
    For example, given the following list of items:

        ["A1", "A11", "A3", "A2", "A10"]

    lexicographical sorting would give us:


        ["A1", "A10", "A11", "A2", "A3"]

    By contrast, "natural" sorting would give us:

        ["A1", "A2", "A3", "A10", "A11"]

    This function thus provides *natural* sorting
    on a single column of a dataframe.

    To accomplish this, we do a natural sort
    on the unique values that are present in the dataframe.
    Then, we reconstitute the entire dataframe
    in the naturally sorted order.

    Natural sorting is provided by the Python package
    [natsort](https://natsort.readthedocs.io/en/master/index.html)

    All keyword arguments to `natsort` should be provided
    after the column name to sort by is provided.
    They are passed through to the `natsorted` function.

    Functional usage syntax:

    ```python
        import pandas as pd
        import janitor as jn

        df = pd.DataFrame(...)

        df = jn.sort_naturally(
            df=df,
            column_name='alphanumeric_column',
        )
    ```

    Method chaining usage syntax:

    ```python
        import pandas as pd
        import janitor

        df = pd.DataFrame(...)

        df = df.sort_naturally(
            column_name='alphanumeric_column',
        )
    ```
    :param df: A pandas DataFrame.
    :param column_name: The column on which natural sorting should take place.
    :param natsorted_kwargs: Keyword arguments to be passed
        to natsort's `natsorted` function.
    :returns: A sorted pandas DataFrame.
    """
    new_order = index_natsorted(df[column_name], **natsorted_kwargs)
    return df.iloc[new_order, :]
