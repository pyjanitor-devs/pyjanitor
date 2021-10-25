from typing import Hashable
import pandas_flavor as pf
import pandas as pd


@pf.register_dataframe_method
def count_cumulative_unique(
    df: pd.DataFrame,
    column_name: Hashable,
    dest_column_name: str,
    case_sensitive: bool = True,
) -> pd.DataFrame:
    """Generates a running total of cumulative unique values in a given column.

    Functional usage syntax:

    ```python
        import pandas as pd
        import janitor as jn

        df = pd.DataFrame(...)

        df = jn.functions.count_cumulative_unique(
            df=df,
            column_name='animals',
            dest_column_name='animals_unique_count',
            case_sensitive=True
        )
    ```

    Method chaining usage example:

    ```python
        import pandas as pd
        import janitor

        df = pd.DataFrame(...)

        df = df.count_cumulative_unique(
            column_name='animals',
            dest_column_name='animals_unique_count',
            case_sensitive=True
        )
    ```

    A new column will be created containing a running
    count of unique values in the specified column.
    If `case_sensitive` is `True`, then the case of
    any letters will matter (i.e., `a != A`);
    otherwise, the case of any letters will not matter.

    This method mutates the original DataFrame.

    :param df: A pandas dataframe.
    :param column_name: Name of the column containing
        values from which a running count of unique values
        will be created.
    :param dest_column_name: The name of the new column containing the
        cumulative count of unique values that will be created.
    :param case_sensitive: Whether or not uppercase and lowercase letters
        will be considered equal (e.g., 'A' != 'a' if `True`).

    :returns: A pandas DataFrame with a new column containing a cumulative
        count of unique values from another column.
    """

    if not case_sensitive:
        # Make it so that the the same uppercase and lowercase
        # letter are treated as one unique value
        df[column_name] = df[column_name].astype(str).map(str.lower)

    df[dest_column_name] = (
        (
            df[[column_name]]
            .drop_duplicates()
            .assign(dummyabcxyz=1)
            .dummyabcxyz.cumsum()
        )
        .reindex(df.index)
        .ffill()
        .astype(int)
    )

    return df
