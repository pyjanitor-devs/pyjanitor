import pandas_flavor as pf
import pandas as pd


@pf.register_dataframe_method
def drop_constant_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Finds and drops the constant columns from a Pandas DataFrame.

    This method does not mutate the original DataFrame.

    Functional usage syntax:

    ```python
    import pandas as pd
    import janitor as jn

    data_dict = {
    "a": [1, 1, 1] * 3,
    "Bell__Chart": [1, 2, 3] * 3,
    "decorated-elephant": [1, 1, 1] * 3,
    "animals": ["rabbit", "leopard", "lion"] * 3,
    "cities": ["Cambridge", "Shanghai", "Basel"] * 3
    }

    df = pd.DataFrame(data_dict)

    df = jn.functions.drop_constant_columns(df)
    ```

    Method chaining usage example:

    ```python
    import pandas as pd
    import janitor

    df = pd.DataFrame(...)

    df = df.drop_constant_columns()
    ```

    :param df: Input Pandas DataFrame
    :returns: The Pandas DataFrame with the constant columns dropped.
    """
    # Find the constant columns
    constant_columns = []
    for col in df.columns:
        if len(df[col].unique()) == 1:
            constant_columns.append(col)

    # Drop constant columns from df and return it
    return df.drop(labels=constant_columns, axis=1)
