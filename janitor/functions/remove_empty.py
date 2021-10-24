import pandas_flavor as pf
import pandas as pd


@pf.register_dataframe_method
def remove_empty(df: pd.DataFrame) -> pd.DataFrame:
    """Drop all rows and columns that are completely null.

    This method also resets the index(by default) since it doesn't make sense
    to preserve the index of a completely empty row.

    This method mutates the original DataFrame.

    Implementation is inspired from [StackOverflow][so].

    [so]: https://stackoverflow.com/questions/38884538/python-pandas-find-all-rows-where-all-values-are-nan

    Functional usage syntax:

    ```python
    df = remove_empty(df)
    ```

    Method chaining syntax:

    ```python
    import pandas as pd
    import janitor
    df = pd.DataFrame(...).remove_empty()
    ```

    :param df: The pandas DataFrame object.
    :returns: A pandas DataFrame.
    """  # noqa: E501
    nanrows = df.index[df.isna().all(axis=1)]
    df = df.drop(index=nanrows).reset_index(drop=True)

    nancols = df.columns[df.isna().all(axis=0)]
    df = df.drop(columns=nancols)

    return df
