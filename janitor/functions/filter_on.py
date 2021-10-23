import pandas_flavor as pf
import pandas as pd


@pf.register_dataframe_method
def filter_on(
    df: pd.DataFrame, criteria: str, complement: bool = False
) -> pd.DataFrame:
    """
    Return a dataframe filtered on a particular criteria.

    This method does not mutate the original DataFrame.

    This is super-sugary syntax that wraps the pandas `.query()` API, enabling
    users to use strings to quickly specify filters for filtering their
    dataframe. The intent is that `filter_on` as a verb better matches the
    intent of a pandas user than the verb `query`.

    Let's say we wanted to filter students based on whether they failed an exam
    or not, which is defined as their score (in the "score" column) being less
    than 50.

    ```python
        df = (pd.DataFrame(...)
              .filter_on('score < 50', complement=False)
              ...)  # chain on more data preprocessing.
    ```

    This stands in contrast to the in-place syntax that is usually used:

    ```python
        df = pd.DataFrame(...)
        df = df[df['score'] < 3]
    ```

    As with the `filter_string` function, a more seamless flow can be expressed
    in the code.

    Functional usage syntax:

    ```python
        df = filter_on(df,
                       'score < 50',
                       complement=False)
    ```

    Method chaining syntax:

    ```python
        df = (pd.DataFrame(...)
              .filter_on('score < 50', complement=False))
    ```

    Credit to Brant Peterson for the name.

    :param df: A pandas DataFrame.
    :param criteria: A filtering criteria that returns an array or Series of
        booleans, on which pandas can filter on.
    :param complement: Whether to return the complement of the filter or not.
    :returns: A filtered pandas DataFrame.
    """
    if complement:
        return df.query("not " + criteria)
    return df.query(criteria)
