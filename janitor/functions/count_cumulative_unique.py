"""Implementation of count_cumulative_unique."""
from typing import Hashable
import pandas_flavor as pf
import pandas as pd

from janitor.utils import check_column


@pf.register_dataframe_method
def count_cumulative_unique(
    df: pd.DataFrame,
    column_name: Hashable,
    dest_column_name: str,
    case_sensitive: bool = True,
) -> pd.DataFrame:
    """Generates a running total of cumulative unique values in a given column.

    A new column will be created containing a running
    count of unique values in the specified column.
    If `case_sensitive` is `True`, then the case of
    any letters will matter (i.e., `a != A`);
    otherwise, the case of any letters will not matter.

    This method does not mutate the original DataFrame.

    Examples:

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({
        ...     "letters": list("aabABb"),
        ...     "numbers": range(4, 10),
        ... })
        >>> df
          letters  numbers
        0       a        4
        1       a        5
        2       b        6
        3       A        7
        4       B        8
        5       b        9
        >>> df.count_cumulative_unique(
        ...     column_name="letters",
        ...     dest_column_name="letters_unique_count",
        ... )
          letters  numbers  letters_unique_count
        0       a        4                     1
        1       a        5                     1
        2       b        6                     2
        3       A        7                     3
        4       B        8                     4
        5       b        9                     4

    :param df: A pandas DataFrame.
    :param column_name: Name of the column containing values from which a
        running count of unique values will be created.
    :param dest_column_name: The name of the new column containing the
        cumulative count of unique values that will be created.
    :param case_sensitive: Whether or not uppercase and lowercase letters
        will be considered equal.
    :returns: A pandas DataFrame with a new column containing a cumulative
        count of unique values from another column.
    """
    check_column(df, column_name)

    if not case_sensitive:
        # Make it so that the the same uppercase and lowercase
        # letter are treated as one unique value
        series = df[column_name].astype("string").str.lower()
    else:
        series = df[column_name]

    dummy_name = "_pyjanitor_dummy_col_"
    count_column = (
        series.drop_duplicates()
        .to_frame()
        .assign(**{dummy_name: 1})[dummy_name]
        .cumsum()
        .reindex(df.index, copy=False)
        .ffill()
        .astype(int)
    )

    return df.assign(**{dest_column_name: count_column})
