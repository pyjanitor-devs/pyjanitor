"""Implementation of count_cumulative_unique."""
from typing import Hashable

import numpy as np
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

    Example: Cumulative counts, ignoring casing.

        >>> df.count_cumulative_unique(
        ...     column_name="letters",
        ...     dest_column_name="letters_unique_count",
        ...     case_sensitive=False,
        ... )
          letters  numbers  letters_unique_count
        0       a        4                     1
        1       a        5                     1
        2       b        6                     2
        3       A        7                     2
        4       B        8                     2
        5       b        9                     2

    :param df: A pandas DataFrame.
    :param column_name: Name of the column containing values from which a
        running count of unique values will be created.
    :param dest_column_name: The name of the new column containing the
        cumulative count of unique values that will be created.
    :param case_sensitive: Whether or not uppercase and lowercase letters
        will be considered equal. Only valid with string-like columns.
    :returns: A pandas DataFrame with a new column containing a cumulative
        count of unique values from another column.
    :raises TypeError: If `case_sensitive` is False when counting a non-string
        `column_name`.
    """
    check_column(df, column_name)
    check_column(df, dest_column_name, present=False)

    counter = df[column_name]
    if not case_sensitive:
        try:
            # Make it so that the the same uppercase and lowercase
            # letter are treated as one unique value
            counter = counter.str.lower()
        except (AttributeError, TypeError) as e:
            # AttributeError is raised by pandas when .str is used on
            # non-string types, e.g. int.
            # TypeError is raised by pandas when .str.lower is used on a
            # forbidden string type, e.g. bytes.
            raise TypeError(
                "case_sensitive=False can only be used with a string-like "
                f"type. Column {column_name} is {counter.dtype} type."
            ) from e

    counter = (
        counter.groupby(counter, sort=False).cumcount().to_numpy(copy=False)
    )
    counter = np.cumsum(counter == 0)

    return df.assign(**{dest_column_name: counter})
