"""Implementation of limit_column_characters."""
import pandas_flavor as pf
import pandas as pd

from janitor.utils import check


@pf.register_dataframe_method
def limit_column_characters(
    df: pd.DataFrame,
    column_length: int,
    col_separator: str = "_",
) -> pd.DataFrame:
    """Truncate column sizes to a specific length.

    This method mutates the original DataFrame.

    Method chaining will truncate all columns to a given length and append
    a given separator character with the index of duplicate columns, except
    for the first distinct column name.

    Example:

        >>> import pandas as pd
        >>> import janitor
        >>> data_dict = {
        ...     "really_long_name": [9, 8, 7],
        ...     "another_really_long_name": [2, 4, 6],
        ...     "another_really_longer_name": list("xyz"),
        ...     "this_is_getting_out_of_hand": list("pqr"),
        ... }
        >>> df = pd.DataFrame(data_dict)
        >>> df  # doctest: +SKIP
           really_long_name  another_really_long_name another_really_longer_name this_is_getting_out_of_hand
        0                 9                         2                          x                           p
        1                 8                         4                          y                           q
        2                 7                         6                          z                           r
        >>> df.limit_column_characters(7)
           really_  another another_1 this_is
        0        9        2         x       p
        1        8        4         y       q
        2        7        6         z       r

    :param df: A pandas DataFrame.
    :param column_length: Character length for which to truncate all columns.
        The column separator value and number for duplicate column name does
        not contribute. Therefore, if all columns are truncated to 10
        characters, the first distinct column will be 10 characters and the
        remaining will be 12 characters (assuming a column separator of one
        character).
    :param col_separator: The separator to use for counting distinct column
        values, for example, `'_'` or `'.'`.
        Supply an empty string (i.e. `''`) to remove the separator.
    :returns: A pandas DataFrame with truncated column lengths.
    """  # noqa: E501

    check("column_length", column_length, [int])
    check("col_separator", col_separator, [str])

    col_names = df.columns
    col_names = [col_name[:column_length] for col_name in col_names]

    col_name_set = set(col_names)
    col_name_count = {}

    # If no columns are duplicates, we can skip the loops below.
    if len(col_name_set) == len(col_names):
        df.columns = col_names
        return df

    for col_name_to_check in col_name_set:
        count = 0
        for idx, col_name in enumerate(col_names):
            if col_name_to_check == col_name:
                col_name_count[idx] = count
                count += 1

    final_col_names = []
    for idx, col_name in enumerate(col_names):
        if col_name_count[idx] > 0:
            col_name_to_append = (
                col_name + col_separator + str(col_name_count[idx])
            )
            final_col_names.append(col_name_to_append)
        else:
            final_col_names.append(col_name)

    df.columns = final_col_names
    return df
