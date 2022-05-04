"""Implementation for expand_column."""
from typing import Hashable

import pandas as pd
import pandas_flavor as pf

from janitor.utils import deprecated_alias


@pf.register_dataframe_method
@deprecated_alias(column="column_name")
def expand_column(
    df: pd.DataFrame,
    column_name: Hashable,
    sep: str = "|",
    concat: bool = True,
) -> pd.DataFrame:
    """Expand a categorical column with multiple labels into dummy-coded columns.

    Super sugary syntax that wraps :py:meth:`pandas.Series.str.get_dummies`.

    This method does not mutate the original DataFrame.

    Functional usage syntax:

        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...         "col1": ["A, B", "B, C, D", "E, F", "A, E, F"],
        ...         "col2": [1, 2, 3, 4],
        ...     }
        ... )
        >>> df = expand_column(
        ...     df,
        ...     column_name="col1",
        ...     sep=", "  # note space in sep
        ... )
        >>> df
              col1  col2  A  B  C  D  E  F
        0     A, B     1  1  1  0  0  0  0
        1  B, C, D     2  0  1  1  1  0  0
        2     E, F     3  0  0  0  0  1  1
        3  A, E, F     4  1  0  0  0  1  1

    Method chaining syntax:

        >>> import pandas as pd
        >>> import janitor
        >>> df = (
        ...     pd.DataFrame(
        ...         {
        ...             "col1": ["A, B", "B, C, D", "E, F", "A, E, F"],
        ...             "col2": [1, 2, 3, 4],
        ...         }
        ...     )
        ...     .expand_column(
        ...         column_name='col1',
        ...         sep=', '
        ...     )
        ... )
        >>> df
              col1  col2  A  B  C  D  E  F
        0     A, B     1  1  1  0  0  0  0
        1  B, C, D     2  0  1  1  1  0  0
        2     E, F     3  0  0  0  0  1  1
        3  A, E, F     4  1  0  0  0  1  1

    :param df: A pandas DataFrame.
    :param column_name: Which column to expand.
    :param sep: The delimiter, same to
        :py:meth:`~pandas.Series.str.get_dummies`'s `sep`, default as `|`.
    :param concat: Whether to return the expanded column concatenated to
        the original dataframe (`concat=True`), or to return it standalone
        (`concat=False`).
    :returns: A pandas DataFrame with an expanded column.
    """
    expanded_df = df[column_name].str.get_dummies(sep=sep)
    if concat:
        df = df.join(expanded_df)
        return df
    return expanded_df
