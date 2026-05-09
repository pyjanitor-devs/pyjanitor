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
    drop_first: bool = False,
) -> pd.DataFrame:
    """Expand a categorical column with multiple labels into dummy-coded columns.

    Super sugary syntax that wraps `pandas.Series.str.get_dummies`.

    This method does not mutate the original DataFrame.

    Examples:
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
        ...     sep=", ",  # note space in sep
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
        >>> df = pd.DataFrame(
        ...     {
        ...         "col1": ["A, B", "B, C, D", "E, F", "A, E, F"],
        ...         "col2": [1, 2, 3, 4],
        ...     }
        ... ).expand_column(column_name="col1", sep=", ")
        >>> df
              col1  col2  A  B  C  D  E  F
        0     A, B     1  1  1  0  0  0  0
        1  B, C, D     2  0  1  1  1  0  0
        2     E, F     3  0  0  0  0  1  1
        3  A, E, F     4  1  0  0  0  1  1

        Drop the first dummy column to avoid multicollinearity in
        downstream regressions, mirroring `pandas.get_dummies(drop_first=True)`:

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame(
        ...     {
        ...         "col1": ["A, B", "B, C, D", "E, F", "A, E, F"],
        ...         "col2": [1, 2, 3, 4],
        ...     }
        ... ).expand_column(column_name="col1", sep=", ", drop_first=True)
        >>> df
              col1  col2  B  C  D  E  F
        0     A, B     1  1  0  0  0  0
        1  B, C, D     2  1  1  1  0  0
        2     E, F     3  0  0  0  1  1
        3  A, E, F     4  0  0  0  1  1

    Args:
        df: A pandas DataFrame.
        column_name: Which column to expand.
        sep: The delimiter, same to
            `pandas.Series.str.get_dummies`'s `sep`.
        concat: Whether to return the expanded column concatenated to
            the original dataframe (`concat=True`), or to return it standalone
            (`concat=False`).
        drop_first: If `True`, drop the first dummy column to avoid the
            collinearity that results from a full one-hot encoding (a
            common preprocessing step before linear regression). Mirrors
            the `drop_first` argument on `pandas.get_dummies`. Note that
            `pandas.Series.str.get_dummies` (the underlying call) does
            not yet expose this argument, so we drop the first column
            after the fact. See issue #368.

    Returns:
        A pandas DataFrame with an expanded column.
    """  # noqa: E501
    expanded_df = df[column_name].str.get_dummies(sep=sep)
    if drop_first and not expanded_df.empty:
        # ``pandas.Series.str.get_dummies`` does not expose ``drop_first``
        # (only ``pandas.get_dummies`` does), so we drop the first column
        # after the fact. Columns coming back from ``str.get_dummies`` are
        # sorted lexicographically, so this is deterministic across calls.
        # Issue #368.
        expanded_df = expanded_df.iloc[:, 1:]
    if concat:
        return df.join(expanded_df)
    return expanded_df
