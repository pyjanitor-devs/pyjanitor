from typing import Hashable
import pandas_flavor as pf
import pandas as pd

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

        df = expand_column(
            df,
            column_name='col_name',
            sep=', '  # note space in sep
        )

    Method chaining syntax:

        import pandas as pd
        import janitor
        df = (
            pd.DataFrame(...)
            .expand_column(
                column_name='col_name',
                sep=', '
            )
        )

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
