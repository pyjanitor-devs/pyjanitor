"""Implementation of remove_columns."""

from typing import Hashable, Iterable, Union

import pandas as pd
import pandas_flavor as pf

from janitor.utils import deprecated_alias, refactored_function


@pf.register_dataframe_method
@refactored_function(
    message=(
        "This function will be deprecated in a 1.x release. "
        "Please use `pd.DataFrame.drop` instead."
    )
)
@deprecated_alias(columns="column_names")
def remove_columns(
    df: pd.DataFrame,
    column_names: Union[str, Iterable[str], Hashable],
) -> pd.DataFrame:
    """Remove the set of columns specified in `column_names`.

    This method does not mutate the original DataFrame.

    Intended to be the method-chaining alternative to `del df[col]`.

    !!!note

        This function will be deprecated in a 1.x release.
        Kindly use `pd.DataFrame.drop` instead.

    Examples:
        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({"a": [2, 4, 6], "b": [1, 3, 5], "c": [7, 8, 9]})
        >>> df
           a  b  c
        0  2  1  7
        1  4  3  8
        2  6  5  9
        >>> df.remove_columns(column_names=['a', 'c'])
           b
        0  1
        1  3
        2  5

    Args:
        df: A pandas DataFrame.
        column_names: The columns to remove.

    Returns:
        A pandas DataFrame.
    """

    return df.drop(columns=column_names)
