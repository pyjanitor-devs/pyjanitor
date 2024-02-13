"""Implementation of the `factorize_columns` function"""

from typing import Any, Hashable, Iterable, Union

import pandas as pd
import pandas_flavor as pf

from janitor.functions.utils import _factorize


@pf.register_dataframe_method
def factorize_columns(
    df: pd.DataFrame,
    column_names: Union[str, Iterable[str], Hashable],
    suffix: str = "_enc",
    **kwargs: Any,
) -> pd.DataFrame:
    """Converts labels into numerical data.

    This method will create a new column with the string `_enc` appended
    after the original column's name.
    This can be overridden with the suffix parameter.

    Internally, this method uses pandas `factorize` method.
    It takes in an optional suffix and keyword arguments also.
    An empty string as suffix will override the existing column.

    This method does not mutate the original DataFrame.

    Examples:
        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({
        ...     "foo": ["b", "b", "a", "c", "b"],
        ...     "bar": range(4, 9),
        ... })
        >>> df
          foo  bar
        0   b    4
        1   b    5
        2   a    6
        3   c    7
        4   b    8
        >>> df.factorize_columns(column_names="foo")
          foo  bar  foo_enc
        0   b    4        0
        1   b    5        0
        2   a    6        1
        3   c    7        2
        4   b    8        0

    Args:
        df: The pandas DataFrame object.
        column_names: A column name or an iterable (list or tuple) of
            column names.
        suffix: Suffix to be used for the new column.
            An empty string suffix means, it will override the existing column.
        **kwargs: Keyword arguments. It takes any of the keyword arguments,
            which the pandas factorize method takes like `sort`, `na_sentinel`,
            `size_hint`.

    Returns:
        A pandas DataFrame.
    """
    df = _factorize(df.copy(), column_names, suffix, **kwargs)
    return df
