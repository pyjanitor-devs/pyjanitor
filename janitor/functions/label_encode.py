"""Implementation of `label_encode` function"""
from typing import Hashable, Iterable, Union
import warnings
import pandas_flavor as pf
import pandas as pd

from janitor.utils import deprecated_alias
from janitor.functions.utils import _factorize


@pf.register_dataframe_method
@deprecated_alias(columns="column_names")
def label_encode(
    df: pd.DataFrame,
    column_names: Union[str, Iterable[str], Hashable],
) -> pd.DataFrame:
    """
    Convert labels into numerical data.

    This method will create a new column with the string `_enc` appended
    after the original column's name.
    Consider this to be syntactic sugar.
    This function uses the `factorize` pandas function under the hood.

    This method behaves differently from
    [`encode_categorical`][janitor.functions.encode_categorical.encode_categorical].
    This method creates a new column of numeric data.
    [`encode_categorical`][janitor.functions.encode_categorical.encode_categorical]
    replaces the dtype of the original column with a *categorical* dtype.

    This method mutates the original DataFrame.

    Example:

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
        >>> df.label_encode(column_names="foo")
          foo  bar  foo_enc
        0   b    4        0
        1   b    5        0
        2   a    6        1
        3   c    7        2
        4   b    8        0

    !!!note

        This function will be deprecated in a 1.x release.
        Please use [`factorize_columns`][janitor.functions.factorize_columns.factorize_columns]
        instead.

    :param df: The pandas DataFrame object.
    :param column_names: A column name or an iterable (list
        or tuple) of column names.
    :returns: A pandas DataFrame.
    """  # noqa: E501
    warnings.warn(
        "`label_encode` will be deprecated in a 1.x release. "
        "Please use `factorize_columns` instead."
    )
    df = _factorize(df, column_names, "_enc")
    return df
