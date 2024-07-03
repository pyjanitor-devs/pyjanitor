"""Implementation of `label_encode` function"""

import warnings
from typing import Hashable, Iterable, Union

import pandas as pd
import pandas_flavor as pf

from janitor.functions.utils import _factorize
from janitor.utils import deprecated_alias, refactored_function


@pf.register_dataframe_method
@refactored_function(
    message=(
        "This function will be deprecated in a 1.x release. "
        "Please use `janitor.factorize_columns` instead."
    )
)
@deprecated_alias(columns="column_names")
def label_encode(
    df: pd.DataFrame,
    column_names: Union[str, Iterable[str], Hashable],
) -> pd.DataFrame:
    """Convert labels into numerical data.

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

    !!!note

        This function will be deprecated in a 1.x release.
        Please use [`factorize_columns`][janitor.functions.factorize_columns.factorize_columns]
        instead.

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
        >>> df.label_encode(column_names="foo")
          foo  bar  foo_enc
        0   b    4        0
        1   b    5        0
        2   a    6        1
        3   c    7        2
        4   b    8        0

    Args:
        df: The pandas DataFrame object.
        column_names: A column name or an iterable (list
            or tuple) of column names.

    Returns:
        A pandas DataFrame.
    """  # noqa: E501
    warnings.warn(
        "`label_encode` will be deprecated in a 1.x release. "
        "Please use `factorize_columns` instead."
    )
    df = _factorize(df, column_names, "_enc")
    return df
