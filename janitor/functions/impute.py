"""Implementation of `impute` function"""

from itertools import product
from typing import Any, Optional

import pandas as pd
import pandas_flavor as pf

from janitor.functions.select import get_index_labels
from janitor.utils import deprecated_alias


@pf.register_dataframe_method
@deprecated_alias(column="column_name")
@deprecated_alias(column_name="column_names")
@deprecated_alias(statistic="statistic_column_name")
def impute(
    df: pd.DataFrame,
    column_names: Any,
    value: Optional[Any] = None,
    statistic_column_name: Optional[str] = None,
) -> pd.DataFrame:
    """Method-chainable imputation of values in a column.

    This method does not mutate the original DataFrame.

    Underneath the hood, this function calls the `.fillna()` method available
    to every `pandas.Series` object.

    Either one of `value` or `statistic_column_name` should be provided.

    If `value` is provided, then all null values in the selected column will
    take on the value provided.

    If `statistic_column_name` is provided, then all null values in the
    selected column(s) will take on the summary statistic value
    of other non-null values.

    Column selection in `column_names` is possible using the
    [`select`][janitor.functions.select.select] syntax.

    Currently supported statistics include:

    - `mean` (also aliased by `average`)
    - `median`
    - `mode`
    - `minimum` (also aliased by `min`)
    - `maximum` (also aliased by `max`)

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({
        ...     "a": [1, 2, 3],
        ...     "sales": np.nan,
        ...     "score": [np.nan, 3, 2],
        ... })
        >>> df
           a  sales  score
        0  1    NaN    NaN
        1  2    NaN    3.0
        2  3    NaN    2.0

        Imputing null values with 0 (using the `value` parameter):

        >>> df.impute(column_names="sales", value=0.0)
           a  sales  score
        0  1    0.0    NaN
        1  2    0.0    3.0
        2  3    0.0    2.0

        Imputing null values with median (using the `statistic_column_name`
        parameter):

        >>> df.impute(column_names="score", statistic_column_name="median")
           a  sales  score
        0  1    NaN    2.5
        1  2    NaN    3.0
        2  3    NaN    2.0

    Args:
        df: A pandas DataFrame.
        column_names: The name of the column(s) on which to impute values.
        value: The value used for imputation, passed into `.fillna` method
            of the underlying pandas Series.
        statistic_column_name: The column statistic to impute.

    Raises:
        ValueError: If both `value` and `statistic_column_name` are
            provided.
        KeyError: If `statistic_column_name` is not one of `mean`,
            `average`, `median`, `mode`, `minimum`, `min`, `maximum`, or
            `max`.

    Returns:
        An imputed pandas DataFrame.
    """
    # Firstly, we check that only one of `value` or `statistic` are provided.
    if (value is None) and (statistic_column_name is None):
        raise ValueError("Kindly specify a value or a statistic_column_name")

    if value is not None and statistic_column_name is not None:
        raise ValueError(
            "Only one of `value` or `statistic_column_name` should be "
            "provided."
        )

    column_names = get_index_labels([column_names], df, axis="columns")

    if value is not None:
        value = dict(product(column_names, [value]))

    else:
        # If statistic is provided, then we compute
        # the relevant summary statistic
        # from the other data.
        funcs = {
            "mean": "mean",
            "average": "mean",  # aliased
            "median": "median",
            "mode": "mode",
            "minimum": "min",
            "min": "min",  # aliased
            "maximum": "max",
            "max": "max",  # aliased
        }
        # Check that the statistic keyword argument is one of the approved.
        if statistic_column_name not in funcs:
            raise KeyError(
                f"`statistic_column_name` must be one of {funcs.keys()}."
            )

        value = dict(product(column_names, [funcs[statistic_column_name]]))

        value = df.agg(value)

        # special treatment for mode
        if statistic_column_name == "mode":
            value = {key: val.at[0] for key, val in value.items()}

    return df.fillna(value=value)
