from typing import Any, Hashable, Optional
import pandas_flavor as pf
import pandas as pd
from scipy.stats import mode

from janitor.utils import deprecated_alias
import numpy as np


@pf.register_dataframe_method
@deprecated_alias(column="column_name")
@deprecated_alias(statistic="statistic_column_name")
def impute(
    df: pd.DataFrame,
    column_name: Hashable,
    value: Optional[Any] = None,
    statistic_column_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Method-chainable imputation of values in a column.

    This method mutates the original DataFrame.

    Underneath the hood, this function calls the `.fillna()` method available
    to every `pandas.Series` object.

    Method-chaining example:

    ```python
        import numpy as np
        import pandas as pd
        import janitor

        data = {
            "a": [1, 2, 3],
            "sales": np.nan,
            "score": [np.nan, 3, 2]}
        df = (
            pd.DataFrame(data)
            # Impute null values with 0
            .impute(column_name='sales', value=0.0)
            # Impute null values with median
            .impute(column_name='score', statistic_column_name='median')
        )
    ```

    Either one of `value` or `statistic_column_name` should be provided.

    If `value` is provided, then all null values in the selected column will
    take on the value provided.

    If `statistic_column_name` is provided, then all null values in the
    selected column will take on the summary statistic value of other non-null
    values.

    Currently supported statistics include:

    - `mean` (also aliased by `average`)
    - `median`
    - `mode`
    - `minimum` (also aliased by `min`)
    - `maximum` (also aliased by `max`)

    :param df: A pandas DataFrame
    :param column_name: The name of the column on which to impute values.
    :param value: (optional) The value to impute.
    :param statistic_column_name: (optional) The column statistic to impute.
    :returns: An imputed pandas DataFrame.
    :raises ValueError: if both `value` and `statistic` are provided.
    :raises KeyError: if `statistic` is not one of `mean`, `average`
        `median`, `mode`, `minimum`, `min`, `maximum`, or `max`.
    """
    # Firstly, we check that only one of `value` or `statistic` are provided.
    if value is not None and statistic_column_name is not None:
        raise ValueError(
            "Only one of `value` or `statistic` should be provided"
        )

    # If statistic is provided, then we compute the relevant summary statistic
    # from the other data.
    funcs = {
        "mean": np.mean,
        "average": np.mean,  # aliased
        "median": np.median,
        "mode": mode,
        "minimum": np.min,
        "min": np.min,  # aliased
        "maximum": np.max,
        "max": np.max,  # aliased
    }
    if statistic_column_name is not None:
        # Check that the statistic keyword argument is one of the approved.
        if statistic_column_name not in funcs.keys():
            raise KeyError(f"`statistic` must be one of {funcs.keys()}")

        value = funcs[statistic_column_name](
            df[column_name].dropna().to_numpy()
        )
        # special treatment for mode, because scipy stats mode returns a
        # moderesult object.
        if statistic_column_name == "mode":
            value = value.mode[0]

    # The code is architected this way - if `value` is not provided but
    # statistic is, we then overwrite the None value taken on by `value`, and
    # use it to set the imputation column.
    if value is not None:
        df[column_name] = df[column_name].fillna(value)
    return df
