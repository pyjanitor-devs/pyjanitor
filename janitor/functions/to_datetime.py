"""Implementation source for `to_datetime`."""

from typing import Any, Hashable

import pandas as pd
import pandas_flavor as pf

from janitor.utils import deprecated_alias, refactored_function


@pf.register_dataframe_method
@deprecated_alias(column="column_name")
@refactored_function(
    message=(
        "This function will be deprecated in a 1.x release. "
        "Please use `jn.transform_columns` instead."
    )
)
def to_datetime(
    df: pd.DataFrame, column_name: Hashable, **kwargs: Any
) -> pd.DataFrame:
    """Convert column to a datetime type, in-place.

    Intended to be the method-chaining equivalent of:

    ```python
    df[column_name] = pd.to_datetime(df[column_name], **kwargs)
    ```

    This method mutates the original DataFrame.

    !!!note

        This function will be deprecated in a 1.x release.
        Please use [`jn.transform_column`][janitor.functions.transform_columns.transform_column]
        instead.

    Examples:
        Converting a string column to datetime type with custom format.

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({'date': ['20200101', '20200202', '20200303']})
        >>> df
               date
        0  20200101
        1  20200202
        2  20200303
        >>> df.to_datetime('date', format='%Y%m%d')
                date
        0 2020-01-01
        1 2020-02-02
        2 2020-03-03

    Read the pandas documentation for [`to_datetime`][pd_docs] for more information.

    [pd_docs]: https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html

    Args:
        df: A pandas DataFrame.
        column_name: Column name.
        **kwargs: Provide any kwargs that `pd.to_datetime` can take.

    Returns:
        A pandas DataFrame with updated datetime data.
    """  # noqa: E501
    df[column_name] = pd.to_datetime(df[column_name], **kwargs)

    return df
