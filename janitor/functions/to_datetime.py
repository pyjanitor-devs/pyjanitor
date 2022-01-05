from typing import Hashable
import pandas_flavor as pf
import pandas as pd

from janitor.utils import deprecated_alias


@pf.register_dataframe_method
@deprecated_alias(column="column_name")
def to_datetime(
    df: pd.DataFrame, column_name: Hashable, **kwargs
) -> pd.DataFrame:
    """Convert column to a datetime type, in-place.

    Intended to be the method-chaining equivalent of:

        df[column_name] = pd.to_datetime(df[column_name], **kwargs)

    This method mutates the original DataFrame.

    Example: Converting a string column to datetime type with custom format.

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

    :param df: A pandas DataFrame.
    :param column_name: Column name.
    :param kwargs: Provide any kwargs that `pd.to_datetime` can take.
    :returns: A pandas DataFrame with updated datetime data.
    """  # noqa: E501
    df[column_name] = pd.to_datetime(df[column_name], **kwargs)

    return df
