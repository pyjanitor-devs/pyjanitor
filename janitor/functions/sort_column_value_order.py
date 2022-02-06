"""Implementation of the `sort_column_value_order` function."""
import pandas_flavor as pf
import pandas as pd
from janitor.utils import check, check_column
from .remove_columns import remove_columns  # noqa: F401


@pf.register_dataframe_method
def sort_column_value_order(
    df: pd.DataFrame, column: str, column_value_order: dict, columns=None
) -> pd.DataFrame:
    """
    This function adds precedence to certain values in a specified column, then
    sorts based on that column and any other specified columns.

    Example:

        >>> import pandas as pd
        >>> import janitor
        >>> import numpy as np
        >>> company_sales = {
        ...     "SalesMonth": ["Jan", "Feb", "Feb", "Mar", "April"],
        ...     "Company1": [150.0, 200.0, 200.0, 300.0, 400.0],
        ...     "Company2": [180.0, 250.0, 250.0, np.nan, 500.0],
        ...     "Company3": [400.0, 500.0, 500.0, 600.0, 675.0],
        ... }
        >>> df = pd.DataFrame.from_dict(company_sales)
        >>> df
          SalesMonth  Company1  Company2  Company3
        0        Jan     150.0     180.0     400.0
        1        Feb     200.0     250.0     500.0
        2        Feb     200.0     250.0     500.0
        3        Mar     300.0       NaN     600.0
        4      April     400.0     500.0     675.0
        >>> df.sort_column_value_order(
        ...     "SalesMonth",
        ...     {"April": 1, "Mar": 2, "Feb": 3, "Jan": 4}
        ... )
          SalesMonth  Company1  Company2  Company3
        4      April     400.0     500.0     675.0
        3        Mar     300.0       NaN     600.0
        1        Feb     200.0     250.0     500.0
        2        Feb     200.0     250.0     500.0
        0        Jan     150.0     180.0     400.0

    :param df: This is our DataFrame that we are manipulating
    :param column: This is a column name as a string we are using to specify
        which column to sort by
    :param column_value_order: This is a dictionary of values that will
        represent precedence of the values in the specified column
    :param columns: This is a list of additional columns that we can sort by
    :raises ValueError: raises error if chosen Column Name is not in
        Dataframe, or if column_value_order dictionary is empty.
    :return: A sorted pandas DataFrame.
    """
    # Validation checks
    check_column(df, column, present=True)
    check("column_value_order", column_value_order, [dict])
    if not column_value_order:
        raise ValueError("column_value_order dictionary cannot be empty")

    df = df.assign(cond_order=df[column].replace(column_value_order))

    sort_by = ["cond_order"]
    if columns is not None:
        sort_by = ["cond_order"] + columns

    df = df.sort_values(sort_by).remove_columns("cond_order")
    return df
