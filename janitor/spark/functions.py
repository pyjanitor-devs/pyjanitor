"""
General purpose data cleaning functions for pyspark.
"""

import re

from pyspark.sql import DataFrame

from .. import functions as janitor_func
from .. import utils as janitor_utils
from . import backend


@backend.register_dataframe_method
def clean_names(
    df: DataFrame,
    case_type: str = "lower",
    remove_special: bool = False,
    strip_underscores: str = None,
) -> DataFrame:
    """
    Clean column names for pyspark dataframe.

    Takes all column names, converts them to lowercase, then replaces all
    spaces with underscores.

    This method does not mutate the original DataFrame.

    Functional usage example:

    .. code-block:: python

        df = clean_names(df)

    Method chaining example:

    .. code-block:: python

        from pyspark.sql import DataFrame
        import janitor.spark
        df = DataFrame(...).clean_names()

    :Example of transformation:

    .. code-block:: python

        Columns before: First Name, Last Name, Employee Status, Subject
        Columns after: first_name, last_name, employee_status, subject

    :param df: Spark DataFrame object.
    :param strip_underscores: (optional) Removes the outer underscores from all
        column names. Default None keeps outer underscores. Values can be
        either 'left', 'right' or 'both' or the respective shorthand 'l', 'r'
        and True.
    :param case_type: (optional) Whether to make columns lower or uppercase.
        Current case may be preserved with 'preserve',
        while snake case conversion (from CamelCase or camelCase only)
        can be turned on using "snake".
        Default 'lower' makes all characters lowercase.
    :param remove_special: (optional) Remove special characters from columns.
        Only letters, numbers and underscores are preserved.
    :returns: A Spark DataFrame.
    """

    cols = df.columns

    cols = [janitor_func._change_case(col, case_type) for col in cols]

    cols = [janitor_func._normalize_1(col) for col in cols]

    if remove_special:
        cols = [janitor_func._remove_special(col) for col in cols]

    cols = [re.sub("_+", "_", col) for col in cols]

    cols = [
        janitor_utils._strip_underscores_func(col, strip_underscores)
        for col in cols
    ]

    cols = [
        f"`{col}` AS `{new_col}`" for col, new_col in zip(df.columns, cols)
    ]

    return df.selectExpr(*cols)
