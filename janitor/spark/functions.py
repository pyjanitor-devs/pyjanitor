"""General purpose data cleaning functions for pyspark."""

import re
from typing import Union

from janitor import utils as janitor_utils
from janitor.functions.utils import (
    _change_case,
    _normalize_1,
    _remove_special,
    _strip_underscores_func,
)

from . import backend

try:
    from pyspark.sql import DataFrame
except ImportError:
    janitor_utils.import_message(
        submodule="spark",
        package="pyspark",
        conda_channel="conda-forge",
        pip_install=True,
    )


@backend.register_dataframe_method
def clean_names(
    df: DataFrame,
    case_type: str = "lower",
    remove_special: bool = False,
    strip_underscores: str = None,
) -> DataFrame:
    """Clean column names for pyspark dataframe.

    Takes all column names, converts them to lowercase, then replaces all
    spaces with underscores.

    This method does not mutate the original DataFrame.

    Examples:
        >>> from pyspark.sql import DataFrame
        >>> import janitor.spark
        >>> df = DataFrame(...).clean_names()  # doctest: +SKIP

        Example of transformation:

        ```
        Columns before: First Name, Last Name, Employee Status, Subject
        Columns after: first_name, last_name, employee_status, subject
        ```

    Args:
        df: Spark DataFrame object.
        strip_underscores: Removes the outer underscores from all
            column names. Default None keeps outer underscores. Values can be
            either 'left', 'right' or 'both' or the respective shorthand 'l',
            'r' and True.
        case_type: Whether to make columns lower or uppercase.
            Current case may be preserved with 'preserve',
            while snake case conversion (from CamelCase or camelCase only)
            can be turned on using "snake".
            Default 'lower' makes all characters lowercase.
        remove_special: Remove special characters from columns.
            Only letters, numbers and underscores are preserved.

    Returns:
        A Spark DataFrame.
    """

    cols = df.columns

    cols = [_change_case(col, case_type) for col in cols]

    cols = [_normalize_1(col) for col in cols]

    if remove_special:
        cols = [_remove_special(col) for col in cols]

    cols = [re.sub("_+", "_", col) for col in cols]  # noqa: PD005

    cols = [_strip_underscores_func(col, strip_underscores) for col in cols]

    cols = [
        f"`{col}` AS `{new_col}`" for col, new_col in zip(df.columns, cols)
    ]

    return df.selectExpr(*cols)


@backend.register_dataframe_method
def update_where(
    df: DataFrame,
    conditions: str,
    target_column_name: str,
    target_val: Union[str, int, float],
) -> DataFrame:
    """Add multiple conditions to update a column in the dataframe.

    This method does not mutate the original DataFrame.

    Examples:

        >>> import pandas as pd
        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.getOrCreate()
        >>> import janitor.spark
        >>> data = {
        ...     "a": [1, 2, 3, 4],
        ...     "b": [5, 6, 7, 8],
        ...     "c": [0, 0, 0, 0]
        ... }
        >>> df = spark.createDataFrame(pd.DataFrame(data))
        >>> df = (
        ...     df
        ...     .update_where(
        ...         conditions="a > 2 AND b < 8",
        ...         target_column_name='c',
        ...         target_val=10
        ...     )
        ... )  # doctest: +SKIP
        >>> df.show()  # doctest: +SKIP
        +---+---+---+
        |  a|  b|  c|
        +---+---+---+
        |  1|  5|  0|
        |  2|  6|  0|
        |  3|  7| 10|
        |  4|  8|  0|
        +---+---+---+

    Args:
        df: Spark DataFrame object.
        conditions: Spark SQL string condition used to update a target
            column and target value
        target_column_name: Column to be updated. If column does not exist
            in dataframe, a new column will be created; note that entries that
            do not get set in the new column will be null.
        target_val: Value to be updated

    Returns:
        An updated spark DataFrame.
    """

    # String without quotes are treated as column name
    if isinstance(target_val, str):
        target_val = f"'{target_val}'"

    if target_column_name in df.columns:
        # `{col]` is to enable non-standard column name,
        # i.e. column name with special characters etc.
        select_stmts = [
            f"`{col}`" for col in df.columns if col != target_column_name
        ] + [
            f"""
            CASE
                WHEN {conditions} THEN {target_val}
                ELSE `{target_column_name}`
            END AS `{target_column_name}`
            """
        ]
        # This is to retain the ordering
        col_order = df.columns
    else:
        select_stmts = [f"`{col}`" for col in df.columns] + [
            f"""
            CASE
                WHEN {conditions} THEN {target_val} ELSE NULL
            END AS `{target_column_name}`
            """
        ]
        col_order = df.columns + [target_column_name]

    return df.selectExpr(*select_stmts).select(*col_order)
