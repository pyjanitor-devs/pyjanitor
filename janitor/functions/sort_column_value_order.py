import pandas_flavor as pf
import pandas as pd


@pf.register_dataframe_method
def sort_column_value_order(
    df: pd.DataFrame, column: str, column_value_order: dict, columns=None
) -> pd.DataFrame:
    """
    This function adds precedence to certain values in a specified column, then
    sorts based on that column and any other specified columns.

    Functional usage syntax:

    ```python

        import pandas as pd
        import janitor as jn

        df = pd.DataFrame(...)

        jn.sort_column_value_order(
            column,
            column_value_order = {col1: number, ...}
            columns
        )
    ```

    Method chaining usage syntax:

    ```python

        import pandas as pd
        import janitor

        df.sort_column_value_order(
            column,
            column_value_order = {col1: number, ...}
            columns
        )
    ```

    :param df: This is our DataFrame that we are manipulating
    :param column: This is a column name as a string we are using to specify
        which column to sort by
    :param column_value_order: This is a dictionary of values that will
        represent precedence of the values in the specified column
    :param columns: This is a list of additional columns that we can sort by
    :raises ValueError: raises error if chosen Column Name is not in
        Dataframe, or if column_value_order dictionary is empty.
    :return: This function returns a Pandas DataFrame
    """
    if len(column_value_order) > 0:
        if column in df.columns:
            df["cond_order"] = df[column].replace(column_value_order)
            if columns is None:
                new_df = df.sort_values("cond_order")
                del new_df["cond_order"]
            else:
                new_df = df.sort_values(columns + ["cond_order"])
                del new_df["cond_order"]
            return new_df
        else:
            raise ValueError("Column Name not in DataFrame")
    else:
        raise ValueError("column_value_order dictionary cannot be empty")
