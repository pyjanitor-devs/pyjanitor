from typing import Hashable
import pandas_flavor as pf
import pandas as pd


@pf.register_dataframe_method
def drop_duplicate_columns(
    df: pd.DataFrame, column_name: Hashable, nth_index: int = 0
) -> pd.DataFrame:
    """Remove a duplicated column specified by column_name, its index.

    This method does not mutate the original DataFrame.

    Column order 0 is to remove the first column,
           order 1 is to remove the second column, and etc

    The corresponding tidyverse R's library is:
    `select(-<column_name>_<nth_index + 1>)`

    Method chaining syntax:



        df = pd.DataFrame({
            "a": range(10),
            "b": range(10),
            "A": range(10, 20),
            "a*": range(20, 30),
        }).clean_names(remove_special=True)

        # remove a duplicated second 'a' column
        df.drop_duplicate_columns(column_name="a", nth_index=1)



    :param df: A pandas DataFrame
    :param column_name: Column to be removed
    :param nth_index: Among the duplicated columns,
        select the nth column to drop.
    :return: A pandas DataFrame
    """
    cols = df.columns.to_list()
    col_indexes = [
        col_idx
        for col_idx, col_name in enumerate(cols)
        if col_name == column_name
    ]

    # given that a column could be duplicated,
    # user could opt based on its order
    removed_col_idx = col_indexes[nth_index]
    # get the column indexes without column that is being removed
    filtered_cols = [
        c_i for c_i, c_v in enumerate(cols) if c_i != removed_col_idx
    ]

    return df.iloc[:, filtered_cols]
