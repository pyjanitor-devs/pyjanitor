import warnings
import pandas_flavor as pf
import pandas as pd

from janitor.utils import check


@pf.register_dataframe_method
def row_to_names(
    df: pd.DataFrame,
    row_number: int = None,
    remove_row: bool = False,
    remove_rows_above: bool = False,
    reset_index: bool = False,
) -> pd.DataFrame:
    """Elevates a row to be the column names of a DataFrame.

    This method mutates the original DataFrame.

    Contains options to remove the elevated row from the DataFrame along with
    removing the rows above the selected row.

    Method chaining usage:



        df = (
            pd.DataFrame(...)
            .row_to_names(
                row_number=0,
                remove_row=False,
                remove_rows_above=False,
                reset_index=False,
            )
        )

    :param df: A pandas DataFrame.
    :param row_number: The row containing the variable names
    :param remove_row: Whether the row should be removed from the DataFrame.
        Defaults to False.
    :param remove_rows_above: Whether the rows above the selected row should
        be removed from the DataFrame. Defaults to False.
    :param reset_index: Whether the index should be reset on the returning
        DataFrame. Defaults to False.
    :returns: A pandas DataFrame with set column names.
    """
    # :Setup:

    # ```python

    #     import pandas as pd
    #     import janitor
    #     data_dict = {
    #         "a": [1, 2, 3] * 3,
    #         "Bell__Chart": [1, 2, 3] * 3,
    #         "decorated-elephant": [1, 2, 3] * 3,
    #         "animals": ["rabbit", "leopard", "lion"] * 3,
    #         "cities": ["Cambridge", "Shanghai", "Basel"] * 3
    #     }

    # :Example: Move first row to column names:

    # ```python

    #     example_dataframe = pd.DataFrame(data_dict)
    #     example_dataframe.row_to_names(0)

    # :Output:

    # ```python

    #        1  1  1   rabbit  Cambridge
    #     0  1  1  1   rabbit  Cambridge
    #     1  2  2  2  leopard   Shanghai
    #     2  3  3  3     lion      Basel
    #     3  1  1  1   rabbit  Cambridge
    #     4  2  2  2  leopard   Shanghai
    #     5  3  3  3     lion      Basel
    #     6  1  1  1   rabbit  Cambridge
    #     7  2  2  2  leopard   Shanghai

    # :Example: Move first row to column names and
    #  remove row while resetting the index:

    # ```python

    #     example_dataframe = pd.DataFrame(data_dict)
    #     example_dataframe.row_to_names(0, remove_row=True,\
    #       reset_index=True)

    # :Output:

    # ```python

    #       1   1   1   rabbit  Cambridge
    #   0   2   2   2   leopard Shanghai
    #   1   3   3   3   lion    Basel
    #   2   1   1   1   rabbit  Cambridge
    #   3   2   2   2   leopard Shanghai
    #   4   3   3   3   lion    Basel
    #   5   1   1   1   rabbit  Cambridge
    #   6   2   2   2   leopard Shanghai
    #   7   3   3   3   lion    Basel

    # :Example: Move first row to column names and remove
    #   row without resetting the index:

    # ```python

    #     example_dataframe = pd.DataFrame(data_dict)
    #     example_dataframe.row_to_names(0, remove_row=True)

    # :Output:

    # ```python

    #        1  1  1   rabbit  Cambridge
    #     1  2  2  2  leopard   Shanghai
    #     2  3  3  3     lion      Basel
    #     3  1  1  1   rabbit  Cambridge
    #     4  2  2  2  leopard   Shanghai
    #     5  3  3  3     lion      Basel
    #     6  1  1  1   rabbit  Cambridge
    #     7  2  2  2  leopard   Shanghai
    #     8  3  3  3     lion      Basel

    # :Example: Move first row to column names, remove row
    #   and remove rows above selected row without resetting
    #   index:

    # ```python

    #     example_dataframe = pd.DataFrame(data_dict)
    #     example_dataframe.row_to_names(2, remove_row=True, \
    #       remove_rows_above=True, reset_index= True)

    # :Output:

    # ```python

    #       3   3   3   lion    Basel
    #   0   1   1   1   rabbit  Cambridge
    #   1   2   2   2   leopard Shanghai
    #   2   3   3   3   lion    Basel
    #   3   1   1   1   rabbit  Cambridge
    #   4   2   2   2   leopard Shanghai
    #   5   3   3   3   lion    Basel

    # :Example: Move first row to column names, remove row,
    # and remove rows above selected row without resetting
    # index:

    # ```python

    #     example_dataframe = pd.DataFrame(data_dict)
    #     example_dataframe.row_to_names(2, remove_row=True, \
    #       remove_rows_above=True)

    # :Output:

    # ```python

    #        3  3  3     lion      Basel
    #     3  1  1  1   rabbit  Cambridge
    #     4  2  2  2  leopard   Shanghai
    #     5  3  3  3     lion      Basel
    #     6  1  1  1   rabbit  Cambridge
    #     7  2  2  2  leopard   Shanghai
    #     8  3  3  3     lion      Basel

    check("row_number", row_number, [int])

    warnings.warn(
        "The function row_to_names will, in the official 1.0 release, "
        "change its behaviour to reset the dataframe's index by default. "
        "You can prepare for this change right now by explicitly setting "
        "`reset_index=True` when calling on `row_to_names`."
    )

    df.columns = df.iloc[row_number, :]
    df.columns.name = None

    if remove_row:
        df = df.drop(df.index[row_number])

    if remove_rows_above:
        df = df.drop(df.index[range(row_number)])

    if reset_index:
        df = df.reset_index(drop=["index"])

    return df
