from janitor.testing_utils.fixtures import dataframe


def test_row_to_names(dataframe):
    df = dataframe.row_to_names(2)
    assert df.columns[0] == 3
    assert df.columns[1] == 3.234_612_5
    assert df.columns[2] == 3
    assert df.columns[3] == "lion"
    assert df.columns[4] == "Basel"


def test_row_to_names_delete_this_row(dataframe):
    df = dataframe.row_to_names(2, remove_row=True)
    assert df.iloc[2, 0] == 1
    assert df.iloc[2, 1] == 1.234_523_45
    assert df.iloc[2, 2] == 1
    assert df.iloc[2, 3] == "rabbit"
    assert df.iloc[2, 4] == "Cambridge"


def test_row_to_names_delete_above(dataframe):
    df = dataframe.row_to_names(2, remove_rows_above=True)
    assert df.iloc[0, 0] == 3
    assert df.iloc[0, 1] == 3.234_612_5
    assert df.iloc[0, 2] == 3
    assert df.iloc[0, 3] == "lion"
    assert df.iloc[0, 4] == "Basel"
