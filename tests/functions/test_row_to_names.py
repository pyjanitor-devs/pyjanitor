import pytest


@pytest.mark.functions
def test_row_to_names(dataframe):
    df = dataframe.row_to_names(2)
    assert df.columns[0] == 3
    assert df.columns[1] == 3.234_612_5
    assert df.columns[2] == 3
    assert df.columns[3] == "lion"
    assert df.columns[4] == "Basel"


@pytest.mark.functions
def test_row_to_names_delete_this_row(dataframe):
    df = dataframe.row_to_names(2, remove_row=True)
    assert df.iloc[2, 0] == 1
    assert df.iloc[2, 1] == 1.234_523_45
    assert df.iloc[2, 2] == 1
    assert df.iloc[2, 3] == "rabbit"
    assert df.iloc[2, 4] == "Cambridge"
    assert df.iloc[2].name == 2

@pytest.mark.functions
def test_row_to_names_delete_the_row_without_resetting_index(dataframe):
    df = dataframe.row_to_names(2, remove_row=True, reset_index=False)
    assert df.iloc[2, 0] == 1
    assert df.iloc[2, 1] == 1.234_523_45
    assert df.iloc[2, 2] == 1
    assert df.iloc[2, 3] == "rabbit"
    assert df.iloc[2, 4] == "Cambridge"
    assert df.iloc[2].name == 3

@pytest.mark.functions
def test_row_to_names_delete_above(dataframe):
    df = dataframe.row_to_names(2, remove_rows_above=True)
    assert df.iloc[0, 0] == 3
    assert df.iloc[0, 1] == 3.234_612_5
    assert df.iloc[0, 2] == 3
    assert df.iloc[0, 3] == "lion"
    assert df.iloc[0, 4] == "Basel"
    assert df.iloc[0].name == 0

@pytest.mark.functions
def test_row_to_names_delete_above_without_resetting_index(dataframe):
    df = dataframe.row_to_names(2, remove_rows_above=True, reset_index=False)
    assert df.iloc[0, 0] == 3
    assert df.iloc[0, 1] == 3.234_612_5
    assert df.iloc[0, 2] == 3
    assert df.iloc[0, 3] == "lion"
    assert df.iloc[0, 4] == "Basel"
    assert df.iloc[0].name == 2
