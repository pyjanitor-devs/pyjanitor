import pytest


@pytest.mark.functions
def test_limit_column_characters(dataframe):
    df = dataframe.limit_column_characters(1)
    assert df.columns[0] == "a"
    assert df.columns[1] == "B"
    assert df.columns[2] == "d"
    assert df.columns[3] == "a_1"
    assert df.columns[4] == "c"


@pytest.mark.functions
def test_limit_column_characters_different_positions(dataframe):
    df = dataframe
    df.columns = ["first", "first", "second", "second", "first"]
    df.limit_column_characters(3)

    assert df.columns[0] == "fir"
    assert df.columns[1] == "fir_1"
    assert df.columns[2] == "sec"
    assert df.columns[3] == "sec_1"
    assert df.columns[4] == "fir_2"


@pytest.mark.functions
def test_limit_column_characters_different_positions_different_separator(
    dataframe,
):
    df = dataframe
    df.columns = ["first", "first", "second", "second", "first"]
    df.limit_column_characters(3, ".")

    assert df.columns[0] == "fir"
    assert df.columns[1] == "fir.1"
    assert df.columns[2] == "sec"
    assert df.columns[3] == "sec.1"
    assert df.columns[4] == "fir.2"


@pytest.mark.functions
def test_limit_column_characters_all_unique(dataframe):
    df = dataframe.limit_column_characters(2)
    assert df.columns[0] == "a"
    assert df.columns[1] == "Be"
    assert df.columns[2] == "de"
    assert df.columns[3] == "an"
    assert df.columns[4] == "ci"
