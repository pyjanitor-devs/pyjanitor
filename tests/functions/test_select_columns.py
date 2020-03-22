import pandas as pd
import pytest


@pytest.mark.functions
@pytest.mark.parametrize(
    "invert,expected",
    [
        (False, ["a", "Bell__Chart", "cities"]),
        (True, ["decorated-elephant", "animals@#$%^"]),
    ],
)
def test_select_columns(dataframe, invert, expected):
    columns = ["a", "Bell__Chart", "cities"]
    df = dataframe.select_columns(search_column_names=columns, invert=invert)

    pd.testing.assert_frame_equal(df, dataframe[expected])


@pytest.mark.functions
@pytest.mark.parametrize(
    "invert,expected",
    [
        (False, ["Bell__Chart", "a", "animals@#$%^"]),
        (True, ["decorated-elephant", "cities"]),
    ],
)
def test_select_columns_glob_inputs(dataframe, invert, expected):
    columns = ["Bell__Chart", "a*"]
    df = dataframe.select_columns(search_column_names=columns, invert=invert)

    pd.testing.assert_frame_equal(df, dataframe[expected])


@pytest.mark.functions
@pytest.mark.parametrize(
    "columns",
    [
        ["a", "Bell__Chart", "foo"],
        ["a", "Bell__Chart", "foo", "bar"],
        ["a*", "Bell__Chart", "foo"],
        ["a*", "Bell__Chart", "foo", "bar"],
    ],
)
def test_select_columns_missing_columns(dataframe, columns):
    """Check that passing non-existent column names or search strings raises NameError"""
    with pytest.raises(NameError):
        dataframe.select_columns(search_column_names=columns)


@pytest.mark.functions
@pytest.mark.parametrize(
    "columns", ["a", ("a", "Bell__Chart"), {"a", "Bell__Chart"}]
)
def test_select_columns_input(dataframe, columns):
    """Check that passing an iterable that is not a list raises TypeError."""
    with pytest.raises(TypeError):
        dataframe.select_columns(search_column_names=columns)
