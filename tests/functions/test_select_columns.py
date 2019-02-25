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
    df = dataframe.select_columns(search_cols=columns, invert=invert)

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
    df = dataframe.select_columns(search_cols=columns, invert=invert)

    pd.testing.assert_frame_equal(df, dataframe[expected])
