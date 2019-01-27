import pandas as pd
import pytest
from janitor.testing_utils.fixtures import dataframe

@pytest.mark.parametrize(
    "invert,expected",
    [
        (False, ["a", "Bell__Chart", "cities"]),
        (True, ["decorated-elephant", "animals@#$%^"]),
    ],
)
def test_select_columns(dataframe, invert, expected):
    columns = ["a", "Bell__Chart", "cities"]
    df = dataframe.select_columns(columns=columns, invert=invert)

    pd.testing.assert_frame_equal(df, dataframe[expected])
