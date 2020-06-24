import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from janitor.functions import update_where


@pytest.mark.functions
def test_update_where(dataframe):
    """
    Test that it accepts conditional parameters
    """
    assert_frame_equal(
        dataframe.update_where(
            (dataframe["decorated-elephant"] == 1)
            & (dataframe["animals@#$%^"] == "rabbit"),
            "cities",
            "Durham",
        ),
        dataframe.replace("Cambridge", "Durham"),
    )


def test_update_where_query():
    """Test that function works with pandas query-style string expression."""
    df = pd.DataFrame(
        {"a": [1, 2, 3, 4], "b": [5, 6, 7, 8], "c": [0, 0, 0, 0]}
    )
    expected = pd.DataFrame(
        {"a": [1, 2, 3, 4], "b": [5, 6, 7, 8], "c": [0, 0, 10, 0]}
    )
    result = update_where(
        df, conditions="a > 2 and b < 8", target_column_name="c", target_val=10
    )

    assert_frame_equal(result, expected)
