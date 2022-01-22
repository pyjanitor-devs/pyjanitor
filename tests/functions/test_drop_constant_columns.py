"""Tests for drop_constant_columns."""
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal


@pytest.mark.functions
def test_drop_constant_columns(df_constant_columns):
    """Test that executes drop_constant_columns function."""
    processed_df = df_constant_columns.drop_constant_columns()
    expected_col_list = ["Bell__Chart", "decorated-elephant", "cities"]
    assert processed_df.columns.to_list() == expected_col_list
    data = {
        "Bell__Chart": [1.234_523_45, 2.456_234, 3.234_612_5] * 3,
        "decorated-elephant": [1, 2, 3] * 3,
        "cities": ["Cambridge", "Shanghai", "Basel"] * 3,
    }
    expected_df = pd.DataFrame(data)
    assert_frame_equal(processed_df, expected_df)
