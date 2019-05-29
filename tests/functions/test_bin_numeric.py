import pytest
from hypothesis import given

from janitor.testing_utils.strategies import df_strategy


@pytest.mark.functions
@given(df=df_strategy())
def test_bin_numeric_expected_columns(df):

    df = df.bin_numeric(from_column_name="a", to_column_name="a_bin")
    expected_columns = [
        "a",
        "Bell__Chart",
        "decorated-elephant",
        "animals@#$%^",
        "cities",
        "a_bin",
    ]

    assert set(df.columns) == set(expected_columns)


@pytest.mark.functions
@given(df=df_strategy())
def test_bin_numeric_num_labels(df):

    with pytest.raises(ValueError):
        labels = ["a", "b", "c", "d", "e"]
        df.bin_numeric(
            from_column_name="a",
            to_column_name="a_bin",
            num_bins=6,
            labels=labels,
        )
