import pytest
from hypothesis import given
from hypothesis import settings

from janitor.testing_utils.strategies import df_strategy


@pytest.mark.functions
@given(df=df_strategy())
@settings(deadline=None)
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
@settings(deadline=None)
def test_bin_numeric_kwargs_has_no_retbins(df):
    with pytest.raises(ValueError):
        labels = ["a", "b", "c", "d", "e"]
        df.bin_numeric(
            from_column_name="a",
            to_column_name="a_bin",
            bins=5,
            labels=labels,
            retbins=True,
        )
