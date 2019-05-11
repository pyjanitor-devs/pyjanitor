import pytest
from hypothesis import given

from janitor.errors import JanitorError
from janitor.testing_utils.strategies import (categoricaldf_strategy,
                                              df_strategy,)


@pytest.mark.functions
@given(df=categoricaldf_strategy())
def test_encode_categorical(df):
    df = df.encode_categorical("names")
    assert df["names"].dtypes == "category"


@pytest.mark.functions
@given(df=df_strategy())
def test_encode_categorical_missing_column(df):
    with pytest.raises(AssertionError):
        df.encode_categorical("aloha")


@pytest.mark.functions
@given(df=df_strategy())
def test_encode_categorical_missing_columns(df):
    with pytest.raises(AssertionError):
        df.encode_categorical(["animals@#$%^", "cities", "aloha"])


@pytest.mark.functions
@given(df=df_strategy())
def test_encode_categorical_invalid_input(df):
    with pytest.raises(JanitorError):
        df.encode_categorical(1)
