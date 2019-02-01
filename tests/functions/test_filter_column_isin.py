from janitor.testing_utils.strategies import categoricaldf_strategy, names
import pytest
from hypothesis import given, assume
import pandas as pd


@pytest.mark.test
@pytest.mark.functions
@given(
    df=categoricaldf_strategy(),
    iterable=names()
)
def test_filter_column_isin(df, iterable):
    assume(len(iterable) >= 1)
    assume(len(df) >= 1)
    df = df.filter_column_isin("names", iterable)
    assert set(df['names']).issubset(iterable)


# @pytest.mark.test
# @pytest.mark.functions
# @given(
#     df=categoricaldf_strategy()
# )
# def test_filter_column_isin(df):
#     print(df)
#     df = df.filter_column_isin("names", ['John'])
#     assert set(df['names']).issubset(['John'])
