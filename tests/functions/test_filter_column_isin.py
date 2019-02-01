import pandas as pd
import pytest
from hypothesis import assume, given

from janitor.testing_utils.strategies import (
    categoricaldf_strategy,
    names_strategy,
)


@pytest.mark.functions
@given(df=categoricaldf_strategy(), iterable=names_strategy())
def test_filter_column_isin(df, iterable):
    assume(len(iterable) >= 1)
    df = df.filter_column_isin("names", iterable)
    assert set(df["names"]).issubset(iterable)
