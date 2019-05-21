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
    """
    `filter_column_isin` should return the property that the column of
    interest's set of values should be a subset of the iterable provided.
    This encompasses a few scenarios:

    - Each element in iterable is present in the column.
    - No elements of iterable are present in the column.
    - A subset of elements in iterable are present in the column.

    All 3 cases can be caught by using subsets.
    """
    with pytest.raises(ValueError):
        assert df.filter_column_isin("names", [])
    df = df.filter_column_isin("names", iterable)
    assert set(df["names"]).issubset(iterable)


@pytest.mark.functions
@given(df=categoricaldf_strategy(), iterable=names_strategy())
def test_complement(df, iterable):
    df = df.filter_column_isin("names", iterable, complement=True)
    assert not set(df["names"]).issubset(iterable)

