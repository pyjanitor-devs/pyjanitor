import pytest
from hypothesis import assume
from hypothesis import given
from hypothesis import settings

from janitor.testing_utils.strategies import categoricaldf_strategy
from janitor.testing_utils.strategies import names_strategy


@pytest.mark.functions
@given(df=categoricaldf_strategy(), iterable=names_strategy())
@settings(deadline=None)
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
    assume(len(iterable) >= 1)
    df = df.filter_column_isin("names", iterable)
    assert set(df["names"]).issubset(iterable)
