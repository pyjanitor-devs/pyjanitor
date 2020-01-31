"""
Tests for sort_naturally.

Some places where this test suite could be improved:

- Replace example-based test with Hypothesis-generated property-based test. [intermediate]
- Provide another example-based test of something
that needs to be naturally rather than lexiographically sorted.
"""
import janitor
from natsort import natsorted
import pytest
import pandas as pd


@pytest.fixture
def well_dataframe():
    data = {
        "Well": ["A21", "A3", "A21", "B2", "B51", "B12"],
        "Value": [1, 2, 13, 3, 4, 7],
    }
    df = pd.DataFrame(data)
    return df


def test_sort_naturally(well_dataframe):
    sorted_df = well_dataframe.sort_naturally("Well")
    assert sorted_df["Well"].tolist() == natsorted(well_dataframe["Well"])
