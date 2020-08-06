"""
Tests for sort_naturally.

Some places where this test suite could be improved:

- Replace example-based test
with Hypothesis-generated property-based test. [intermediate]
- Provide another example-based test of something
that needs to be naturally rather than lexiographically sorted.
"""
import pandas as pd
import pytest
from natsort import natsorted
from pandas.testing import assert_frame_equal

import janitor  # noqa: F401


@pytest.fixture
def well_dataframe():
    data = {
        "Well": ["A21", "A3", "A21", "B2", "B51", "B12"],
        "Value": [1, 2, 13, 3, 4, 7],
    }
    df = pd.DataFrame(data)
    return df


def test_sort_naturally(well_dataframe):
    """Example-based test for sort_naturally.

    We check that:

    - the resultant dataframe is sorted identically
    to what natsorted would provide,
    - the data in the dataframe are not corrupted.
    """
    sorted_df = well_dataframe.sort_naturally("Well")
    assert sorted_df["Well"].tolist() == natsorted(well_dataframe["Well"])
    assert_frame_equal(sorted_df.sort_index(), well_dataframe)
