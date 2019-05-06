import pandas as pd
import pytest


def test_bin_numeric(dataframe):
    """
    Test that it accepts conditional parameters
    """
    pd.testing.assert_frame_equal(
        dataframe.bin_numeric(
            from_column="a", to_column="new", bins=[0, 2, 3], labels=["0-2", ">2-3"]
        ),
        dataframe.new=pd.DataFrame({"new": ["0-2", "0-2", ">2-3"]}),
    )