"""Tests for `to_datetime` function."""
import numpy as np
import pandas as pd
import pytest


@pytest.mark.functions
def test_to_datetime():
    """Checks to_datetime functionality is as expected."""

    df = pd.DataFrame(
        {"date1": ["20190101", "20190102", "20190304", np.nan]}
    ).to_datetime("date1", format="%Y%m%d")
    assert df["date1"].dtype == np.dtype("datetime64[ns]")
    assert df["date1"].iloc[0].isoformat() == "2019-01-01T00:00:00"
