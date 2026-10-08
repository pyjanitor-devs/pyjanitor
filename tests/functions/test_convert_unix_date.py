import os

import pandas as pd
import pytest


@pytest.mark.skipif(os.name == "nt", reason="Skip *nix-specific tests on Windows")
def test_convert_unix_date():
    unix = [
        1_284_101_486,
        -2_147_483_648,
        2_147_483_648,
    ]
    df = pd.DataFrame(unix, columns=["dates"]).convert_unix_date("dates")

    assert df["dates"].dtype == "M8[s]"
    expected = pd.to_datetime(
        ["2010-09-10 06:51:26", "1901-12-13 20:45:52", "2038-01-19 03:14:08"]
    )
    assert (df["dates"] == expected).all()


@pytest.mark.skipif(os.name == "nt", reason="Skip *nix-specific tests on Windows")
def test_convert_unix_date_milliseconds():
    """Timestamps past the nanosecond range in seconds are read as milliseconds."""
    df = pd.DataFrame({"dates": [1_284_101_488_000]}).convert_unix_date("dates")

    assert df["dates"].iloc[0] == pd.Timestamp("2010-09-10 06:51:28")
