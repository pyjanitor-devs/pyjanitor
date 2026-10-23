import pandas as pd
import pytest


@pytest.mark.functions
def test_convert_matlab_date():
    mlab = [
        733_301.0,
        729_159.0,
        734_471.0,
        737_299.563_296_356_5,
        737_300.000_000_000_0,
    ]
    df = pd.DataFrame(mlab, columns=["dates"]).convert_matlab_date("dates")

    assert pd.api.types.is_datetime64_dtype(df["dates"])
    expected = pd.to_datetime(
        [
            "2007-09-16",
            "1996-05-14",
            "2010-11-29",
            "2018-08-27 13:31:08.805199120",
            "2018-08-28",
        ],
        format="mixed",
    )
    assert (df["dates"] == expected).all()
