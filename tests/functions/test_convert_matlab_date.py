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

    assert df["dates"].dtype == "M8[ns]"
