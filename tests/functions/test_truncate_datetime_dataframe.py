from datetime import datetime

import numpy as np
import pandas as pd

from janitor.functions.truncate_datetime import _truncate_datetime


def test_truncate_datetime_dataframe():
    """Test for truncate_datetime_dataframe"""
    x = datetime.now()

    time = {"datetime: ": [x], "Missing Data: ": np.NAN}

    df = pd.DataFrame.from_dict(time)
    df.truncate_datetime_dataframe("Year")

    assert df["datetime: "][0] == _truncate_datetime("year", x)
