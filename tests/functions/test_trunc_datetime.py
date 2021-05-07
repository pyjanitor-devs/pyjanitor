import numpy as np
import pandas as pd
import pytest
from datetime import datetime
from pandas.testing import assert_frame_equal
from janitor import trunc_datetime


def test_trunc_datetime():
        x = datetime.now()
        print(x)
        x = trunc_datetime("month", x)
        time = {
                'Year': [x.year],
                'Month': [x.month],
                'Day': [x.day],
                'Hour': [x.hour],
                'Minute': [x.minute],
                'Second': [x.second],
            }

        assert time['Day'] == 1
        assert time['Month'] == datetime.now().month
