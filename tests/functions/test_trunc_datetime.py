from datetime import datetime
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

        # time[] returns datetime object, needs indexing.
        assert time['Day'][0] == 1
        assert time['Month'][0] == datetime.now().month
