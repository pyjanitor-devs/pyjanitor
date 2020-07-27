import pytest
import pandas as pd
from janitor.timeseries import fill_missing_timestamps


@pytest.mark.timeseries
def test_datatypes_check():
    with pytest.raises(TypeError):
        assert fill_missing_timestamps(
            4, '1T', None, None, None
        )
        assert fill_missing_timestamps(
            pd.DataFrame([]), 2, None, None, None
        )
        assert fill_missing_timestamps(
            pd.DataFrame([]), '1B', 1, None, None
        )
        assert fill_missing_timestamps(
            pd.DataFrame([]), '1B', 'DateTime', '2020-01-01', None
        )
        assert fill_missing_timestamps(
            pd.DataFrame([]), '1B', 'DateTime', pd.Timestamp(2020, 1, 1), '2020-02-01'
        )

