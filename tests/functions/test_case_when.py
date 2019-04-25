import pandas as pd
import pytest

def test_case_when(dataframe):
    """
    Test that it accepts conditional parameters
    """
    pd.testing.assert_frame_equal(
    dataframe.case_when((dataframe['decorated-elephant'] == 1) & (dataframe['animals@#$%^'] == 'rabbit'), 'cities', 'Durham'),
    dataframe.replace('Cambridge', "Durham"))