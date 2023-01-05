import numpy as np
import pandas as pd
import pytest

from pandas.testing import assert_frame_equal
from pandas import NA

@pytest.mark.functions
def test_empty_args(dataframe):
    """Test output if args is empty"""
    assert_frame_equal(dataframe.mutate(), dataframe)


