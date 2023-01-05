import numpy as np
import pandas as pd
import pytest

from pandas.testing import assert_frame_equal
from pandas import NA

@pytest.mark.functions
def test_empty_args(dataframe):
    """Test output if args is empty"""
    assert_frame_equal(dataframe.mutate(), dataframe)


@pytest.mark.functions
def test_dict_args(dataframe):
    """Raise if arg is not a dict"""
    with pytest.raises(TypeError):
        dataframe.mutate(1)