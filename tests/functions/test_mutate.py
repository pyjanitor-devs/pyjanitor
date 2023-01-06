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
    with pytest.raises(TypeError, match="Argument 0 in the mutate function.+"):
        dataframe.mutate(1)

@pytest.mark.functions
def test_dict_args_val(dataframe):
    """
    Raise if arg is a dict,
    key exist in the columns,
    but the value is not a string or callable
    """
    with pytest.raises(TypeError, match="func for a in argument 0.+"):
        dataframe.mutate({"a":1})

@pytest.mark.functions
def test_dict_nested(dataframe):
    """
    Raise if func in nested dict 
    is a wrong type
    """
    with pytest.raises(TypeError, match="func in nested dictionary for a in argument 0.+"):
        dataframe.mutate({"a":{"b":1}})

@pytest.mark.functions
def test_dict_str(dataframe):
    """Test output for dict"""
    expected = dataframe.assign(a = dataframe.a.transform('sqrt'))
    actual = dataframe.mutate({"a":"sqrt"})
    assert_frame_equal(expected, actual)

@pytest.mark.functions
def test_dict_callable(dataframe):
    """Test output for dict"""
    expected = dataframe.assign(a = dataframe.a.transform(np.sqrt))
    actual = dataframe.mutate({"a":np.sqrt})
    assert_frame_equal(expected, actual)

@pytest.mark.functions
def test_dict_nested(dataframe):
    """Test output for dict"""
    expected = dataframe.assign(b = dataframe.a.transform('sqrt'))
    actual = dataframe.mutate({"a":{"b":"sqrt"}})
    assert_frame_equal(expected, actual)