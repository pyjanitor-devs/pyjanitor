import numpy as np
import pytest

from pandas.testing import assert_frame_equal


@pytest.mark.functions
def test_empty_args(dataframe):
    """Test output if args is empty"""
    assert_frame_equal(dataframe.mutate(), dataframe)


@pytest.mark.functions
def test_dict_args_error(dataframe):
    """Raise if arg is not a dict/tuple"""
    with pytest.raises(TypeError, match="Argument 0 in the mutate function.+"):
        dataframe.mutate(1)


@pytest.mark.functions
def test_dict_args_val_error(dataframe):
    """
    Raise if arg is a dict,
    key exist in the columns,
    but the value is not a string or callable
    """
    with pytest.raises(TypeError, match="func for a in argument 0.+"):
        dataframe.mutate({"a": 1})


@pytest.mark.functions
def test_dict_nested_error(dataframe):
    """
    Raise if func in nested dict
    is a wrong type
    """
    with pytest.raises(
        TypeError, match="func in nested dictionary for a in argument 0.+"
    ):
        dataframe.mutate({"a": {"b": 1}})


@pytest.mark.functions
def test_tuple_length_error(dataframe):
    """Raise if length of tuple is not 3"""
    with pytest.raises(
        ValueError, match="The tuple length of argument 0 should be 3,.+"
    ):
        dataframe.mutate(("a", "sum"))


@pytest.mark.functions
def test_tuple_name_error(dataframe):
    """Raise if name is provided, and is not a string"""
    with pytest.raises(
        TypeError, match="The third value in the tuple argument 0.+"
    ):
        dataframe.mutate(("a", "sum", 1))


@pytest.mark.functions
def test_tuple_func_error(dataframe):
    """Raise if func is provided, and is not a string/callable/list/tuple"""
    with pytest.raises(
        TypeError, match="The second value in the tuple argument 0.+"
    ):
        dataframe.mutate(("a", 1, "name"))


@pytest.mark.functions
def test_dict_str(dataframe):
    """Test output for dict"""
    expected = dataframe.assign(a=dataframe.a.transform("sqrt"))
    actual = dataframe.mutate({"a": "sqrt"})
    assert_frame_equal(expected, actual)


@pytest.mark.functions
def test_dict_str_grouped(dataframe):
    """Test output for dict on a groupby"""
    expected = dataframe.assign(
        a=dataframe.groupby("decorated-elephant").a.transform("sum")
    )
    actual = dataframe.mutate({"a": "sum"}, by="decorated-elephant")
    assert_frame_equal(expected, actual)


@pytest.mark.functions
def test_dict_callable(dataframe):
    """Test output for dict"""
    expected = dataframe.assign(a=dataframe.a.transform(np.sqrt))
    actual = dataframe.mutate({"a": np.sqrt})
    assert_frame_equal(expected, actual)


@pytest.mark.functions
def test_dict_callable_grouped(dataframe):
    """Test output for dict on a groupby"""
    expected = dataframe.assign(
        a=dataframe.groupby("decorated-elephant").a.transform(np.sum)
    )
    actual = dataframe.mutate(
        {"a": lambda f: f.transform("sum")},
        by={"by": "decorated-elephant", "sort": False},
    )
    assert_frame_equal(expected, actual)


@pytest.mark.functions
def test_dict_nested(dataframe):
    """Test output for dict"""
    expected = dataframe.assign(b=dataframe.a.transform("sqrt"))
    actual = dataframe.mutate({"a": {"b": "sqrt"}})
    assert_frame_equal(expected, actual)


@pytest.mark.functions
def test_dict_nested_grouped_str(dataframe):
    """Test output for dict on a groupby"""
    expected = dataframe.assign(
        b=dataframe.groupby("decorated-elephant").a.transform("sum")
    )
    actual = dataframe.mutate({"a": {"b": "sum"}}, by="decorated-elephant")
    assert_frame_equal(expected, actual)


@pytest.mark.functions
def test_dict_nested_grouped_callable(dataframe):
    """Test output for dict on a groupby"""
    expected = dataframe.assign(
        b=dataframe.groupby("decorated-elephant").a.transform(np.sum)
    )
    actual = dataframe.mutate(
        {"a": {"b": lambda f: f.transform(np.sum)}},
        by={"by": "decorated-elephant", "sort": False},
    )
    assert_frame_equal(expected, actual)
