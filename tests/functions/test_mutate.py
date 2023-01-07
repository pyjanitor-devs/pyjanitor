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
    with pytest.raises(
        TypeError, match="The function for column a in argument 0.+"
    ):
        dataframe.mutate({"a": 1})


@pytest.mark.functions
def test_dict_nested_error(dataframe):
    """
    Raise if func in nested dict
    is a wrong type
    """
    with pytest.raises(
        TypeError,
        match="The function in the nested dictionary "
        "for column a in argument 0.+",
    ):
        dataframe.mutate({"a": {"b": 1}})


@pytest.mark.functions
def test_tuple_length_error_max(dataframe):
    """Raise if length of tuple is > 3"""
    with pytest.raises(
        ValueError, match=r"Argument 0 should have a maximum length of 3.+"
    ):
        dataframe.mutate(("a", "sum", "sum", "sum"))


@pytest.mark.functions
def test_tuple_length_error_min(dataframe):
    """Raise if length of tuple is < 2"""
    with pytest.raises(
        ValueError, match=r"Argument 0 should have a minimum length of 2.+"
    ):
        dataframe.mutate(("a",))


@pytest.mark.functions
def test_tuple_name_error(dataframe):
    """Raise if name is provided, and is not a string"""
    with pytest.raises(
        TypeError,
        match=r"The names \(position 2 in the tuple\) for argument 0.+",
    ):
        dataframe.mutate(("a", "sum", 1))


@pytest.mark.functions
def test_tuple_func_error(dataframe):
    """Raise if func is not a string/callable/list/tuple"""
    with pytest.raises(
        TypeError,
        match=r"The function \(position 1 in the tuple\) for argument 0.+",
    ):
        dataframe.mutate(("a", 1, "name"))


@pytest.mark.functions
def test_tuple_func_seq_error(dataframe):
    """Raise if func is a list/tuple, and its content is not str/callable"""
    with pytest.raises(
        TypeError, match=r"Entry 1 in the function sequence for argument 0.+"
    ):
        dataframe.mutate(("a", [np.sum, 1], "name"))


args = [{"a": "sqrt"}, {"a": np.sqrt}, ("a", "sqrt")]


@pytest.mark.parametrize("test_input", args)
@pytest.mark.functions
def test_args_various(dataframe, test_input):
    """Test output for various arguments"""
    expected = dataframe.assign(a=dataframe.a.transform("sqrt"))
    actual = dataframe.mutate(test_input)
    assert_frame_equal(expected, actual)


args = [
    ({"a": "sum"}, "decorated-elephant"),
    ({"a": lambda f: f.transform("sum")}, "decorated-elephant"),
    (("a", "sum"), "decorated-elephant"),
]


@pytest.mark.parametrize("test_input,by", args)
@pytest.mark.functions
def test_args_various_grouped(dataframe, test_input, by):
    """Test output for various arguments"""
    expected = dataframe.assign(
        a=dataframe.groupby("decorated-elephant").a.transform("sum")
    )
    actual = dataframe.mutate(test_input, by=by)
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


@pytest.mark.functions
def test_tuple_func_str_rename(dataframe):
    """Test output for tuple string function"""
    expected = dataframe.assign(a_sqrt=dataframe.a.transform("sqrt"))
    actual = dataframe.mutate(("a", "sqrt", "{_col}_{_fn}"))
    assert_frame_equal(expected, actual)
