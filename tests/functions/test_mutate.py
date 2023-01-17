import numpy as np
import pytest

from pandas.testing import assert_frame_equal
from pandas.api.types import is_numeric_dtype


@pytest.mark.functions
def test_empty_args(dataframe):
    """Test output if args is empty"""
    assert_frame_equal(dataframe.mutate(), dataframe)


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


args = [("a", lambda f: np.sqrt(f)), ("a", "sqrt"), ("a", np.sqrt)]


@pytest.mark.parametrize("test_input", args)
@pytest.mark.functions
def test_args_various(dataframe, test_input):
    """Test output for various arguments"""
    expected = dataframe.assign(a=dataframe.a.transform("sqrt"))
    actual = dataframe.mutate(test_input)
    assert_frame_equal(expected, actual)


args = [
    (("a", "sum"), "decorated-elephant"),
    (("a", lambda f: f.sum()), "decorated-elephant"),
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
def test_tuple_func_str_rename(dataframe):
    """Test output for tuple string function"""
    expected = dataframe.assign(a_sqrt=dataframe.a.transform("sqrt"))
    actual = dataframe.mutate(("a", "sqrt", "{_col}_{_fn}"))
    assert_frame_equal(expected, actual)


@pytest.mark.functions
def test_tuple_func_callable_rename(dataframe):
    """Test output for tuple string function"""
    expected = dataframe.assign(a_sqrt=dataframe.a.transform("sqrt"))
    actual = dataframe.mutate(("a", np.sqrt, "{_col}_{_fn}"))
    assert_frame_equal(expected, actual)


@pytest.mark.functions
def test_tuple_func_list(dataframe):
    """Test output for tuple list of functions"""
    expected = dataframe.assign(a=dataframe.a.transform("sqrt").sum())
    actual = dataframe.mutate(("a", [np.sqrt, np.sum]))
    assert_frame_equal(expected, actual)


@pytest.mark.functions
def test_tuple_func_list_rename(dataframe):
    """Test output for tuple list of functions"""
    expected = dataframe.assign(
        a_sqrt=dataframe.a.transform("sqrt"),
        **{"a_<lambda>": dataframe.a.sum()},
    )
    actual = dataframe.mutate(
        ("a", ["sqrt", lambda f: f.sum()], "{_col}_{_fn}")
    )
    assert_frame_equal(expected, actual)


@pytest.mark.functions
def test_tuple_func_list_grouped(dataframe):
    """Test output for tuple list of functions"""
    grp = dataframe.groupby("decorated-elephant")
    expected = dataframe.assign(
        a_sum=grp.a.transform("sum"), a_mean=grp.a.transform("mean")
    )
    func = lambda f: f.transform("mean")  # noqa: E731
    func.__name__ = "mean"
    actual = dataframe.mutate(
        ("a", ["sum", func], "{_col}_{_fn}"), by="decorated-elephant"
    )
    assert_frame_equal(expected, actual)


@pytest.mark.functions
def test_tuple_func_list_grouped_dupes(dataframe):
    """Test output for tuple list of functions"""
    grp = dataframe.groupby("decorated-elephant")
    expected = dataframe.assign(
        a_sum0=grp.a.transform(np.sum), a_sum1=grp.a.transform("sum")
    )
    actual = dataframe.mutate(
        ("a", ["sum", np.sum], "{_col}_{_fn}"), by="decorated-elephant"
    )
    assert_frame_equal(expected, actual)


@pytest.mark.functions
def test_tuple_func_list_dupes(dataframe):
    """Test output for tuple list of functions"""
    A = dataframe.select_dtypes("number").add_suffix("0").transform(np.sqrt)
    B = dataframe.select_dtypes("number").add_suffix("1").transform("sqrt")
    C = dataframe.select_dtypes("number").transform(abs)
    expected = {**A, **B, **C}
    expected = dataframe.assign(**expected)
    actual = dataframe.mutate(
        (dataframe.dtypes.map(is_numeric_dtype), ["sqrt", np.sqrt, abs])
    )
    assert_frame_equal(expected.sort_index(axis=1), actual.sort_index(axis=1))
