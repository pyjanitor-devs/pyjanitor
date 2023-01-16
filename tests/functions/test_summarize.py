import numpy as np
import pytest
import pandas as pd

from pandas.testing import assert_frame_equal
from pandas.api.types import is_numeric_dtype


@pytest.mark.functions
def test_dict_args_error(dataframe):
    """Raise if arg is not a dict/tuple"""
    with pytest.raises(
        TypeError, match="Argument 0 in the summarize function.+"
    ):
        dataframe.summarize(1)


@pytest.mark.functions
def test_dict_args_val_error(dataframe):
    """
    Raise if arg is a dict,
    key exist in the columns,
    but the value is not a string or callable
    """
    with pytest.raises(TypeError, match="func for a in argument 0.+"):
        dataframe.summarize({"a": 1})


func = lambda grp: grp.Revenue.sum() / grp.Quantity.sum()  # noqa: E731


@pytest.mark.functions
def test_dict_agg_error(dataframe):
    """
    Raise if func triggers an attributeerror/valueerror
    """
    with pytest.raises(AttributeError):
        dataframe.summarize({"a": func})


@pytest.mark.functions
def test_tuple_agg_error(dataframe):
    """
    Raise if func triggers an attributeerror/valueerror
    """
    with pytest.raises(AttributeError):
        dataframe.summarize(("a", func))


@pytest.mark.functions
def test_tuple_length_error_max(dataframe):
    """Raise if length of tuple is > 3"""
    with pytest.raises(
        ValueError, match=r"Argument 0 should have a maximum length of 3.+"
    ):
        dataframe.summarize(("a", "sum", "sum", "sum"))


@pytest.mark.functions
def test_tuple_length_error_min(dataframe):
    """Raise if length of tuple is < 2"""
    with pytest.raises(
        ValueError, match=r"Argument 0 should have a minimum length of 2.+"
    ):
        dataframe.summarize(("a",))


@pytest.mark.functions
def test_tuple_name_error(dataframe):
    """Raise if name is provided, and is not a string"""
    with pytest.raises(
        TypeError,
        match=r"The names \(position 2 in the tuple\) for argument 0.+",
    ):
        dataframe.summarize(("a", "sum", 1))


@pytest.mark.functions
def test_tuple_func_error(dataframe):
    """Raise if func is not a string/callable/list/tuple"""
    with pytest.raises(
        TypeError,
        match=r"The function \(position 1 in the tuple\) for argument 0.+",
    ):
        dataframe.summarize(("a", 1, "name"))


@pytest.mark.functions
def test_tuple_func_seq_error(dataframe):
    """Raise if func is a list/tuple, and its content is not str/callable"""
    with pytest.raises(
        TypeError, match=r"Entry 1 in the function sequence for argument 0.+"
    ):
        dataframe.summarize(("a", [np.sum, 1], "name"))


args = [
    {"a": lambda df: df.a.sum()},
    ("a", "sum"),
    ("a", np.sum),
    {"a": lambda f: np.sum(f.a)},
]


@pytest.mark.parametrize("test_input", args)
@pytest.mark.functions
def test_args_various(dataframe, test_input):
    """Test output for various arguments"""
    expected = dataframe.agg({"a": ["sum"]}).reset_index(drop=True)
    actual = dataframe.summarize(test_input)
    assert_frame_equal(expected, actual)


args = [("a", "sum", "{_col}_{_fn}"), ("a", np.sum, "{_col}_{_fn}")]


@pytest.mark.parametrize("test_input", args)
@pytest.mark.functions
def test_tuples_rename(dataframe, test_input):
    """Test output for various arguments"""
    expected = (
        dataframe.agg({"a": ["sum"]})
        .rename(columns={"a": "a_sum"})
        .reset_index(drop=True)
    )
    actual = dataframe.summarize(test_input)
    assert_frame_equal(expected, actual)


@pytest.mark.functions
def test_tuple_func_list(dataframe):
    """Test output for tuple list of functions"""
    expected = dataframe.agg({"a": ["sum"]}).reset_index(drop=True)
    actual = dataframe.summarize(("a", ["mean", "sum"]))
    assert_frame_equal(expected, actual)


@pytest.mark.functions
def test_tuple_func_list_rename(dataframe):
    """Test output for tuple list of functions"""
    expected = (
        dataframe.a.agg(["mean", "sum"])
        .add_prefix("a_")
        .to_frame()
        .T.reset_index(drop=True)
    )
    actual = dataframe.summarize(
        ("a", ["mean", "sum"], "{_col}_{_fn}")
    ).astype(float)
    assert_frame_equal(expected, actual)


@pytest.mark.functions
def test_tuple_func_list_grouped(dataframe):
    """Test output for tuple list of functions"""
    grp = dataframe.groupby("decorated-elephant")
    expected = grp.agg(a_sum=("a", "sum"), a_mean=("a", "mean"))
    actual = dataframe.summarize(
        ("a", ["sum", "mean"], "{_col}_{_fn}"), by="decorated-elephant"
    )
    assert_frame_equal(expected, actual)


@pytest.mark.functions
def test_tuple_func_list_grouped_dupes(dataframe):
    """Test output for tuple list of functions"""
    grp = dataframe.groupby("decorated-elephant")
    expected = grp.agg(a_sum0=("a", "sum"), a_sum1=("a", "sum"))
    actual = dataframe.summarize(
        ("a", ["sum", np.sum], "{_col}_{_fn}"), by={"by": "decorated-elephant"}
    )
    assert_frame_equal(expected, actual)


@pytest.mark.functions
def test_tuple_func_list_dupes(dataframe):
    """Test output for tuple list of functions"""
    # horrible hack
    expected = {
        (f"{key}0", f"{key}1", key): value.agg(["sum", np.sum, "mean"]).array
        for key, value in dataframe.select_dtypes("number").items()
    }
    expected = pd.concat(
        [pd.DataFrame([val], columns=key) for key, val in expected.items()],
        axis=1,
    )
    actual = dataframe.summarize(
        (dataframe.dtypes.map(is_numeric_dtype), ["sum", np.sum, np.mean])
    ).astype(float)
    assert_frame_equal(expected.sort_index(axis=1), actual.sort_index(axis=1))
