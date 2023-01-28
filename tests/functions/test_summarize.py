import numpy as np
import pytest
import pandas as pd

from pandas.testing import assert_frame_equal
from pandas.api.types import is_numeric_dtype
from janitor import SD


func = lambda grp: grp.Revenue.sum() / grp.Quantity.sum()  # noqa: E731


@pytest.mark.functions
def test_SD_agg_error(dataframe):
    """
    Raise if func triggers an attributeerror/valueerror
    """
    with pytest.raises(AttributeError):
        dataframe.summarize(SD("a").add_fns(func))


@pytest.mark.functions
def test_SD_name_error(dataframe):
    """Raise if name is provided, and is not a string"""
    with pytest.raises(
        TypeError,
        match=r"name should be.+",
    ):
        dataframe.summarize(SD("a").add_fns("sum").rename(1))


@pytest.mark.functions
def test_SD_name_dupe_error(dataframe):
    """Raise if names_glue already exists"""
    with pytest.raises(
        ValueError,
        match=r"A name has already been assigned",
    ):
        dataframe.summarize(SD("a").add_fns("sum").rename("1").rename("name"))


@pytest.mark.functions
def test_SD_func_dupe_error(dataframe):
    """Raise if func already exists"""
    with pytest.raises(
        ValueError,
        match=r"A function has already been assigned",
    ):
        dataframe.summarize(
            SD("a").add_fns("sum").add_fns(np.sum).rename("name")
        )


@pytest.mark.functions
def test_SD_no_func_error(dataframe):
    """Raise if func is not provided"""
    with pytest.raises(
        ValueError,
        match=r"Kindly provide a function for Argument 0",
    ):
        dataframe.summarize(SD("a"))


@pytest.mark.functions
def test_SD_func_error(dataframe):
    """Raise if func is a wrong type"""
    with pytest.raises(
        TypeError,
        match=r"Function should be.+",
    ):
        dataframe.summarize(SD("a").add_fns(1).rename("name"))


@pytest.mark.functions
def test_SD_func_seq_error(dataframe):
    """Raise if func is a list and contains wrong type"""
    with pytest.raises(
        TypeError,
        match=r"Function in list/tuple.+",
    ):
        dataframe.summarize(SD("a").add_fns(["sum", 1]).rename("name"))


args = [
    lambda df: df.sum(),
    "sum",
    np.sum,
]


@pytest.mark.parametrize("func", args)
@pytest.mark.functions
def test_args_various(dataframe, func):
    """Test output for various arguments"""
    expected = dataframe.agg({"a": ["sum"]}).reset_index(drop=True)
    actual = dataframe.summarize(SD("a").add_fns(func))
    assert_frame_equal(expected, actual)


args = [("a", "sum", "{_col}_{_fn}"), ("a", np.sum, "{_col}_{_fn}")]


@pytest.mark.parametrize("cols,fn,names", args)
@pytest.mark.functions
def test_args_rename(dataframe, cols, fn, names):
    """Test output for various arguments"""
    expected = (
        dataframe.agg({"a": ["sum"]})
        .rename(columns={"a": "a_sum"})
        .reset_index(drop=True)
    )
    actual = dataframe.summarize(SD(cols).add_fns(fn).rename(names))
    assert_frame_equal(expected, actual)


@pytest.mark.functions
def test_args_func_list(dataframe):
    """Test output for list of functions"""
    expected = dataframe.agg({"a": ["sum"]}).reset_index(drop=True)
    actual = dataframe.summarize(SD("a").add_fns(["mean", "sum"]))
    assert_frame_equal(expected, actual)


@pytest.mark.functions
def test_args_func_list_rename(dataframe):
    """Test output for list of functions"""
    expected = (
        dataframe.a.agg(["mean", "sum"])
        .add_prefix("a_")
        .to_frame()
        .T.reset_index(drop=True)
    )
    actual = dataframe.summarize(
        SD("a").add_fns(["mean", "sum"]).rename("{_col}_{_fn}")
    ).astype(float)
    assert_frame_equal(expected, actual)


@pytest.mark.functions
def test_args_func_list_grouped(dataframe):
    """Test output for list of functions"""
    grp = dataframe.groupby("decorated-elephant")
    expected = grp.agg(a_sum=("a", "sum"), a_mean=("a", "mean"))
    actual = dataframe.summarize(
        SD("a").add_fns(["sum", "mean"]).rename("{_col}_{_fn}"),
        by="decorated-elephant",
    )

    assert_frame_equal(expected, actual)


@pytest.mark.functions
def test_args_func_list_grouped_dupes(dataframe):
    """Test output for list of functions"""
    grp = dataframe.groupby("decorated-elephant")
    expected = grp.agg(a_sum0=("a", "sum"), a_sum1=("a", "sum"))
    actual = dataframe.summarize(
        SD("a").add_fns(["sum", "sum"]).rename("{_col}_{_fn}"),
        by={"by": "decorated-elephant"},
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
        SD(dataframe.dtypes.map(is_numeric_dtype)).add_fns(
            ["sum", np.sum, np.mean]
        )
    ).astype(float)
    assert_frame_equal(expected.sort_index(axis=1), actual.sort_index(axis=1))


@pytest.mark.functions
def test_dataframe():
    """Test output if a dataframe is returned"""
    df = [
        {"A": "foo", "B": "one", "C": -0.575247, "D": 1.346061},
        {"A": "bar", "B": "one", "C": 0.254161, "D": 1.511763},
        {"A": "foo", "B": "two", "C": -1.143704, "D": 1.627081},
        {"A": "bar", "B": "three", "C": 0.215897, "D": -0.990582},
        {"A": "foo", "B": "two", "C": 1.193555, "D": -0.441652},
        {"A": "bar", "B": "two", "C": -0.077118, "D": 1.211526},
        {"A": "foo", "B": "one", "C": -0.40853, "D": 0.26852},
        {"A": "foo", "B": "three", "C": -0.862495, "D": 0.02458},
    ]

    df = pd.DataFrame(df)
    expected = df.groupby("A").C.describe().add_prefix("C_")
    actual = df.summarize(SD("C").add_fns("describe"), by="A")
    assert_frame_equal(expected, actual)
