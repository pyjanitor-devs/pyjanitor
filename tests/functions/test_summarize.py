import re

import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_string_dtype
from pandas.testing import assert_frame_equal

from janitor import col

func = lambda grp: grp.Revenue.sum() / grp.Quantity.sum()  # noqa: E731


@pytest.mark.functions
def test_Column_agg_error(dataframe):
    """
    Raise if func triggers an attributeerror/valueerror
    """
    with pytest.raises(AttributeError):
        dataframe.summarize(col("a").compute(func))


@pytest.mark.functions
def test_Column_type_error(dataframe):
    """
    Raise if column is not an instance of janitor.col
    """
    with pytest.raises(
        TypeError, match="Argument 0 in the summarize function.+"
    ):
        dataframe.summarize(dataframe.a.mean())


@pytest.mark.functions
def test_Column_by_error(dataframe):
    """
    Raise if `by` is a col class, and has a function
    assigned to it
    """
    with pytest.raises(
        ValueError, match="Function assignment is not required within by"
    ):
        dataframe.summarize(
            col("a").compute("sum"), by=col("b").compute("size")
        )


@pytest.mark.functions
def test_Column_exclude_error(dataframe):
    """
    Raise if a label is to be excluded,
    but a func is already assigned
    """
    with pytest.raises(
        ValueError,
        match="exclude should be applied before a function is assigned",
    ):
        dataframe.summarize(col("a", "b").compute("sum").exclude("a"))


@pytest.mark.functions
def test_Column_name_no_func_error(dataframe):
    """Raise if name is provided, and there is no function"""
    with pytest.raises(
        ValueError,
        match="rename is only applicable "
        "if there is a function to be applied.",
    ):
        dataframe.summarize(col("a").rename("1"))


@pytest.mark.functions
def test_Column_name_error(dataframe):
    """Raise if name is provided, and is not a string"""
    with pytest.raises(
        TypeError,
        match=r"names should be.+",
    ):
        dataframe.summarize(col("a").compute("sum").rename(1))


@pytest.mark.functions
def test_Column_name_dupe_error(dataframe):
    """Raise if names already exists"""
    with pytest.raises(
        ValueError,
        match=r"A name has already been assigned",
    ):
        dataframe.summarize(col("a").compute("sum").rename("1").rename("name"))


@pytest.mark.functions
def test_Column_func_dupe_error(dataframe):
    """Raise if func already exists"""
    with pytest.raises(
        ValueError,
        match=r"A function has already been assigned",
    ):
        dataframe.summarize(
            col("a").compute("sum").compute(np.sum).rename("name")
        )


@pytest.mark.functions
def test_Column_no_func_error(dataframe):
    """Raise if func is not provided"""
    with pytest.raises(
        ValueError,
        match=r"Kindly provide a function for Argument 0",
    ):
        dataframe.summarize(col("a"))


@pytest.mark.functions
def test_Column_func_error(dataframe):
    """Raise if func is a wrong type"""
    with pytest.raises(
        TypeError,
        match=r"Function should be.+",
    ):
        dataframe.summarize(col("a").compute("sum", 1).rename("name"))


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
    actual = dataframe.summarize(col("a").compute(func))
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
    actual = dataframe.summarize(col(cols).compute(fn).rename(names))
    assert_frame_equal(expected, actual)


@pytest.mark.functions
def test_args_func_list(dataframe):
    """Test output for list of functions"""
    expected = dataframe.agg({"a": ["mean", "sum"]})
    actual = (
        dataframe.summarize(col("a").compute("mean", "sum"))
        .stack(level=1)
        .droplevel(0)
    )
    assert_frame_equal(expected, actual)


@pytest.mark.functions
def test_args_func_list_lambda(dataframe):
    """Test output for list of functions"""
    expected = dataframe.groupby("cities").agg(
        {"a": [lambda f: f.mean(), lambda f: f.sum()]}
    )
    actual = dataframe.summarize(
        col("a").compute(lambda f: f.mean(), lambda f: f.sum()),
        by={"by": "cities"},
    )
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
        col("a").compute("mean", "sum").rename("{_col}_{_fn}")
    ).astype(float)
    assert_frame_equal(expected, actual)


@pytest.mark.functions
def test_args_func_list_grouped(dataframe):
    """Test output for list of functions"""
    grp = dataframe.groupby("decorated-elephant")
    expected = grp.agg(a_sum=("a", "sum"), a_mean=("a", "mean"))
    actual = dataframe.summarize(
        col("a").compute("sum", "mean").rename("{_col}_{_fn}"),
        by="decorated-elephant",
    )

    assert_frame_equal(expected, actual)


@pytest.mark.functions
def test_args_func_list_grouped_dupes(dataframe):
    """Raise for duplicate agg label"""
    with pytest.raises(
        ValueError,
        match="a_sum already exists as a label for an aggregated column",
    ):
        dataframe.summarize(
            col("a").compute("sum", "sum").rename("{_col}_{_fn}"),
            by={"by": "decorated-elephant"},
        )


@pytest.mark.functions
def test_args_func_list_grouped_dupes_tuple(dataframe):
    """Raise for duplicate agg label"""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "('a', 'sum') already exists as a label for an aggregated column"
        ),
    ):
        dataframe.summarize(
            col("a").compute("sum", "sum"),
            by={"by": "decorated-elephant"},
        )


@pytest.fixture
def df():
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

    return pd.DataFrame(df)


@pytest.mark.functions
def test_dataframe(df):
    """Test output if a dataframe is returned"""
    expected = df.groupby("A").C.describe()
    actual = df.summarize(
        col("A", "C").exclude("A").compute("describe"),
        by=col(is_string_dtype).exclude("B"),
    )

    assert_frame_equal(expected, actual)


@pytest.mark.functions
def test_dataframe_dupes(df):
    """Raise if duplicate labels already exist"""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "('C', 'mean') already exists as a label for an aggregated column"
        ),
    ):
        df.summarize(
            col("A", "C").exclude("A").compute("mean", "describe"),
            by=col(is_string_dtype).exclude("B"),
        )


@pytest.mark.functions
def test_dataframe_dupes1(df):
    """Raise if duplicate labels already exist"""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "('C', 'mean') already exists as a label for an aggregated column"
        ),
    ):
        df.summarize(
            col("A", "C").exclude("A").compute("describe", "mean"),
            by=col(is_string_dtype).exclude("B"),
        )
