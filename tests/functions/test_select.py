import pandas as pd
import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from janitor.functions.utils import DropLabel


@pytest.fixture
def dataframe():
    """Base DataFrame"""
    arrays = [
        ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
        ["one", "two", "one", "two", "one", "two", "one", "two"],
    ]
    tuples = list(zip(*arrays))
    index = pd.MultiIndex.from_tuples(tuples, names=["A", "B"])
    return pd.DataFrame(
        np.random.randint(9, size=(8, 2)),
        index=index,
        columns=["col1", "col2"],
    )


def test_args_and_rows_and_columns(dataframe):
    """
    Raise if args and rows/columns are provided.
    """
    with pytest.raises(
        ValueError,
        match="Either provide variable args with the axis parameter,.+",
    ):
        dataframe.select("*", columns="*")


def test_args_invert(dataframe):
    """Raise if args and invert is not a boolean"""
    with pytest.raises(TypeError, match="invert should be one of.+"):
        dataframe.select("col1", invert=1, axis="columns")


def test_args_axis(dataframe):
    """Raise assertionerror if args and axis is not index/columns"""
    with pytest.raises(AssertionError):
        dataframe.select("col1", axis=1)


def test_invert_no_args(dataframe):
    """Raise Error if no args and invert is True"""
    with pytest.raises(
        ValueError,
        match="`invert` is applicable only for the variable args parameter.",
    ):
        dataframe.select(columns="col1", invert=True)


def test_select_all_columns(dataframe):
    """Test output for select"""
    actual = dataframe.select(columns="*")
    assert_frame_equal(actual, dataframe)


def test_select_all_rows(dataframe):
    """Test output for select"""
    actual = dataframe.select(rows="*")
    assert_frame_equal(actual, dataframe)


def test_select_rows_only(dataframe):
    """Test output for rows only"""
    actual = dataframe.select(rows={"B": "two"})
    expected = dataframe.loc(axis=0)[(slice(None), "two")]
    assert_frame_equal(actual, expected)


def test_select_rows_only_args(dataframe):
    """Test output for rows only"""
    actual = dataframe.select({"B": "two"}, axis="index")
    expected = dataframe.loc(axis=0)[(slice(None), "two")]
    assert_frame_equal(actual, expected)


def test_select_rows_scalar_(dataframe):
    """Test output for rows only"""
    actual = dataframe.select(rows="bar")
    expected = dataframe.xs("bar", axis=0, level=0, drop_level=False)
    assert_frame_equal(actual, expected)


def test_select_columns_only(dataframe):
    """Test output for columns only"""
    actual = dataframe.select(columns=["col1", "col2"])
    expected = dataframe.loc[:, :]
    assert_frame_equal(actual, expected)


def test_select_columns_only_args(dataframe):
    """Test output for columns only"""
    actual = dataframe.select("col1", "col2", axis="columns")
    expected = dataframe.loc[:, :]
    assert_frame_equal(actual, expected)


def test_select_single_column(dataframe):
    """Test output for columns only"""
    actual = dataframe.select(columns="col1")
    expected = dataframe.loc[:, ["col1"]]
    assert_frame_equal(actual, expected)


def test_select_single_row(dataframe):
    """Test output for row only"""
    actual = dataframe.select(rows=("bar", "one"))
    expected = dataframe.loc[[("bar", "one")]]
    assert_frame_equal(actual, expected)


def test_select_columns_scalar(dataframe):
    """Test output for columns only"""
    actual = dataframe.select(columns="col*")
    expected = dataframe.loc[:, :]
    assert_frame_equal(actual, expected)


def test_select_rows_and_columns(dataframe):
    """Test output for both rows and columns"""
    actual = dataframe.select(
        rows=DropLabel(lambda df: df.eval('A == "foo"')),
        columns=DropLabel(slice("col2", None)),
    )
    expected = dataframe.loc[["bar", "baz", "qux"], ["col1"]]
    assert_frame_equal(actual, expected)
