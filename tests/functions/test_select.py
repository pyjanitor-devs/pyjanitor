import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from janitor.functions.select import DropLabel, select


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


def test_import_does_not_register_legacy_select():
    """Do not override a pandas select method during janitor import."""
    code = textwrap.dedent(
        """
        import warnings

        import pandas as pd

        def pandas_select(self, *args, **kwargs):
            result = self.loc[:, list(args)].copy()
            for name in kwargs:
                result[name] = 0
            return result

        pd.DataFrame.select = pandas_select
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import janitor

        assert pd.DataFrame.select is pandas_select
        assert not any("overriding" in str(warning.message) for warning in caught)

        frame = pd.DataFrame({"a": [1], "b": [2]})
        result = frame.select("a", computed=pd.col("b") * 2)
        assert list(result.columns) == ["a", "computed"]
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_series_axis(dataframe):
    """Raise if object is a  Series and axis is a column"""
    with pytest.raises(
        ValueError, match="columns axis is not supported for pd.Series.select"
    ):
        select(dataframe.iloc[:, 0], "bar", axis="columns")


def test_series_select(dataframe):
    """Test output when selecting on a Series"""
    series = dataframe.iloc[:, 0]
    actual = select(series, index="bar")
    expected = series.loc[["bar"]]
    assert_series_equal(actual, expected)


def test_args_and_rows_and_columns(dataframe):
    """
    Raise if args and rows/columns are provided.
    """
    with pytest.raises(
        ValueError,
        match="Either provide variable args with the axis parameter,.+",
    ):
        select(dataframe, "*", columns="*")


def test_args_invert(dataframe):
    """Raise if args and invert is not a boolean"""
    with pytest.raises(TypeError, match="invert should be one of.+"):
        select(dataframe, "col1", invert=1, axis="columns")


def test_args_axis(dataframe):
    """Raise ValueError if args and axis is not index/columns"""
    with pytest.raises(ValueError, match="axis should be either 'index' or 'columns'."):
        select(dataframe, "col1", axis=1)


def test_invert(dataframe):
    "Test output if invert is provided."
    actual = select(dataframe, columns=["col1"], index=("bar", "one"), invert=True)
    expected = dataframe.loc[("bar", "two") :, ["col2"]]
    assert_frame_equal(actual, expected)


def test_invert_args(dataframe):
    "Test output if invert is provided."
    actual = select(dataframe, ("bar", "one"), axis="index", invert=True)
    expected = dataframe.loc[("bar", "two") :, :]
    assert_frame_equal(actual, expected)


def test_select_all_columns(dataframe):
    """Test output for select"""
    actual = select(dataframe, columns="*")
    assert_frame_equal(actual, dataframe)


def test_select_all_rows(dataframe):
    """Test output for select"""
    actual = select(dataframe, index="*")
    assert_frame_equal(actual, dataframe)


def test_select_rows_only(dataframe):
    """Test output for rows only"""
    actual = select(dataframe, index={"B": "two"})
    expected = dataframe.loc(axis=0)[(slice(None), "two")]
    assert_frame_equal(actual, expected)


def test_select_rows_only_args(dataframe):
    """Test output for rows only"""
    actual = select(dataframe, {"B": "two"}, axis="index")
    expected = dataframe.loc(axis=0)[(slice(None), "two")]
    assert_frame_equal(actual, expected)


def test_select_rows_scalar_(dataframe):
    """Test output for rows only"""
    actual = select(dataframe, index="bar")
    expected = dataframe.xs("bar", axis=0, level=0, drop_level=False)
    assert_frame_equal(actual, expected)


def test_select_columns_only(dataframe):
    """Test output for columns only"""
    actual = select(dataframe, columns=["col1", "col2"])
    expected = dataframe.loc[:, :]
    assert_frame_equal(actual, expected)


@pytest.mark.parametrize(
    "selector",
    [
        [],
        np.array([], dtype=object),
        pd.Index([]),
        pd.Series([], dtype=object),
    ],
)
def test_empty_selectors(dataframe, selector):
    """Empty list-like selectors select nothing on either axis."""
    assert_frame_equal(dataframe.select_columns(selector), dataframe.iloc[:, []])
    assert_frame_equal(dataframe.select_rows(selector), dataframe.iloc[[], :])
    assert_frame_equal(select(dataframe, columns=selector), dataframe.iloc[:, []])
    assert_frame_equal(select(dataframe, index=selector), dataframe.iloc[[], :])
    assert_frame_equal(dataframe.select_columns(selector, invert=True), dataframe)
    assert_frame_equal(dataframe.select_rows(selector, invert=True), dataframe)


def test_empty_drop_label_selects_everything(dataframe):
    """An empty DropLabel drops nothing on either axis."""
    assert_frame_equal(dataframe.select_columns(DropLabel([])), dataframe)
    assert_frame_equal(dataframe.select_rows(DropLabel([])), dataframe)
    assert_frame_equal(
        dataframe.select_columns(DropLabel([]), invert=True),
        dataframe.iloc[:, []],
    )
    assert_frame_equal(
        dataframe.select_rows(DropLabel([]), invert=True),
        dataframe.iloc[[], :],
    )


def test_empty_dict_selects_nothing_on_flat_and_multiindex_columns():
    """An empty dictionary selects nothing on either column index type."""
    flat = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    multi = flat.copy()
    multi.columns = pd.MultiIndex.from_tuples([("x", "a"), ("x", "b")])

    for frame in (flat, multi):
        assert_frame_equal(frame.select_columns([]), frame.iloc[:, []])
        assert_frame_equal(frame.select_columns(DropLabel([])), frame)
        assert_frame_equal(frame.select_columns({}), frame.iloc[:, []])


def test_empty_tuple_keeps_label_semantics():
    """An empty tuple remains a label selector, not an empty list-like."""
    flat = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    multi = flat.copy()
    multi.columns = pd.MultiIndex.from_tuples([("x", "a"), ("x", "b")])

    with pytest.raises(KeyError):
        flat.select_columns(())
    assert_frame_equal(multi.select_columns(()), multi)


def test_empty_selectors_with_duplicate_labels_and_drop_label():
    """Empty selectors preserve duplicate labels and combined selection."""
    dataframe = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "a"])
    assert_frame_equal(dataframe.select_columns([]), dataframe.iloc[:, []])
    assert_frame_equal(dataframe.select_columns(DropLabel([])), dataframe)
    assert_frame_equal(
        dataframe.select_columns(["a", DropLabel([])]),
        dataframe,
    )


def test_select_columns_only_args(dataframe):
    """Test output for columns only"""
    actual = select(dataframe, "col1", "col2", axis="columns")
    expected = dataframe.loc[:, :]
    assert_frame_equal(actual, expected)


def test_select_single_column(dataframe):
    """Test output for columns only"""
    actual = select(dataframe, columns="col1")
    expected = dataframe.loc[:, ["col1"]]
    assert_frame_equal(actual, expected)


def test_select_single_row(dataframe):
    """Test output for row only"""
    actual = select(dataframe, index=("bar", "one"))
    expected = dataframe.loc[[("bar", "one")]]
    assert_frame_equal(actual, expected)


def test_select_columns_scalar(dataframe):
    """Test output for columns only"""
    actual = select(dataframe, columns="col*")
    expected = dataframe.loc[:, :]
    assert_frame_equal(actual, expected)


def test_select_rows_and_columns(dataframe):
    """Test output for both rows and columns"""
    actual = select(
        dataframe,
        index=DropLabel(lambda df: df.eval('A == "foo"')),
        columns=DropLabel(slice("col2", None)),
    )
    expected = dataframe.loc[["bar", "baz", "qux"], ["col1"]]
    assert_frame_equal(actual, expected)


def test_select_rows_and_columns_expr(dataframe):
    """Test output for both rows and columns"""
    actual = select(
        dataframe,
        index=~(pd.col("col1") == 2),
        columns=DropLabel(slice("col2", None)),
    )
    expected = dataframe.loc[~(pd.col("col1") == 2), ["col1"]]
    assert_frame_equal(actual, expected)


def test_expr_index_error(dataframe):
    """Raise if boolean from expression has wrong length"""
    with pytest.raises(
        IndexError, match="The boolean array output from the Expression.+"
    ):
        select(dataframe, columns=pd.col("col1") == 2)


def test_expr_boolean_error(dataframe):
    """Raise if expression does not evaluate to a boolean"""
    with pytest.raises(
        ValueError, match="The output of the Expression should be a 1-D boolean array."
    ):
        select(
            dataframe,
            columns=pd.col("col1") * 2,
        )
