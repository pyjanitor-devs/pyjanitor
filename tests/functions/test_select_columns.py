import pandas as pd
import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from janitor.functions.utils import IndexLabel


@pytest.mark.functions
@pytest.mark.parametrize(
    "invert,expected",
    [
        (False, ["a", "Bell__Chart", "cities"]),
        (True, ["decorated-elephant", "animals@#$%^"]),
    ],
)
def test_select_column_names(dataframe, invert, expected):
    "Base DataFrame"
    columns = ["a", "Bell__Chart", "cities"]
    df = dataframe.select_columns(columns, invert=invert)

    assert_frame_equal(df, dataframe[expected])


@pytest.mark.functions
@pytest.mark.parametrize(
    "invert,expected",
    [
        (False, ["Bell__Chart", "a", "animals@#$%^"]),
        (True, ["decorated-elephant", "cities"]),
    ],
)
def test_select_column_names_glob_inputs(dataframe, invert, expected):
    "Base DataFrame"
    columns = ["Bell__Chart", "a*"]
    df = dataframe.select_columns(columns, invert=invert)

    assert_frame_equal(df, dataframe[expected])


@pytest.mark.functions
@pytest.mark.parametrize(
    "columns",
    [
        ["a", "Bell__Chart", "foo"],
        ["a", "Bell__Chart", "foo", "bar"],
        ["a*", "Bell__Chart", "foo"],
        ["a*", "Bell__Chart", "foo", "bar"],
    ],
)
def test_select_column_names_missing_columns(dataframe, columns):
    """Check that passing non-existent column names or search strings raises KeyError"""  # noqa: E501
    with pytest.raises(KeyError):
        dataframe.select_columns(columns)


@pytest.mark.functions
@pytest.mark.parametrize(
    "invert,expected",
    [
        (False, ["Bell__Chart", "a", "decorated-elephant"]),
        (True, ["animals@#$%^", "cities"]),
    ],
)
def test_select_unique_columns(dataframe, invert, expected):
    """Test that only unique columns are returned."""
    columns = ["Bell__*", slice("a", "decorated-elephant")]
    df = dataframe.select_columns(columns, invert=invert)

    assert_frame_equal(df, dataframe[expected])


@pytest.mark.functions
@pytest.mark.parametrize(
    "invert,expected",
    [
        (False, ["Bell__Chart", "decorated-elephant"]),
        (True, ["a", "animals@#$%^", "cities"]),
    ],
)
def test_select_callable_columns(dataframe, invert, expected):
    """Test that columns are returned when a callable is passed."""

    def columns(x):
        return "-" in x.name or "_" in x.name

    df = dataframe.select_columns(columns, invert=invert)

    assert_frame_equal(df, dataframe[expected])


@pytest.fixture
def multiindex():
    """pytest fixture."""
    arrays = [
        ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
        pd.Categorical(
            ["one", "two", "one", "two", "one", "two", "one", "two"]
        ),
    ]
    index = pd.MultiIndex.from_arrays(arrays, names=["first", "second"])
    return pd.DataFrame(np.random.randn(4, 8), columns=index)


def test_multiindex(multiindex):
    """
    Test output for a MultiIndex and tuple passed.
    """
    assert_frame_equal(
        multiindex.select_columns(("bar", "one")),
        multiindex.loc[:, [("bar", "one")]],
    )


def test_multiindex_indexlabel(multiindex):
    """
    Test output for a MultiIndex with IndexLabel.
    """
    ix = IndexLabel(("bar", "one"))
    assert_frame_equal(
        multiindex.select_columns(ix), multiindex.loc[:, [("bar", "one")]]
    )


def test_multiindex_scalar(multiindex):
    """
    Test output for a MultiIndex with a scalar.
    """
    ix = IndexLabel("bar")
    assert_frame_equal(
        multiindex.select_columns(ix),
        multiindex.xs(key="bar", level=0, axis=1, drop_level=False),
    )


def test_multiindex_multiple_labels(multiindex):
    """
    Test output for a MultiIndex with multiple labels.
    """
    ix = [IndexLabel(("bar", "one")), IndexLabel(("baz", "two"))]
    assert_frame_equal(
        multiindex.select_columns(ix),
        multiindex.loc[:, [("bar", "one"), ("baz", "two")]],
    )


def test_multiindex_level(multiindex):
    """
    Test output for a MultiIndex on a level.
    """
    ix = IndexLabel("one", level=1)
    assert_frame_equal(
        multiindex.select_columns(ix),
        multiindex.xs(key="one", axis=1, level=1, drop_level=False),
    )


def test_multiindex_slice(multiindex):
    """
    Test output for a MultiIndex on a slice.
    """
    ix = [slice("bar", "foo")]
    ix = IndexLabel(ix)
    assert_frame_equal(
        multiindex.select_columns(ix), multiindex.loc[:, "bar":"foo"]
    )


def test_multiindex_bool(multiindex):
    """
    Test output for a MultiIndex on a boolean.
    """
    bools = [True, True, True, False, False, False, True, True]
    ix = IndexLabel([bools], level="first")
    assert_frame_equal(multiindex.select_columns(ix), multiindex.loc[:, bools])


def test_multiindex_invert(multiindex):
    """
    Test output for a MultiIndex when `invert` is True.
    """
    bools = np.array([True, True, True, False, False, False, True, True])
    ix = IndexLabel([bools], level="first")
    assert_frame_equal(
        multiindex.select_columns(ix, invert=True), multiindex.loc[:, ~bools]
    )


def test_errors_MultiIndex(multiindex):
    """
    Raise if `level` is a mix of str and int
    """
    ix = IndexLabel(("bar", "one"), level=["first", 1])
    msg = "All entries in the `level` parameter "
    msg += "should be either strings or integers."
    with pytest.raises(TypeError, match=msg):
        multiindex.select_columns(ix)


def test_errors_MultiIndex1(multiindex):
    """
    Raise if `level` is a str and not found
    """
    ix = IndexLabel("one", level="1")
    with pytest.raises(ValueError, match="Level 1 not found"):
        multiindex.select_columns(ix)


def test_errors_MultiIndex2(multiindex):
    """
    Raise if `level` is an int and less than 0
    """
    ix = IndexLabel("one", level=-200)
    msg = "Too many levels: Index has only 2 levels, "
    msg += "-200 is not a valid level number"
    with pytest.raises(IndexError, match=msg):
        multiindex.select_columns(ix)


def test_errors_MultiIndex3(multiindex):
    """
    Raise if `level` is an int
    and not less than the actual number
    of levels of the MultiIndex
    """
    ix = IndexLabel("one", level=2)
    msg = "Too many levels: Index has only 2 levels, "
    msg += "not 3"
    with pytest.raises(IndexError, match=msg):
        multiindex.select_columns(ix)


def test_errors_MultiIndex4(multiindex):
    """
    Raise if `level` is an int/string
    and duplicated
    """
    ix = IndexLabel("one", level=[1, 1])
    msg = "Entries in `level` should be unique; "
    msg += "1 exists multiple times."
    with pytest.raises(ValueError, match=msg):
        multiindex.select_columns(ix)


def test_errors_MultiIndex5(multiindex):
    """
    Raise if `IndexLabel` is combined
    with other selection options
    """
    ix = IndexLabel("one")
    msg = "`IndexLabel` cannot be combined "
    msg += "with other selection options."
    with pytest.raises(NotImplementedError, match=msg):
        multiindex.select_columns(ix, "bar")
