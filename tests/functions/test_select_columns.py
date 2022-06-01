import pandas as pd
import pytest
import re
from pandas.testing import assert_frame_equal


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
def df_tuple():
    "pytest fixture."
    frame = pd.DataFrame(
        {
            "A": {0: "a", 1: "b", 2: "c"},
            "B": {0: 1, 1: 3, 2: 5},
            "C": {0: 2, 1: 4, 2: 6},
        }
    )
    frame.columns = [list("ABC"), list("DEF")]
    return frame


def test_multiindex(df_tuple):
    """
    Test output for a MultiIndex and tuple passed.
    """
    assert_frame_equal(
        df_tuple.select_columns(("A", "D")), df_tuple.loc[:, [("A", "D")]]
    )


def test_level_callable(df_tuple):
    """
    Test output if level is supplied for a callable.
    """
    expected = df_tuple.select_columns(
        lambda df: df.name.startswith("A"), level=0
    )
    actual = df_tuple.xs("A", axis=1, drop_level=False, level=0)
    assert_frame_equal(actual, expected)


def test_level_regex(df_tuple):
    """
    Test output if level is supplied for a regex
    """
    expected = df_tuple.select_columns(re.compile("D"), level=1)
    actual = df_tuple.xs("D", axis=1, drop_level=False, level=1)
    assert_frame_equal(actual, expected)


def test_level_slice(df_tuple):
    """
    Test output if level is supplied for a slice
    """
    expected = df_tuple.select_columns(slice("F", "D"), level=1)
    assert_frame_equal(df_tuple, expected)


def test_level_str(df_tuple):
    """
    Test output if level is supplied for a string.
    """
    expected = df_tuple.select_columns("A", level=0, invert=True)
    assert_frame_equal(df_tuple.drop(columns="A", axis=1, level=0), expected)
