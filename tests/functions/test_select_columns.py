import pytest
import pandas as pd
from pandas.testing import assert_frame_equal


@pytest.mark.functions
@pytest.mark.parametrize(
    "invert,expected",
    [
        (False, ["a", "Bell__Chart", "cities"]),
        (True, ["decorated-elephant", "animals@#$%^"]),
    ],
)
def test_select_columns(dataframe, invert, expected):
    columns = ["a", "Bell__Chart", "cities"]
    df = dataframe.select_columns(search_column_names=columns, invert=invert)

    assert_frame_equal(df, dataframe[expected])


@pytest.mark.functions
@pytest.mark.parametrize(
    "invert,expected",
    [
        (False, ["Bell__Chart", "a", "animals@#$%^"]),
        (True, ["decorated-elephant", "cities"]),
    ],
)
def test_select_columns_glob_inputs(dataframe, invert, expected):
    columns = ["Bell__Chart", "a*"]
    df = dataframe.select_columns(search_column_names=columns, invert=invert)

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
def test_select_columns_missing_columns(dataframe, columns):
    """Check that passing non-existent column names or search strings raises KeyError"""  # noqa: E501
    with pytest.raises(KeyError):
        dataframe.select_columns(search_column_names=columns)


@pytest.mark.functions
@pytest.mark.parametrize(
    "columns",
    [
        pytest.param(
            "a",
            marks=pytest.mark.xfail(
                reason="`select_columns` now accepts strings"
            ),
        ),
        pytest.param(
            ("a", "Bell__Chart"),
            marks=pytest.mark.xfail(
                reason="`select_columns` converts list-like into lists"
            ),
        ),
        pytest.param(
            {"a", "Bell__Chart"},
            marks=pytest.mark.xfail(
                reason="`select_columns` converts list-like into lists"
            ),
        ),
    ],
)
def test_select_columns_input(dataframe, columns):
    """Check that passing an iterable that is not a list raises TypeError."""
    with pytest.raises(TypeError):
        dataframe.select_columns(search_column_names=columns)


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
    df = dataframe.select_columns(search_column_names=columns, invert=invert)

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

    df = dataframe.select_columns(search_column_names=columns, invert=invert)

    assert_frame_equal(df, dataframe[expected])


@pytest.mark.xfail(reason="Allow tuples which are acceptable in MultiIndex.")
def test_MultiIndex():
    """
    Raise ValueError if columns is a MultiIndex.
    """
    df = pd.DataFrame(
        {
            "A": {0: "a", 1: "b", 2: "c"},
            "B": {0: 1, 1: 3, 2: 5},
            "C": {0: 2, 1: 4, 2: 6},
        }
    )

    df.columns = [list("ABC"), list("DEF")]

    with pytest.raises(ValueError):
        df.select_columns("A")
