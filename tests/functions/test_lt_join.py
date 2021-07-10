import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal


@pytest.fixture
def left_df():
    return pd.DataFrame({"col_a": [1, 2, 3], "col_b": ["A", "B", "C"]})


@pytest.fixture
def right_df():
    return pd.DataFrame({"col_a": [0, 2, 3], "col_c": ["Z", "X", "Y"]})


@pytest.fixture
def sequence():
    return [1, 2, 3]


@pytest.fixture
def multiIndex_df():
    frame = pd.DataFrame({"col_a": [0, 2, 3], "col_b": ["Z", "X", "Y"]})
    frame.columns = [["A", "col_a"], ["B", "col_b"]]
    return frame


def test_df_MultiIndex(multiIndex_df, right_df):
    """Raise ValueError if `df` has MultiIndex columns"""
    with pytest.raises(
        ValueError,
        match="MultiIndex columns are not supported for non-equi joins.",
    ):
        multiIndex_df.lt_join(right_df, "col_a", "col_a")


def test_right_MultiIndex(left_df, multiIndex_df):
    """Raise ValueError if `right` has MultiIndex columns"""
    with pytest.raises(
        ValueError,
        match="MultiIndex columns are not supported for non-equi joins.",
    ):
        left_df.lt_join(multiIndex_df, "col_a", "col_a")


def test_right_not_Series(left_df, sequence):
    """Raise TypeError if `right` is not DataFrame/Series"""
    with pytest.raises(TypeError):
        left_df.lt_join(sequence, "col_a", "col_a")


def test_right_unnamed_Series(left_df, sequence):
    """Raise ValueError if `right` is not a named Series"""
    sequence = pd.Series(sequence)
    with pytest.raises(
        ValueError,
        match="Unnamed Series are not supported for non-equi joins.",
    ):
        left_df.lt_join(sequence, "col_a", "col_a")


def test_wrong_type_sort_by_appearance(left_df, right_df):
    """Raise TypeError if wrong type is provided for `sort_by_appearance`."""
    with pytest.raises(TypeError):
        left_df.lt_join(right_df, "col_a", "col_a", sort_by_appearance="True")


def test_wrong_column_presence_right(left_df, right_df):
    """Raise ValueError if column is not found in `right`."""
    with pytest.raises(ValueError):
        left_df.lt_join(right_df, "col_a", "col_b")


def test_wrong_column_presence_df(left_df, right_df):
    """Raise ValueError if column is not found in `df`."""
    with pytest.raises(ValueError):
        left_df.lt_join(right_df, "col_c", "col_a")


def test_wrong_column_type_df(left_df, right_df):
    """Raise ValueError if wrong type is provided for column."""
    with pytest.raises(TypeError):
        left_df.lt_join(right_df, 1, "col_a")
        left_df.lt_join(right_df, "col_a", 2)


def test_wrong_type_suffixes(left_df, right_df):
    """Raise TypeError if `suffixes` is not a tuple."""
    with pytest.raises(TypeError):
        left_df.lt_join(right_df, "col_a", "col_a", suffixes=None)


def test_wrong_length_suffixes(left_df, right_df):
    """Raise ValueError if `suffixes` length != 2."""
    with pytest.raises(
        ValueError, match="`suffixes` argument must be a 2-length tuple"
    ):
        left_df.lt_join(right_df, "col_a", "col_a", suffixes=("_x",))


def test_suffixes_None(left_df, right_df):
    """Raise ValueError if `suffixes` is (None, None)."""
    with pytest.raises(
        ValueError, match="At least one of the suffixes should be non-null."
    ):
        left_df.lt_join(right_df, "col_a", "col_a", suffixes=(None, None))


def test_wrong_type_suffix(left_df, right_df):
    """
    Raise TypeError if one of the `suffixes`
    is not None or a string type.
    """
    with pytest.raises(TypeError):
        left_df.lt_join(right_df, "col_a", "col_a", suffixes=("_x", 1))


def test_suffix_already_exists_df(left_df, right_df):
    """Raise ValueError if label with suffix already exists."""
    left_df["col_a_x"] = 2
    with pytest.raises(ValueError):
        left_df.lt_join(right_df, "col_a", "col_a")


def test_suffix_already_exists_right(left_df, right_df):
    """Raise ValueError if label with suffix already exists."""
    right_df["col_a_y"] = 2
    with pytest.raises(ValueError):
        left_df.lt_join(right_df, "col_a", "col_a")


def test_column_same_type(left_df, right_df):
    """Raise ValueError if both columns are not of the same type."""
    with pytest.raises(ValueError):
        left_df.lt_join(right_df, "col_a", "col_c")


various = [
    (
        pd.DataFrame({"col_a": [4, 6, 7.5], "col_c": ["Z", "X", "Y"]}),
        pd.DataFrame({"col_a": [1, 2, 3], "col_b": ["A", "B", "C"]}),
        "col_a",
        "col_a",
        False,
        pd.DataFrame([], columns=("col_a_x", "col_c", "col_a_y", "col_b")),
    ),
    (
        pd.DataFrame({"col_a": [4, 6, 7.5], "col_c": ["Z", "X", "Y"]}),
        pd.DataFrame({"col_a": [1, 2, 3], "col_b": ["A", "B", "C"]}),
        "col_c",
        "col_b",
        True,
        pd.DataFrame([], columns=("col_a_x", "col_c", "col_a_y", "col_b")),
    ),
    (
        pd.DataFrame({"col_a": [4, 6, 7.5], "col_c": ["Z", "X", "Y"]}),
        pd.Series([1, 2, 3], name="col_a"),
        "col_a",
        "col_a",
        True,
        pd.DataFrame([], columns=("col_a_x", "col_c", "col_a_y")),
    ),
    (
        pd.DataFrame(
            [
                {"x": "b", "y": 1, "v": 1},
                {"x": "b", "y": 3, "v": 2},
                {"x": "b", "y": 6, "v": 3},
                {"x": "a", "y": 1, "v": 4},
                {"x": "a", "y": 3, "v": 5},
                {"x": "a", "y": 6, "v": 6},
                {"x": "c", "y": 1, "v": 7},
                {"x": "c", "y": 3, "v": 8},
                {"x": "c", "y": 6, "v": 9},
                {"x": "c", "y": np.nan, "v": 9},
            ]
        ),
        pd.DataFrame(
            [
                {"x": "c", "v": 8, "foo": 4},
                {"x": "b", "v": 7, "foo": 2},
                {"x": "b", "v": 7, "foo": None},
            ]
        ),
        "y",
        "foo",
        True,
        pd.DataFrame(
            [
                {
                    "x_x": "b",
                    "y": 1.0,
                    "v_x": 1,
                    "x_y": "c",
                    "v_y": 8,
                    "foo": 4.0,
                },
                {
                    "x_x": "b",
                    "y": 1.0,
                    "v_x": 1,
                    "x_y": "b",
                    "v_y": 7,
                    "foo": 2.0,
                },
                {
                    "x_x": "b",
                    "y": 3.0,
                    "v_x": 2,
                    "x_y": "c",
                    "v_y": 8,
                    "foo": 4.0,
                },
                {
                    "x_x": "a",
                    "y": 1.0,
                    "v_x": 4,
                    "x_y": "c",
                    "v_y": 8,
                    "foo": 4.0,
                },
                {
                    "x_x": "a",
                    "y": 1.0,
                    "v_x": 4,
                    "x_y": "b",
                    "v_y": 7,
                    "foo": 2.0,
                },
                {
                    "x_x": "a",
                    "y": 3.0,
                    "v_x": 5,
                    "x_y": "c",
                    "v_y": 8,
                    "foo": 4.0,
                },
                {
                    "x_x": "c",
                    "y": 1.0,
                    "v_x": 7,
                    "x_y": "c",
                    "v_y": 8,
                    "foo": 4.0,
                },
                {
                    "x_x": "c",
                    "y": 1.0,
                    "v_x": 7,
                    "x_y": "b",
                    "v_y": 7,
                    "foo": 2.0,
                },
                {
                    "x_x": "c",
                    "y": 3.0,
                    "v_x": 8,
                    "x_y": "c",
                    "v_y": 8,
                    "foo": 4.0,
                },
            ]
        ),
    ),
    (
        pd.DataFrame(
            [
                {"x": "b", "y": 1, "v": 1},
                {"x": "b", "y": 3, "v": 2},
                {"x": "b", "y": 6, "v": 3},
                {"x": "a", "y": 1, "v": 4},
                {"x": "a", "y": 3, "v": 5},
                {"x": "a", "y": 6, "v": 6},
                {"x": "c", "y": 1, "v": 7},
                {"x": "c", "y": 3, "v": 8},
                {"x": "c", "y": 6, "v": 9},
                {"x": "c", "y": np.nan, "v": 9},
            ]
        ),
        pd.DataFrame(
            [
                {"x": "c", "v": 8, "foo": 4},
                {"x": "b", "v": 7, "foo": 2},
                {"x": "b", "v": 7, "foo": None},
            ]
        ),
        "x",
        "x",
        True,
        pd.DataFrame(
            [
                {
                    "x_x": "b",
                    "y": 1.0,
                    "v_x": 1,
                    "x_y": "c",
                    "v_y": 8,
                    "foo": 4.0,
                },
                {
                    "x_x": "b",
                    "y": 3.0,
                    "v_x": 2,
                    "x_y": "c",
                    "v_y": 8,
                    "foo": 4.0,
                },
                {
                    "x_x": "b",
                    "y": 6.0,
                    "v_x": 3,
                    "x_y": "c",
                    "v_y": 8,
                    "foo": 4.0,
                },
                {
                    "x_x": "a",
                    "y": 1.0,
                    "v_x": 4,
                    "x_y": "c",
                    "v_y": 8,
                    "foo": 4.0,
                },
                {
                    "x_x": "a",
                    "y": 1.0,
                    "v_x": 4,
                    "x_y": "b",
                    "v_y": 7,
                    "foo": 2.0,
                },
                {
                    "x_x": "a",
                    "y": 1.0,
                    "v_x": 4,
                    "x_y": "b",
                    "v_y": 7,
                    "foo": np.nan,
                },
                {
                    "x_x": "a",
                    "y": 3.0,
                    "v_x": 5,
                    "x_y": "c",
                    "v_y": 8,
                    "foo": 4.0,
                },
                {
                    "x_x": "a",
                    "y": 3.0,
                    "v_x": 5,
                    "x_y": "b",
                    "v_y": 7,
                    "foo": 2.0,
                },
                {
                    "x_x": "a",
                    "y": 3.0,
                    "v_x": 5,
                    "x_y": "b",
                    "v_y": 7,
                    "foo": np.nan,
                },
                {
                    "x_x": "a",
                    "y": 6.0,
                    "v_x": 6,
                    "x_y": "c",
                    "v_y": 8,
                    "foo": 4.0,
                },
                {
                    "x_x": "a",
                    "y": 6.0,
                    "v_x": 6,
                    "x_y": "b",
                    "v_y": 7,
                    "foo": 2.0,
                },
                {
                    "x_x": "a",
                    "y": 6.0,
                    "v_x": 6,
                    "x_y": "b",
                    "v_y": 7,
                    "foo": np.nan,
                },
            ]
        ),
    ),
]


@pytest.mark.parametrize(
    "left_df, right_df, left_on, right_on, appearance, actual", various
)
def test_various_scenarios(
    left_df, right_df, left_on, right_on, appearance, actual
):
    """Test various scenarios for lt_join"""
    expected = left_df.lt_join(
        right_df, left_on, right_on, sort_by_appearance=appearance
    )
    assert_frame_equal(expected, actual)
