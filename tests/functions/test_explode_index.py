import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

df = [1, 2, 3]


@pytest.fixture
def df_checks():
    """fixture dataframe"""
    return pd.DataFrame(
        {
            "fam_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        }
    )


@pytest.fixture
def df_multi():
    """MultiIndex dataframe fixture."""
    return pd.DataFrame(
        {
            ("name", "a"): {0: "Wilbur", 1: "Petunia", 2: "Gregory"},
            ("names", "aa"): {0: 67, 1: 80, 2: 64},
            ("more_names", "aaa"): {0: 56, 1: 90, 2: 50},
        }
    )


def test_type_axis(df_checks):
    """Raise TypeError if wrong type is provided for axis."""
    with pytest.raises(TypeError, match="axis should be one of.+"):
        df_checks.explode_index(axis=1)


def test_axis_values(df_checks):
    """Raise ValueError if wrong value is provided for axis."""
    msg = "axis should be either index or columns."
    with pytest.raises(ValueError, match=msg):
        df_checks.explode_index(axis="INDEX")


def test_names_sep_pattern(df_checks):
    """Raise ValueError if both names_sep and names_pattern is provided."""
    msg = "Provide argument for either names_sep or names_pattern, not both."
    with pytest.raises(ValueError, match=msg):
        df_checks.explode_index(
            axis="columns", names_sep="_", names_pattern=r"(.+)_(.+)"
        )


def test_names_sep_pattern_both_none(df_checks):
    """Raise ValueError if neither names_sep nor names_pattern is provided."""
    msg = "Provide argument for either names_sep or names_pattern."
    with pytest.raises(ValueError, match=msg):
        df_checks.explode_index(
            axis="columns", names_sep=None, names_pattern=None
        )


def test_names_sep_typeerror(df_checks):
    """Raise TypeError if names_sep is a wrong type."""
    with pytest.raises(TypeError, match="names_sep should be one of.+"):
        df_checks.explode_index(axis="columns", names_sep=1)


def test_names_pattern_typeerror(df_checks):
    """Raise TypeError if names_pattern is a wrong type."""
    with pytest.raises(TypeError, match="names_pattern should be one of.+"):
        df_checks.explode_index(names_pattern=1)


def test_level_names_typeerror(df_checks):
    """Raise TypeError if level_names is a wrong type."""
    with pytest.raises(TypeError, match="level_names should be one of.+"):
        df_checks.explode_index(names_sep="_", level_names="new_level")


def test_multiindex(df_multi):
    """Test output if df.columns is a multiindex"""
    actual = df_multi.explode_index(names_sep="_")
    assert_frame_equal(df_multi, actual)


def test_names_sep(df_checks):
    """test output if names_sep"""
    actual = df_checks.explode_index(names_sep="_", level_names=["a", "b"])
    expected = pd.DataFrame(
        {
            ("fam", "id"): [1, 1, 1, 2, 2, 2, 3, 3, 3],
        }
    )
    expected.columns.names = ["a", "b"]
    assert_frame_equal(actual, expected)


def test_names_pattern(df_checks):
    """test output if names_pattern"""
    actual = df_checks.explode_index(names_pattern=r"(?P<a>.+)_(?P<b>.+)")
    expected = pd.DataFrame(
        {
            ("fam", "id"): [1, 1, 1, 2, 2, 2, 3, 3, 3],
        }
    )
    expected.columns.names = ["a", "b"]
    assert_frame_equal(actual, expected)
