import polars as pl
import pytest
from polars.testing import assert_frame_equal

from janitor.polars import pivot_longer_spec


@pytest.fixture
def df_checks():
    """fixture dataframe"""
    return pl.DataFrame(
        {
            "famid": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "birth": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "ht1": [2.8, 2.9, 2.2, 2, 1.8, 1.9, 2.2, 2.3, 2.1],
            "ht2": [3.4, 3.8, 2.9, 3.2, 2.8, 2.4, 3.3, 3.4, 2.9],
        }
    )


spec = {".name": ["ht1", "ht2"], ".value": ["ht", "ht"], "age": [1, 2]}
spec = pl.DataFrame(spec)


def test_spec_is_a_dataframe(df_checks):
    """Raise Error if spec is not a DataFrame."""
    with pytest.raises(
        TypeError,
        match="spec should be one of.+",
    ):
        df_checks.pipe(pivot_longer_spec, spec={".name": "name"})


def test_spec_columns_has_dot_name(df_checks):
    """Raise KeyError if '.name' not in spec's columns."""
    with pytest.raises(
        KeyError,
        match="Kindly ensure the spec DataFrame has a `.name` column.",
    ):
        df_checks.pipe(
            pivot_longer_spec,
            spec=spec.rename({".name": "name"}),
        )


def test_spec_columns_has_dot_value(df_checks):
    """Raise KeyError if '.value' not in spec's columns."""
    with pytest.raises(
        KeyError,
        match="Kindly ensure the spec DataFrame has a `.value` column.",
    ):
        df_checks.pipe(
            pivot_longer_spec,
            spec=spec.rename({".value": "name"}),
        )


def test_spec_columns_dot_name_unique(df_checks):
    """Raise ValueError if '.name' column is not unique."""
    with pytest.raises(
        ValueError, match="The labels in the `.name` column should be unique.+"
    ):
        df_checks.pipe(
            pivot_longer_spec,
            spec=spec.with_columns(pl.Series(".name", ["ht2", "ht2"])),
        )


def test_spec_columns_index(df_checks):
    """Raise ValueError if the columns in spec already exist in the dataframe."""
    with pytest.raises(
        ValueError,
        match=r"Labels \('birth',\) in the spec dataframe already exist.+",
    ):
        df_checks.pipe(
            pivot_longer_spec,
            spec=spec.with_columns(pl.Series("birth", ["ht1", "ht2"])),
        )


actual = [
    {"famid": 1, "birth": 1, "age": 1, "ht": 2.8},
    {"famid": 1, "birth": 1, "age": 2, "ht": 3.4},
    {"famid": 1, "birth": 2, "age": 1, "ht": 2.9},
    {"famid": 1, "birth": 2, "age": 2, "ht": 3.8},
    {"famid": 1, "birth": 3, "age": 1, "ht": 2.2},
    {"famid": 1, "birth": 3, "age": 2, "ht": 2.9},
    {"famid": 2, "birth": 1, "age": 1, "ht": 2.0},
    {"famid": 2, "birth": 1, "age": 2, "ht": 3.2},
    {"famid": 2, "birth": 2, "age": 1, "ht": 1.8},
    {"famid": 2, "birth": 2, "age": 2, "ht": 2.8},
    {"famid": 2, "birth": 3, "age": 1, "ht": 1.9},
    {"famid": 2, "birth": 3, "age": 2, "ht": 2.4},
    {"famid": 3, "birth": 1, "age": 1, "ht": 2.2},
    {"famid": 3, "birth": 1, "age": 2, "ht": 3.3},
    {"famid": 3, "birth": 2, "age": 1, "ht": 2.3},
    {"famid": 3, "birth": 2, "age": 2, "ht": 3.4},
    {"famid": 3, "birth": 3, "age": 1, "ht": 2.1},
    {"famid": 3, "birth": 3, "age": 2, "ht": 2.9},
]
actual = pl.DataFrame(actual).sort(by=pl.all())


def test_pivot_longer_spec(df_checks):
    """
    Test output if a specification is passed.
    """
    expected = (
        df_checks.pipe(pivot_longer_spec, spec=spec)
        .select("famid", "birth", "age", "ht")
        .sort(by=pl.all())
    )

    assert_frame_equal(actual, expected)
