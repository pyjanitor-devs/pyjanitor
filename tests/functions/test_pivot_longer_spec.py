import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from janitor import pivot_longer_spec


@pytest.fixture
def df_checks():
    """fixture dataframe"""
    return pd.DataFrame(
        {
            "famid": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "birth": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "ht1": [2.8, 2.9, 2.2, 2, 1.8, 1.9, 2.2, 2.3, 2.1],
            "ht2": [3.4, 3.8, 2.9, 3.2, 2.8, 2.4, 3.3, 3.4, 2.9],
        }
    )


spec = {".name": ["ht1", "ht2"], ".value": ["ht", "ht"], "age": [1, 2]}
spec = pd.DataFrame(spec)


def test_spec_columns_not_unique(df_checks):
    """Raise ValueError if the spec's columns is not unique."""
    with pytest.raises(
        ValueError, match="Kindly ensure the spec's columns is unique."
    ):
        df_checks.pipe(
            pivot_longer_spec,
            spec=spec.set_axis(labels=[".name", ".name", "age"], axis=1),
        )


def test_spec_columns_has_dot_name(df_checks):
    """Raise KeyError if '.name' not in spec's columns."""
    with pytest.raises(
        KeyError,
        match="Kindly ensure the spec dataframe has a `.name` column.",
    ):
        df_checks.pipe(
            pivot_longer_spec,
            spec=spec.set_axis(labels=[".value", ".blabla", "age"], axis=1),
        )


def test_spec_columns_has_dot_value(df_checks):
    """Raise KeyError if '.valje' not in spec's columns."""
    with pytest.raises(
        KeyError,
        match="Kindly ensure the spec dataframe has a `.value` column.",
    ):
        df_checks.pipe(
            pivot_longer_spec,
            spec=spec.set_axis(labels=[".name", ".blabla", "age"], axis=1),
        )


def test_spec_columns_dot_name_unique(df_checks):
    """Raise ValueError if '.name' column is not unique."""
    with pytest.raises(
        ValueError, match="The labels in the `.name` column should be unique.+"
    ):
        df_checks.pipe(
            pivot_longer_spec, spec=spec.assign(**{".name": ["ht2", "ht2"]})
        )


def test_spec_columns_index(df_checks):
    """Raise ValueError if the columns in spec already exist in the dataframe."""
    with pytest.raises(
        ValueError,
        match=r"Labels \('birth',\) in the spec dataframe already exist.+",
    ):
        df_checks.pipe(
            pivot_longer_spec, spec=spec.assign(birth=["ht2", "ht2"])
        )


def test_sort_by_appearance(df_checks):
    """Raise error if sort_by_appearance is not boolean."""
    with pytest.raises(
        TypeError, match="sort_by_appearance should be one of.+"
    ):
        df_checks.pipe(pivot_longer_spec, spec=spec, sort_by_appearance=1)


def test_ignore_index(df_checks):
    """Raise error if ignore_index is not boolean."""
    with pytest.raises(TypeError, match="ignore_index should be one of.+"):
        df_checks.pipe(pivot_longer_spec, spec=spec, ignore_index=1)


def test_pivot_longer_spec(df_checks):
    """
    Test output if a specification is passed.
    """
    actual = df_checks.pipe(pivot_longer_spec, spec=spec)
    expected = pd.wide_to_long(
        df_checks, stubnames="ht", i=["famid", "birth"], j="age"
    ).reset_index()
    assert_frame_equal(
        actual.sort_values(actual.columns.tolist(), ignore_index=True),
        expected.sort_values(actual.columns.tolist(), ignore_index=True),
    )


def test_pivot_longer_spec_dot_value_only(df_checks):
    """
    Test output if a specification is passed,
    and it is just .name and .value columns
    in the specification DataFrame.
    """
    specs = {".name": ["ht1", "ht2"], ".value": ["ht", "ht"]}
    specs = pd.DataFrame(specs)
    actual = df_checks.pipe(pivot_longer_spec, spec=specs)
    expected = (
        pd.wide_to_long(
            df_checks, stubnames="ht", i=["famid", "birth"], j="age"
        )
        .reset_index()
        .drop(columns="age")
    )
    assert_frame_equal(
        actual.sort_values(actual.columns.tolist(), ignore_index=True),
        expected.sort_values(actual.columns.tolist(), ignore_index=True),
    )
