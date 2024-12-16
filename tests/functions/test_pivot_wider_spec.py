import re

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from janitor import pivot_wider_spec


@pytest.fixture
def df_checks():
    """pytest fixture"""
    return pd.DataFrame(
        [
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
    )


spec = {".name": ["ht1", "ht2"], ".value": ["ht", "ht"], "age": [1, 2]}
spec = pd.DataFrame(spec)


def test_spec_is_a_dataframe(df_checks):
    """Raise Error if spec is not a DataFrame."""
    with pytest.raises(
        TypeError,
        match="spec should be one of.+",
    ):
        df_checks.pipe(pivot_wider_spec, spec={".name": "name"})


def test_spec_columns_has_dot_name(df_checks):
    """Raise KeyError if '.name' not in spec's columns."""
    with pytest.raises(
        KeyError,
        match="Kindly ensure the spec DataFrame has a `.name` column.",
    ):
        df_checks.pipe(
            pivot_wider_spec,
            spec=spec.set_axis(labels=[".value", ".blabla", "age"], axis=1),
        )


def test_spec_columns_has_dot_value(df_checks):
    """Raise KeyError if '.value' not in spec's columns."""
    with pytest.raises(
        KeyError,
        match="Kindly ensure the spec DataFrame has a `.value` column.",
    ):
        df_checks.pipe(
            pivot_wider_spec,
            spec=spec.set_axis(labels=[".name", ".blabla", "age"], axis=1),
        )


def test_spec_columns_name_value_order(df_checks):
    """
    Raise ValueError if '.name' and '.value'
    are not the first two labels
    in spec's columns.
    """
    msg = "The first two columns of the spec DataFrame "
    msg += "should be '.name' and '.value',.+"
    with pytest.raises(
        ValueError,
        match=msg,
    ):
        df_checks.pipe(
            pivot_wider_spec,
            spec=spec.loc[:, [".value", ".name", "age"]],
        )


def test_spec_columns_len_2(df_checks):
    """
    Raise ValueError if '.name' and '.value'
    are the only columns in spec.
    """
    msg = "Kindly provide the column(s) "
    msg += "to use to make new frame’s columns"
    with pytest.raises(
        ValueError,
        match=re.escape(msg),
    ):
        df_checks.pipe(
            pivot_wider_spec,
            spec=spec.loc[:, [".name", ".value"]],
        )


def test_spec_columns_not_unique(df_checks):
    """Raise ValueError if the spec's columns is not unique."""
    with pytest.raises(
        ValueError, match="Kindly ensure the spec's columns is unique."
    ):
        df_checks.pipe(
            pivot_wider_spec,
            spec=spec.set_axis(labels=[".name", ".name", "age"], axis=1),
        )


def test_pivot_wider_spec(df_checks):
    """
    Test output
    """
    expected = (
        df_checks.pivot(index=["famid", "birth"], columns="age", values="ht")
        .add_prefix("ht")
        .rename_axis(columns=None)
        .reset_index()
    )
    actual = df_checks.pipe(
        pivot_wider_spec, spec=spec, index=["famid", "birth"]
    )
    assert_frame_equal(
        actual.sort_values(expected.columns.tolist(), ignore_index=True),
        expected.sort_values(expected.columns.tolist(), ignore_index=True),
    )
