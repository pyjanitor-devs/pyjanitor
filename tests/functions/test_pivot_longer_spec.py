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


def test_spec_is_a_dataframe(df_checks):
    """Raise Error if spec is not a DataFrame."""
    with pytest.raises(
        TypeError,
        match="spec should be one of.+",
    ):
        df_checks.pipe(pivot_longer_spec, spec={".name": "name"})


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
        match="Kindly ensure the spec DataFrame has a `.name` column.",
    ):
        df_checks.pipe(
            pivot_longer_spec,
            spec=spec.set_axis(labels=[".value", ".blabla", "age"], axis=1),
        )


def test_spec_columns_has_dot_value(df_checks):
    """Raise KeyError if '.value' not in spec's columns."""
    with pytest.raises(
        KeyError,
        match="Kindly ensure the spec DataFrame has a `.value` column.",
    ):
        df_checks.pipe(
            pivot_longer_spec,
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
            pivot_longer_spec,
            spec=spec.loc[:, [".value", ".name", "age"]],
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
    msg = r"Labels \(\'birth',\)\ in the spec DataFrame already exist.+"
    with pytest.raises(
        ValueError,
        match=msg,
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


def test_df_columns_is_unique(df_checks):
    """Raise error if df_columns_is_unique is not boolean."""
    with pytest.raises(
        TypeError, match="df_columns_is_unique should be one of.+"
    ):
        df_checks.pipe(pivot_longer_spec, spec=spec, df_columns_is_unique=1)


def test_pivot_longer_spec(df_checks):
    """
    Test output if a specification is passed.
    """
    actual = df_checks.pipe(pivot_longer_spec, spec=spec)
    expected = pd.wide_to_long(
        df_checks, stubnames="ht", i=["famid", "birth"], j="age"
    ).reset_index()
    assert_frame_equal(
        actual.loc[:, expected.columns.tolist()].sort_values(
            actual.columns.tolist(), ignore_index=True
        ),
        expected.sort_values(actual.columns.tolist(), ignore_index=True),
    )


def test_pivot_longer_spec_dot_value_only(df_checks):
    """
    Test output if a specification is passed,
    and it is just .name and .value columns
    in the specification DataFrame.
    """
    specs = {
        ".name": ["ht1", "ht2"],
        ".value": ["ht", "ht"],
    }
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


def test_duplicated_columns():
    """Test output for duplicated columns."""
    rows = [["credit", 1, 1, 2, 3]]
    columns = ["Type", "amount", "active", "amount", "active"]

    df = pd.DataFrame(rows, columns=columns)
    df = df.set_index("Type")

    actual = pd.DataFrame(
        {"amount": [1, 2], "active": [1, 3]},
        index=pd.Index(["credit", "credit"], name="Type"),
    ).loc[:, ["amount", "active"]]
    specs = {
        ".name": ["amount", "active"],
        ".value": ["amount", "active"],
    }
    specs = pd.DataFrame(specs)
    expected = df.pipe(
        pivot_longer_spec,
        spec=specs,
        ignore_index=False,
        df_columns_is_unique=False,
    ).loc[:, ["amount", "active"]]

    assert_frame_equal(actual, expected)
