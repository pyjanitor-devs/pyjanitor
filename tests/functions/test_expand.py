import pandas as pd
import pytest
from hypothesis import given, settings
from pandas.testing import assert_frame_equal

import janitor  # noqa: F401
from janitor.testing_utils.strategies import (
    df_strategy,
)
from janitor.functions.expand_grid import _build_pandas_objects_for_expand

@pytest.fixture
def df():
    """fixture dataframe"""
    return pd.DataFrame(
        {
            "famid": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "birth": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "ht1": [2.8, 2.9, 2.2, 2, 1.8, 1.9, 2.2, 2.3, 2.1],
            "ht2": [3.4, 3.8, 2.9, 3.2, 2.8, 2.4, 3.3, 3.4, 2.9],
        }
    )


def test_columns_wrong_type(df):
    """Raise Error if wrong type is provided."""
    msg = "The arguments to the variable columns parameter.+"
    with pytest.raises(TypeError, match=msg):
        df.expand({"famid", "birth"})


def test_by(df):
    """Raise if by is not the right type."""
    msg = "The argument to the by parameter.+"
    with pytest.raises(TypeError, match=msg):
        df.expand("famid", by={"a": 2})


@settings(deadline=None, max_examples=10)
@given(df=df_strategy())
def test_various(df):
    """Test `expand` output for various inputs."""
    mapping = {"year": range(1, 5)}
    expected = df.expand(
        "a",
        "cities",
        ["decorated-elephant", "animals@#$%^"],
        mapping,
        pd.RangeIndex(start=1, stop=5, name="rangeindex"),
        lambda df: df["a"].rename("lambda"),
    )
    A = df["a"].drop_duplicates()
    B = df["cities"].drop_duplicates()
    C = df.loc[:, ["decorated-elephant", "animals@#$%^"]].drop_duplicates()
    D = pd.Series(range(1, 5), name="year")
    actual = (
        pd.merge(A, B, how="cross")
        .merge(
            C,
            how="cross",
        )
        .merge(D, how="cross")
        .merge(D.rename("rangeindex"), how="cross")
        .merge(df["a"].rename("lambda"), how="cross")
    )

    assert_frame_equal(actual, expected)


@settings(deadline=None, max_examples=10)
@given(df=df_strategy())
def test_various_sorted(df):
    """Test `expand` output for various inputs."""
    mapping = {"year": range(1, 5)}
    expected = df.expand(
        "a",
        "cities",
        ["decorated-elephant", "animals@#$%^"],
        mapping,
        pd.RangeIndex(start=1, stop=5, name="rangeindex"),
        lambda df: df["a"].rename("lambda"),
        sort=True,
    )
    A = df["a"].drop_duplicates()
    B = df["cities"].drop_duplicates()
    C = df.loc[:, ["decorated-elephant", "animals@#$%^"]].drop_duplicates()
    D = pd.Series(range(1, 5), name="year")
    actual = (
        pd.merge(A, B, how="cross")
        .merge(
            C,
            how="cross",
        )
        .merge(D, how="cross")
        .merge(D.rename("rangeindex"), how="cross")
        .merge(df["a"].rename("lambda"), how="cross")
    )
    headers = actual.columns.tolist()
    actual = actual.sort_values(headers, ignore_index=True)

    assert_frame_equal(actual, expected)


def test_expand_by():
    """
    Test `expand` with `by`
    """
    # https://stackoverflow.com/a/44870793/7175713

    output = [
        {
            "dealid": 1,
            "acquirer": "FirmA",
            "target": "FirmB",
            "vendor": "FirmC",
        },
        {
            "dealid": 1,
            "acquirer": "FirmA",
            "target": "FirmB",
            "vendor": "FirmE",
        },
        {"dealid": 1, "acquirer": "FirmA", "target": None, "vendor": "FirmC"},
        {"dealid": 1, "acquirer": "FirmA", "target": None, "vendor": "FirmE"},
        {
            "dealid": 1,
            "acquirer": "FirmD",
            "target": "FirmB",
            "vendor": "FirmC",
        },
        {
            "dealid": 1,
            "acquirer": "FirmD",
            "target": "FirmB",
            "vendor": "FirmE",
        },
        {"dealid": 1, "acquirer": "FirmD", "target": None, "vendor": "FirmC"},
        {"dealid": 1, "acquirer": "FirmD", "target": None, "vendor": "FirmE"},
        {
            "dealid": 2,
            "acquirer": "FirmA",
            "target": "FirmF",
            "vendor": "FirmC",
        },
        {
            "dealid": 2,
            "acquirer": "FirmA",
            "target": "FirmF",
            "vendor": "FirmE",
        },
        {"dealid": 2, "acquirer": "FirmA", "target": None, "vendor": "FirmC"},
        {"dealid": 2, "acquirer": "FirmA", "target": None, "vendor": "FirmE"},
        {
            "dealid": 2,
            "acquirer": "FirmD",
            "target": "FirmF",
            "vendor": "FirmC",
        },
        {
            "dealid": 2,
            "acquirer": "FirmD",
            "target": "FirmF",
            "vendor": "FirmE",
        },
        {"dealid": 2, "acquirer": "FirmD", "target": None, "vendor": "FirmC"},
        {"dealid": 2, "acquirer": "FirmD", "target": None, "vendor": "FirmE"},
        {
            "dealid": 2,
            "acquirer": "FirmG",
            "target": "FirmF",
            "vendor": "FirmC",
        },
        {
            "dealid": 2,
            "acquirer": "FirmG",
            "target": "FirmF",
            "vendor": "FirmE",
        },
        {"dealid": 2, "acquirer": "FirmG", "target": None, "vendor": "FirmC"},
        {"dealid": 2, "acquirer": "FirmG", "target": None, "vendor": "FirmE"},
    ]
    sorter = [*output[0].keys()]
    expected = pd.DataFrame(output).sort_values(sorter)

    input = [
        {
            "dealid": 1,
            "acquirer": "FirmA",
            "target": "FirmB",
            "vendor": "FirmC",
        },
        {"dealid": 1, "acquirer": "FirmD", "target": None, "vendor": "FirmE"},
        {"dealid": 2, "acquirer": "FirmA", "target": None, "vendor": "FirmC"},
        {"dealid": 2, "acquirer": "FirmD", "target": None, "vendor": "FirmE"},
        {
            "dealid": 2,
            "acquirer": "FirmG",
            "target": "FirmF",
            "vendor": "FirmE",
        },
    ]
    df = pd.DataFrame(input)

    actual = (
        df.expand("acquirer", "target", "vendor", by="dealid")
        .sort_values(sorter)
        .reset_index()
    )
    assert_frame_equal(actual, expected)


def test_expand_grouped():
    """
    Test `expand` with `DataFrameGroupBy`
    """
    # https://stackoverflow.com/a/44870793/7175713

    output = [
        {
            "dealid": 1,
            "acquirer": "FirmA",
            "target": "FirmB",
            "vendor": "FirmC",
        },
        {
            "dealid": 1,
            "acquirer": "FirmA",
            "target": "FirmB",
            "vendor": "FirmE",
        },
        {"dealid": 1, "acquirer": "FirmA", "target": None, "vendor": "FirmC"},
        {"dealid": 1, "acquirer": "FirmA", "target": None, "vendor": "FirmE"},
        {
            "dealid": 1,
            "acquirer": "FirmD",
            "target": "FirmB",
            "vendor": "FirmC",
        },
        {
            "dealid": 1,
            "acquirer": "FirmD",
            "target": "FirmB",
            "vendor": "FirmE",
        },
        {"dealid": 1, "acquirer": "FirmD", "target": None, "vendor": "FirmC"},
        {"dealid": 1, "acquirer": "FirmD", "target": None, "vendor": "FirmE"},
        {
            "dealid": 2,
            "acquirer": "FirmA",
            "target": "FirmF",
            "vendor": "FirmC",
        },
        {
            "dealid": 2,
            "acquirer": "FirmA",
            "target": "FirmF",
            "vendor": "FirmE",
        },
        {"dealid": 2, "acquirer": "FirmA", "target": None, "vendor": "FirmC"},
        {"dealid": 2, "acquirer": "FirmA", "target": None, "vendor": "FirmE"},
        {
            "dealid": 2,
            "acquirer": "FirmD",
            "target": "FirmF",
            "vendor": "FirmC",
        },
        {
            "dealid": 2,
            "acquirer": "FirmD",
            "target": "FirmF",
            "vendor": "FirmE",
        },
        {"dealid": 2, "acquirer": "FirmD", "target": None, "vendor": "FirmC"},
        {"dealid": 2, "acquirer": "FirmD", "target": None, "vendor": "FirmE"},
        {
            "dealid": 2,
            "acquirer": "FirmG",
            "target": "FirmF",
            "vendor": "FirmC",
        },
        {
            "dealid": 2,
            "acquirer": "FirmG",
            "target": "FirmF",
            "vendor": "FirmE",
        },
        {"dealid": 2, "acquirer": "FirmG", "target": None, "vendor": "FirmC"},
        {"dealid": 2, "acquirer": "FirmG", "target": None, "vendor": "FirmE"},
    ]
    sorter = [*output[0].keys()]
    expected = pd.DataFrame(output).sort_values(sorter)

    input = [
        {
            "dealid": 1,
            "acquirer": "FirmA",
            "target": "FirmB",
            "vendor": "FirmC",
        },
        {"dealid": 1, "acquirer": "FirmD", "target": None, "vendor": "FirmE"},
        {"dealid": 2, "acquirer": "FirmA", "target": None, "vendor": "FirmC"},
        {"dealid": 2, "acquirer": "FirmD", "target": None, "vendor": "FirmE"},
        {
            "dealid": 2,
            "acquirer": "FirmG",
            "target": "FirmF",
            "vendor": "FirmE",
        },
    ]
    df = pd.DataFrame(input)

    actual = (
        df.groupby("dealid")
        .expand("acquirer", "target", "vendor")
        .sort_values(sorter)
        .reset_index()
    )
    assert_frame_equal(actual, expected)


def test_build_pandas_objects_for_expand_scalar_unique():
    """Verify scalar columns use unique() and return expected named Series."""
    df = pd.DataFrame({"a": [1, 1, 2, 3], "b": ["x", "x", "y", "z"]})
    result = _build_pandas_objects_for_expand(df, ("a", "b"))

    assert len(result) == 2
    pd.testing.assert_series_equal(
        result[0], pd.Series([1, 2, 3], name="a")
    )
    pd.testing.assert_series_equal(
        result[1], pd.Series(["x", "y", "z"], name="b")
    )

def test_build_pandas_objects_for_expand_object_unhashable_fallback():
    """Verify object columns with unhashable types fall back safely to drop_duplicates."""
    df = pd.DataFrame({"a": [[1, 2], [1, 2], [3, 4]]})

    # Should NOT raise TypeError: unhashable type: 'list'
    result = _build_pandas_objects_for_expand(df, ("a",))

    assert len(result) == 1
    pd.testing.assert_series_equal(
        result[0].reset_index(drop=True),
        pd.Series([[1, 2], [3, 4]], name="a"),
    )

def test_expand_object_with_lists():
    """Integration test to verify public df.expand() handles object dtypes with lists."""
    df = pd.DataFrame({"a": [[1, 2], [1, 2], [3, 4]], "b": [1, 1, 2]})
    result = df.expand("a", "b")

    # Verify expand output shape and execution without crashing
    assert len(result) == 4
    assert list(result.columns) == ["a", "b"]
