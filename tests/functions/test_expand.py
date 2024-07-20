import pandas as pd
import pytest
from hypothesis import given, settings
from pandas.testing import assert_frame_equal

import janitor  # noqa: F401
from janitor.testing_utils.strategies import (
    df_strategy,
)


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
