import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal


@pytest.fixture
def df():
    return pd.DataFrame(
        [
            {
                "rank": 1,
                "pet_type": np.nan,
                "breed": "Boston Terrier",
                "owner": "sam",
            },
            {
                "rank": 2,
                "pet_type": np.nan,
                "breed": "Retrievers (Labrador)",
                "owner": "ogor",
            },
            {
                "rank": 3,
                "pet_type": np.nan,
                "breed": "Retrievers (Golden)",
                "owner": "nathan",
            },
            {
                "rank": 4,
                "pet_type": np.nan,
                "breed": "French Bulldogs",
                "owner": np.nan,
            },
            {
                "rank": 5,
                "pet_type": np.nan,
                "breed": "Bulldogs",
                "owner": np.nan,
            },
            {
                "rank": 6,
                "pet_type": "Dog",
                "breed": "Beagles",
                "owner": np.nan,
            },
            {
                "rank": 1,
                "pet_type": np.nan,
                "breed": "Persian",
                "owner": np.nan,
            },
            {
                "rank": 2,
                "pet_type": np.nan,
                "breed": "Maine Coon",
                "owner": "ragnar",
            },
            {
                "rank": 3,
                "pet_type": np.nan,
                "breed": "Ragdoll",
                "owner": np.nan,
            },
            {
                "rank": 4,
                "pet_type": np.nan,
                "breed": "Exotic",
                "owner": np.nan,
            },
            {
                "rank": 5,
                "pet_type": np.nan,
                "breed": "Siamese",
                "owner": np.nan,
            },
            {
                "rank": 6,
                "pet_type": "Cat",
                "breed": "American Short",
                "owner": "adaora",
            },
        ]
    )


def test_fill_column(df):
    """Fill down on a single column."""
    expected = df.copy()
    expected.loc[:, "pet_type"] = expected.loc[:, "pet_type"].ffill()
    assert_frame_equal(df.fill_direction(**{"pet_type": "down"}), expected)


def test_fill_column_up(df):
    """Fill up on a single column."""
    expected = df.copy()
    expected.loc[:, "pet_type"] = expected.loc[:, "pet_type"].bfill()
    assert_frame_equal(df.fill_direction(**{"pet_type": "up"}), expected)


def test_fill_column_updown(df):
    """Fill upwards, then downwards on a single column."""
    expected = df.copy()
    expected.loc[:, "pet_type"] = expected.loc[:, "pet_type"].bfill().ffill()
    assert_frame_equal(df.fill_direction(**{"pet_type": "updown"}), expected)


def test_fill_column_down_up(df):
    """Fill downwards, then upwards on a single column."""
    expected = df.copy()
    expected.loc[:, "pet_type"] = expected.loc[:, "pet_type"].ffill().bfill()
    assert_frame_equal(df.fill_direction(**{"pet_type": "downup"}), expected)


def test_fill_multiple_columns(df):
    """Fill on multiple columns with a single direction."""
    expected = df.copy()
    expected.loc[:, ["pet_type", "owner"]] = expected.loc[
        :, ["pet_type", "owner"]
    ].ffill()
    assert_frame_equal(
        df.fill_direction(**{"pet_type": "down", "owner": "down"}), expected
    )


def test_fill_multiple_columns_multiple_directions(df):
    """Fill on multiple columns with different directions."""
    expected = df.copy()
    expected.loc[:, "pet_type"] = expected.loc[:, "pet_type"].ffill()
    expected.loc[:, "owner"] = expected.loc[:, "owner"].bfill()
    assert_frame_equal(
        df.fill_direction(**{"pet_type": "down", "owner": "up"}), expected
    )


def test_wrong_column_name(df):
    """Raise Value Error if wrong column name is provided."""
    with pytest.raises(ValueError):
        df.fill_direction(**{"PetType": "down"})


def test_wrong_column_type(df):
    """Raise Value Error if wrong type is provided for column_name."""
    with pytest.raises(TypeError):
        df.fill_direction(**{1: "down"})


def test_wrong_direction(df):
    """Raise Value Error if wrong direction is provided."""
    with pytest.raises(ValueError):
        df.fill_direction(**{"pet_type": "upanddawn"})


def test_wrong_direction_type(df):
    """Raise Type Error if wrong type is provided for direction."""
    with pytest.raises(TypeError):
        df.fill_direction(**{"pet_type": 1})


@pytest.mark.xfail(reason="limit is deprecated")
def test_wrong_type_limit(df):
    """Raise TypeError if limit is wrong type."""
    with pytest.raises(TypeError):
        df.fill_direction(**{"pet_type": "up"}, limit="one")


def test_empty_directions(df):
    """Return dataframe if `directions` is empty."""
    assert_frame_equal(df.fill_direction(), df)
