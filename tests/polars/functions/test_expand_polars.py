import polars as pl
import pytest
from polars.testing import assert_frame_equal

import janitor.polars  # noqa: F401


@pytest.fixture
def df():
    """pytest fixture"""
    return pl.DataFrame(
        dict(
            group=(1, 2, 1, 2),
            item_id=(1, 2, 2, 3),
            item_name=("a", "a", "b", "b"),
            value1=(1, None, 3, 4),
            value2=range(4, 8),
        )
    )


def test_column_None(df):
    """Test output if *columns is empty."""
    assert_frame_equal(df.expand(), df)


def test_empty_groups(df):
    """Raise TypeError if wrong column type is passed."""
    msg = "The argument passed to the columns parameter "
    msg += "should either be a string, a column selector, "
    msg += "a polars expression, or a polars Series; instead got.+"
    with pytest.raises(TypeError, match=msg):
        df.complete("group", {})


def test_type_sort(df):
    """Raise TypeError if `sort` is not boolean."""
    with pytest.raises(TypeError):
        df.complete("group", "item_id", sort=11)


def test_expand_1(df):
    """
    Test output for janitor.expand.
    """
    expected = df.expand("group", "item_id", "item_name", sort=True)
    actual = (
        df.select(pl.col("group").unique())
        .join(df.select(pl.col("item_id").unique()), how="cross")
        .join(df.select(pl.col("item_name").unique()), how="cross")
        .sort(by=pl.all())
    )
    assert_frame_equal(actual, expected)
