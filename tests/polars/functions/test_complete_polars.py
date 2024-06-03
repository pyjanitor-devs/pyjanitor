import polars as pl
import polars.selectors as cs
import pytest
from polars.testing import assert_frame_equal

import janitor.polars  # noqa: F401


@pytest.fixture
def fill_df():
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


@pytest.fixture
def taxonomy_df():
    """pytest fixture"""
    return pl.DataFrame(
        {
            "Year": [1999, 2000, 2004, 1999, 2004],
            "Taxon": [
                "Saccharina",
                "Saccharina",
                "Saccharina",
                "Agarum",
                "Agarum",
            ],
            "Abundance": [4, 5, 2, 1, 8],
        }
    )


def test_column_None(fill_df):
    """Test output if *columns is empty."""
    assert_frame_equal(fill_df.janitor.complete(), fill_df)


def test_empty_groups(fill_df):
    """Raise TypeError if wrong column type is passed."""
    msg = "The argument passed to the columns parameter "
    msg += "should either be a string, a column selector, "
    msg += "or a polars expression, instead got.+"
    with pytest.raises(TypeError, match=msg):
        fill_df.janitor.complete("group", {})


def test_type_sort(fill_df):
    """Raise TypeError if `sort` is not boolean."""
    with pytest.raises(TypeError):
        fill_df.janitor.complete("group", "item_id", sort=11)


def test_type_explicit(fill_df):
    """Raise TypeError if `explicit` is not boolean."""
    with pytest.raises(TypeError):
        fill_df.janitor.complete("group", "item_id", explicit=11)


def test_complete_1(fill_df):
    """
    Test output for janitor.complete.
    """
    trimmed = fill_df.lazy().select(~cs.starts_with("value"))
    result = trimmed.janitor.complete(
        cs.by_name("group"),
        pl.struct("item_id", "item_name").alias("rar").unique(),
        fill_value=0,
        explicit=False,
        sort=True,
    )
    expected = pl.DataFrame(
        [
            {"group": 1, "item_id": 1, "item_name": "a"},
            {"group": 1, "item_id": 2, "item_name": "a"},
            {"group": 1, "item_id": 2, "item_name": "b"},
            {"group": 1, "item_id": 3, "item_name": "b"},
            {"group": 2, "item_id": 1, "item_name": "a"},
            {"group": 2, "item_id": 2, "item_name": "a"},
            {"group": 2, "item_id": 2, "item_name": "b"},
            {"group": 2, "item_id": 3, "item_name": "b"},
        ]
    )
    assert_frame_equal(result.collect(), expected)


def test_groupby_complete():
    """
    Test output if by is present
    """
    # https://stackoverflow.com/q/77123843/7175713
    data = [
        {"Grid Cell": 1, "Site": "A", "Date": "1999-01-01", "Value": -2.45},
        {"Grid Cell": 1, "Site": "A", "Date": "1999-02-01", "Value": -3.72},
        {"Grid Cell": 1, "Site": "A", "Date": "1999-03-01", "Value": 1.34},
        {"Grid Cell": 1, "Site": "A", "Date": "1999-04-01", "Value": 4.56},
        {"Grid Cell": 1, "Site": "B", "Date": "1999-01-01", "Value": 0.23},
        {"Grid Cell": 1, "Site": "B", "Date": "1999-02-01", "Value": 3.26},
        {"Grid Cell": 1, "Site": "B", "Date": "1999-03-01", "Value": 6.76},
        {"Grid Cell": 2, "Site": "C", "Date": "2000-01-01", "Value": -7.45},
        {"Grid Cell": 2, "Site": "C", "Date": "2000-02-01", "Value": -6.43},
        {"Grid Cell": 2, "Site": "C", "Date": "2000-03-01", "Value": -2.18},
        {"Grid Cell": 2, "Site": "D", "Date": "2000-01-01", "Value": -10.72},
        {"Grid Cell": 2, "Site": "D", "Date": "2000-02-01", "Value": -8.97},
        {"Grid Cell": 2, "Site": "D", "Date": "2000-03-01", "Value": -5.32},
        {"Grid Cell": 2, "Site": "D", "Date": "2000-04-01", "Value": -1.73},
    ]

    df = pl.LazyFrame(data)
    expected = (
        df.janitor.complete("Date", "Site", by="Grid Cell")
        .select("Grid Cell", "Site", "Date", "Value")
        .sort(by=pl.all())
    )

    actual = [
        {"Grid Cell": 1, "Site": "A", "Date": "1999-01-01", "Value": -2.45},
        {"Grid Cell": 1, "Site": "A", "Date": "1999-02-01", "Value": -3.72},
        {"Grid Cell": 1, "Site": "A", "Date": "1999-03-01", "Value": 1.34},
        {"Grid Cell": 1, "Site": "A", "Date": "1999-04-01", "Value": 4.56},
        {"Grid Cell": 1, "Site": "B", "Date": "1999-01-01", "Value": 0.23},
        {"Grid Cell": 1, "Site": "B", "Date": "1999-02-01", "Value": 3.26},
        {"Grid Cell": 1, "Site": "B", "Date": "1999-03-01", "Value": 6.76},
        {"Grid Cell": 1, "Site": "B", "Date": "1999-04-01", "Value": None},
        {"Grid Cell": 2, "Site": "C", "Date": "2000-01-01", "Value": -7.45},
        {"Grid Cell": 2, "Site": "C", "Date": "2000-02-01", "Value": -6.43},
        {"Grid Cell": 2, "Site": "C", "Date": "2000-03-01", "Value": -2.18},
        {"Grid Cell": 2, "Site": "C", "Date": "2000-04-01", "Value": None},
        {"Grid Cell": 2, "Site": "D", "Date": "2000-01-01", "Value": -10.72},
        {"Grid Cell": 2, "Site": "D", "Date": "2000-02-01", "Value": -8.97},
        {"Grid Cell": 2, "Site": "D", "Date": "2000-03-01", "Value": -5.32},
        {"Grid Cell": 2, "Site": "D", "Date": "2000-04-01", "Value": -1.73},
    ]

    actual = pl.DataFrame(actual).sort(by=pl.all())

    assert_frame_equal(expected.collect(), actual)


# https://tidyr.tidyverse.org/reference/complete.html
def test_complete_2(fill_df):
    """Test output for janitor.complete."""
    result = fill_df.janitor.complete(
        "group",
        pl.struct("item_id", "item_name").alias("rar").unique(),
        fill_value={"value1": 0, "value2": 99},
        explicit=False,
        sort=True,
    )
    expected = pl.DataFrame(
        [
            {
                "group": 1,
                "item_id": 1,
                "item_name": "a",
                "value1": 1,
                "value2": 4,
            },
            {
                "group": 1,
                "item_id": 2,
                "item_name": "a",
                "value1": 0,
                "value2": 99,
            },
            {
                "group": 1,
                "item_id": 2,
                "item_name": "b",
                "value1": 3,
                "value2": 6,
            },
            {
                "group": 1,
                "item_id": 3,
                "item_name": "b",
                "value1": 0,
                "value2": 99,
            },
            {
                "group": 2,
                "item_id": 1,
                "item_name": "a",
                "value1": 0,
                "value2": 99,
            },
            {
                "group": 2,
                "item_id": 2,
                "item_name": "a",
                "value1": None,
                "value2": 5,
            },
            {
                "group": 2,
                "item_id": 2,
                "item_name": "b",
                "value1": 0,
                "value2": 99,
            },
            {
                "group": 2,
                "item_id": 3,
                "item_name": "b",
                "value1": 4,
                "value2": 7,
            },
        ]
    )

    assert_frame_equal(result, expected)


# https://stackoverflow.com/questions/48914323/tidyr-complete-cases-nesting-misunderstanding
def test_complete_multiple_groupings():
    """Test that `janitor.complete` gets the correct output for multiple groupings."""
    df3 = pl.DataFrame(
        {
            "project_id": [1, 1, 1, 1, 2, 2, 2],
            "meta": ["A", "A", "B", "B", "A", "B", "C"],
            "domain1": ["d", "e", "h", "i", "d", "i", "k"],
            "question_count": [3, 3, 3, 3, 2, 2, 2],
            "tag_count": [2, 1, 3, 2, 1, 1, 2],
        }
    )

    output3 = pl.DataFrame(
        {
            "project_id": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            "meta": ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C"],
            "domain1": ["d", "d", "e", "e", "h", "h", "i", "i", "k", "k"],
            "question_count": [3, 2, 3, 2, 3, 2, 3, 2, 3, 2],
            "tag_count": [2, 1, 1, 0, 3, 0, 2, 1, 0, 2],
        }
    )

    result = df3.janitor.complete(
        pl.struct("meta", "domain1").alias("bar").unique(),
        pl.struct("project_id", "question_count").alias("foo").unique(),
        fill_value={"tag_count": 0},
        sort=True,
    ).select("project_id", "meta", "domain1", "question_count", "tag_count")
    assert_frame_equal(result, output3)


def test_complete_3(fill_df):
    """
    Test output for janitor.complete
    """
    assert_frame_equal(
        fill_df.janitor.complete("group", sort=True).sort("group"),
        fill_df.sort("group"),
    )
