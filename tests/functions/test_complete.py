import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from pandas.testing import assert_frame_equal
from janitor.testing_utils.strategies import (
    df_strategy,
    categoricaldf_strategy,
)
from string import ascii_lowercase


@given(df=df_strategy())
def test_column_None(df):
    """Test output if *columns is empty."""
    assert_frame_equal(df.complete(), df)


@given(df=df_strategy())
def test_MultiIndex(df):
    """Raise ValueError if `df` has MultiIndex columns."""
    top = range(df.columns.size)
    df.columns = pd.MultiIndex.from_arrays([top, df.columns])
    with pytest.raises(ValueError):
        df.complete("a", "cities")


@given(df=df_strategy())
def test_empty_groups(df):
    """Raise ValueError if any of the groups is empty."""
    with pytest.raises(ValueError):
        df.complete("a", {})


@given(df=df_strategy())
def test_dict_not_list_like(df):
    """
    Raise ValueError if `*columns`
    is a dictionary, and the value
    is not list-like.
    """
    with pytest.raises(ValueError):
        df.complete("a", {"cities": "cities"})


@given(df=df_strategy())
def test_dict_not_1D(df):
    """
    Raise ValueError if `*columns`
    is a dictionary, and the value
    is not 1D array.
    """
    with pytest.raises(ValueError):
        df.complete("a", {"cities": df})


@given(df=df_strategy())
def test_dict_empty(df):
    """
    Raise ValueError if `*columns`
    is a dictionary, and the value
    is an empty array.
    """
    with pytest.raises(ValueError):
        df.complete("a", {"cities": pd.Series([], dtype=int)})


@given(df=df_strategy())
def test_duplicate_groups(df):
    """Raise ValueError if there are duplicate groups."""
    with pytest.raises(ValueError):
        df.complete("a", "cities", ("a", "cities"))


@given(df=df_strategy())
def test_type_groups(df):
    """Raise TypeError if grouping is not a permitted type."""
    with pytest.raises(TypeError):
        df.complete("a", "cities", {1, 2, 3})


@given(df=df_strategy())
def test_type_by(df):
    """Raise TypeError if `by`` is not a permitted type."""
    with pytest.raises(TypeError):
        df.complete("a", "cities", by=1)


@given(df=df_strategy())
def test_groups_not_found(df):
    """Raise ValueError if group does not exist."""
    with pytest.raises(ValueError):
        df.complete("a", "cities", ("a", "cties"))


@given(df=df_strategy())
def test_by_not_found(df):
    """Raise ValueError if `by` does not exist."""
    with pytest.raises(ValueError):
        df.complete("a", "cities", by=["Bell__Chart", "elephant"])


@given(df=categoricaldf_strategy())
def test_all_strings_no_nulls(df):
    """
    Test `complete` output when *columns
    is all strings and `df` is unique.
    """
    cols = ["names", "numbers"]
    df = df.assign(dummy=1, names=[*ascii_lowercase[: len(df)]])
    result = df.complete(*cols)
    result = result.sort_values(cols, ignore_index=True)
    columns = df.columns
    expected = (
        df.set_index(cols)
        .unstack(cols[-1])
        .stack(dropna=False)
        .reset_index()
        .reindex(columns=columns)
    )

    assert_frame_equal(result, expected)


@given(df=categoricaldf_strategy())
def test_dict(df):
    """
    Test `complete` output when *columns
    is a dictionary.
    """
    cols = ["names", "numbers"]
    df = df.assign(names=[*ascii_lowercase[: len(df)]])
    new_numbers = {"numbers": lambda df: range(df.min(), df.max() + 1)}
    cols = ["numbers", "names"]
    result = df.complete(new_numbers, "names")
    result = result.sort_values(cols, ignore_index=True)
    columns = df.columns
    new_index = range(df.numbers.min(), df.numbers.max() + 1)
    new_index = pd.MultiIndex.from_product([new_index, df.names], names=cols)
    expected = (
        df.set_index(cols)
        .reindex(new_index)
        .reset_index()
        .reindex(columns=columns)
    )

    assert_frame_equal(result, expected)


@given(df=categoricaldf_strategy())
def test_dict_extension_array(df):
    """
    Test `complete` output when *columns
    is a dictionary, and value is
    a pandas array.
    """
    cols = ["names", "numbers"]
    df = df.assign(names=[*ascii_lowercase[: len(df)]])
    new_numbers = range(df.numbers.min(), df.numbers.max() + 1)
    new_numbers = pd.array(new_numbers)
    new_numbers = {"numbers": new_numbers}
    cols = ["numbers", "names"]
    result = df.complete(new_numbers, "names")
    result = result.sort_values(cols, ignore_index=True)
    columns = df.columns
    new_index = range(df.numbers.min(), df.numbers.max() + 1)
    new_index = pd.MultiIndex.from_product([new_index, df.names], names=cols)
    expected = (
        df.set_index(cols)
        .reindex(new_index)
        .reset_index()
        .reindex(columns=columns)
    )

    assert_frame_equal(result, expected)


@given(df=categoricaldf_strategy())
def test_dict_numpy(df):
    """
    Test `complete` output when *columns
    is a dictionary, and value is
    a numpy array.
    """
    cols = ["names", "numbers"]
    df = df.assign(names=[*ascii_lowercase[: len(df)]])
    new_numbers = np.arange(df.numbers.min(), df.numbers.max() + 1)
    new_numbers = {"numbers": new_numbers}
    cols = ["numbers", "names"]
    result = df.complete(new_numbers, "names")
    result = result.sort_values(cols, ignore_index=True)
    columns = df.columns
    new_index = range(df.numbers.min(), df.numbers.max() + 1)
    new_index = pd.MultiIndex.from_product([new_index, df.names], names=cols)
    expected = (
        df.set_index(cols)
        .reindex(new_index)
        .reset_index()
        .reindex(columns=columns)
    )

    assert_frame_equal(result, expected)


@given(df=categoricaldf_strategy())
def test_dict_Index(df):
    """
    Test `complete` output when *columns
    is a dictionary, and value is
    a pandas array.
    """
    cols = ["names", "numbers"]
    df = df.assign(names=[*ascii_lowercase[: len(df)]])
    new_numbers = pd.RangeIndex(
        start=df.numbers.min(), stop=df.numbers.max() + 1
    )
    new_numbers = {"numbers": new_numbers}
    cols = ["numbers", "names"]
    result = df.complete(new_numbers, "names")
    result = result.sort_values(cols, ignore_index=True)
    columns = df.columns
    new_index = range(df.numbers.min(), df.numbers.max() + 1)
    new_index = pd.MultiIndex.from_product([new_index, df.names], names=cols)
    expected = (
        df.set_index(cols)
        .reindex(new_index)
        .reset_index()
        .reindex(columns=columns)
    )

    assert_frame_equal(result, expected)


@given(df=categoricaldf_strategy())
def test_dict_duplicated(df):
    """
    Test `complete` output when *columns
    is a dictionary, and value is
    duplicated.
    """
    cols = ["names", "numbers"]
    df = df.assign(names=[*ascii_lowercase[: len(df)]])
    new_numbers = pd.RangeIndex(
        start=df.numbers.min(), stop=df.numbers.max() + 1
    )
    new_numbers = new_numbers.append(new_numbers)
    new_numbers = {"numbers": new_numbers}
    cols = ["numbers", "names"]
    result = df.complete(new_numbers, "names")
    result = result.sort_values(cols, ignore_index=True)
    columns = df.columns
    new_index = range(df.numbers.min(), df.numbers.max() + 1)
    new_index = pd.MultiIndex.from_product([new_index, df.names], names=cols)
    expected = (
        df.set_index(cols)
        .reindex(new_index)
        .reset_index()
        .reindex(columns=columns)
    )

    assert_frame_equal(result, expected)


@given(df=categoricaldf_strategy())
def test_single_column(df):
    """Test `complete` output if a single column is provided."""
    result = df.complete("names")
    assert_frame_equal(result, df)


def test_tuple_column():
    """Test `complete` output if a tuple is provided."""
    df = pd.DataFrame(
        {
            "group": [1, 2, 1],
            "item_id": [1, 2, 2],
            "item_name": ["a", "b", "b"],
            "value1": [1, 2, 3],
            "value2": [4, 5, 6],
        }
    )

    result = df.complete("group", ("item_id", "item_name"))
    result = result.sort_values(
        ["group", "item_id", "item_name"], ignore_index=True
    )
    expected = pd.DataFrame(
        {
            "group": [1, 1, 2, 2],
            "item_id": [1, 2, 1, 2],
            "item_name": ["a", "b", "a", "b"],
            "value1": [1.0, 3.0, np.nan, 2.0],
            "value2": [4.0, 6.0, np.nan, 5.0],
        }
    )
    assert_frame_equal(result, expected)


# https://stackoverflow.com/questions/48914323/tidyr-complete-cases-nesting-misunderstanding
def test_complete_multiple_groupings():
    """Test that `complete` gets the correct output for multiple groupings."""
    df3 = pd.DataFrame(
        {
            "project_id": [1, 1, 1, 1, 2, 2, 2],
            "meta": ["A", "A", "B", "B", "A", "B", "C"],
            "domain1": ["d", "e", "h", "i", "d", "i", "k"],
            "question_count": [3, 3, 3, 3, 2, 2, 2],
            "tag_count": [2, 1, 3, 2, 1, 1, 2],
        }
    )

    output3 = pd.DataFrame(
        {
            "project_id": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            "meta": ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C"],
            "domain1": ["d", "d", "e", "e", "h", "h", "i", "i", "k", "k"],
            "question_count": [3, 2, 3, 2, 3, 2, 3, 2, 3, 2],
            "tag_count": [2.0, 1.0, 1.0, 0.0, 3.0, 0.0, 2.0, 1.0, 0.0, 2.0],
        }
    )

    result = (
        df3.complete(
            ("meta", "domain1"),
            ("project_id", "question_count"),
        )
        .fillna({"tag_count": 0})
        .sort_values(
            ["meta", "domain1", "project_id", "question_count"],
            ignore_index=True,
        )
    )
    assert_frame_equal(result, output3)


#  http://imachordata.com/2016/02/05/you-complete-me/
def test_dict_tuple():
    """
    Test if a dictionary and a tuple/list
    are included in the `columns` parameter.
    """

    df = pd.DataFrame(
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

    result = df.complete(
        {"Year": lambda x: range(x.min(), x.max() + 1)},
        ("Taxon", "Abundance"),
    )

    result = result.sort_values(
        ["Year", "Taxon", "Abundance"], ignore_index=True
    )
    expected = pd.DataFrame(
        [
            {"Year": 1999, "Taxon": "Agarum", "Abundance": 1},
            {"Year": 1999, "Taxon": "Agarum", "Abundance": 8},
            {"Year": 1999, "Taxon": "Saccharina", "Abundance": 2},
            {"Year": 1999, "Taxon": "Saccharina", "Abundance": 4},
            {"Year": 1999, "Taxon": "Saccharina", "Abundance": 5},
            {"Year": 2000, "Taxon": "Agarum", "Abundance": 1},
            {"Year": 2000, "Taxon": "Agarum", "Abundance": 8},
            {"Year": 2000, "Taxon": "Saccharina", "Abundance": 2},
            {"Year": 2000, "Taxon": "Saccharina", "Abundance": 4},
            {"Year": 2000, "Taxon": "Saccharina", "Abundance": 5},
            {"Year": 2001, "Taxon": "Agarum", "Abundance": 1},
            {"Year": 2001, "Taxon": "Agarum", "Abundance": 8},
            {"Year": 2001, "Taxon": "Saccharina", "Abundance": 2},
            {"Year": 2001, "Taxon": "Saccharina", "Abundance": 4},
            {"Year": 2001, "Taxon": "Saccharina", "Abundance": 5},
            {"Year": 2002, "Taxon": "Agarum", "Abundance": 1},
            {"Year": 2002, "Taxon": "Agarum", "Abundance": 8},
            {"Year": 2002, "Taxon": "Saccharina", "Abundance": 2},
            {"Year": 2002, "Taxon": "Saccharina", "Abundance": 4},
            {"Year": 2002, "Taxon": "Saccharina", "Abundance": 5},
            {"Year": 2003, "Taxon": "Agarum", "Abundance": 1},
            {"Year": 2003, "Taxon": "Agarum", "Abundance": 8},
            {"Year": 2003, "Taxon": "Saccharina", "Abundance": 2},
            {"Year": 2003, "Taxon": "Saccharina", "Abundance": 4},
            {"Year": 2003, "Taxon": "Saccharina", "Abundance": 5},
            {"Year": 2004, "Taxon": "Agarum", "Abundance": 1},
            {"Year": 2004, "Taxon": "Agarum", "Abundance": 8},
            {"Year": 2004, "Taxon": "Saccharina", "Abundance": 2},
            {"Year": 2004, "Taxon": "Saccharina", "Abundance": 4},
            {"Year": 2004, "Taxon": "Saccharina", "Abundance": 5},
        ]
    )

    assert_frame_equal(result, expected)


def test_complete_groupby():
    """Test output in the presence of a groupby."""
    df = pd.DataFrame(
        {
            "state": ["CA", "CA", "HI", "HI", "HI", "NY", "NY"],
            "year": [2010, 2013, 2010, 2012, 2016, 2009, 2013],
            "value": [1, 3, 1, 2, 3, 2, 5],
        }
    )

    result = df.complete(
        {"year": lambda x: range(x.min(), x.max() + 1)},
        by="state",
    )

    result = result.sort_values(["state", "year"], ignore_index=True)

    expected = pd.DataFrame(
        [
            {"state": "CA", "year": 2010, "value": 1.0},
            {"state": "CA", "year": 2011, "value": np.nan},
            {"state": "CA", "year": 2012, "value": np.nan},
            {"state": "CA", "year": 2013, "value": 3.0},
            {"state": "HI", "year": 2010, "value": 1.0},
            {"state": "HI", "year": 2011, "value": np.nan},
            {"state": "HI", "year": 2012, "value": 2.0},
            {"state": "HI", "year": 2013, "value": np.nan},
            {"state": "HI", "year": 2014, "value": np.nan},
            {"state": "HI", "year": 2015, "value": np.nan},
            {"state": "HI", "year": 2016, "value": 3.0},
            {"state": "NY", "year": 2009, "value": 2.0},
            {"state": "NY", "year": 2010, "value": np.nan},
            {"state": "NY", "year": 2011, "value": np.nan},
            {"state": "NY", "year": 2012, "value": np.nan},
            {"state": "NY", "year": 2013, "value": 5.0},
        ]
    )

    expected = expected.reindex(columns=df.columns)

    assert_frame_equal(result, expected)
