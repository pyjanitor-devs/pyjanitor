from string import ascii_lowercase

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from pandas.testing import assert_frame_equal

from janitor.testing_utils.strategies import categoricaldf_strategy


@pytest.fixture
def MI():
    """MultiIndex fixture. Adapted from Pandas MultiIndexing docs"""

    def mklbl(prefix, n):
        return ["%s%s" % (prefix, i) for i in range(n)]

    miindex = pd.MultiIndex.from_product(
        [mklbl("A", 1), mklbl("B", 2), mklbl("C", 1), mklbl("D", 2)]
    )

    micolumns = pd.MultiIndex.from_tuples(
        [("a", "foo"), ("a", "bar"), ("b", "foo"), ("b", "bah")],
        names=["lvl0", "lvl1"],
    )

    dfmi = (
        pd.DataFrame(
            np.arange(len(miindex) * len(micolumns)).reshape(
                (len(miindex), len(micolumns))
            ),
            index=miindex,
            columns=micolumns,
        )
        .sort_index()
        .sort_index(axis=1)
    )

    return dfmi


def test_multiindex_names_not_found(MI):
    """
    Raise ValueError if the passed label is not found
    """
    MI.index.names = list("ABCD")
    with pytest.raises(
        ValueError, match="group not present in dataframe columns!"
    ):
        MI.complete("group")


@pytest.fixture
def fill_df():
    """pytest fixture"""
    return pd.DataFrame(
        dict(
            group=(1, 2, 1, 2),
            item_id=(1, 2, 2, 3),
            item_name=("a", "a", "b", "b"),
            value1=(1, np.nan, 3, 4),
            value2=range(4, 8),
        )
    )


@pytest.fixture
def taxonomy_df():
    """pytest fixture"""
    return pd.DataFrame(
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
    assert_frame_equal(fill_df.complete(), fill_df)


def test_empty_groups(fill_df):
    """Raise ValueError if any of the groups is empty."""
    with pytest.raises(
        ValueError, match="entry in columns argument cannot be empty"
    ):
        fill_df.complete("group", {})


def test_dict_not_list_like(fill_df):
    """
    Raise ValueError if `*columns`
    is a dictionary, and the value
    is not list-like.
    """
    with pytest.raises(ValueError):
        fill_df.complete("group", {"item_id": "cities"})


def test_dict_not_1D(fill_df):
    """
    Raise ValueError if `*columns`
    is a dictionary, and the value
    is not 1D array.
    """
    with pytest.raises(
        ValueError, match="Kindly provide a 1-D array for item_id."
    ):
        fill_df.complete("group", {"item_id": fill_df})


def test_dict_empty(fill_df):
    """
    Raise ValueError if `*columns`
    is a dictionary, and the value
    is an empty array.
    """
    with pytest.raises(
        ValueError,
        match="Kindly ensure the provided array for group "
        "has at least one value.",
    ):
        fill_df.complete("item_id", {"group": pd.Series([], dtype=int)})


def test_by_not_found(fill_df):
    """Raise ValueError if `by` does not exist."""
    with pytest.raises(ValueError):
        fill_df.complete("group", "item_id", by="name")


def test_group_None(fill_df):
    """Raise ValueError if entry is None."""
    with pytest.raises(
        ValueError, match="label in the columns argument cannot be None."
    ):
        fill_df.complete("group", "item_id", None)


def test_duplicate_groups(fill_df):
    """Raise ValueError if there are duplicate groups."""
    with pytest.raises(
        ValueError, match="item_id should be in only one group."
    ):
        fill_df.complete("group", "item_id", ("item_id", "item_name"))


def test_type_groups(fill_df):
    """Raise TypeError if grouping is not a permitted type."""
    with pytest.raises(TypeError):
        fill_df.complete("group", "item_id", {1, 2, 3})


def test_type_sort(fill_df):
    """Raise TypeError if `sort` is not boolean."""
    with pytest.raises(TypeError):
        fill_df.complete("group", "item_id", sort=11)


def test_groups_not_found(fill_df):
    """Raise ValueError if group does not exist."""
    with pytest.raises(ValueError):
        fill_df.complete("group", ("item_id", "name"))


def test_fill_value(fill_df):
    """Raise ValueError if `fill_value` is not the right data type."""
    with pytest.raises(
        TypeError,
        match="fill_value should either be a dictionary or a scalar value.",
    ):
        fill_df.complete("group", "item_id", fill_value=pd.Series([2, 3, 4]))


def test_fill_value_column(fill_df):
    """Raise ValueError if `fill_value` has a non existent column."""
    with pytest.raises(ValueError):
        fill_df.complete("group", "item_id", fill_value={"cities": 0})


def test_fill_value_dict_scalar(fill_df):
    """
    Raise ValueError if `fill_value` is a dictionary
    and the value is not a scalar.
    """
    with pytest.raises(
        ValueError, match="The value for item_name should be a scalar."
    ):
        fill_df.complete(
            "group", "item_id", fill_value={"item_name": pd.Series([2, 3, 4])}
        )


def test_type_explicit(fill_df):
    """Raise TypeError if `explicit` is not boolean."""
    with pytest.raises(TypeError):
        fill_df.complete("group", "item_id", explicit=11)


@given(df=categoricaldf_strategy())
@settings(deadline=None, max_examples=10)
def test_all_strings_no_nulls(df):
    """
    Test `complete` output when *columns
    is all strings and `df` is unique.
    """
    cols = ["names", "numbers"]
    df = df.assign(dummy=1, names=[*ascii_lowercase[: len(df)]])
    result = df.complete(*cols, sort=True)
    columns = df.columns
    expected = (
        df.set_index(cols)  # noqa: PD013, PD010
        .unstack(cols[-1])  # noqa: PD010
        .stack(dropna=False)  # noqa: PD013
        .reset_index()
        .reindex(columns=columns)
    )

    assert_frame_equal(result, expected)


@given(df=categoricaldf_strategy())
@settings(deadline=None, max_examples=10)
def test_dict_callable(df):
    """
    Test `complete` output when *columns
    is a dictionary.
    """
    cols = ["names", "numbers"]
    df = df.assign(names=[*ascii_lowercase[: len(df)]])
    new_numbers = {
        "numbers": lambda df: range(df.numbers.min(), df.numbers.max() + 1)
    }
    cols = ["numbers", "names"]
    result = df.complete(new_numbers, "names", sort=True)
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


@pytest.mark.xfail(
    reason="CI failure due to dtype mismatch. Tests successful locally."
)
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
    new_index = pd.MultiIndex.from_product([df.names, new_numbers], names=cols)
    new_numbers = {"numbers": pd.array(new_numbers)}
    result = df.complete("names", new_numbers, sort=True)

    expected = (
        df.set_index(cols)
        .reindex(new_index)
        .reset_index()
        .reindex(columns=df.columns)
    )

    assert_frame_equal(result, expected)


@given(df=categoricaldf_strategy())
@settings(deadline=None, max_examples=10)
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
    result = df.complete(new_numbers, "names", sort=True)
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
@settings(deadline=None, max_examples=10)
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
    result = df.complete(new_numbers, "names", sort=True)
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
@settings(deadline=None, max_examples=10)
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
    result = df.complete(new_numbers, "names", sort=True)
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
            "group": [2, 1, 1],
            "item_id": [2, 1, 2],
            "item_name": ["b", "a", "b"],
            "value1": [2, 1, 3],
            "value2": [5, 4, 6],
        }
    )

    result = df.complete("group", ("item_id", "item_name"), sort=True)

    columns = ["group", "item_id", "item_name"]
    expected = (
        df.set_index(columns)
        .unstack("group")
        .stack(dropna=False)
        .reset_index()
        .reindex(columns=df.columns)
        .sort_values(columns, ignore_index=True)
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
            "tag_count": [2, 1, 1, 0, 3, 0, 2, 1, 0, 2],
        }
    )

    result = df3.complete(
        ("meta", "domain1"),
        ("project_id", "question_count"),
        fill_value={"tag_count": 0},
        sort=True,
    ).astype({"tag_count": int})
    assert_frame_equal(result, output3)


def test_fill_value_scalar(taxonomy_df):
    """Test output if the fill_value is a scalar."""
    result = taxonomy_df.complete(
        "Year", "Taxon", fill_value=0, sort=False
    ).astype({"Abundance": int})
    expected = (
        taxonomy_df.encode_categorical(Taxon="appearance")
        .set_index(["Year", "Taxon"])
        .unstack(fill_value=0)
        .stack(dropna=False)
        .reset_index()
        .astype({"Taxon": "object"})
    )

    assert_frame_equal(result, expected)


#  http://imachordata.com/2016/02/05/you-complete-me/
def test_dict_tuple_callable(taxonomy_df):
    """
    Test output if a dictionary and a tuple/list
    are included in the `columns` parameter.
    """

    result = taxonomy_df.complete(
        {"Year": lambda x: range(x.Year.min(), x.Year.max() + 1)},
        ("Taxon", "Abundance"),
        sort=True,
    )

    expected = (
        taxonomy_df.set_index(["Year", "Taxon", "Abundance"])
        .assign(dummy=1)
        .unstack("Year")
        .droplevel(0, 1)
        .reindex(columns=range(1999, 2005))
        .stack(dropna=False)
        .reset_index()
        .iloc[:, :-1]
        .reindex(columns=["Year", "Taxon", "Abundance"])
        .sort_values(["Year", "Taxon", "Abundance"], ignore_index=True)
    )

    assert_frame_equal(result, expected)


def test_dict_tuple(taxonomy_df):
    """
    Test output if a dictionary and a tuple/list
    are included in the `columns` parameter.
    """

    result = taxonomy_df.complete(
        {"Year": [2000, 1999, 2001, 2002, 2003, 2004]},
        ("Taxon", "Abundance"),
        sort=True,
    )

    expected = (
        taxonomy_df.set_index(["Year", "Taxon", "Abundance"])
        .assign(dummy=1)
        .unstack("Year")
        .droplevel(0, 1)
        .reindex(columns=range(1999, 2005))
        .stack(dropna=False)
        .reset_index()
        .iloc[:, :-1]
        .reindex(columns=["Year", "Taxon", "Abundance"])
        .sort_values(["Year", "Taxon", "Abundance"], ignore_index=True)
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
        {"year": lambda x: range(x.year.min(), x.year.max() + 1)},
        by="state",
        sort=True,
    )

    expected = (
        df.set_index("year")
        .groupby("state")
        .apply(lambda x: x.reindex(range(x.index.min(), x.index.max() + 1)))
        .drop(columns="state")
        .reset_index()
    )

    assert_frame_equal(result, expected)


def test_explicit_scalar(fill_df):
    """Test output if fill_value is a scalar, and explicit is False."""
    result = fill_df.complete(
        "group",
        ("item_id", "item_name"),
        fill_value=0,
        explicit=False,
    ).astype({"value2": int})
    columns = ["group", "item_id", "item_name"]
    expected = (
        fill_df.set_index(columns)
        .unstack("group", fill_value=0)
        .stack(dropna=False)
        .reset_index()
        .reindex(columns=fill_df.columns)
        .sort_values(columns, ignore_index=True)
    )
    assert_frame_equal(result, expected, check_dtype=False)


def test_explicit_scalar_cat(fill_df):
    """
    Test output if fill_value is a scalar, explicit is False,
    and one of the columns is of category dtype.
    """
    fill_df = fill_df.astype({"value1": "category"})
    result = fill_df.complete(
        "group",
        ("item_id", "item_name"),
        fill_value=0,
        explicit=False,
    ).astype({"value2": int})
    columns = ["group", "item_id", "item_name"]
    fill_df["value1"] = fill_df["value1"].cat.add_categories([0])
    expected = (
        fill_df.set_index(columns)
        .unstack("group", fill_value=0)
        .stack(dropna=False)
        .reset_index()
        .reindex(columns=fill_df.columns)
        .sort_values(columns, ignore_index=True)
        .astype(
            {
                "value1": pd.CategoricalDtype(
                    categories=[1.0, 3.0, 4.0, 0.0], ordered=False
                )
            }
        )
    )
    assert_frame_equal(result, expected, check_dtype=False)


# https://tidyr.tidyverse.org/reference/complete.html
def test_explicit_dict(fill_df):
    """Test output if fill_value is a dictionary, and explicit is False."""
    result = fill_df.complete(
        "group",
        ("item_id", "item_name"),
        fill_value={"value1": 0, "value2": 99},
        explicit=False,
        sort=True,
    ).astype({"value2": int})
    expected = pd.DataFrame(
        [
            {
                "group": 1,
                "item_id": 1,
                "item_name": "a",
                "value1": 1.0,
                "value2": 4,
            },
            {
                "group": 1,
                "item_id": 2,
                "item_name": "a",
                "value1": 0.0,
                "value2": 99,
            },
            {
                "group": 1,
                "item_id": 2,
                "item_name": "b",
                "value1": 3.0,
                "value2": 6,
            },
            {
                "group": 1,
                "item_id": 3,
                "item_name": "b",
                "value1": 0.0,
                "value2": 99,
            },
            {
                "group": 2,
                "item_id": 1,
                "item_name": "a",
                "value1": 0.0,
                "value2": 99,
            },
            {
                "group": 2,
                "item_id": 2,
                "item_name": "a",
                "value1": np.nan,
                "value2": 5,
            },
            {
                "group": 2,
                "item_id": 2,
                "item_name": "b",
                "value1": 0.0,
                "value2": 99,
            },
            {
                "group": 2,
                "item_id": 3,
                "item_name": "b",
                "value1": 4.0,
                "value2": 7,
            },
        ]
    )

    assert_frame_equal(result, expected, check_dtype=False)


def test_explicit_(fill_df):
    """
    Test output if explicit is False,
    and the columns used for the combination
    are reused in the fill_value.
    """
    trimmed = fill_df.select("value*", axis="columns", invert=True)
    result = trimmed.complete(
        "group",
        ("item_id", "item_name"),
        fill_value=0,
        explicit=False,
        sort=True,
    )
    expected = pd.DataFrame(
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
    assert_frame_equal(result, expected)


def test_nulls(fill_df):
    """
    Test output if nulls are present
    """
    actual = fill_df.complete(["value1"], "value2", sort=True)
    ind = [fill_df.value1.dropna().unique(), fill_df.value2.unique()]
    ind = pd.MultiIndex.from_product(ind, names=["value1", "value2"])
    ind = pd.DataFrame([], index=ind)
    expected = fill_df.merge(
        ind, on=["value1", "value2"], how="outer", sort=True
    )
    assert_frame_equal(actual, expected)


def test_groupby_tuple():
    """Test output for groupby on a tuple of columns."""
    # https://stackoverflow.com/q/77123843/7175713
    data_dict = {
        "Grid Cell": [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2],
        "Site": [
            "A",
            "A",
            "A",
            "A",
            "B",
            "B",
            "B",
            "C",
            "C",
            "C",
            "D",
            "D",
            "D",
            "D",
        ],
        "Date": [
            "1999-01-01",
            "1999-02-01",
            "1999-03-01",
            "1999-04-01",
            "1999-01-01",
            "1999-02-01",
            "1999-03-01",
            "2000-01-01",
            "2000-02-01",
            "2000-03-01",
            "2000-01-01",
            "2000-02-01",
            "2000-03-01",
            "2000-04-01",
        ],
        "Value": [
            -2.45,
            -3.72,
            1.34,
            4.56,
            0.23,
            3.26,
            6.76,
            -7.45,
            -6.43,
            -2.18,
            -10.72,
            -8.97,
            -5.32,
            -1.73,
        ],
    }
    df = pd.DataFrame.from_dict(data_dict)
    expected = df.complete("Date", "Site", by="Grid Cell").sort_values(
        ["Grid Cell", "Site", "Date"], ignore_index=True
    )

    # https://stackoverflow.com/a/77123963/7175713
    def reindex(g):
        idx = pd.MultiIndex.from_product(
            [g["Grid Cell"].unique(), g["Site"].unique(), g["Date"].unique()],
            names=["Grid Cell", "Site", "Date"],
        )
        return g.set_index(["Grid Cell", "Site", "Date"]).reindex(
            idx, fill_value=np.nan
        )

    actual = (
        df.groupby("Grid Cell", group_keys=False).apply(reindex).reset_index()
    )
    assert_frame_equal(expected, actual)


def test_MI_1(MI):
    """
    Test output on multiindex columns
    """
    expected = pd.merge(
        MI.iloc[:2],
        pd.DataFrame({("a", "bar"): range(1, 6)}),
        on=[("a", "bar")],
        how="outer",
        sort=True,
    ).rename_axis(columns=[None, None])
    actual = MI.iloc[:2].complete({("a", "bar"): range(1, 5)})
    assert_frame_equal(actual, expected)
