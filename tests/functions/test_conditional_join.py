import numpy as np
import pandas as pd
import pytest

from hypothesis import given, settings
from pandas.testing import assert_frame_equal
from janitor.testing_utils.strategies import (
    conditional_df,
    conditional_right,
)


@pytest.fixture
def dummy():
    """Test fixture."""
    return pd.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 3],
            "value_1": [2, 5, 7, 1, 3, 4],
            "S": list("ABCDEF"),
        }
    )


@pytest.fixture
def series():
    """Test fixture."""
    return pd.Series([2, 3, 4], name="B")


def test_df_columns_right_columns_both_None(dummy, series):
    """Raise if both df_columns and right_columns is None"""
    with pytest.raises(
        ValueError,
        match="df_columns and right_columns " "cannot both be None.",
    ):
        dummy.conditional_join(
            series, ("id", "B", ">"), df_columns=None, right_columns=None
        )


def test_df_multiindex(dummy, series):
    """Raise ValueError if `df` columns is a MultiIndex."""
    with pytest.raises(
        ValueError,
        match="The number of column levels "
        "from the left and right frames must match.+",
    ):
        dummy.columns = [list("ABC"), list("FGH")]
        dummy.conditional_join(series, (("A", "F"), "non", "=="))


def test_right_df(dummy):
    """Raise TypeError if `right` is not a Series/DataFrame."""
    with pytest.raises(TypeError, match="right should be one of.+"):
        dummy.conditional_join({"non": [2, 3, 4]}, ("id", "non", "=="))


def test_right_series(dummy):
    """Raise ValueError if `right` is not a named Series."""
    with pytest.raises(
        ValueError,
        match="Unnamed Series are not supported for conditional_join.",
    ):
        dummy.conditional_join(pd.Series([2, 3, 4]), ("id", "non", ">="))


def test_check_conditions_exist(dummy, series):
    """Raise ValueError if no condition is provided."""
    with pytest.raises(
        ValueError, match="Kindly provide at least one join condition."
    ):
        dummy.conditional_join(series)


def test_check_condition_type(dummy, series):
    """Raise TypeError if any condition in conditions is not a tuple."""
    with pytest.raises(TypeError, match="condition should be one of.+"):
        dummy.conditional_join(series, ("id", "B", ">"), ["A", "B"])


def test_indicator_type(dummy, series):
    """Raise TypeError if indicator is not a boolean/string."""
    with pytest.raises(TypeError, match="indicator should be one of.+"):
        dummy.conditional_join(series, ("id", "B", ">"), indicator=1)


def test_indicator_exists(dummy, series):
    """Raise ValueError if indicator is a dup of an existing column name."""
    with pytest.raises(
        ValueError,
        match="Cannot use name of an " "existing column for indicator column",
    ):
        dummy.conditional_join(series, ("id", "B", ">"), indicator="id")


def test_check_condition_length(dummy, series):
    """Raise ValueError if any condition is not length 3."""
    with pytest.raises(
        ValueError, match="condition should have only three elements;.+"
    ):
        dummy.conditional_join(series, ("id", "B", "C", "<"))


def test_check_left_on_type(dummy, series):
    """Raise TypeError if left_on is not a hashable."""
    with pytest.raises(TypeError, match="left_on should be one of.+"):
        dummy.conditional_join(series, ([1], "B", "<"))


def test_check_right_on_type(dummy, series):
    """Raise TypeError if right_on is not a hashable."""
    with pytest.raises(TypeError, match="right_on should be one of.+"):
        dummy.conditional_join(series, ("id", {1}, "<"))


def test_check_op_type(dummy, series):
    """Raise TypeError if the operator is not a string."""
    with pytest.raises(TypeError, match="operator should be one of.+"):
        dummy.conditional_join(series, ("id", "B", 1))


def test_check_column_exists_df(dummy, series):
    """
    Raise ValueError if `left_on`
    can not be found in `df`.
    """
    with pytest.raises(
        ValueError, match=".not present in dataframe columns.+"
    ):
        dummy.conditional_join(series, ("C", "B", "<"))


def test_check_column_exists_right(dummy, series):
    """
    Raise ValueError if `right_on`
    can not be found in `right`.
    """
    with pytest.raises(
        ValueError, match=".+not present in dataframe columns.+"
    ):
        dummy.conditional_join(series, ("id", "A", ">="))


def test_check_op_correct(dummy, series):
    """
    Raise ValueError if `op` is not any of
     `!=`, `<`, `>`, `>=`, `<=`.
    """
    with pytest.raises(
        ValueError, match="The conditional join operator should be one of.+"
    ):
        dummy.conditional_join(series, ("id", "B", "=!"))


def test_equi_only(dummy):
    """
    Raise ValueError if only an equi-join is present.
    """
    with pytest.raises(
        ValueError, match="Equality only joins are not supported"
    ):
        dummy.conditional_join(
            dummy.rename(columns={"S": "Strings"}), ("S", "Strings", "==")
        )


def test_check_how_type(dummy, series):
    """
    Raise TypeError if `how` is not a string.
    """
    with pytest.raises(TypeError, match="how should be one of.+"):
        dummy.conditional_join(series, ("id", "B", "<"), how=1)


def test_check_how_value(dummy, series):
    """
    Raise ValueError if `how` is not one of
    `inner`, `left`, or `right`.
    """
    with pytest.raises(ValueError, match="'how' should be one of.+"):
        dummy.conditional_join(series, ("id", "B", "<"), how="INNER")


def test_check_sort_by_appearance_type(dummy, series):
    """
    Raise TypeError if `sort_by_appearance` is not a boolean.
    """
    with pytest.raises(
        TypeError, match="sort_by_appearance should be one of.+"
    ):
        dummy.conditional_join(
            series, ("id", "B", "<"), sort_by_appearance="True"
        )


def test_df_columns(dummy):
    """
    Raise TypeError if `df_columns`is a dictionary,
    and the columns is a MultiIndex.
    """
    with pytest.raises(
        ValueError,
        match="Column renaming with a dictionary is not supported.+",
    ):
        dummy.columns = [list("ABC"), list("FGH")]
        dummy.conditional_join(
            dummy,
            (("A", "F"), ("A", "F"), ">="),
            df_columns={("A", "F"): ("C", "D")},
        )


def test_check_use_numba_type(dummy, series):
    """
    Raise TypeError if `use_numba` is not a string.
    """
    with pytest.raises(TypeError, match="use_numba should be one of.+"):
        dummy.conditional_join(series, ("id", "B", "<"), use_numba=1)


def test_check_keep_type(dummy, series):
    """
    Raise TypeError if `keep` is not a string.
    """
    with pytest.raises(TypeError, match="keep should be one of.+"):
        dummy.conditional_join(series, ("id", "B", "<"), keep=1)


def test_check_keep_value(dummy, series):
    """
    Raise ValueError if `keep` is not one of
    `all`, `first`, or `last`.
    """
    with pytest.raises(ValueError, match="'keep' should be one of.+"):
        dummy.conditional_join(series, ("id", "B", "<"), keep="ALL")


def test_unequal_categories(dummy):
    """
    Raise ValueError if the dtypes are both categories
    and do not match.
    """
    match = "'S' and 'Strings' should have the same categories,"
    match = match + " and the same order."
    with pytest.raises(ValueError, match=match):
        dummy.astype({"S": "category"}).conditional_join(
            dummy.rename(columns={"S": "Strings"}).encode_categorical(
                Strings="appearance"
            ),
            ("S", "Strings", "=="),
            ("id", "value_1", "<="),
        )


def test_dtype_not_permitted(dummy, series):
    """
    Raise ValueError if dtype of column in `df`
    is not an acceptable type.
    """
    dummy["F"] = pd.Timedelta("1 days")
    match = "conditional_join only supports string, "
    match = match + "category, numeric, or date dtypes.+"
    with pytest.raises(ValueError, match=match):
        dummy.conditional_join(series, ("F", "B", "<"))


def test_dtype_str(dummy, series):
    """
    Raise ValueError if dtype of column in `df`
    does not match the dtype of column from `right`.
    """
    with pytest.raises(
        ValueError, match="Both columns should have the same type.+"
    ):
        dummy.conditional_join(series, ("S", "B", "<"))


def test_dtype_strings_non_equi(dummy):
    """
    Raise ValueError if the dtypes are both strings
    on a non-equi operator.
    """
    match = "non-equi joins are supported only "
    match = match + "for datetime and numeric dtypes.+"
    with pytest.raises(
        ValueError,
        match=match,
    ):
        dummy.conditional_join(
            dummy.rename(columns={"S": "Strings"}), ("S", "Strings", "<")
        )


def test_dtype_category_non_equi():
    """
    Raise ValueError if dtype is category,
    and op is non-equi.
    """
    match = (
        "non-equi joins are supported only for datetime and numeric dtypes.+"
    )
    with pytest.raises(ValueError, match=match):
        left = pd.DataFrame({"A": [1, 2, 3]}, dtype="category")
        right = pd.DataFrame({"B": [1, 2, 3]}, dtype="category")
        left.conditional_join(right, ("A", "B", "<"))


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_single_condition_less_than_floats_keep_first(df, right):
    """Test output for a single condition. "<"."""

    df = df.sort_values("B").dropna(subset=["B"])
    expected = pd.merge_asof(
        df[["B"]],
        right[["Numeric"]].sort_values("Numeric").dropna(subset=["Numeric"]),
        left_on="B",
        right_on="Numeric",
        direction="forward",
        allow_exact_matches=False,
    )
    expected.index = range(len(expected))
    actual = (
        df[["B"]]
        .conditional_join(
            right[["Numeric"]].sort_values("Numeric"),
            ("B", "Numeric", "<"),
            how="left",
            sort_by_appearance=False,
            keep="first",
        )
        .sort_values(["B", "Numeric"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_single_condition_less_than_floats_keep_last(df, right):
    """Test output for a single condition. "<"."""

    df = df.sort_values("B").dropna(subset=["B"])
    right = right.sort_values("Numeric").dropna(subset=["Numeric"])
    expected = pd.merge_asof(
        df[["B"]],
        right[["Numeric"]],
        left_on="B",
        right_on="Numeric",
        direction="backward",
        allow_exact_matches=False,
    )
    expected.index = range(len(expected))
    actual = (
        df[["B"]]
        .conditional_join(
            right[["Numeric"]],
            ("B", "Numeric", ">"),
            how="left",
            sort_by_appearance=False,
            keep="last",
        )
        .sort_values(["B", "Numeric"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_single_condition_less_than_floats(df, right):
    """Test output for a single condition. "<"."""

    expected = (
        df[["B"]]
        .merge(right[["Numeric"]], how="cross")
        .loc[lambda df: df.B.lt(df.Numeric)]
        .sort_values(["B", "Numeric"], ignore_index=True)
    )
    actual = (
        df[["B"]]
        .conditional_join(
            right[["Numeric"]],
            ("B", "Numeric", "<"),
            how="inner",
            sort_by_appearance=False,
        )
        .sort_values(["B", "Numeric"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@settings(deadline=None)
@pytest.mark.turtle
@given(df=conditional_df(), right=conditional_right())
def test_single_condition_less_than_floats_keep_first_numba(df, right):
    """Test output for a single condition. "<"."""

    df = df.sort_values("B").dropna(subset=["B"])
    right = right.sort_values("Numeric").dropna(subset=["Numeric"])
    expected = pd.merge_asof(
        df[["B"]],
        right[["Numeric"]],
        left_on="B",
        right_on="Numeric",
        direction="forward",
        allow_exact_matches=False,
    )
    expected.index = range(len(expected))
    actual = (
        df[["B"]]
        .conditional_join(
            right[["Numeric"]],
            ("B", "Numeric", "<"),
            how="left",
            sort_by_appearance=False,
            keep="first",
            use_numba=True,
        )
        .sort_values(["B", "Numeric"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@settings(deadline=None)
@pytest.mark.turtle
@given(df=conditional_df(), right=conditional_right())
def test_single_condition_less_than_floats_keep_last_numba(df, right):
    """Test output for a single condition. "<"."""

    df = df.sort_values("B").dropna(subset=["B"])
    right = right.sort_values("Numeric").dropna(subset=["Numeric"])
    expected = pd.merge_asof(
        df[["B"]],
        right[["Numeric"]],
        left_on="B",
        right_on="Numeric",
        direction="backward",
        allow_exact_matches=False,
    ).sort_values(["B", "Numeric"], ascending=[True, False], ignore_index=True)
    expected.index = range(len(expected))
    actual = (
        df[["B"]]
        .conditional_join(
            right[["Numeric"]],
            ("B", "Numeric", ">"),
            how="left",
            sort_by_appearance=False,
            keep="last",
            use_numba=True,
        )
        .sort_values(
            ["B", "Numeric"], ascending=[True, False], ignore_index=True
        )
    )

    assert_frame_equal(expected, actual)


@settings(deadline=None)
@pytest.mark.turtle
@given(df=conditional_df(), right=conditional_right())
def test_single_condition_less_than_ints_extension_array_numba_first_match(
    df, right
):
    """Test output for a single condition. "<"."""

    df = df.assign(A=df["A"].astype("Int64"))
    right = right.assign(Integers=right["Integers"].astype(pd.Int64Dtype()))

    expected = (
        df[["A"]]
        .assign(index=df.index)
        .merge(right[["Integers"]], how="cross")
        .loc[lambda df: df.A < df.Integers]
        .groupby("index")
        .head(1)
        .drop(columns="index")
        .reset_index(drop=True)
        .sort_values(["A", "Integers"], ignore_index=True)
    )

    actual = (
        df[["A"]]
        .conditional_join(
            right[["Integers"]],
            ("A", "Integers", "<"),
            how="inner",
            keep="first",
            sort_by_appearance=False,
            use_numba=True,
        )
        .sort_values(["A", "Integers"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@settings(deadline=None)
@pytest.mark.turtle
@given(df=conditional_df(), right=conditional_right())
def test_single_condition_less_than_ints_extension_array_numba_last_match(
    df, right
):
    """Test output for a single condition. "<"."""

    df = df.assign(A=df["A"].astype("Int64"))
    right = right.assign(Integers=right["Integers"].astype(pd.Int64Dtype()))

    expected = (
        df[["A"]]
        .assign(index=df.index)
        .merge(right[["Integers"]], how="cross")
        .loc[lambda df: df.A < df.Integers]
        .groupby("index")
        .tail(1)
        .drop(columns="index")
        .reset_index(drop=True)
        .sort_values(["A", "Integers"], ignore_index=True)
    )

    actual = (
        df[["A"]]
        .conditional_join(
            right[["Integers"]],
            ("A", "Integers", "<"),
            how="inner",
            keep="last",
            sort_by_appearance=False,
            use_numba=True,
        )
        .sort_values(["A", "Integers"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_single_condition_less_than_ints(df, right):
    """Test output for a single condition. "<"."""

    expected = (
        df[["A"]]
        .merge(right[["Integers"]], how="cross")
        .loc[lambda df: df.A.lt(df.Integers)]
        .sort_values(["A", "Integers"], ignore_index=True)
    )

    actual = (
        df[["A"]]
        .conditional_join(
            right[["Integers"]],
            ("A", "Integers", "<"),
            how="inner",
            sort_by_appearance=False,
        )
        .sort_values(["A", "Integers"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_single_condition_less_than_ints_numba(df, right):
    """Test output for a single condition. "<"."""

    expected = (
        df[["A"]]
        .merge(right[["Integers"]], how="cross")
        .loc[lambda df: df.A.lt(df.Integers)]
        .sort_values(["A", "Integers"], ignore_index=True)
    )

    actual = (
        df[["A"]]
        .conditional_join(
            right[["Integers"]],
            ("A", "Integers", "<"),
            how="inner",
            sort_by_appearance=False,
            use_numba=True,
        )
        .sort_values(["A", "Integers"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_single_condition_less_than_ints_extension_array(df, right):
    """Test output for a single condition. "<"."""

    df = df.assign(A=df["A"].astype("Int64"))
    right = right.assign(Integers=right["Integers"].astype(pd.Int64Dtype()))

    expected = (
        df[["A"]]
        .assign(index=df.index)
        .merge(right[["Integers"]], how="cross")
        .loc[lambda df: df.A < df.Integers]
        .groupby("index")
        .head(1)
        .drop(columns="index")
        .reset_index(drop=True)
    )

    actual = df[["A"]].conditional_join(
        right[["Integers"]],
        ("A", "Integers", "<"),
        how="inner",
        keep="first",
        sort_by_appearance=False,
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_single_condition_less_than_ints_extension_array_numba(df, right):
    """Test output for a single condition. "<"."""

    df = df.assign(A=df["A"].astype("Int64"))
    right = right.assign(Integers=right["Integers"].astype(pd.Int64Dtype()))

    expected = (
        df[["A"]]
        .assign(index=df.index)
        .merge(right[["Integers"]], how="cross")
        .loc[lambda df: df.A < df.Integers]
        .groupby("index")
        .head(1)
        .drop(columns="index")
        .reset_index(drop=True)
        .sort_values(["A", "Integers"], ignore_index=True)
    )

    actual = (
        df[["A"]]
        .conditional_join(
            right[["Integers"]],
            ("A", "Integers", "<"),
            how="inner",
            keep="first",
            sort_by_appearance=False,
            use_numba=True,
        )
        .sort_values(["A", "Integers"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_single_condition_less_than_equal(df, right):
    """Test output for a single condition. "<=". DateTimes"""

    expected = (
        df[["E"]]
        .assign(index=df.index)
        .merge(right[["Dates"]], how="cross")
        .loc[lambda df: df.E.le(df.Dates)]
        .groupby("index")
        .tail(1)
        .drop(columns="index")
        .reset_index(drop=True)
    )

    actual = df[["E"]].conditional_join(
        right[["Dates"]],
        ("E", "Dates", "<="),
        how="inner",
        keep="last",
        sort_by_appearance=False,
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_single_condition_less_than_equal_numba(df, right):
    """Test output for a single condition. "<=". DateTimes"""

    expected = (
        df[["E"]]
        .assign(index=df.index)
        .merge(right[["Dates"]], how="cross")
        .loc[lambda df: df.E.le(df.Dates)]
        .groupby("index")
        .tail(1)
        .drop(columns="index")
        .reset_index(drop=True)
        .sort_values(["E", "Dates"], ignore_index=True)
    )

    actual = (
        df[["E"]]
        .conditional_join(
            right[["Dates"]],
            ("E", "Dates", "<="),
            how="inner",
            keep="last",
            use_numba=True,
            sort_by_appearance=False,
        )
        .sort_values(["E", "Dates"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_single_condition_less_than_date(df, right):
    """Test output for a single condition. "<". Dates"""

    expected = (
        df[["E"]]
        .merge(right[["Dates"]], how="cross")
        .loc[lambda df: df.E.lt(df.Dates)]
        .sort_values(["E", "Dates"], ignore_index=True)
    )
    actual = (
        df[["E"]]
        .conditional_join(
            right[["Dates"]],
            ("E", "Dates", "<"),
            how="inner",
            sort_by_appearance=False,
        )
        .sort_values(["E", "Dates"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_single_condition_less_than_date_numba(df, right):
    """Test output for a single condition. "<". Dates"""

    expected = (
        df[["E"]]
        .merge(right[["Dates"]], how="cross")
        .loc[lambda df: df.E.lt(df.Dates)]
        .sort_values(["E", "Dates"], ignore_index=True)
    )
    actual = (
        df[["E"]]
        .conditional_join(
            right[["Dates"]],
            ("E", "Dates", "<"),
            how="inner",
            sort_by_appearance=False,
            use_numba=True,
        )
        .sort_values(["E", "Dates"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_single_condition_greater_than_datetime(df, right):
    """Test output for a single condition. ">". Datetimes"""

    expected = (
        df[["E"]]
        .merge(right[["Dates"]], how="cross")
        .loc[lambda df: df.E.gt(df.Dates)]
        .sort_values(["E", "Dates"], ignore_index=True)
    )
    actual = (
        df[["E"]]
        .conditional_join(
            right[["Dates"]],
            ("E", "Dates", ">"),
            how="inner",
            sort_by_appearance=False,
        )
        .sort_values(["E", "Dates"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_single_condition_greater_than_datetime_numba(df, right):
    """Test output for a single condition. ">". Datetimes"""

    expected = (
        df[["E"]]
        .merge(right[["Dates"]], how="cross")
        .loc[lambda df: df.E.gt(df.Dates)]
        .sort_values(["E", "Dates"], ignore_index=True)
    )
    actual = (
        df[["E"]]
        .conditional_join(
            right[["Dates"]],
            ("E", "Dates", ">"),
            how="inner",
            use_numba=True,
            sort_by_appearance=False,
        )
        .sort_values(["E", "Dates"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_single_condition_greater_than_ints(df, right):
    """Test output for a single condition. ">="."""

    expected = (
        df[["A"]]
        .assign(index=df.index)
        .merge(right[["Integers"]], how="cross")
        .loc[lambda df: df.A.ge(df.Integers)]
        .groupby("index")
        .head(1)
        .drop(columns="index")
        .reset_index(drop=True)
    )

    actual = df[["A"]].conditional_join(
        right[["Integers"]],
        ("A", "Integers", ">="),
        how="inner",
        keep="first",
        sort_by_appearance=False,
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_single_condition_greater_than_ints_numba(df, right):
    """Test output for a single condition. ">="."""

    expected = (
        df[["A"]]
        .assign(index=df.index)
        .merge(right[["Integers"]], how="cross")
        .loc[lambda df: df.A.ge(df.Integers)]
        .groupby("index")
        .head(1)
        .drop(columns="index")
        .reset_index(drop=True)
        .sort_values(["A", "Integers"], ignore_index=True)
    )

    actual = (
        df[["A"]]
        .conditional_join(
            right[["Integers"]],
            ("A", "Integers", ">="),
            how="inner",
            keep="first",
            sort_by_appearance=False,
            use_numba=True,
        )
        .sort_values(["A", "Integers"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_single_condition_greater_than_floats_floats(df, right):
    """Test output for a single condition. ">"."""

    expected = (
        df[["B"]]
        .assign(index=df.index)
        .merge(right[["Numeric"]], how="cross")
        .loc[lambda df: df.B.gt(df.Numeric)]
        .groupby("index")
        .tail(1)
        .drop(columns="index")
        .reset_index(drop=True)
        .sort_values(["B", "Numeric"], ignore_index=True)
    )
    actual = (
        df[["B"]]
        .conditional_join(
            right[["Numeric"]],
            ("B", "Numeric", ">"),
            how="inner",
            keep="last",
            sort_by_appearance=False,
        )
        .sort_values(["B", "Numeric"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_single_condition_greater_than_floats_floats_numba(df, right):
    """Test output for a single condition. ">"."""

    expected = (
        df[["B"]]
        .assign(index=df.index)
        .merge(right[["Numeric"]], how="cross")
        .loc[lambda df: df.B.gt(df.Numeric)]
        .groupby("index")
        .tail(1)
        .drop(columns="index")
        .reset_index(drop=True)
        .sort_values(["B", "Numeric"], ignore_index=True)
    )
    actual = (
        df[["B"]]
        .conditional_join(
            right[["Numeric"]],
            ("B", "Numeric", ">"),
            how="inner",
            keep="last",
            sort_by_appearance=False,
            use_numba=True,
        )
        .sort_values(["B", "Numeric"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_single_condition_greater_than_ints_extension_array(df, right):
    """Test output for a single condition. ">"."""

    df = df.astype({"A": "Int64"})
    right = right.astype({"Integers": "Int64"})
    expected = (
        df[["A"]]
        .merge(right[["Integers"]], how="cross")
        .loc[lambda df: df.A > df.Integers]
        .sort_values(["A", "Integers"], ignore_index=True)
    )

    actual = (
        df[["A"]]
        .conditional_join(
            right[["Integers"]],
            ("A", "Integers", ">"),
            how="inner",
            sort_by_appearance=False,
        )
        .sort_values(["A", "Integers"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_single_condition_greater_than_ints_extension_array_numba(df, right):
    """Test output for a single condition. ">"."""

    df = df.astype({"A": "Int64"})
    right = right.astype({"Integers": "Int64"})
    expected = (
        df[["A"]]
        .merge(right[["Integers"]], how="cross")
        .loc[lambda df: df.A > df.Integers]
        .sort_values(["A", "Integers"], ignore_index=True)
    )

    actual = (
        df[["A"]]
        .conditional_join(
            right[["Integers"]],
            ("A", "Integers", ">"),
            how="inner",
            sort_by_appearance=False,
            use_numba=True,
        )
        .sort_values(["A", "Integers"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_single_condition_not_equal_ints(df, right):
    """Test output for a single condition. "!="."""

    expected = (
        df[["A"]]
        .merge(right[["Integers"]], how="cross")
        .loc[lambda df: df.A != df.Integers]
        .sort_values(["A", "Integers"], ignore_index=True)
    )

    actual = (
        df[["A"]]
        .conditional_join(
            right[["Integers"]],
            ("A", "Integers", "!="),
            how="inner",
            sort_by_appearance=False,
        )
        .sort_values(["A", "Integers"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_single_condition_not_equal_ints_numba(df, right):
    """Test output for a single condition. "!="."""

    expected = (
        df[["A"]]
        .merge(right[["Integers"]], how="cross")
        .loc[lambda df: df.A != df.Integers]
        .sort_values(["A", "Integers"], ignore_index=True)
    )

    actual = (
        df[["A"]]
        .conditional_join(
            right[["Integers"]],
            ("A", "Integers", "!="),
            how="inner",
            sort_by_appearance=False,
            use_numba=True,
        )
        .sort_values(["A", "Integers"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_single_condition_not_equal_floats_only(df, right):
    """Test output for a single condition. "!="."""

    expected = (
        df[["B"]]
        .assign(index=df.index)
        .merge(right[["Numeric"]], how="cross")
        .loc[lambda df: df.B != df.Numeric]
        .groupby("index")
        .tail(1)
        .drop(columns="index")
        .reset_index(drop=True)
    )

    actual = df[["B"]].conditional_join(
        right[["Numeric"]],
        ("B", "Numeric", "!="),
        how="inner",
        keep="last",
        sort_by_appearance=False,
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_single_condition_not_equal_floats_only_numba(df, right):
    """Test output for a single condition. "!="."""

    expected = (
        df[["B"]]
        .assign(index=df.index)
        .merge(right[["Numeric"]], how="cross")
        .loc[lambda df: df.B != df.Numeric]
        .groupby("index")
        .tail(1)
        .drop(columns="index")
        .reset_index(drop=True)
        .sort_values(["B", "Numeric"], ignore_index=True)
    )

    actual = (
        df[["B"]]
        .conditional_join(
            right[["Numeric"]],
            ("B", "Numeric", "!="),
            how="inner",
            keep="last",
            sort_by_appearance=False,
            use_numba=True,
        )
        .sort_values(["B", "Numeric"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_single_condition_not_equal_datetime(df, right):
    """Test output for a single condition. "!="."""

    expected = (
        df[["E"]]
        .assign(index=df.index)
        .merge(right[["Dates"]], how="cross")
        .loc[lambda df: df.E != df.Dates]
        .groupby("index")
        .head(1)
        .drop(columns="index")
        .reset_index(drop=True)
    )

    actual = df[["E"]].conditional_join(
        right[["Dates"]],
        ("E", "Dates", "!="),
        how="inner",
        keep="first",
        sort_by_appearance=False,
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_single_condition_not_equal_datetime_numba(df, right):
    """Test output for a single condition. "!="."""

    expected = (
        df[["E"]]
        .assign(index=df.index)
        .merge(right[["Dates"]], how="cross")
        .loc[lambda df: df.E != df.Dates]
        .groupby("index")
        .head(1)
        .drop(columns="index")
        .reset_index(drop=True)
        .sort_values(["E", "Dates"], ignore_index=True)
    )

    actual = (
        df[["E"]]
        .conditional_join(
            right[["Dates"]],
            ("E", "Dates", "!="),
            how="inner",
            keep="first",
            use_numba=True,
            sort_by_appearance=False,
        )
        .sort_values(["E", "Dates"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_how_left(df, right):
    """Test output when `how==left`. "<="."""

    expected = (
        df[["A"]]
        .assign(index=np.arange(len(df)))
        .merge(right[["Integers"]], how="cross")
        .loc[lambda df: df.A <= df.Integers]
    )
    expected = expected.set_index("index")
    expected.index.name = None
    expected = (
        df[["A"]]
        .merge(
            expected[["Integers"]],
            left_index=True,
            right_index=True,
            how="left",
            indicator=True,
            sort=False,
        )
        .sort_values(["A", "Integers"], ignore_index=True)
        .reset_index(drop=True)
    )
    actual = (
        df[["A"]]
        .conditional_join(
            right[["Integers"]],
            ("A", "Integers", "<="),
            how="left",
            indicator=True,
        )
        .sort_values(["A", "Integers"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_how_left_multiindex(df, right):
    """Test output when `how==left`. "<="."""

    expected = (
        df[["A"]]
        .assign(index=np.arange(len(df)))
        .merge(right.Integers.rename("A"), how="cross")
        .loc[lambda df: df.A_x <= df.A_y]
    )
    expected = expected.set_index("index")
    expected.index.name = None
    expected = (
        df[["A"]]
        .merge(
            expected[["A_y"]],
            left_index=True,
            right_index=True,
            how="left",
            indicator=True,
            sort=False,
        )
        .sort_values(["A", "A_y"], ignore_index=True)
        .reset_index(drop=True)
    )
    actual = (
        df[["A"]]
        .conditional_join(
            right.Integers.rename("A"),
            ("A", "A", "<="),
            how="left",
            indicator=True,
        )
        .collapse_levels()
        .rename(columns={"left_A": "A", "right_A": "A_y"})
        .select_columns("A", "A_y", "_merge")
        .sort_values(["A", "A_y"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_how_left_sort(df, right):
    """Test output when `how==left`. "<="."""

    expected = (
        df[["A"]]
        .assign(index=np.arange(len(df)))
        .merge(right[["Integers"]], how="cross")
        .loc[lambda df: df.A <= df.Integers]
    )
    expected = expected.set_index("index")
    expected.index.name = None
    expected = (
        df[["A"]]
        .merge(
            expected[["Integers"]],
            left_index=True,
            right_index=True,
            how="left",
            indicator=True,
            sort=False,
        )
        .sort_values(["A", "Integers"], ignore_index=True)
        .reset_index(drop=True)
    )
    actual = (
        df[["A"]]
        .conditional_join(
            right[["Integers"]],
            ("A", "Integers", "<="),
            how="left",
            indicator=True,
            sort_by_appearance=True,
        )
        .sort_values(["A", "Integers"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
@pytest.mark.turtle
def test_how_right(df, right):
    """Test output when `how==right`. ">"."""

    expected = df.merge(
        right.assign(index=np.arange(len(right))), how="cross"
    ).loc[lambda df: df.E.gt(df.Dates)]
    expected = expected.set_index("index")
    expected.index.name = None
    expected = (
        expected[["E"]]
        .merge(
            right[["Dates"]],
            how="right",
            left_index=True,
            right_index=True,
            sort=False,
            indicator=True,
        )
        .sort_values(["E", "Dates"], ignore_index=True)
        .reset_index(drop=True)
    )
    actual = (
        df[["E"]]
        .conditional_join(
            right[["Dates"]], ("E", "Dates", ">"), how="right", indicator=True
        )
        .sort_values(["E", "Dates"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
@pytest.mark.turtle
def test_how_right_sort(df, right):
    """Test output when `how==right`. ">"."""

    expected = df.merge(
        right.assign(index=np.arange(len(right))), how="cross"
    ).loc[lambda df: df.E.gt(df.Dates)]
    expected = expected.set_index("index")
    expected.index.name = None
    expected = (
        expected[["E"]]
        .merge(
            right[["Dates"]],
            how="right",
            left_index=True,
            right_index=True,
            sort=False,
            indicator=True,
        )
        .sort_values(["E", "Dates"], ignore_index=True)
        .reset_index(drop=True)
    )
    actual = (
        df[["E"]]
        .conditional_join(
            right[["Dates"]],
            ("E", "Dates", ">"),
            how="right",
            indicator=True,
            sort_by_appearance=True,
        )
        .sort_values(["E", "Dates"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_dual_conditions_gt_and_lt_dates(df, right):
    """Test output for interval conditions."""

    middle, left_on, right_on = ("E", "Dates", "Dates_Right")
    expected = (
        df[["E"]]
        .merge(right[["Dates", "Dates_Right"]], how="cross")
        .loc[
            lambda df: df.E.between(
                df.Dates, df.Dates_Right, inclusive="neither"
            )
        ]
        .sort_values(["E", "Dates", "Dates_Right"], ignore_index=True)
    )

    actual = (
        df[["E"]]
        .conditional_join(
            right[["Dates", "Dates_Right"]],
            (middle, left_on, ">"),
            (middle, right_on, "<"),
            how="inner",
            sort_by_appearance=False,
        )
        .sort_values(["E", "Dates", "Dates_Right"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
@pytest.mark.turtle
def test_dual_conditions_ge_and_le_dates(df, right):
    """Test output for interval conditions."""

    expected = (
        df[["E"]]
        .merge(right[["Dates", "Dates_Right"]], how="cross")
        .loc[
            lambda df: df.E.between(df.Dates, df.Dates_Right, inclusive="both")
        ]
        .sort_values(["E", "Dates", "Dates_Right"], ignore_index=True)
    )

    actual = (
        df[["E"]]
        .conditional_join(
            right[["Dates", "Dates_Right"]],
            ("E", "Dates", ">="),
            ("E", "Dates_Right", "<="),
            how="inner",
            sort_by_appearance=False,
        )
        .sort_values(["E", "Dates", "Dates_Right"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
@pytest.mark.turtle
def test_dual_conditions_le_and_ge_dates(df, right):
    """Test output for interval conditions, if "<" comes before ">"."""

    expected = (
        df[["E"]]
        .merge(right[["Dates", "Dates_Right"]], how="cross")
        .loc[
            lambda df: df.E.between(df.Dates, df.Dates_Right, inclusive="both")
        ]
        .sort_values(["E", "Dates", "Dates_Right"], ignore_index=True)
    )
    actual = (
        df[["E"]]
        .conditional_join(
            right[["Dates", "Dates_Right"]],
            ("E", "Dates_Right", "<="),
            ("E", "Dates", ">="),
            how="inner",
            sort_by_appearance=False,
        )
        .sort_values(["E", "Dates", "Dates_Right"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_dual_conditions_ge_and_le_dates_right_open(df, right):
    """Test output for interval conditions."""

    expected = (
        df[["E"]]
        .merge(right[["Dates", "Dates_Right"]], how="cross")
        .loc[
            lambda df: df.E.between(
                df.Dates, df.Dates_Right, inclusive="right"
            )
        ]
        .sort_values(["E", "Dates", "Dates_Right"], ignore_index=True)
    )

    actual = (
        df[["E"]]
        .conditional_join(
            right[["Dates", "Dates_Right"]],
            ("E", "Dates", ">"),
            ("E", "Dates_Right", "<="),
            how="inner",
            sort_by_appearance=False,
        )
        .sort_values(["E", "Dates", "Dates_Right"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_dual_conditions_ge_and_le_numbers(df, right):
    """Test output for interval conditions, for numeric dtypes."""

    expected = (
        df[["B"]]
        .merge(right[["Numeric", "Floats"]], how="cross")
        .loc[lambda df: df.B.between(df.Numeric, df.Floats, inclusive="both")]
        .sort_values(["B", "Numeric", "Floats"], ignore_index=True)
    )

    actual = (
        df[["B"]]
        .conditional_join(
            right[["Numeric", "Floats"]],
            ("B", "Numeric", ">="),
            ("B", "Floats", "<="),
            how="inner",
            sort_by_appearance=False,
        )
        .sort_values(["B", "Numeric", "Floats"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
@pytest.mark.turtle
def test_dual_conditions_le_and_ge_numbers(df, right):
    """
    Test output for interval conditions,
    for numeric dtypes,
    if "<" comes before ">".
    """

    expected = (
        df[["B"]]
        .merge(right[["Numeric", "Floats"]], how="cross")
        .loc[lambda df: df.B.between(df.Numeric, df.Floats, inclusive="both")]
        .sort_values(["B", "Numeric", "Floats"], ignore_index=True)
    )

    actual = (
        df[["B"]]
        .conditional_join(
            right[["Numeric", "Floats"]],
            ("B", "Floats", "<="),
            ("B", "Numeric", ">="),
            how="inner",
            sort_by_appearance=False,
        )
        .sort_values(["B", "Numeric", "Floats"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_dual_conditions_gt_and_lt_numbers(df, right):
    """Test output for interval conditions."""

    expected = (
        df[["B"]]
        .merge(right[["Numeric", "Floats"]], how="cross")
        .loc[
            lambda df: df.B.between(df.Numeric, df.Floats, inclusive="neither")
        ]
        .sort_values(["B", "Numeric", "Floats"], ignore_index=True)
    )

    actual = (
        df[["B"]]
        .conditional_join(
            right[["Numeric", "Floats"]],
            ("B", "Floats", "<"),
            ("B", "Numeric", ">"),
            how="inner",
            sort_by_appearance=False,
        )
        .sort_values(["B", "Numeric", "Floats"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
@pytest.mark.turtle
def test_dual_conditions_gt_and_lt_numbers_left_open(df, right):
    """Test output for interval conditions."""

    expected = (
        df[["B"]]
        .merge(right[["Numeric", "Floats"]], how="cross")
        .loc[lambda df: df.B.between(df.Numeric, df.Floats, inclusive="left")]
        .sort_values(["B", "Numeric", "Floats"], ignore_index=True)
    )

    actual = (
        df[["B"]]
        .conditional_join(
            right[["Numeric", "Floats"]],
            ("B", "Floats", "<"),
            ("B", "Numeric", ">="),
            how="inner",
            sort_by_appearance=False,
        )
        .sort_values(["B", "Numeric", "Floats"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_dual_conditions_gt_and_lt_numbers_(df, right):
    """
    Test output for multiple conditions.
    """

    expected = (
        right[["Numeric", "Floats"]]
        .merge(df[["B"]], how="cross")
        .loc[
            lambda df: df.B.between(df.Numeric, df.Floats, inclusive="neither")
        ]
        .sort_values(["Numeric", "Floats", "B"], ignore_index=True)
    )

    actual = (
        right[["Numeric", "Floats"]]
        .conditional_join(
            df[["B"]],
            ("Floats", "B", ">"),
            ("Numeric", "B", "<"),
            how="inner",
            sort_by_appearance=False,
        )
        .sort_values(["Numeric", "Floats", "B"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
@pytest.mark.turtle
def test_dual_conditions_gt_and_lt_numbers_left_join(df, right):
    """
    Test output for multiple conditions, and how is `left`.
    """
    expected = (
        df[["B"]]
        .assign(index=np.arange(len(df)))
        .merge(right[["Numeric", "Floats"]], how="cross")
        .loc[
            lambda df: df.B.between(df.Numeric, df.Floats, inclusive="neither")
        ]
    )
    expected = expected.set_index("index")
    expected.index.name = None
    expected = (
        df[["B"]]
        .merge(
            expected[["Numeric", "Floats"]],
            left_index=True,
            right_index=True,
            indicator=True,
            how="left",
            sort=False,
        )
        .reset_index(drop=True)
    ).sort_values(["B", "Numeric", "Floats"], ignore_index=True)

    actual = (
        df[["B"]]
        .conditional_join(
            right[["Numeric", "Floats"]],
            ("B", "Numeric", ">"),
            ("B", "Floats", "<"),
            how="left",
            sort_by_appearance=True,
            indicator=True,
        )
        .sort_values(["B", "Numeric", "Floats"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_dual_conditions_gt_and_lt_numbers_right_join(df, right):
    """
    Test output for multiple conditions, and how is `right`.
    """

    expected = (
        df[["B"]]
        .merge(
            right[["Numeric", "Floats"]].assign(index=np.arange(len(right))),
            how="cross",
        )
        .loc[
            lambda df: df.B.between(df.Numeric, df.Floats, inclusive="neither")
        ]
    )
    expected = expected.set_index("index")
    expected.index.name = None
    expected = (
        expected[["B"]]
        .merge(
            right[["Numeric", "Floats"]],
            left_index=True,
            right_index=True,
            indicator=True,
            how="right",
            sort=False,
        )
        .sort_values(["Numeric", "Floats", "B"], ignore_index=True)
        .reset_index(drop=True)
    )

    actual = (
        df[["B"]]
        .conditional_join(
            right[["Numeric", "Floats"]],
            ("B", "Numeric", ">"),
            ("B", "Floats", "<"),
            how="right",
            indicator=True,
        )
        .sort_values(["Numeric", "Floats", "B"], ignore_index=True)
    )
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_dual_ne_extension(df, right):
    """
    Test output for multiple conditions. Extension Arrays. `!=`
    """

    filters = ["A", "Integers", "B", "Numeric"]
    df = df.astype({"A": "Int64"})
    right = right.astype({"Integers": "Int64"})
    expected = df.merge(right, how="cross")
    expected = (
        expected.loc[
            expected.A.ne(expected.Integers) & expected.B.ne(expected.Numeric),
            filters,
        ]
        .reset_index(drop=True)
        .sort_values(filters, ignore_index=True)
    )

    actual = (
        df.conditional_join(
            right,
            ("A", "Integers", "!="),
            ("B", "Numeric", "!="),
            how="inner",
            sort_by_appearance=True,
        )
        .filter(filters)
        .sort_values(filters, ignore_index=True)
    )
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_dual_ne(df, right):
    """
    Test output for multiple conditions. `!=`
    """

    filters = ["A", "B", "Integers", "Numeric"]

    expected = df[["A", "B"]].merge(
        right[["Integers", "Numeric"]], how="cross"
    )
    expected = expected.loc[
        expected.A.ne(expected.Integers) & expected.B.ne(expected.Numeric)
    ].sort_values(filters, ignore_index=True)

    actual = (
        df[["A", "B"]]
        .conditional_join(
            right[["Integers", "Numeric"]],
            ("A", "Integers", "!="),
            ("B", "Numeric", "!="),
            how="inner",
            sort_by_appearance=False,
        )
        .sort_values(filters, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_dual_ne_numba_extension(df, right):
    """
    Test output for multiple conditions. Extension Arrays. `!=`
    """

    filters = ["A", "Integers", "B", "Numeric"]
    df = df.astype({"A": "Int64"})
    right = right.astype({"Integers": "Int64"})
    expected = df.merge(right, how="cross")
    expected = (
        expected.loc[
            expected.A.ne(expected.Integers) & expected.B.ne(expected.Numeric),
            filters,
        ]
        .reset_index(drop=True)
        .sort_values(filters, ignore_index=True)
    )

    actual = (
        df.conditional_join(
            right,
            ("A", "Integers", "!="),
            ("B", "Numeric", "!="),
            how="inner",
            use_numba=True,
            sort_by_appearance=True,
        )
        .filter(filters)
        .sort_values(filters, ignore_index=True)
    )
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_dual_ne_dates(df, right):
    """
    Test output for multiple conditions. `!=`
    """

    filters = ["A", "Integers", "E", "Dates"]
    expected = (
        df[["A", "E"]]
        .merge(right[["Integers", "Dates"]], indicator=True, how="cross")
        .loc[lambda df: df.A.ne(df.Integers) & df.E.ne(df.Dates)]
        .sort_values(filters, ignore_index=True)
    )

    actual = (
        df[["A", "E"]]
        .conditional_join(
            right[["Integers", "Dates"]],
            ("A", "Integers", "!="),
            ("E", "Dates", "!="),
            how="inner",
            sort_by_appearance=False,
            indicator=True,
        )
        .sort_values(filters, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_dual_ne_numba_dates(df, right):
    """
    Test output for multiple conditions. `!=`
    """

    filters = ["A", "Integers", "E", "Dates"]
    expected = (
        df[["A", "E"]]
        .merge(right[["Integers", "Dates"]], how="cross")
        .loc[lambda df: df.A.ne(df.Integers) & df.E.ne(df.Dates)]
        .sort_values(filters, ignore_index=True)
    )

    actual = (
        df[["A", "E"]]
        .conditional_join(
            right[["Integers", "Dates"]],
            ("A", "Integers", "!="),
            ("E", "Dates", "!="),
            how="inner",
            use_numba=True,
            sort_by_appearance=False,
        )
        .sort_values(filters, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
@pytest.mark.turtle
def test_multiple_ne_dates(df, right):
    """
    Test output for multiple conditions. `!=`
    """

    filters = ["A", "E", "B", "Integers", "Dates", "Numeric"]
    expected = (
        df[["A", "E", "B"]]
        .merge(right[["Integers", "Dates", "Numeric"]], how="cross")
        .loc[
            lambda df: df.A.ne(df.Integers)
            & df.E.ne(df.Dates)
            & df.B.ne(df.Numeric)
        ]
        .sort_values(filters, ignore_index=True)
    )

    actual = (
        df[["A", "E", "B"]]
        .conditional_join(
            right[["Integers", "Dates", "Numeric"]],
            ("A", "Integers", "!="),
            ("E", "Dates", "!="),
            ("B", "Numeric", "!="),
            how="inner",
            sort_by_appearance=False,
        )
        .sort_values(filters, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_dual_conditions_eq_and_ne(df, right):
    """Test output for equal and not equal conditions."""

    columns = ["B", "Numeric", "E", "Dates"]
    expected = (
        df.dropna(subset=["B"])
        .merge(
            right.dropna(subset=["Numeric"]), left_on="B", right_on="Numeric"
        )
        .loc[lambda df: df.E.ne(df.Dates), columns]
        .sort_values(columns, ignore_index=True)
    )

    actual = (
        df.dropna(subset=["B"])
        .conditional_join(
            right.dropna(subset=["Numeric"]),
            ("B", "Numeric", "=="),
            ("E", "Dates", "!="),
            how="inner",
            sort_by_appearance=False,
        )
        .filter(columns)
        .sort_values(columns, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_dual_conditions_ne_and_eq(df, right):
    """Test output for equal and not equal conditions."""

    filters = ["A", "E", "Integers", "Dates"]
    expected = (
        df[["A", "E"]]
        .merge(right[["Integers", "Dates"]], left_on="E", right_on="Dates")
        .loc[lambda df: df.A.ne(df.Integers)]
        .sort_values(filters, ignore_index=True)
    )

    actual = (
        df[["A", "E"]]
        .conditional_join(
            right[["Integers", "Dates"]],
            ("A", "Integers", "!="),
            ("E", "Dates", "=="),
            how="inner",
            sort_by_appearance=False,
        )
        .sort_values(filters, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
@pytest.mark.turtle
def test_gt_lt_ne_conditions(df, right):
    """
    Test output for multiple conditions.
    """

    filters = ["A", "B", "E", "Integers", "Numeric", "Dates"]
    expected = (
        df[["A", "B", "E"]]
        .merge(right[["Integers", "Numeric", "Dates"]], how="cross")
        .loc[
            lambda df: df.A.gt(df.Integers)
            & df.B.lt(df.Numeric)
            & df.E.ne(df.Dates)
        ]
        .sort_values(filters, ignore_index=True)
    )

    actual = (
        df[["A", "B", "E"]]
        .conditional_join(
            right[["Integers", "Numeric", "Dates"]],
            ("A", "Integers", ">"),
            ("B", "Numeric", "<"),
            ("E", "Dates", "!="),
            how="inner",
            sort_by_appearance=False,
        )
        .sort_values(filters, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
@pytest.mark.turtle
def test_gt_lt_ne_numba_conditions(df, right):
    """
    Test output for multiple conditions.
    """

    filters = ["A", "B", "E", "Integers", "Numeric", "Dates"]
    expected = (
        df[["A", "B", "E"]]
        .merge(right[["Integers", "Numeric", "Dates"]], how="cross")
        .loc[
            lambda df: df.A.gt(df.Integers)
            & df.B.lt(df.Numeric)
            & df.E.ne(df.Dates)
        ]
        .sort_values(filters, ignore_index=True)
    )

    actual = (
        df[["A", "B", "E"]]
        .conditional_join(
            right[["Integers", "Numeric", "Dates"]],
            ("A", "Integers", ">"),
            ("B", "Numeric", "<"),
            ("E", "Dates", "!="),
            how="inner",
            use_numba=True,
            sort_by_appearance=False,
        )
        .sort_values(filters, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_gt_ne_conditions(df, right):
    """
    Test output for multiple conditions.
    """

    filters = ["A", "E", "Integers", "Dates"]
    expected = (
        df[["A", "E"]]
        .merge(right[["Integers", "Dates"]], how="cross")
        .loc[lambda df: df.A.gt(df.Integers) & df.E.ne(df.Dates)]
        .sort_values(filters, ignore_index=True)
    )

    actual = (
        df[["A", "E"]]
        .conditional_join(
            right[["Integers", "Dates"]],
            ("A", "Integers", ">"),
            ("E", "Dates", "!="),
            how="inner",
            sort_by_appearance=False,
        )
        .sort_values(filters, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_gt_ne_numba_conditions(df, right):
    """
    Test output for multiple conditions.
    """

    filters = ["A", "E", "Integers", "Dates"]
    expected = (
        df[["A", "E"]]
        .merge(right[["Integers", "Dates"]], how="cross")
        .loc[lambda df: df.A.gt(df.Integers) & df.E.ne(df.Dates)]
        .sort_values(filters, ignore_index=True)
    )

    actual = (
        df[["A", "E"]]
        .conditional_join(
            right[["Integers", "Dates"]],
            ("A", "Integers", ">"),
            ("E", "Dates", "!="),
            how="inner",
            use_numba=True,
            sort_by_appearance=False,
        )
        .sort_values(filters, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_le_ne_conditions(df, right):
    """
    Test output for multiple conditions.
    """

    filters = ["A", "E", "Integers", "Dates"]
    expected = (
        df[["A", "E"]]
        .merge(right[["Integers", "Dates"]], how="cross")
        .loc[lambda df: df.A.le(df.Integers) & df.E.ne(df.Dates)]
        .sort_values(filters, ignore_index=True)
    )

    actual = (
        df[["A", "E"]]
        .conditional_join(
            right[["Integers", "Dates"]],
            ("A", "Integers", "<="),
            ("E", "Dates", "!="),
            how="inner",
            sort_by_appearance=False,
        )
        .sort_values(filters, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_le_ne_numba_conditions(df, right):
    """
    Test output for multiple conditions.
    """

    filters = ["A", "E", "Integers", "Dates"]
    expected = (
        df[["A", "E"]]
        .merge(right[["Integers", "Dates"]], how="cross")
        .loc[lambda df: df.A.le(df.Integers) & df.E.ne(df.Dates)]
        .sort_values(filters, ignore_index=True)
    )

    actual = (
        df[["A", "E"]]
        .conditional_join(
            right[["Integers", "Dates"]],
            ("A", "Integers", "<="),
            ("E", "Dates", "!="),
            how="inner",
            use_numba=True,
            sort_by_appearance=False,
        )
        .sort_values(filters, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
@pytest.mark.turtle
def test_gt_lt_ne_start(df, right):
    """
    Test output for multiple conditions.
    """

    filters = ["A", "B", "E", "Integers", "Numeric", "Dates"]
    expected = (
        df[["A", "B", "E"]]
        .merge(right[["Integers", "Numeric", "Dates"]], how="cross")
        .loc[
            lambda df: df.A.gt(df.Integers)
            & df.B.lt(df.Numeric)
            & df.E.ne(df.Dates)
        ]
        .sort_values(filters, ignore_index=True)
    )

    actual = (
        df[["A", "B", "E"]]
        .conditional_join(
            right[["Integers", "Numeric", "Dates"]],
            ("E", "Dates", "!="),
            ("A", "Integers", ">"),
            ("B", "Numeric", "<"),
            how="inner",
            sort_by_appearance=False,
        )
        .sort_values(filters, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
@pytest.mark.turtle
def test_ge_le_ne_extension_array(df, right):
    """
    Test output for multiple conditions.
    """

    filters = ["A", "B", "E", "Integers", "Numeric", "Dates"]
    df = df.assign(A=df["A"].astype("Int64"))
    right = right.assign(Integers=right["Integers"].astype(pd.Int64Dtype()))

    expected = df[["A", "B", "E"]].merge(
        right[["Integers", "Numeric", "Dates"]], how="cross"
    )
    expected = expected.loc[
        expected.A.ne(expected.Integers)
        & expected.B.lt(expected.Numeric)
        & expected.E.ge(expected.Dates),
    ].sort_values(filters, ignore_index=True)

    actual = (
        df[["A", "B", "E"]]
        .conditional_join(
            right[["Integers", "Numeric", "Dates"]],
            ("E", "Dates", ">="),
            ("A", "Integers", "!="),
            ("B", "Numeric", "<"),
            how="inner",
            sort_by_appearance=False,
        )
        .sort_values(filters, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
@pytest.mark.turtle
def test_ge_le_ne_extension_array_numba(df, right):
    """
    Test output for multiple conditions.
    """

    filters = ["A", "B", "E", "Integers", "Numeric", "Dates"]
    df = df.assign(A=df["A"].astype("Int64"))
    right = right.assign(Integers=right["Integers"].astype(pd.Int64Dtype()))

    expected = df[["A", "B", "E"]].merge(
        right[["Integers", "Numeric", "Dates"]], how="cross"
    )
    expected = expected.loc[
        expected.A.ne(expected.Integers)
        & expected.B.lt(expected.Numeric)
        & expected.E.ge(expected.Dates),
    ].sort_values(filters, ignore_index=True)

    actual = (
        df[["A", "B", "E"]]
        .conditional_join(
            right[["Integers", "Numeric", "Dates"]],
            ("E", "Dates", ">="),
            ("A", "Integers", "!="),
            ("B", "Numeric", "<"),
            how="inner",
            use_numba=True,
            sort_by_appearance=False,
        )
        .sort_values(filters, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_ge_lt_ne_extension(df, right):
    """
    Test output for multiple conditions.
    """

    filters = ["A", "B", "E", "Integers", "Numeric", "Dates", "Dates_Right"]
    df = df.assign(A=df["A"].astype("Int64"))
    right = right.assign(Integers=right["Integers"].astype(pd.Int64Dtype()))

    expected = df[["A", "B", "E"]].merge(
        right[["Integers", "Numeric", "Dates", "Dates_Right"]], how="cross"
    )
    expected = expected.loc[
        expected.A.lt(expected.Integers)
        & expected.B.ne(expected.Numeric)
        & expected.E.ge(expected.Dates)
        & expected.E.ne(expected.Dates_Right),
    ].sort_values(filters, ignore_index=True)

    actual = (
        df[["A", "B", "E"]]
        .conditional_join(
            right[["Integers", "Numeric", "Dates", "Dates_Right"]],
            ("E", "Dates", ">="),
            ("B", "Numeric", "!="),
            ("A", "Integers", "<"),
            ("E", "Dates_Right", "!="),
            how="inner",
            sort_by_appearance=False,
        )
        .sort_values(filters, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_ge_lt_ne_numba_extension(df, right):
    """
    Test output for multiple conditions.
    """

    filters = ["A", "B", "E", "Integers", "Numeric", "Dates", "Dates_Right"]
    df = df.assign(A=df["A"].astype("Int64"))
    right = right.assign(Integers=right["Integers"].astype(pd.Int64Dtype()))

    expected = df[["A", "B", "E"]].merge(
        right[["Integers", "Numeric", "Dates", "Dates_Right"]], how="cross"
    )
    expected = expected.loc[
        expected.A.lt(expected.Integers)
        & expected.B.ne(expected.Numeric)
        & expected.E.ge(expected.Dates)
        & expected.E.ne(expected.Dates_Right),
    ].sort_values(filters, ignore_index=True)

    actual = (
        df[["A", "B", "E"]]
        .conditional_join(
            right[["Integers", "Numeric", "Dates", "Dates_Right"]],
            ("E", "Dates", ">="),
            ("B", "Numeric", "!="),
            ("A", "Integers", "<"),
            ("E", "Dates_Right", "!="),
            how="inner",
            use_numba=True,
            sort_by_appearance=False,
        )
        .sort_values(filters, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_eq_ge_and_le_numbers(df, right):
    """Test output for multiple conditions."""

    columns = ["B", "A", "E", "Floats", "Integers", "Dates"]
    expected = (
        df.merge(
            right, left_on="B", right_on="Floats", how="inner", sort=False
        )
        .loc[lambda df: df.A.ge(df.Integers) & df.E.le(df.Dates), columns]
        .sort_values(columns, ignore_index=True)
    )

    actual = (
        df[["B", "A", "E"]]
        .conditional_join(
            right[["Floats", "Integers", "Dates"]],
            ("B", "Floats", "=="),
            ("A", "Integers", ">="),
            ("E", "Dates", "<="),
            how="inner",
            sort_by_appearance=False,
        )
        .sort_values(columns, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_dual_ge_and_le_diff_numbers_numba(df, right):
    """Test output for multiple conditions."""

    columns = ["A", "E", "Integers", "Dates"]
    expected = (
        df.merge(
            right,
            how="cross",
        )
        .loc[lambda df: df.A.le(df.Integers) & df.E.gt(df.Dates), columns]
        .sort_values(columns, ignore_index=True)
    )

    actual = (
        df[["A", "E"]]
        .conditional_join(
            right[["Integers", "Dates"]],
            ("A", "Integers", "<="),
            ("E", "Dates", ">"),
            how="inner",
            use_numba=True,
            sort_by_appearance=False,
        )
        .sort_values(columns, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_dual_ge_and_le_diff_numbers(df, right):
    """Test output for multiple conditions."""

    columns = ["A", "E", "Integers", "Dates"]
    expected = (
        df.merge(
            right,
            how="cross",
        )
        .loc[lambda df: df.A.le(df.Integers) & df.E.gt(df.Dates), columns]
        .sort_values(columns, ignore_index=True)
    )

    actual = (
        df[["A", "E"]]
        .conditional_join(
            right[["Integers", "Dates"]],
            ("A", "Integers", "<="),
            ("E", "Dates", ">"),
            how="inner",
            sort_by_appearance=False,
        )
        .sort_values(columns, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_ge_lt_ne_extension_variant(df, right):
    """
    Test output for multiple conditions.
    """

    filters = ["A", "Integers", "B", "Numeric", "E", "Dates", "Dates_Right"]
    df = df.assign(A=df["A"].astype("Int64"))
    right = right.assign(Integers=right["Integers"].astype(pd.Int64Dtype()))

    expected = df.merge(right, how="cross")
    expected = expected.loc[
        expected.A.ne(expected.Integers)
        & expected.B.lt(expected.Numeric)
        & expected.E.ge(expected.Dates)
        & expected.E.ne(expected.Dates_Right),
        filters,
    ].sort_values(filters, ignore_index=True)

    actual = (
        df.conditional_join(
            right,
            ("E", "Dates", ">="),
            ("B", "Numeric", "<"),
            ("A", "Integers", "!="),
            ("E", "Dates_Right", "!="),
            how="inner",
            sort_by_appearance=False,
        )
        .filter(filters)
        .sort_values(filters, ignore_index=True)
    )
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_ge_lt_ne_extension_variant_numba(df, right):
    """
    Test output for multiple conditions.
    """

    filters = ["A", "Integers", "B", "Numeric", "E", "Dates", "Dates_Right"]
    df = df.assign(A=df["A"].astype("Int64"))
    right = right.assign(Integers=right["Integers"].astype(pd.Int64Dtype()))

    expected = df.merge(right, how="cross")
    expected = expected.loc[
        expected.A.ne(expected.Integers)
        & expected.B.lt(expected.Numeric)
        & expected.E.ge(expected.Dates)
        & expected.E.ne(expected.Dates_Right),
        filters,
    ].sort_values(filters, ignore_index=True)

    actual = (
        df.conditional_join(
            right,
            ("E", "Dates", ">="),
            ("B", "Numeric", "<"),
            ("A", "Integers", "!="),
            ("E", "Dates_Right", "!="),
            how="inner",
            use_numba=True,
            sort_by_appearance=False,
        )
        .filter(filters)
        .sort_values(filters, ignore_index=True)
    )
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_ge_eq_and_le_numbers_variant(df, right):
    """Test output for multiple conditions."""

    columns = ["B", "A", "E", "Floats", "Integers", "Dates"]
    expected = (
        df.merge(
            right, left_on="B", right_on="Floats", how="inner", sort=False
        )
        .loc[lambda df: df.A.ge(df.Integers) & df.E.le(df.Dates), columns]
        .sort_values(columns, ignore_index=True)
    )
    expected = expected.filter(columns)
    actual = (
        df[["B", "A", "E"]]
        .conditional_join(
            right[["Floats", "Integers", "Dates"]],
            ("A", "Integers", ">="),
            ("E", "Dates", "<="),
            ("B", "Floats", "=="),
            how="inner",
            sort_by_appearance=True,
        )
        .sort_values(columns, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
@pytest.mark.turtle
def test_multiple_eqs_variant(df, right):
    """Test output for multiple conditions."""

    columns = ["B", "A", "E", "Floats", "Integers", "Dates"]
    expected = (
        df.merge(
            right,
            left_on=["B", "A"],
            right_on=["Floats", "Integers"],
            how="inner",
            sort=False,
        )
        .loc[lambda df: df.E.ne(df.Dates), columns]
        .sort_values(columns, ignore_index=True)
    )

    actual = (
        df[["B", "A", "E"]]
        .conditional_join(
            right[["Floats", "Integers", "Dates"]],
            ("E", "Dates", "!="),
            ("B", "Floats", "=="),
            ("A", "Integers", "=="),
            how="inner",
            sort_by_appearance=False,
        )
        .sort_values(columns, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
@pytest.mark.turtle
def test_multiple_eqs_variant_numba(df, right):
    """Test output for multiple conditions."""

    columns = ["B", "A", "E", "Floats", "Integers", "Dates"]
    expected = (
        df.merge(
            right,
            left_on=["B", "A"],
            right_on=["Floats", "Integers"],
            how="inner",
            sort=False,
        )
        .loc[lambda df: df.E.ne(df.Dates), columns]
        .sort_values(columns, ignore_index=True)
    )

    actual = (
        df[["B", "A", "E"]]
        .conditional_join(
            right[["Floats", "Integers", "Dates"]],
            ("E", "Dates", "!="),
            ("B", "Floats", "=="),
            ("A", "Integers", "=="),
            how="inner",
            use_numba=True,
            sort_by_appearance=False,
        )
        .sort_values(columns, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_dual_ge_and_le_range_numbers(df, right):
    """Test output for multiple conditions."""

    columns = ["A", "E", "Integers", "Dates"]
    expected = (
        df.merge(
            right,
            how="cross",
        )
        .loc[lambda df: df.A.ge(df.Integers) & df.E.lt(df.Dates), columns]
        .sort_values(columns, ignore_index=True)
    )

    actual = (
        df[["A", "E"]]
        .conditional_join(
            right[["Integers", "Dates"]],
            ("E", "Dates", "<"),
            ("A", "Integers", ">="),
            how="inner",
            sort_by_appearance=False,
        )
        .sort_values(columns, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_dual_ge_and_le_range_numbers_numba(df, right):
    """Test output for multiple conditions."""

    columns = ["A", "E", "Integers", "Dates"]
    expected = (
        df.merge(
            right,
            how="cross",
        )
        .loc[lambda df: df.A.ge(df.Integers) & df.E.lt(df.Dates), columns]
        .sort_values(columns, ignore_index=True)
    )

    actual = (
        df[["A", "E"]]
        .conditional_join(
            right[["Integers", "Dates"]],
            ("E", "Dates", "<"),
            ("A", "Integers", ">="),
            how="inner",
            use_numba=True,
            sort_by_appearance=False,
        )
        .sort_values(columns, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_dual_ge_and_le_range_numbers_df_columns_only(df, right):
    """Test output for multiple conditions and select df only."""

    columns = ["A", "E"]
    expected = (
        df.merge(
            right,
            how="cross",
        )
        .loc[lambda df: df.A.ge(df.Integers) & df.E.lt(df.Dates), columns]
        .sort_values(columns, ignore_index=True)
    )

    actual = (
        df[["A", "E"]]
        .conditional_join(
            right[["Integers", "Dates"]],
            ("E", "Dates", "<"),
            ("A", "Integers", ">="),
            how="inner",
            use_numba=False,
            right_columns=None,
            sort_by_appearance=False,
        )
        .sort_values(columns, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_dual_ge_and_le_range_numbers_right_only(df, right):
    """Test output for multiple conditions and select right only."""

    columns = ["Integers", "Dates"]
    expected = (
        df.merge(
            right,
            how="cross",
        )
        .loc[lambda df: df.A.ge(df.Integers) & df.E.lt(df.Dates), columns]
        .sort_values(columns, ignore_index=True)
    )

    actual = (
        df[["A", "E"]]
        .conditional_join(
            right[["Integers", "Dates"]],
            ("E", "Dates", "<"),
            ("A", "Integers", ">="),
            how="inner",
            df_columns=None,
            sort_by_appearance=False,
        )
        .sort_values(columns, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
@pytest.mark.turtle
def test_ge_eq_and_le_numbers(df, right):
    """Test output for multiple conditions."""

    columns = ["B", "A", "E", "Floats", "Integers", "Dates"]
    expected = (
        df.merge(
            right, left_on="B", right_on="Floats", how="inner", sort=False
        )
        .loc[lambda df: df.A.ge(df.Integers) & df.E.le(df.Dates), columns]
        .sort_values(columns, ignore_index=True)
    )

    actual = (
        df[["B", "A", "E"]]
        .conditional_join(
            right[["Floats", "Integers", "Dates"]],
            ("A", "Integers", ">="),
            ("E", "Dates", "<="),
            ("B", "Floats", "=="),
            how="inner",
            sort_by_appearance=False,
        )
        .sort_values(columns, ignore_index=True)
    )
    actual = actual.filter(columns)
    assert_frame_equal(expected, actual)


@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
@pytest.mark.turtle
def test_ge_eq_and_le_numbers_numba(df, right):
    """Test output for multiple conditions."""

    columns = ["B", "A", "E", "Floats", "Integers", "Dates"]
    expected = (
        df.merge(
            right, left_on="B", right_on="Floats", how="inner", sort=False
        )
        .loc[lambda df: df.A.ge(df.Integers) & df.E.le(df.Dates), columns]
        .sort_values(columns, ignore_index=True)
    )

    actual = (
        df[["B", "A", "E"]]
        .conditional_join(
            right[["Floats", "Integers", "Dates"]],
            ("A", "Integers", ">="),
            ("E", "Dates", "<="),
            ("B", "Floats", "=="),
            how="inner",
            use_numba=True,
            sort_by_appearance=False,
        )
        .sort_values(columns, ignore_index=True)
    )
    actual = actual.filter(columns)
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_multiple_non_equi(df, right):
    """Test output for multiple conditions."""

    columns = ["B", "A", "E", "Floats", "Integers", "Dates"]
    expected = (
        df.merge(
            right,
            how="cross",
        )
        .loc[
            lambda df: df.A.ge(df.Integers)
            & df.E.le(df.Dates)
            & df.B.lt(df.Floats),
            columns,
        ]
        .sort_values(columns, ignore_index=True)
    )

    actual = (
        df[["B", "A", "E"]]
        .conditional_join(
            right[["Floats", "Integers", "Dates"]],
            ("A", "Integers", ">="),
            ("E", "Dates", "<="),
            ("B", "Floats", "<"),
            how="inner",
            sort_by_appearance=False,
        )
        .sort_values(columns, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
@pytest.mark.turtle
def test_multiple_non_equii(df, right):
    """Test output for multiple conditions."""

    columns = ["B", "A", "E", "Floats", "Integers", "Dates", "Numeric"]
    expected = (
        df.merge(
            right,
            how="cross",
        )
        .loc[
            lambda df: df.A.ge(df.Integers)
            & df.E.le(df.Dates)
            & df.B.lt(df.Floats)
            & df.B.gt(df.Numeric)
        ]
        .sort_values(columns, ignore_index=True)
    )
    expected = expected.filter(columns)
    actual = (
        df[["B", "A", "E"]]
        .conditional_join(
            right[["Floats", "Integers", "Dates", "Numeric"]],
            ("A", "Integers", ">="),
            ("E", "Dates", "<="),
            ("B", "Floats", "<"),
            ("B", "Numeric", ">"),
            how="inner",
            sort_by_appearance=False,
        )
        .sort_values(columns, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_multiple_non_eqi(df, right):
    """Test output for multiple conditions."""

    columns = ["B", "A", "E", "Floats", "Integers", "Dates"]
    expected = (
        df.merge(
            right,
            how="cross",
        )
        .loc[
            lambda df: df.A.ge(df.Integers)
            & df.E.gt(df.Dates)
            & df.B.gt(df.Floats)
        ]
        .sort_values(columns, ignore_index=True)
        .filter(columns)
        .rename(columns={"B": "b", "Floats": "floats"})
    )

    actual = df.conditional_join(
        right,
        ("A", "Integers", ">="),
        ("E", "Dates", ">"),
        ("B", "Floats", ">"),
        how="inner",
        sort_by_appearance=False,
        df_columns={"B": "b", "A": "A", "E": "E"},
        right_columns={
            "Floats": "floats",
            "Integers": "Integers",
            "Dates": "Dates",
        },
    ).sort_values(
        ["b", "A", "E", "floats", "Integers", "Dates"], ignore_index=True
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_multiple_non_eqi_numba(df, right):
    """Test output for multiple conditions."""

    columns = ["B", "A", "E", "Floats", "Integers", "Dates"]
    expected = (
        df.merge(
            right,
            how="cross",
        )
        .loc[
            lambda df: df.A.ge(df.Integers)
            & df.E.gt(df.Dates)
            & df.B.gt(df.Floats)
        ]
        .sort_values(columns, ignore_index=True)
        .filter(columns)
        .rename(columns={"B": "b", "Floats": "floats"})
    )

    actual = df.conditional_join(
        right,
        ("A", "Integers", ">="),
        ("E", "Dates", ">"),
        ("B", "Floats", ">"),
        how="inner",
        use_numba=True,
        sort_by_appearance=False,
        df_columns={"B": "b", "A": "A", "E": "E"},
        right_columns={
            "Floats": "floats",
            "Integers": "Integers",
            "Dates": "Dates",
        },
    ).sort_values(
        ["b", "A", "E", "floats", "Integers", "Dates"], ignore_index=True
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_multiple_non_eq(df, right):
    """Test output for multiple conditions."""

    expected = (
        df[["B", "A", "E"]]
        .assign(index=df.index)
        .merge(
            right[["Floats", "Integers", "Dates"]],
            how="cross",
        )
        .loc[
            lambda df: df.B.le(df.Floats)
            & df.A.lt(df.Integers)
            & df.E.lt(df.Dates)
        ]
        .groupby("index")
        .head(1)
        .drop(columns="index")
        .reset_index(drop=True)
    )

    actual = df[["B", "A", "E"]].conditional_join(
        right[["Floats", "Integers", "Dates"]],
        ("B", "Floats", "<="),
        ("A", "Integers", "<"),
        ("E", "Dates", "<"),
        how="inner",
        keep="first",
        sort_by_appearance=False,
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_multiple_non_eq_numba(df, right):
    """Test output for multiple conditions."""

    expected = (
        df[["B", "A", "E"]]
        .assign(index=df.index)
        .merge(
            right[["Floats", "Integers", "Dates"]],
            how="cross",
        )
        .loc[
            lambda df: df.B.le(df.Floats)
            & df.A.lt(df.Integers)
            & df.E.lt(df.Dates)
        ]
        .groupby("index")
        .head(1)
        .drop(columns="index")
        .reset_index(drop=True)
        .sort_values(
            ["B", "A", "E", "Floats", "Integers", "Dates"], ignore_index=True
        )
    )

    actual = (
        df[["B", "A", "E"]]
        .conditional_join(
            right[["Floats", "Integers", "Dates"]],
            ("B", "Floats", "<="),
            ("A", "Integers", "<"),
            ("E", "Dates", "<"),
            how="inner",
            keep="first",
            use_numba=True,
            sort_by_appearance=False,
        )
        .sort_values(
            ["B", "A", "E", "Floats", "Integers", "Dates"], ignore_index=True
        )
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_multiple_eqs(df, right):
    """Test output for multiple conditions."""

    columns = ["B", "A", "E", "Floats", "Integers", "Dates"]
    expected = (
        df.merge(
            right,
            left_on=["B", "A"],
            right_on=["Floats", "Integers"],
            how="inner",
            sort=False,
        )
        .loc[lambda df: df.E.ne(df.Dates), columns]
        .sort_values(columns, ignore_index=True)
    )
    expected = expected.filter(columns)
    actual = (
        df[["B", "A", "E"]]
        .conditional_join(
            right[["Floats", "Integers", "Dates"]],
            ("E", "Dates", "!="),
            ("B", "Floats", "=="),
            ("A", "Integers", "=="),
            how="inner",
            sort_by_appearance=False,
        )
        .sort_values(columns, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None)
@given(df=conditional_df(), right=conditional_right())
def test_eq_strings(df, right):
    """Test output for joins on strings."""

    columns = ["C", "A", "Strings", "Integers"]
    expected = df.merge(
        right,
        left_on="C",
        right_on="Strings",
        how="inner",
        sort=False,
    )
    expected = expected.loc[
        expected.A >= expected.Integers, columns
    ].sort_values(columns, ignore_index=True)

    actual = df.conditional_join(
        right,
        ("C", "Strings", "=="),
        ("A", "Integers", ">="),
        how="inner",
        sort_by_appearance=False,
        df_columns=["C", "A"],
        right_columns=["Strings", "Integers"],
    ).sort_values(columns, ignore_index=True)

    assert_frame_equal(expected, actual)


def test_extension_array_eq():
    """Extension arrays when matching on equality."""
    df1 = pd.DataFrame(
        {"id": [1, 1, 1, 2, 2, 3], "value_1": [2, 5, 7, 1, 3, 4]}
    )
    df1 = df1.astype({"value_1": "Int64"})
    df2 = pd.DataFrame(
        {
            "id": [1, 1, 1, 1, 2, 2, 2, 3],
            "value_2A": [0, 3, 7, 12, 0, 2, 3, 1],
            "value_2B": [1, 5, 9, 15, 1, 4, 6, 3],
        }
    )
    df2 = df2.astype({"value_2A": "Int64"})
    expected = df1.conditional_join(
        df2,
        ("id", "id", "=="),
        ("value_1", "value_2A", ">"),
        use_numba=False,
        sort_by_appearance=False,
    )
    expected = (
        expected.drop(columns=("right", "id"))
        .droplevel(axis=1, level=0)
        .sort_values(["id", "value_1", "value_2A"], ignore_index=True)
    )
    actual = (
        df1.merge(df2, on="id")
        .loc[lambda df: df.value_1.gt(df.value_2A)]
        .sort_values(["id", "value_1", "value_2A"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


def test_extension_array_eq_range():
    """Extension arrays when matching on equality."""
    df1 = pd.DataFrame(
        {"id": [1, 1, 1, 2, 2, 3], "value_1": [2, 5, 7, 1, 3, 4]}
    )
    df1 = df1.astype({"value_1": "Int64"})
    df2 = pd.DataFrame(
        {
            "id": [1, 1, 1, 1, 2, 2, 2, 3],
            "value_2A": [0, 3, 7, 12, 0, 2, 3, 1],
            "value_2B": [1, 5, 9, 15, 1, 4, 6, 3],
        }
    )
    df2 = df2.astype({"value_2A": "Int64", "value_2B": "Int64"})
    expected = df1.conditional_join(
        df2,
        ("id", "id", "=="),
        ("value_1", "value_2A", ">"),
        ("value_1", "value_2B", "<"),
        sort_by_appearance=True,
    )
    expected = expected.drop(columns=("right", "id")).droplevel(
        axis=1, level=0
    )
    actual = (
        df1.merge(df2, on="id")
        .loc[
            lambda df: df.value_1.gt(df.value_2A) & df.value_1.lt(df.value_2B)
        ]
        .reset_index(drop=True)
    )

    assert_frame_equal(expected, actual)


def test_left_empty():
    """Test nulls for equality merge."""
    df1 = pd.DataFrame({"A": [np.nan, np.nan], "B": [2, 3]})
    df2 = pd.DataFrame({"A": [2.0, 2.0], "B": [3, 2]})
    actual = (
        df1.merge(df2, on="A", sort=False)
        .loc[lambda df: df.B_x <= df.B_y]
        .reset_index(drop=True)
    )
    actual.columns = list("ABC")
    expected = df1.conditional_join(
        df2, ("A", "A", "=="), ("B", "B", "<=")
    ).drop(columns=("right", "A"))
    expected.columns = list("ABC")

    assert_frame_equal(expected, actual)


def test_right_empty():
    """Test nulls for equality merge."""
    df2 = pd.DataFrame({"A": [np.nan, np.nan], "B": [2, 3]})
    df1 = pd.DataFrame({"A": [2.0, 2.0], "B": [3, 2]})
    actual = (
        df1.merge(df2, on="A", sort=False)
        .loc[lambda df: df.B_x <= df.B_y]
        .reset_index(drop=True)
    )
    actual.columns = list("ABC")
    expected = df1.conditional_join(
        df2, ("A", "A", "=="), ("B", "B", "<=")
    ).drop(columns=("right", "A"))
    expected.columns = list("ABC")

    assert_frame_equal(expected, actual)


def test_no_match():
    """
    Test output for equality merge,
     where binary search is triggered,
     and there are no matches.
    """
    df1 = pd.DataFrame({"A": [1, 2, 2, 3], "B": range(0, 4)})
    df2 = pd.DataFrame({"A": [1, 2, 2, 3], "B": range(4, 8)})
    actual = (
        df1.merge(df2, on="A", sort=False)
        .loc[lambda df: df.B_x > df.B_y]
        .reset_index(drop=True)
    )
    actual.columns = list("ABC")
    expected = df1.conditional_join(
        df2, ("A", "A", "=="), ("B", "B", ">")
    ).drop(columns=("right", "A"))
    expected.columns = list("ABC")

    assert_frame_equal(expected, actual)
