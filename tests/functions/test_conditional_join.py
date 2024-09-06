import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from pandas import Timedelta
from pandas.testing import assert_frame_equal, assert_index_equal

from janitor import get_join_indices
from janitor.testing_utils.strategies import (
    conditional_df,
    conditional_right,
)

# # turn on to view dataframes from failed tests
pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("max_colwidth", None)


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


def test_conditional_join():
    """Execution test for conditional_join.

    This example is lifted directly from the conditional_join docstring.
    """
    df1 = pd.DataFrame({"value_1": [2, 5, 7, 1, 3, 4]})
    df2 = pd.DataFrame(
        {
            "value_2A": [0, 3, 7, 12, 0, 2, 3, 1],
            "value_2B": [1, 5, 9, 15, 1, 4, 6, 3],
        }
    )
    df1.conditional_join(
        df2, ("value_1", "value_2A", ">"), ("value_1", "value_2B", "<")
    )


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


def test_check_force_type(dummy, series):
    """
    Raise TypeError if `force` is not boolean.
    """
    with pytest.raises(TypeError, match="force should be one of.+"):
        dummy.conditional_join(series, ("id", "B", "<"), force=1)


def test_check_how_value(dummy, series):
    """
    Raise ValueError if `how` is not one of
    `inner`, `left`, or `right`, or `outer`.
    """
    with pytest.raises(ValueError, match="'how' should be one of.+"):
        dummy.conditional_join(series, ("id", "B", "<"), how="INNER")


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
    Raise TypeError if `use_numba` is not a boolean.
    """
    with pytest.raises(TypeError, match="use_numba should be one of.+"):
        dummy.conditional_join(series, ("id", "B", "<"), use_numba=1)


def test_check_use_numba_equi_join(dummy):
    """
    Raise TypeError if `use_numba` is True,
    there is an equi join,
    and the dtype is not a datetime or number.
    """
    with pytest.raises(
        TypeError, match="Only numeric, timedelta and datetime types.+"
    ):
        dummy.conditional_join(
            dummy, ("S", "S", "=="), ("id", "id", ">"), use_numba=True
        )


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_check_use_numba_equi_join_no_le_or_ge(df, right):
    """
    Raise ValueError if `use_numba` is True,
    there is an equi join,
    and there is no less than/greater than join.
    """
    with pytest.raises(
        ValueError, match="At least one less than or greater than.+"
    ):
        df.conditional_join(
            right,
            ("E", "Dates", "!="),
            ("A", "Integers", "=="),
            ("B", "Numeric", "!="),
            use_numba=True,
        )


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


def test_dtype_not_permitted(dummy, series):
    """
    Raise TypeError if dtype of column in `df`
    is not an acceptable type.
    """
    dummy["F"] = pd.IntervalIndex.from_tuples(
        [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60)]
    )
    match = "Only numeric, timedelta and datetime types "
    match = "are supported in a non equi-join, "
    match = "or if use_numba is set to True.+"
    with pytest.raises(TypeError, match=match):
        dummy.conditional_join(series, ("F", "B", "<"))


def test_dtype_str(dummy, series):
    """
    Raise TypeError if dtype of column in `df`
    does not match the dtype of column from `right`.
    """
    match = "Only numeric, timedelta and datetime types "
    match = "are supported in a non equi-join, "
    match = "or if use_numba is set to True.+"
    with pytest.raises(TypeError, match=match):
        dummy.conditional_join(series, ("S", "B", "<"))


def test_dtype_strings_non_equi(dummy):
    """
    Raise TypeError if the dtypes are both strings
    on a non-equi operator.
    """
    match = "Only numeric, timedelta and datetime types "
    match = "are supported in a non equi-join, "
    match = "or if use_numba is set to True.+"
    with pytest.raises(
        TypeError,
        match=match,
    ):
        dummy.conditional_join(
            dummy.rename(columns={"S": "Strings"}), ("S", "Strings", "<")
        )


def test_dtype_category_non_equi():
    """
    Raise TypeError if dtype is category,
    and op is non-equi.
    """
    match = "Only numeric, timedelta and datetime types "
    match = "are supported in a non equi-join, "
    match = "or if use_numba is set to True.+"
    with pytest.raises(TypeError, match=match):
        left = pd.DataFrame({"A": [1, 2, 3]}, dtype="category")
        right = pd.DataFrame({"B": [1, 2, 3]}, dtype="category")
        left.conditional_join(right, ("A", "B", "<"))


def test_dtype_different_non_equi():
    """
    Raise TypeError if dtype is different,
    and op is non-equi.
    """
    match = "Both columns should have the same type.+"
    with pytest.raises(TypeError, match=match):
        left = pd.DataFrame({"A": [1, 2, 3]}, dtype="int64")
        right = pd.DataFrame({"B": [1, 2, 3]}, dtype="int8")
        left.conditional_join(right, ("A", "B", "<"))


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
            keep="first",
        )
        .sort_values(["B", "Numeric"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_single_condition_greater_than_floats_keep_last(df, right):
    """Test output for a single condition. "<"."""

    df = df.sort_values("B").dropna(subset=["B"])
    expected = pd.merge_asof(
        df[["B"]],
        right[["Numeric"]].sort_values("Numeric").dropna(subset=["Numeric"]),
        left_on="B",
        right_on="Numeric",
        direction="backward",
        allow_exact_matches=False,
    )
    expected.index = range(len(expected))
    actual = (
        df[["B"]]
        .conditional_join(
            right[["Numeric"]].sort_values("Numeric"),
            ("B", "Numeric", ">"),
            how="left",
            keep="last",
        )
        .sort_values(["B", "Numeric"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_single_condition_greater_than_floats_keep_last_numba(df, right):
    """
    Test the functionality of conditional_join with a single
    'greater than' condition on floating-point data,
    while keeping the last match using Numba.

    This test sorts and filters dataframes 'df' and 'right'
    by columns 'B' and 'Numeric' respectively,
    removing NaN values.
    It then performs a backward merge_asof operation
    on these sorted dataframes.
    The expected outcome is a dataframe
    where each row from 'df' is merged
    with the last row from 'right'
      where 'Numeric' is greater than 'B'.

    The actual outcome is produced
    by the conditional_join method
    with a 'greater than' condition,
    left join type, sorted by appearance,
    keeping the last match,
    and utilizing Numba for performance optimization.
    The test asserts that the actual dataframe matches
    the expected dataframe,
    ensuring correct functionality of the conditional_join
    under these specific parameters.
    """

    df = df.sort_values("B").dropna(subset=["B"])
    expected = pd.merge_asof(
        df[["B"]],
        right[["Numeric"]].sort_values("Numeric").dropna(subset=["Numeric"]),
        left_on="B",
        right_on="Numeric",
        direction="backward",
        allow_exact_matches=False,
    )
    expected.index = range(len(expected))
    actual = (
        df[["B"]]
        .conditional_join(
            right[["Numeric"]].sort_values("Numeric"),
            ("B", "Numeric", ">"),
            how="left",
            keep="last",
            use_numba=True,
        )
        .sort_values(["B", "Numeric"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_single_condition_less_than_floats_keep_last(df, right):
    """
    Test the functionality of conditional_join
    with a single 'greater than' condition on floating-point data,
    while keeping the last match using Numba.

    This test sorts and filters dataframes 'df' and 'right'
    by columns 'B' and 'Numeric' respectively, removing NaN values.
    It then performs a backward merge_asof operation on these sorted dataframes.
    The expected outcome is a dataframe where each row from 'df'
    is merged with the last row from 'right' where 'Numeric' is greater than 'B'.

    The actual outcome is produced by the conditional_join method
    with a 'greater than' condition, left join type, sorted by appearance,
    keeping the last match, without utilizing Numba for performance optimization.
    The test asserts that the actual dataframe matches the expected dataframe,
    ensuring correct functionality of the conditional_join
    under these specific parameters.
    """

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
            keep="last",
        )
        .sort_values(["B", "Numeric"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
        )
        .sort_values(["B", "Numeric"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
            keep="first",
            use_numba=True,
        )
        .sort_values(["B", "Numeric"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@settings(deadline=None, max_examples=10)
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
            keep="last",
            use_numba=True,
        )
        .sort_values(
            ["B", "Numeric"], ascending=[True, False], ignore_index=True
        )
    )

    assert_frame_equal(expected, actual)


@settings(deadline=None, max_examples=10)
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
            use_numba=True,
        )
        .sort_values(["A", "Integers"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@settings(deadline=None, max_examples=10)
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
            use_numba=True,
        )
        .sort_values(["A", "Integers"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
        )
        .sort_values(["A", "Integers"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
            use_numba=True,
        )
        .sort_values(["A", "Integers"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
            use_numba=True,
        )
        .sort_values(["A", "Integers"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
        )
        .sort_values(["E", "Dates"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
        )
        .sort_values(["E", "Dates"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
            use_numba=True,
        )
        .sort_values(["E", "Dates"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
        )
        .sort_values(["E", "Dates"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
        )
        .sort_values(["E", "Dates"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
            use_numba=True,
        )
        .sort_values(["A", "Integers"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
        )
        .sort_values(["B", "Numeric"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
            use_numba=True,
        )
        .sort_values(["B", "Numeric"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
        )
        .sort_values(["A", "Integers"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
            use_numba=True,
        )
        .sort_values(["A", "Integers"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
        )
        .sort_values(["A", "Integers"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
            use_numba=True,
        )
        .sort_values(["A", "Integers"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
            use_numba=True,
        )
        .sort_values(["B", "Numeric"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
        )
        .sort_values(["E", "Dates"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
@settings(deadline=None, max_examples=10)
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
        .select("A", "A_y", "_merge", axis="columns")
        .sort_values(["A", "A_y"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
        )
        .sort_values(["A", "Integers"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@settings(deadline=None, max_examples=10)
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
        .sort_index(axis="columns")
        .reset_index(drop=True)
    )
    actual = (
        df[["E"]]
        .conditional_join(
            right[["Dates"]], ("E", "Dates", ">"), how="right", indicator=True
        )
        .sort_values(["E", "Dates"], ignore_index=True)
        .sort_index(axis="columns")
    )

    assert_frame_equal(expected, actual)


@settings(deadline=None, max_examples=10)
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
        .sort_index(axis="columns")
        .reset_index(drop=True)
    )
    actual = (
        df[["E"]]
        .conditional_join(
            right[["Dates"]],
            ("E", "Dates", ">"),
            how="right",
            indicator=True,
        )
        .sort_values(["E", "Dates"], ignore_index=True)
        .sort_index(axis="columns")
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
        )
        .sort_values(["E", "Dates", "Dates_Right"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_dual_conditions_gt_and_lt_dates_numba(df, right):
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
            use_numba=True,
        )
        .sort_values(["E", "Dates", "Dates_Right"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@settings(deadline=None, max_examples=10)
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
        )
        .sort_values(["E", "Dates", "Dates_Right"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
@pytest.mark.turtle
def test_dual_conditions_ge_and_le_dates_numba(df, right):
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
            use_numba=True,
        )
        .sort_values(["E", "Dates", "Dates_Right"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@settings(deadline=None, max_examples=10)
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
        )
        .sort_values(["E", "Dates", "Dates_Right"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
@pytest.mark.turtle
def test_dual_conditions_le_and_ge_dates_numba(df, right):
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
            use_numba=True,
        )
        .sort_values(["E", "Dates", "Dates_Right"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
        )
        .sort_values(["E", "Dates", "Dates_Right"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_dual_conditions_ge_and_le_dates_right_open_numba(df, right):
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
            use_numba=True,
        )
        .sort_values(["E", "Dates", "Dates_Right"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
        )
        .sort_values(["B", "Numeric", "Floats"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_dual_conditions_ge_and_le_numbers_numba(df, right):
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
            use_numba=True,
        )
        .sort_values(["B", "Numeric", "Floats"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@settings(deadline=None, max_examples=10)
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
        )
        .sort_values(["B", "Numeric", "Floats"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
@pytest.mark.turtle
def test_dual_conditions_le_and_ge_numbers_numba(df, right):
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
            use_numba=True,
        )
        .sort_values(["B", "Numeric", "Floats"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
        )
        .sort_values(["B", "Numeric", "Floats"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@settings(deadline=None, max_examples=10)
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
        )
        .sort_values(["B", "Numeric", "Floats"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
        )
        .sort_values(["Numeric", "Floats", "B"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_dual_conditions_gt_and_lt_numbers_numba_(df, right):
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
            use_numba=True,
        )
        .sort_values(["Numeric", "Floats", "B"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@settings(deadline=None, max_examples=10)
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
            indicator=True,
        )
        .sort_values(["B", "Numeric", "Floats"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
        .sort_index(axis="columns")
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
        .sort_index(axis="columns")
    )
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=2)
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
        )
        .sort_values(filters, ignore_index=True)
        .loc[:, filters]
    )
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
        )
        .sort_values(filters, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
        )
        .filter(filters)
        .sort_values(filters, ignore_index=True)
        .loc[:, filters]
    )
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
            indicator=True,
        )
        .sort_values(filters, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
        )
        .sort_values(filters, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@settings(deadline=None, max_examples=10)
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
        )
        .sort_values(filters, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
        )
        .sort_values(columns, ignore_index=True)
        .loc[:, columns]
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
        )
        .sort_values(filters, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@settings(deadline=None, max_examples=10)
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
        )
        .sort_values(filters, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@settings(deadline=None, max_examples=10)
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
        )
        .sort_values(filters, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
        )
        .sort_values(filters, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
        )
        .sort_values(filters, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
        )
        .sort_values(filters, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
        )
        .sort_values(filters, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@settings(deadline=None, max_examples=10)
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
        )
        .sort_values(filters, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@settings(deadline=None, max_examples=10)
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
        )
        .sort_values(filters, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@settings(deadline=None, max_examples=10)
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
        )
        .sort_values(filters, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
        )
        .sort_values(filters, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
        )
        .sort_values(filters, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
        )
        .sort_values(columns, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
        )
        .sort_values(columns, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
        )
        .sort_values(columns, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
        )
        .sort_values(filters, ignore_index=True)
        .loc[:, filters]
    )
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
        )
        .sort_values(filters, ignore_index=True)
        .loc[:, filters]
    )
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
        )
        .sort_values(columns, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_multiple_ge_eq_and_le_numbers(df, right):
    """Test output for multiple conditions."""

    columns = ["B", "A", "E", "Floats", "Integers", "Dates", "Numeric"]
    expected = (
        df.merge(
            right, left_on="B", right_on="Floats", how="inner", sort=False
        )
        .loc[
            lambda df: df.A.ge(df.Integers)
            & df.E.le(df.Dates)
            & df.B.gt(df.Numeric),
            columns,
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
            ("B", "Floats", "=="),
            ("B", "Numeric", ">"),
            how="inner",
        )
        .sort_values(columns, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_ge_eq_and_multiple_le_numbers(df, right):
    """Test output for multiple conditions."""

    columns = ["B", "A", "E", "Floats", "Integers", "Dates", "Numeric"]
    expected = (
        df.merge(
            right, left_on="B", right_on="Floats", how="inner", sort=False
        )
        .loc[
            lambda df: df.A.ge(df.Integers)
            & df.E.le(df.Dates)
            & df.B.lt(df.Numeric),
            columns,
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
            ("B", "Floats", "=="),
            ("B", "Numeric", "<"),
            how="inner",
        )
        .sort_values(columns, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@settings(deadline=None, max_examples=10)
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
        )
        .sort_values(columns, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_dual_ge_and_le_range_numbers(df, right):
    """Test output for multiple conditions."""

    columns = ["A", "E", "Integers", "Dates_Right"]
    expected = (
        df.merge(
            right,
            how="cross",
        )
        .loc[
            lambda df: df.A.ge(df.Integers) & df.E.lt(df.Dates_Right), columns
        ]
        .sort_values(columns, ignore_index=True)
    )

    actual = (
        df[["A", "E"]]
        .conditional_join(
            right[["Integers", "Dates_Right"]],
            ("E", "Dates_Right", "<"),
            ("A", "Integers", ">="),
            how="inner",
        )
        .sort_values(columns, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_dual_ge_and_le_range_numbers_numba(df, right):
    """Test output for multiple conditions."""

    columns = ["A", "E", "Integers", "Dates_Right"]
    expected = (
        df.merge(
            right,
            how="cross",
        )
        .loc[
            lambda df: df.A.ge(df.Integers) & df.E.lt(df.Dates_Right), columns
        ]
        .sort_values(columns, ignore_index=True)
    )

    actual = (
        df[["A", "E"]]
        .conditional_join(
            right[["Integers", "Dates_Right"]],
            ("E", "Dates_Right", "<"),
            ("A", "Integers", ">="),
            how="inner",
            use_numba=True,
        )
        .sort_values(columns, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
        )
        .sort_values(columns, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
        )
        .sort_values(columns, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@settings(deadline=None, max_examples=10)
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
        )
        .sort_values(columns, ignore_index=True)
    )
    actual = actual.filter(columns)
    assert_frame_equal(expected, actual)


@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
@pytest.mark.turtle
def test_ge_eq_and_le_numbers_force(df, right):
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
            force=True,
        )
        .sort_values(columns, ignore_index=True)
    )
    actual = actual.filter(columns)
    assert_frame_equal(expected, actual)


@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
@pytest.mark.turtle
def test_ge_eq_and_le_numbers_variant_numba(df, right):
    """Test output for multiple conditions."""

    columns = ["B", "A", "E", "Floats", "Integers", "Dates"]
    expected = (
        df.merge(
            right, left_on="B", right_on="Floats", how="inner", sort=False
        )
        .loc[lambda df: df.A.lt(df.Integers) & df.E.gt(df.Dates), columns]
        .sort_values(columns, ignore_index=True)
    )

    actual = (
        df[["B", "A", "E"]]
        .conditional_join(
            right[["Floats", "Integers", "Dates"]],
            ("A", "Integers", "<"),
            ("E", "Dates", ">"),
            ("B", "Floats", "=="),
            how="inner",
        )
        .sort_values(columns, ignore_index=True)
    )
    actual = actual.filter(columns)
    assert_frame_equal(expected, actual)


@settings(deadline=None, max_examples=10)
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
        )
        .sort_values(columns, ignore_index=True)
    )
    actual = actual.filter(columns)
    assert_frame_equal(expected, actual)


@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
@pytest.mark.turtle
def test_ge_eq_and_le_integers_numba(df, right):
    """Test output for multiple conditions."""

    columns = ["B", "A", "E", "Floats", "Integers", "Dates"]
    expected = (
        df.merge(
            right, left_on="A", right_on="Integers", how="inner", sort=False
        )
        .loc[lambda df: df.B.ge(df.Floats) & df.E.le(df.Dates), columns]
        .sort_values(columns, ignore_index=True)
    )

    actual = (
        df[["B", "A", "E"]]
        .conditional_join(
            right[["Floats", "Integers", "Dates"]],
            ("A", "Integers", "=="),
            ("E", "Dates", "<="),
            ("B", "Floats", ">="),
            how="inner",
            use_numba=True,
        )
        .sort_values(columns, ignore_index=True)
    )
    actual = actual.filter(columns)
    assert_frame_equal(expected, actual)


@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
@pytest.mark.turtle
def test_ge_eq_and_lt_integers_numba(df, right):
    """Test output for multiple conditions."""

    columns = ["B", "A", "E", "Floats", "Integers", "Dates"]
    expected = (
        df.merge(
            right, left_on="A", right_on="Integers", how="inner", sort=False
        )
        .loc[lambda df: df.B.lt(df.Floats) & df.E.ge(df.Dates), columns]
        .sort_values(columns, ignore_index=True)
    )

    actual = (
        df[["B", "A", "E"]]
        .conditional_join(
            right[["Floats", "Integers", "Dates"]],
            ("A", "Integers", "=="),
            ("E", "Dates", ">="),
            ("B", "Floats", "<"),
            how="inner",
            use_numba=True,
        )
        .sort_values(columns, ignore_index=True)
    )
    actual = actual.filter(columns)
    assert_frame_equal(expected, actual)


@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
@pytest.mark.turtle
def test_gt_eq_integers_numba(df, right):
    """Test output for multiple conditions."""

    columns = ["A", "E", "Integers", "Dates"]
    expected = (
        df.merge(
            right, left_on="A", right_on="Integers", how="inner", sort=False
        )
        .loc[lambda df: df.E.gt(df.Dates), columns]
        .sort_values(columns, ignore_index=True)
    )

    actual = (
        df[["A", "E"]]
        .conditional_join(
            right[["Integers", "Dates"]],
            ("A", "Integers", "=="),
            ("E", "Dates", ">"),
            how="inner",
            use_numba=True,
        )
        .sort_values(columns, ignore_index=True)
    )
    actual = actual.filter(columns)
    assert_frame_equal(expected, actual)


@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
@pytest.mark.turtle
def test_gt_eq_dates_numba(df, right):
    """Test output for multiple conditions."""

    columns = ["A", "E", "Integers", "Dates"]
    expected = (
        df.dropna(subset=["E"])
        .merge(
            right.dropna(subset=["Dates"]),
            left_on="E",
            right_on="Dates",
            how="inner",
            sort=False,
        )
        .loc[lambda df: df.A.gt(df.Integers), columns]
        .sort_values(columns, ignore_index=True)
    )

    actual = (
        df[["A", "E"]]
        .conditional_join(
            right[["Integers", "Dates"]],
            ("A", "Integers", ">"),
            ("E", "Dates", "=="),
            how="inner",
            use_numba=True,
        )
        .sort_values(columns, ignore_index=True)
    )
    actual = actual.filter(columns)
    assert_frame_equal(expected, actual)


@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
@pytest.mark.turtle
def test_lt_eq_integers_numba(df, right):
    """Test output for multiple conditions."""

    columns = ["A", "E", "Integers", "Dates"]
    expected = (
        df.merge(
            right, left_on="A", right_on="Integers", how="inner", sort=False
        )
        .loc[lambda df: df.E.lt(df.Dates), columns]
        .sort_values(columns, ignore_index=True)
    )

    actual = (
        df[["A", "E"]]
        .conditional_join(
            right[["Integers", "Dates"]],
            ("A", "Integers", "=="),
            ("E", "Dates", "<"),
            how="inner",
            use_numba=True,
        )
        .sort_values(columns, ignore_index=True)
    )
    actual = actual.filter(columns)
    assert_frame_equal(expected, actual)


@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
@pytest.mark.turtle
def test_lt_eq_dates_numba(df, right):
    """Test output for multiple conditions."""

    columns = ["A", "E", "Integers", "Dates"]
    expected = (
        df.dropna(subset=["E"])
        .merge(
            right.dropna(subset=["Dates"]),
            left_on="E",
            right_on="Dates",
            how="inner",
            sort=False,
        )
        .loc[lambda df: df.A.lt(df.Integers), columns]
        .sort_values(columns, ignore_index=True)
    )

    actual = (
        df[["A", "E"]]
        .conditional_join(
            right[["Integers", "Dates"]],
            ("A", "Integers", "<"),
            ("E", "Dates", "=="),
            how="inner",
            use_numba=True,
        )
        .sort_values(columns, ignore_index=True)
    )
    actual = actual.filter(columns)
    assert_frame_equal(expected, actual)


@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
@pytest.mark.turtle
def test_ge_eq_and_le_dates_numba(df, right):
    """Test output for multiple conditions."""

    columns = ["B", "A", "E", "Floats", "Integers", "Dates"]
    expected = (
        df.dropna(subset=["E"])
        .merge(
            right.dropna(subset=["Dates"]),
            left_on="E",
            right_on="Dates",
            how="inner",
            sort=False,
        )
        .loc[lambda df: df.B.gt(df.Floats) & df.A.lt(df.Integers), columns]
        .sort_values(columns, ignore_index=True)
    )

    actual = (
        df[["B", "A", "E"]]
        .conditional_join(
            right[["Floats", "Integers", "Dates"]],
            ("A", "Integers", "<"),
            ("E", "Dates", "=="),
            ("B", "Floats", ">"),
            how="inner",
            use_numba=True,
        )
        .sort_values(columns, ignore_index=True)
    )
    actual = actual.filter(columns)
    assert_frame_equal(expected, actual)


@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
@pytest.mark.turtle
def test_ge_eq_and_le_datess_numba(df, right):
    """Test output for multiple conditions."""

    columns = ["B", "A", "E", "Floats", "Integers", "Dates", "Numeric"]
    expected = (
        df.dropna(subset=["E"])
        .merge(
            right.dropna(subset=["Dates"]),
            left_on="E",
            right_on="Dates",
            how="inner",
            sort=False,
        )
        .loc[
            lambda df: df.B.gt(df.Floats)
            & df.A.lt(df.Integers)
            & df.B.ne(df.Numeric),
            columns,
        ]
        .sort_values(columns, ignore_index=True)
    )

    actual = (
        df[["B", "A", "E"]]
        .conditional_join(
            right[["Floats", "Integers", "Dates", "Numeric"]],
            ("A", "Integers", "<"),
            ("E", "Dates", "=="),
            ("B", "Floats", ">"),
            ("B", "Numeric", "!="),
            how="inner",
            use_numba=True,
        )
        .sort_values(columns, ignore_index=True)
    )
    actual = actual.filter(columns)
    assert_frame_equal(expected, actual)


@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
@pytest.mark.turtle
def test_ge_eq_and_le_datess_numba_indices(df, right):
    """compare join indices for multiple conditions."""

    expected = (
        df.reset_index()
        .dropna(subset=["E"])
        .merge(
            right.dropna(subset=["Dates"]),
            left_on="E",
            right_on="Dates",
            how="inner",
            sort=False,
        )
        .loc[
            lambda df: df.B.gt(df.Floats)
            & df.A.lt(df.Integers)
            & df.B.ne(df.Numeric),
            "index",
        ]
    )
    expected = pd.Index(expected)

    actual, _ = get_join_indices(
        df[["B", "A", "E"]].dropna(subset=["E"]),
        right[["Floats", "Integers", "Dates", "Numeric"]].dropna(
            subset=["Dates"]
        ),
        [
            ("A", "Integers", "<"),
            ("E", "Dates", "=="),
            ("B", "Floats", ">"),
            ("B", "Numeric", "!="),
        ],
        use_numba=True,
    )
    actual = df.index[actual]
    assert_index_equal(expected, actual, check_names=False)


@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
@pytest.mark.turtle
def test_eq_indices(df, right):
    """compare join indices for single condition."""

    expected = (
        df.reset_index()
        .dropna(subset=["E"])
        .merge(
            right.dropna(subset=["Dates"]),
            left_on="E",
            right_on="Dates",
            how="inner",
            sort=False,
        )
        .loc[:, "index"]
    )
    expected = pd.Index(expected)

    actual, _ = get_join_indices(
        df,
        right,
        [
            ("E", "Dates", "=="),
        ],
    )
    actual = df.index[actual]
    assert_index_equal(expected, actual, check_names=False)


@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
@pytest.mark.turtle
def test_eq_indices_ragged_arrays(df, right):
    """compare join indices for single condition."""

    expected = (
        df.assign(lindex=range(len(df)))
        .dropna(subset=["E"])
        .merge(
            right.assign(rindex=range(len(right))).dropna(subset=["Dates"]),
            left_on="E",
            right_on="Dates",
            how="inner",
            sort=False,
        )
        .loc[:, ["lindex", "rindex"]]
        .sort_values(["lindex", "rindex"])
    )
    rindex = pd.Index(expected["rindex"])
    lindex = pd.Index(expected["lindex"])

    lactual, ractual = get_join_indices(
        df,
        right,
        [
            ("E", "Dates", "=="),
        ],
        return_ragged_arrays=True,
    )
    if isinstance(ractual, list):
        ractual = [right.index[arr] for arr in ractual]
        ractual = np.concatenate(ractual)
    ractual = pd.Index(ractual)
    lactual = pd.Index(lactual)
    if isinstance(ractual, list):
        ractual = [right.index[arr] for arr in ractual]
        lengths = [len(arr) for arr in ractual]
        ractual = np.concatenate(ractual)
        lactual = pd.Index(lactual).repeat(lengths)
    ractual = pd.Index(ractual)
    lactual = pd.Index(lactual)
    sorter = np.lexsort((ractual, lactual))
    lactual = lactual[sorter]
    ractual = ractual[sorter]
    sorter = np.lexsort((rindex, lindex))
    lindex = lindex[sorter]
    rindex = rindex[sorter]
    assert_index_equal(rindex, ractual, check_names=False)
    assert_index_equal(lindex, lactual, check_names=False)


@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
@pytest.mark.turtle
def test_le_indices_ragged_arrays(df, right):
    """compare join indices for single condition."""
    expected = (
        df.assign(lindex=range(len(df)))
        .merge(
            right.assign(rindex=range(len(right))),
            how="cross",
        )
        .loc[lambda df: df.E.le(df.Dates), ["lindex", "rindex"]]
    )
    rindex = pd.Index(expected["rindex"])
    lindex = pd.Index(expected["lindex"])

    lactual, ractual = get_join_indices(
        df,
        right,
        [
            ("E", "Dates", "<="),
        ],
        return_ragged_arrays=True,
    )
    if isinstance(ractual, list):
        ractual = [right.index[arr] for arr in ractual]
        lengths = [len(arr) for arr in ractual]
        ractual = np.concatenate(ractual)
        lactual = pd.Index(lactual).repeat(lengths)
    ractual = pd.Index(ractual)
    lactual = pd.Index(lactual)
    ractual = pd.Index(ractual)
    lactual = pd.Index(lactual)
    sorter = np.lexsort((ractual, lactual))
    lactual = lactual[sorter]
    ractual = ractual[sorter]
    sorter = np.lexsort((rindex, lindex))
    lindex = lindex[sorter]
    rindex = rindex[sorter]
    assert_index_equal(rindex, ractual, check_names=False)
    assert_index_equal(lindex, lactual, check_names=False)


@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
@pytest.mark.turtle
def test_lt_indices_ragged_arrays(df, right):
    """compare join indices for single condition."""

    expected = (
        df.assign(lindex=range(len(df)))
        .merge(
            right.assign(rindex=range(len(right))),
            how="cross",
        )
        .loc[lambda df: df.E.lt(df.Dates), ["lindex", "rindex"]]
    )
    rindex = pd.Index(expected["rindex"])
    lindex = pd.Index(expected["lindex"])

    lactual, ractual = get_join_indices(
        df,
        right,
        [
            ("E", "Dates", "<"),
        ],
        return_ragged_arrays=True,
    )
    if isinstance(ractual, list):
        ractual = [right.index[arr] for arr in ractual]
        lengths = [len(arr) for arr in ractual]
        ractual = np.concatenate(ractual)
        lactual = pd.Index(lactual).repeat(lengths)
    ractual = pd.Index(ractual)
    lactual = pd.Index(lactual)
    sorter = np.lexsort((ractual, lactual))
    lactual = lactual[sorter]
    ractual = ractual[sorter]
    sorter = np.lexsort((rindex, lindex))
    lindex = lindex[sorter]
    rindex = rindex[sorter]
    assert_index_equal(rindex, ractual, check_names=False)
    assert_index_equal(lindex, lactual, check_names=False)


@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
@pytest.mark.turtle
def test_gt_indices_ragged_arrays(df, right):
    """compare join indices for single condition."""

    expected = (
        df.assign(lindex=range(len(df)))
        .merge(
            right.assign(rindex=range(len(right))),
            how="cross",
        )
        .loc[lambda df: df.E.gt(df.Dates), ["lindex", "rindex"]]
    )
    rindex = pd.Index(expected["rindex"])
    lindex = pd.Index(expected["lindex"])

    lactual, ractual = get_join_indices(
        df,
        right,
        [
            ("E", "Dates", ">"),
        ],
        return_ragged_arrays=True,
    )
    if isinstance(ractual, list):
        ractual = [right.index[arr] for arr in ractual]
        lengths = [len(arr) for arr in ractual]
        ractual = np.concatenate(ractual)
        lactual = pd.Index(lactual).repeat(lengths)
    ractual = pd.Index(ractual)
    lactual = pd.Index(lactual)
    sorter = np.lexsort((ractual, lactual))
    lactual = lactual[sorter]
    ractual = ractual[sorter]
    sorter = np.lexsort((rindex, lindex))
    lindex = lindex[sorter]
    rindex = rindex[sorter]
    assert_index_equal(rindex, ractual, check_names=False)
    assert_index_equal(lindex, lactual, check_names=False)


@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
@pytest.mark.turtle
def test_ge_indices_ragged_arrays(df, right):
    """compare join indices for single condition."""

    expected = (
        df.assign(lindex=range(len(df)))
        .dropna(subset=["E"])
        .merge(
            right.assign(rindex=range(len(right))).dropna(subset=["Dates"]),
            how="cross",
        )
        .loc[lambda df: df.E.ge(df.Dates), ["lindex", "rindex"]]
    )
    rindex = pd.Index(expected["rindex"])
    lindex = pd.Index(expected["lindex"])

    lactual, ractual = get_join_indices(
        df,
        right,
        [
            ("E", "Dates", ">="),
        ],
        return_ragged_arrays=True,
    )
    if isinstance(ractual, list):
        ractual = [right.index[arr] for arr in ractual]
        lengths = [len(arr) for arr in ractual]
        ractual = np.concatenate(ractual)
        lactual = pd.Index(lactual).repeat(lengths)
    ractual = pd.Index(ractual)
    lactual = pd.Index(lactual)
    sorter = np.lexsort((ractual, lactual))
    lactual = lactual[sorter]
    ractual = ractual[sorter]
    sorter = np.lexsort((rindex, lindex))
    lindex = lindex[sorter]
    rindex = rindex[sorter]
    assert_index_equal(rindex, ractual, check_names=False)
    assert_index_equal(lindex, lactual, check_names=False)


@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
@pytest.mark.turtle
def test_le_gt_indices_ragged_arrays(df, right):
    """compare join indices for range join."""

    expected = (
        df.assign(lindex=range(len(df)))
        .merge(
            right.assign(rindex=range(len(right))),
            how="cross",
        )
        .loc[
            lambda df: df.E.le(df.Dates) & df.B.gt(df.Numeric),
            ["lindex", "rindex"],
        ]
    )
    rindex = pd.Index(expected["rindex"])
    lindex = pd.Index(expected["lindex"])

    lactual, ractual = get_join_indices(
        df,
        right,
        [("E", "Dates", "<="), ("B", "Numeric", ">")],
        return_ragged_arrays=True,
    )
    if isinstance(ractual, list):
        ractual = [right.index[arr] for arr in ractual]
        lengths = [len(arr) for arr in ractual]
        ractual = np.concatenate(ractual)
        lactual = pd.Index(lactual).repeat(lengths)
    ractual = pd.Index(ractual)
    lactual = pd.Index(lactual)
    sorter = np.lexsort((ractual, lactual))
    lactual = lactual[sorter]
    ractual = ractual[sorter]
    sorter = np.lexsort((rindex, lindex))
    lindex = lindex[sorter]
    rindex = rindex[sorter]
    assert_index_equal(rindex, ractual, check_names=False)
    assert_index_equal(lindex, lactual, check_names=False)


@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
@pytest.mark.turtle
def test_le_ge_indices_ragged_arrays(df, right):
    """compare join indices for range join."""

    expected = (
        df.assign(lindex=range(len(df)))
        .merge(
            right.assign(rindex=range(len(right))),
            how="cross",
        )
        .loc[
            lambda df: df.E.le(df.Dates) & df.B.ge(df.Numeric),
            ["lindex", "rindex"],
        ]
    )
    rindex = pd.Index(expected["rindex"])
    lindex = pd.Index(expected["lindex"])

    lactual, ractual = get_join_indices(
        df,
        right,
        [("E", "Dates", "<="), ("B", "Numeric", ">=")],
        return_ragged_arrays=True,
    )
    if isinstance(ractual, list):
        ractual = [right.index[arr] for arr in ractual]
        lengths = [len(arr) for arr in ractual]
        ractual = np.concatenate(ractual)
        lactual = pd.Index(lactual).repeat(lengths)
    ractual = pd.Index(ractual)
    lactual = pd.Index(lactual)
    sorter = np.lexsort((ractual, lactual))
    lactual = lactual[sorter]
    ractual = ractual[sorter]
    sorter = np.lexsort((rindex, lindex))
    lindex = lindex[sorter]
    rindex = rindex[sorter]
    assert_index_equal(rindex, ractual, check_names=False)
    assert_index_equal(lindex, lactual, check_names=False)


@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
@pytest.mark.turtle
def test_ge_le_indices_ragged_arrays(df, right):
    """compare join indices for range join."""

    expected = (
        df.assign(lindex=range(len(df)))
        .merge(
            right.assign(rindex=range(len(right))),
            how="cross",
        )
        .loc[
            lambda df: df.E.ge(df.Dates) & df.B.le(df.Numeric),
            ["lindex", "rindex"],
        ]
    )
    rindex = pd.Index(expected["rindex"])
    lindex = pd.Index(expected["lindex"])

    lactual, ractual = get_join_indices(
        df,
        right,
        [("E", "Dates", ">="), ("B", "Numeric", "<=")],
        return_ragged_arrays=True,
    )
    if isinstance(ractual, list):
        ractual = [right.index[arr] for arr in ractual]
        lengths = [len(arr) for arr in ractual]
        ractual = np.concatenate(ractual)
        lactual = pd.Index(lactual).repeat(lengths)
    ractual = pd.Index(ractual)
    lactual = pd.Index(lactual)
    sorter = np.lexsort((ractual, lactual))
    lactual = lactual[sorter]
    ractual = ractual[sorter]
    sorter = np.lexsort((rindex, lindex))
    lindex = lindex[sorter]
    rindex = rindex[sorter]
    assert_index_equal(rindex, ractual, check_names=False)
    assert_index_equal(lindex, lactual, check_names=False)


@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
@pytest.mark.turtle
def test_range_indices_ragged_arrays(df, right):
    """compare join indices for range join."""

    expected = (
        df.assign(lindex=range(len(df)))
        .merge(
            right.assign(rindex=range(len(right))),
            how="cross",
        )
        .loc[
            lambda df: df.E.lt(df.Dates) & df.B.gt(df.Numeric),
            ["lindex", "rindex"],
        ]
    )
    rindex = pd.Index(expected["rindex"])
    lindex = pd.Index(expected["lindex"])

    lactual, ractual = get_join_indices(
        df,
        right,
        [("E", "Dates", "<"), ("B", "Numeric", ">")],
        return_ragged_arrays=True,
    )
    if isinstance(ractual, list):
        ractual = [right.index[arr] for arr in ractual]
        lengths = [len(arr) for arr in ractual]
        ractual = np.concatenate(ractual)
        lactual = pd.Index(lactual).repeat(lengths)
    ractual = pd.Index(ractual)
    lactual = pd.Index(lactual)
    sorter = np.lexsort((ractual, lactual))
    lactual = lactual[sorter]
    ractual = ractual[sorter]
    sorter = np.lexsort((rindex, lindex))
    lindex = lindex[sorter]
    rindex = rindex[sorter]
    assert_index_equal(rindex, ractual, check_names=False)
    assert_index_equal(lindex, lactual, check_names=False)


@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
@pytest.mark.turtle
def test_ge_eq_and_le_datess_indices(df, right):
    """compare join indices for multiple conditions."""
    expected = (
        df.reset_index()
        .merge(
            right,
            left_on="E",
            right_on="Dates",
            how="inner",
            sort=False,
        )
        .loc[
            lambda df: df.B.gt(df.Floats)
            & df.A.lt(df.Integers)
            & df.B.ne(df.Numeric),
            "index",
        ]
    )
    expected = pd.Index(expected)

    actual, _ = get_join_indices(
        df[["B", "A", "E"]],
        right[["Floats", "Integers", "Dates", "Numeric"]],
        [
            ("A", "Integers", "<"),
            ("E", "Dates", "=="),
            ("B", "Floats", ">"),
            ("B", "Numeric", "!="),
        ],
    )
    actual = df.index[actual]
    assert_index_equal(expected, actual, check_names=False)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
        )
        .sort_values(columns, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_multiple_non_equi_numba_(df, right):
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
        )
        .sort_values(columns, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@settings(deadline=None, max_examples=10)
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
            & df.B.gt(df.Numeric),
            columns,
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
        )
        .sort_values(columns, ignore_index=True)
        .loc[:, columns]
    )

    assert_frame_equal(expected, actual)


@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
@pytest.mark.turtle
def test_multiple_non_equii_numba_(df, right):
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
            & df.B.gt(df.Numeric),
            columns,
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
        )
        .sort_values(columns, ignore_index=True)
        .loc[:, columns]
    )

    assert_frame_equal(expected, actual)


@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
@pytest.mark.turtle
def test_multiple_non_equii_col_syntax(df, right):
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
            & df.B.gt(df.Numeric),
            columns,
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
        )
        .sort_values(columns, ignore_index=True)
        .loc[:, columns]
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
        .sort_index(axis="columns")
    )

    actual = (
        df.rename(columns={"B": "b"})
        .conditional_join(
            right.rename(
                columns={
                    "Floats": "floats",
                }
            ),
            ("A", "Integers", ">="),
            ("E", "Dates", ">"),
            ("b", "floats", ">"),
            how="inner",
        )
        .loc[:, ["b", "A", "E", "floats", "Integers", "Dates"]]
        .sort_values(
            ["b", "A", "E", "floats", "Integers", "Dates"], ignore_index=True
        )
        .sort_index(axis="columns")
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
        .sort_index(axis="columns")
    )

    actual = (
        df.rename(columns={"B": "b"})
        .conditional_join(
            right.rename(
                columns={
                    "Floats": "floats",
                }
            ),
            ("A", "Integers", ">="),
            ("E", "Dates", ">"),
            ("b", "floats", ">"),
            how="inner",
        )
        .loc[:, ["b", "A", "E", "floats", "Integers", "Dates"]]
        .sort_values(
            ["b", "A", "E", "floats", "Integers", "Dates"], ignore_index=True
        )
        .sort_index(axis="columns")
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_multiple_non_eq(df, right):
    """Test output for multiple conditions."""
    columns = ["A", "Integers", "E", "Dates", "B", "Floats"]
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
        .groupby("index", sort=False)
        .head(1)
        .drop(columns="index")
        .sort_values(columns, ignore_index=True)
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
        )
        .sort_values(columns, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
        .groupby("index", sort=False)
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
        )
        .sort_values(
            ["B", "A", "E", "Floats", "Integers", "Dates"], ignore_index=True
        )
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_multiple_non_eq_first(df, right):
    """Test output for multiple conditions - grab only the first match."""
    columns = ["A", "Integers", "E", "Dates", "B", "Floats"]
    expected = (
        df[["B", "A", "E"]]
        .assign(index=df.index)
        .merge(
            right[["Floats", "Integers", "Dates"]],
            how="cross",
        )
        .loc[
            lambda df: df.B.le(df.Floats)
            & df.A.gt(df.Integers)
            & df.E.lt(df.Dates)
        ]
        .groupby("index", sort=False)
        .head(1)
        .drop(columns="index")
        .sort_values(columns, ignore_index=True)
    )

    actual = (
        df[["B", "A", "E"]]
        .conditional_join(
            right[["Floats", "Integers", "Dates"]],
            ("B", "Floats", "<="),
            ("A", "Integers", ">"),
            ("E", "Dates", "<"),
            how="inner",
            keep="first",
        )
        .sort_values(columns, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_multiple_non_eq_first_numba(df, right):
    """Test output for multiple conditions - grab only the first match."""
    columns = ["A", "Integers", "E", "Dates", "B", "Floats"]
    expected = (
        df[["B", "A", "E"]]
        .assign(index=df.index)
        .merge(
            right[["Floats", "Integers", "Dates"]],
            how="cross",
        )
        .loc[
            lambda df: df.B.le(df.Floats)
            & df.A.gt(df.Integers)
            & df.E.lt(df.Dates)
        ]
        .groupby("index", sort=False)
        .head(1)
        .drop(columns="index")
        .sort_values(columns, ignore_index=True)
    )

    actual = (
        df[["B", "A", "E"]]
        .conditional_join(
            right[["Floats", "Integers", "Dates"]],
            ("B", "Floats", "<="),
            ("A", "Integers", ">"),
            ("E", "Dates", "<"),
            how="inner",
            keep="first",
            use_numba=True,
        )
        .sort_values(columns, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_multiple_non_eq_last(df, right):
    """Test output for multiple conditions - grab only the last match."""
    columns = ["A", "Integers", "E", "Dates", "B", "Floats"]
    expected = (
        df[["B", "A", "E"]]
        .assign(index=df.index)
        .merge(
            right[["Floats", "Integers", "Dates"]],
            how="cross",
        )
        .loc[
            lambda df: df.B.le(df.Floats)
            & df.A.gt(df.Integers)
            & df.E.lt(df.Dates)
        ]
        .groupby("index", sort=False)
        .tail(1)
        .drop(columns="index")
        .sort_values(columns, ignore_index=True)
    )

    actual = (
        df[["B", "A", "E"]]
        .conditional_join(
            right[["Floats", "Integers", "Dates"]],
            ("B", "Floats", "<="),
            ("A", "Integers", ">"),
            ("E", "Dates", "<"),
            how="inner",
            keep="last",
        )
        .sort_values(columns, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_multiple_non_eq_last_numba(df, right):
    """Test output for multiple conditions - grab only the last match."""
    columns = ["A", "Integers", "E", "Dates", "B", "Floats"]
    expected = (
        df[["B", "A", "E"]]
        .assign(index=df.index)
        .merge(
            right[["Floats", "Integers", "Dates"]],
            how="cross",
        )
        .loc[
            lambda df: df.B.le(df.Floats)
            & df.A.gt(df.Integers)
            & df.E.lt(df.Dates)
        ]
        .groupby("index", sort=False)
        .tail(1)
        .drop(columns="index")
        .sort_values(columns, ignore_index=True)
    )

    actual = (
        df[["B", "A", "E"]]
        .conditional_join(
            right[["Floats", "Integers", "Dates"]],
            ("B", "Floats", "<="),
            ("A", "Integers", ">"),
            ("E", "Dates", "<"),
            how="inner",
            keep="last",
            use_numba=True,
        )
        .sort_values(columns, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_dual_non_eq_last(df, right):
    """Test output for dual conditions - grab only the last match."""
    columns = ["A", "Integers", "E", "Dates"]
    expected = (
        df[["A", "E"]]
        .assign(index=df.index)
        .merge(
            right[["Integers", "Dates"]],
            how="cross",
        )
        .loc[lambda df: df.A.gt(df.Integers) & df.E.lt(df.Dates)]
        .groupby("index", sort=False)
        .tail(1)
        .drop(columns="index")
        .sort_values(columns, ignore_index=True)
    )

    actual = (
        df[["A", "E"]]
        .conditional_join(
            right[["Integers", "Dates"]],
            ("A", "Integers", ">"),
            ("E", "Dates", "<"),
            how="inner",
            keep="last",
        )
        .sort_values(columns, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_dual_non_eq_last_numba(df, right):
    """Test output for dual conditions - grab only the last match."""
    columns = ["A", "Integers", "E", "Dates"]
    expected = (
        df[["A", "E"]]
        .assign(index=df.index)
        .merge(
            right[["Integers", "Dates"]],
            how="cross",
        )
        .loc[lambda df: df.A.gt(df.Integers) & df.E.lt(df.Dates)]
        .groupby("index", sort=False)
        .tail(1)
        .drop(columns="index")
        .sort_values(columns, ignore_index=True)
    )

    actual = (
        df[["A", "E"]]
        .conditional_join(
            right[["Integers", "Dates"]],
            ("A", "Integers", ">"),
            ("E", "Dates", "<"),
            how="inner",
            keep="last",
            use_numba=True,
        )
        .sort_values(columns, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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
        )
        .sort_values(columns, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_multiple_eqs_numba_range(df, right):
    """Test output for multiple conditions."""

    columns = ["B", "A", "E", "Floats", "Integers", "Dates"]
    expected = (
        df.merge(
            right,
            left_on=["A"],
            right_on=["Integers"],
            how="inner",
            sort=False,
        )
        .loc[lambda df: df.E.lt(df.Dates) & df.B.gt(df.Floats), columns]
        .sort_values(columns, ignore_index=True)
    )
    expected = expected.filter(columns)
    actual = (
        df[["B", "A", "E"]]
        .conditional_join(
            right[["Floats", "Integers", "Dates"]],
            ("E", "Dates", "<"),
            ("B", "Floats", ">"),
            ("A", "Integers", "=="),
            how="inner",
            use_numba=True,
        )
        .sort_values(columns, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_multiple_eqs_outer(df, right):
    """Test output for multiple conditions."""

    columns = ["B", "A", "E", "Floats", "Integers", "Dates"]
    expected = df.merge(
        right,
        left_on=["B", "A"],
        right_on=["Floats", "Integers"],
        how="inner",
        sort=False,
        indicator=True,
    ).loc[lambda df: df.E.ne(df.Dates), columns + ["_merge"]]
    contents = [expected]
    top = df.loc(axis=1)[["B", "A", "E"]].merge(
        expected.loc(axis=1)[["B", "A", "E"]], indicator=True, how="left"
    )
    top = top.loc[top._merge == "left_only"]
    if not top.empty:
        contents.append(top)
    bottom = expected.loc(axis=1)[["Floats", "Integers", "Dates"]].merge(
        right.loc(axis=1)[["Floats", "Integers", "Dates"]],
        indicator=True,
        how="right",
    )
    bottom = bottom.loc[bottom._merge == "right_only"]
    if not bottom.empty:
        contents.append(bottom)

    expected = pd.concat(contents)
    expected = expected.sort_values(columns, ignore_index=True).sort_index(
        axis="columns"
    )
    actual = (
        df[["B", "A", "E"]]
        .conditional_join(
            right[["Floats", "Integers", "Dates"]].assign(B=right.Floats),
            ("E", "Dates", "!="),
            ("B", "B", "=="),
            ("A", "Integers", "=="),
            how="outer",
            indicator=True,
        )
        .select(("right", "B"), axis="columns", invert=True)
        .droplevel(axis=1, level=0)
        .rename(columns={"": "_merge"})
        .sort_values(columns, ignore_index=True)
        .sort_index(axis="columns")
    )
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_multiple_eqs_col_syntax(df, right):
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
        )
        .sort_values(columns, ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
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


def test_extension_array_eq_force():
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
        force=True,
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


def test_extension_array_eq_numba():
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
        use_numba=True,
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


def test_extension_array_eq_range_numba():
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
        use_numba=True,
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


def test_no_match_equi_numba():
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
        df2, ("A", "A", "=="), ("B", "B", ">"), use_numba=True
    ).drop(columns=("right", "A"))
    expected.columns = list("ABC")

    assert_frame_equal(expected, actual)


def test_timedelta_dtype():
    """
    Test output on timedelta
    """
    A = {
        "l": {
            0: Timedelta("0 days 00:00:00"),
            1: Timedelta("0 days 00:51:00"),
            2: Timedelta("0 days 00:57:00"),
            3: Timedelta("0 days 01:16:00"),
            4: Timedelta("0 days 01:29:00"),
        },
        "r": {
            0: Timedelta("0 days 00:51:00"),
            1: Timedelta("0 days 00:57:00"),
            2: Timedelta("0 days 01:16:00"),
            3: Timedelta("0 days 01:29:00"),
            4: Timedelta("0 days 01:30:00"),
        },
    }

    A = pd.DataFrame(A)

    B = {
        "ll": {
            0: Timedelta("0 days 00:00:00"),
            1: Timedelta("0 days 00:19:00"),
            2: Timedelta("0 days 00:28:00"),
            3: Timedelta("0 days 01:21:00"),
            4: Timedelta("0 days 01:23:00"),
        },
        "rr": {
            0: Timedelta("0 days 00:19:00"),
            1: Timedelta("0 days 00:28:00"),
            2: Timedelta("0 days 01:21:00"),
            3: Timedelta("0 days 01:23:00"),
            4: Timedelta("0 days 01:30:00"),
        },
    }

    B = pd.DataFrame(B)

    expected = A.conditional_join(B, ("l", "ll", ">="), ("r", "rr", "<="))
    actual = A.merge(B, how="cross").loc[lambda f: f.l.ge(f.ll) & f.r.le(f.rr)]
    actual.index = range(len(actual))

    assert_frame_equal(expected, actual)


# https://stackoverflow.com/q/61948103/7175713
def test_numba_equi_extension_array():
    """
    Test output for equi join and numba
    """
    df1 = pd.DataFrame(
        {"id": [1, 1, 1, 2, 2, 3], "value_1": [2, 5, 7, 1, 3, 4]}
    )
    df2 = pd.DataFrame(
        {
            "id": [1, 1, 1, 1, 2, 2, 2, 3],
            "value_2A": [0, 3, 7, 12, 0, 2, 3, 1],
            "value_2B": [1, 9, 5, 15, 1, 6, 4, 3],
        }
    )
    df1["value_1"] = df1["value_1"].astype(pd.Int64Dtype())
    df2["value_2A"] = df2["value_2A"].astype(pd.Int64Dtype())
    df2["value_2B"] = df2["value_2B"].astype(pd.Int64Dtype())
    expected = df1.merge(df2, on="id").query("value_2A < value_1 < value_2B")
    expected.index = range(expected.index.size)
    actual = df1.conditional_join(
        df2,
        ("id", "id", "=="),
        ("value_1", "value_2A", ">"),
        ("value_1", "value_2B", "<"),
        right_columns="value*",
        use_numba=True,
    )

    assert_frame_equal(expected, actual)
