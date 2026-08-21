import operator
from itertools import permutations
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from pandas import Timedelta
from pandas.testing import assert_frame_equal

import janitor as jn
from janitor.functions._conditional_join import _le_ge_1_or_more
from janitor.testing_utils.strategies import (
    conditional_df,
    conditional_right,
)

# ## turn on to view dataframes from failed tests
# pd.set_option("display.max_columns", None)
# pd.set_option("display.expand_frame_repr", False)
# pd.set_option("max_colwidth", None)


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


def test_conditional_join_does_not_mutate_inputs():
    """Index and column metadata remain unchanged after a join."""
    left = pd.DataFrame(
        {"value": [1, 2, 3], "payload": list("abc")},
        index=pd.Index([10, 20, 30], name="left_index"),
    )
    right = pd.DataFrame(
        {"value": [2, 3, 4], "payload": list("xyz")},
        index=pd.Index([40, 50, 60], name="right_index"),
    )
    expected_left = left.copy()
    expected_right = right.copy()

    left.conditional_join(right, ("value", "value", "<"), how="outer")

    assert_frame_equal(left, expected_left)
    assert_frame_equal(right, expected_right)


def test_df_columns_right_columns_both_None(dummy, series):
    """Raise if both df_columns and right_columns is None"""
    with pytest.raises(
        ValueError,
        match="df_columns and right_columns cannot both be None.",
    ):
        dummy.conditional_join(
            series, ("id", "B", ">"), df_columns=None, right_columns=None
        )


def test_df_multiindex(dummy, series):
    """Raise ValueError if `df` columns is a MultiIndex."""
    with pytest.raises(
        ValueError,
        match="The number of column levels from the left and right frames must match.+",
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
    with pytest.raises(ValueError, match="Kindly provide at least one join condition."):
        dummy.conditional_join(series)


def test_check_condition_type(dummy, series):
    """Raise TypeError if any condition in conditions is not a tuple."""
    with pytest.raises(TypeError, match="condition should be one of.+"):
        dummy.conditional_join(series, ("id", "B", ">"), ["A", "B"])


def test_indicator_type(dummy, series):
    """Raise TypeError if indicator is not a boolean/string."""
    with pytest.raises(TypeError, match="indicator should be one of.+"):
        dummy.conditional_join(series, ("id", "B", ">"), indicator=1)


def test_join_positions_type(dummy, series):
    """Raise TypeError if include_join_positions is not a boolean/string."""
    with pytest.raises(TypeError, match="include_join_positions should be one of.+"):
        dummy.conditional_join(series, ("id", "B", ">"), include_join_positions=1)


def test_join_return_building_blocks(dummy, series):
    """Raise TypeError if return_building_blocks is not a boolean."""
    with pytest.raises(TypeError, match="return_building_blocks should be one of.+"):
        jn.get_join_indices(dummy, series, ("id", "B", ">"), return_building_blocks=1)


def test_join_algorithm_type(dummy, series):
    """Raise TypeError if join_algorithm is not a str."""
    with pytest.raises(TypeError, match="join_algorithm should be one of.+"):
        jn.conditional_join(dummy, series, ("id", "B", ">"), join_algorithm=1)


def test_join_algorithm_options(dummy, series):
    """Raise Value if join_algorithm is not default/regions."""
    with pytest.raises(
        ValueError, match="join_algorithm should be either default or regions.+"
    ):
        jn.conditional_join(dummy, series, ("id", "B", ">"), join_algorithm="region")


def test_join_aggs_reverse(dummy, series):
    """Raise TypeError if reverse is not a boolean."""
    with pytest.raises(TypeError, match="reverse should be one of.+"):
        dummy.join_agg(series, ("id", "B", "<"), reverse=1, aggfunc=[("B", "sum")])


def test_indicator_exists(dummy, series):
    """Raise ValueError if indicator is a dup of an existing column name."""
    with pytest.raises(
        ValueError,
        match="Cannot use name of an existing column for indicator column",
    ):
        dummy.conditional_join(series, ("id", "B", ">"), indicator="id")


def test_join_positions_how(dummy, series):
    """Raise ValueError if include_join_positions and how!=inner."""
    with pytest.raises(
        ValueError,
        match="include_join_positions is valid only if.+",
    ):
        dummy.conditional_join(
            series, ("id", "B", ">"), include_join_positions=True, how="left"
        )


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
    with pytest.raises(ValueError, match=".not present in dataframe columns.+"):
        dummy.conditional_join(series, ("C", "B", "<"))


def test_check_column_exists_right(dummy, series):
    """
    Raise ValueError if `right_on`
    can not be found in `right`.
    """
    with pytest.raises(ValueError, match=".+not present in dataframe columns.+"):
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
    with pytest.raises(ValueError, match="Equality only joins are not supported."):
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
    with pytest.raises(TypeError, match="Only numeric, timedelta and datetime types.+"):
        dummy.conditional_join(
            dummy, ("S", "S", "=="), ("id", "id", ">"), use_numba=True
        )


def test_check_aggfunc_type(dummy, series):
    """
    Raise TypeError if `aggfunc` is not a list.
    """
    with pytest.raises(TypeError, match="aggfunc should be one of.+"):
        dummy.join_agg(series, ("id", "B", "<"), aggfunc=1)


def test_check_aggfunc_ne(dummy, series):
    """
    Raise TypeError if all join conditions are !=
    """
    with pytest.raises(
        NotImplementedError,
        match="aggfunc is not supported when all the join operators.+",
    ):
        dummy.join_agg(series, ("id", "B", "!="), aggfunc=[("B", "sum")])


def test_check_aggfunc_sub(dummy, series):
    """
    Raise TypeError if entry in `aggfunc` is not a tuple.
    """
    with pytest.raises(TypeError, match="entry in aggfunc should be one of.+"):
        dummy.join_agg(series, ("id", "B", "<"), aggfunc=[1])


def test_check_aggfunc_sub_size(dummy, series):
    """
    Raise TypeError if entry in `aggfunc` is a tuple of len != 2.
    """
    with pytest.raises(
        ValueError, match="The tuple in an aggfunc should be 2 elements.+"
    ):
        dummy.join_agg(series, ("id", "B", "<"), aggfunc=[("B",)])


def test_check_aggfunc_column(dummy, series):
    """
    Raise Error if `column` is not in right_dataframe.
    """
    with pytest.raises(
        KeyError, match="BB in aggfunc does not exist in the right dataframe"
    ):
        dummy.join_agg(
            series,
            ("id", "B", "<"),
            aggfunc=[("BB", "sum")],
        )


def test_check_aggfunc_unsupported(dummy, series):
    """
    Raise Error if `aggfunc` is not in default aggs.
    """
    with pytest.raises(
        ValueError, match="The aggregation function for B should be one of.+"
    ):
        dummy.join_agg(
            series,
            ("id", "B", "<"),
            aggfunc=[("B", "summ")],
        )


def test_check_aggfunc_numeric(dummy):
    """
    Raise Error if `aggfunc` column is not numeric.
    """
    ser = pd.DatetimeIndex(["1970-01-01"], name="B").to_series()
    with pytest.raises(ValueError, match="sum is supported only for numeric columns"):
        dummy.join_agg(
            ser,
            ("id", "B", "<"),
            aggfunc=[("B", "sum")],
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
    with pytest.raises(ValueError, match="At least one less than or greater than.+"):
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
    match += "are supported in a non equi-join, "
    match += "or if use_numba is set to True.+"
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


@pytest.mark.parametrize("op", ["<", "<=", ">", ">="])
@pytest.mark.parametrize("keep", ["first", "last"])
def test_single_inequality_unsorted_right_keep(op, keep):
    """Keep uses original right-row order when the join key is unsorted."""
    left = pd.DataFrame({"left": [1.0, 3.0, 5.0, 7.0]})
    right = pd.DataFrame(
        {
            "right": [5.0, np.nan, 1.0, 7.0, 3.0],
            "right_row": ["zero", "null", "two", "three", "four"],
        }
    )
    compare = {
        "<": operator.lt,
        "<=": operator.le,
        ">": operator.gt,
        ">=": operator.ge,
    }[op]

    expected = []
    for left_value in left["left"]:
        matches = []
        for right_position, right_value in enumerate(right["right"]):
            if pd.notna(right_value) and compare(left_value, right_value):
                matches.append(right_position)
        if matches:
            position = min(matches) if keep == "first" else max(matches)
            expected.append((left_value, right.loc[position, "right_row"]))

    actual = left.conditional_join(
        right,
        ("left", "right", op),
        keep=keep,
    )
    assert list(actual[["left", "right_row"]].itertuples(index=False, name=None)) == (
        expected
    )


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
    Test single join output
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
    Test single join output
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
        .sort_values(["B", "Numeric"], ascending=[True, False], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@settings(deadline=None, max_examples=10)
@pytest.mark.turtle
@given(df=conditional_df(), right=conditional_right())
def test_single_condition_less_than_ints_extension_array_numba_first_match(df, right):
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
def test_single_condition_less_than_ints_extension_array_numba_last_match(df, right):
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
        .sort_values(["B", "Numeric"], ignore_index=True)
    )

    actual = (
        df[["B"]]
        .conditional_join(
            right[["Numeric"]],
            ("B", "Numeric", "!="),
            how="inner",
            keep="last",
        )
        .sort_values(["B", "Numeric"], ignore_index=True)
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
        .sort_values(["E", "Dates"], ignore_index=True)
    )

    actual = (
        df[["E"]]
        .conditional_join(
            right[["Dates"]],
            ("E", "Dates", "!="),
            how="inner",
            keep="first",
        )
        .sort_values(["E", "Dates"], ignore_index=True)
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
        .select_columns("A", "A_y", "_merge")
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

    expected = df.merge(right.assign(index=np.arange(len(right))), how="cross").loc[
        lambda df: df.E.gt(df.Dates)
    ]
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

    expected = df.merge(right.assign(index=np.arange(len(right))), how="cross").loc[
        lambda df: df.E.gt(df.Dates)
    ]
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
        .loc[lambda df: df.E.between(df.Dates, df.Dates_Right, inclusive="neither")]
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
def test_dual_conditions_gt_and_lt_dates_regions(df, right):
    """Test output for interval conditions."""

    middle, left_on, right_on = ("E", "Dates", "Dates_Right")
    expected = (
        df[["E"]]
        .merge(right[["Dates", "Dates_Right"]], how="cross")
        .loc[lambda df: df.E.between(df.Dates, df.Dates_Right, inclusive="neither")]
        .sort_values(["E", "Dates", "Dates_Right"], ignore_index=True)
    )

    actual = (
        df[["E"]]
        .conditional_join(
            right[["Dates", "Dates_Right"]],
            (middle, left_on, ">"),
            (middle, right_on, "<"),
            how="inner",
            join_algorithm="regions",
        )
        .sort_values(["E", "Dates", "Dates_Right"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_dual_conditions_gt_and_ge_dates(df, right):
    """Test output for multiple conditions."""

    expected = (
        df[["E"]]
        .merge(right[["Dates", "Dates_Right"]], how="cross")
        .query("E>Dates and E>=Dates_Right")
        .sort_values(["E", "Dates", "Dates_Right"], ignore_index=True)
    )

    actual = (
        df[["E"]]
        .conditional_join(
            right[["Dates", "Dates_Right"]],
            ("E", "Dates", ">"),
            ("E", "Dates_Right", ">="),
            how="inner",
        )
        .sort_values(["E", "Dates", "Dates_Right"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_dual_conditions_gt_and_ge_dates_regions(df, right):
    """Test output for multiple conditions."""

    expected = (
        df[["E"]]
        .merge(right[["Dates", "Dates_Right"]], how="cross")
        .query("E>Dates and E>=Dates_Right")
        .sort_values(["E", "Dates", "Dates_Right"], ignore_index=True)
    )

    actual = (
        df[["E"]]
        .conditional_join(
            right[["Dates", "Dates_Right"]],
            ("E", "Dates", ">"),
            ("E", "Dates_Right", ">="),
            how="inner",
            join_algorithm="regions",
        )
        .sort_values(["E", "Dates", "Dates_Right"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_dual_conditions_gt_and_ge_dates_first(df, right):
    """Test output for multiple conditions."""

    expected = (
        df[["E"]]
        .reset_index()
        .merge(right[["Dates", "Dates_Right"]], how="cross")
        .query("E>Dates and E>=Dates_Right")
        .groupby("index", as_index=False)
        .head(1)
        .drop(columns="index")
        .reset_index(drop=True)
    )
    expected = expected.sort_values(expected.columns.tolist(), ignore_index=True)

    actual = (
        df[["E"]]
        .conditional_join(
            right[["Dates", "Dates_Right"]],
            ("E", "Dates", ">"),
            ("E", "Dates_Right", ">="),
            how="inner",
            keep="first",
        )
        .sort_values(expected.columns.tolist(), ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_dual__dates_first(df, right):
    """Test output for multiple conditions."""

    expected = (
        df[["E"]]
        .reset_index()
        .merge(right[["Dates", "Dates_Right"]], how="cross")
        .query("E>Dates and E<=Dates_Right")
        .groupby("index", as_index=False)
        .head(1)
        .drop(columns="index")
        .reset_index(drop=True)
    )
    expected = expected.sort_values(expected.columns.tolist(), ignore_index=True)

    actual = (
        df[["E"]]
        .conditional_join(
            right[["Dates", "Dates_Right"]],
            ("E", "Dates", ">"),
            ("E", "Dates_Right", "<="),
            how="inner",
            keep="first",
        )
        .sort_values(expected.columns.tolist(), ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_dual__dates_last(df, right):
    """Test output for multiple conditions."""

    expected = (
        df[["E"]]
        .reset_index()
        .merge(right[["Dates", "Dates_Right"]], how="cross")
        .query("E>Dates and E<=Dates_Right")
        .groupby("index", as_index=False)
        .tail(1)
        .drop(columns="index")
        .reset_index(drop=True)
    )
    expected = expected.sort_values(expected.columns.tolist(), ignore_index=True)

    actual = (
        df[["E"]]
        .conditional_join(
            right[["Dates", "Dates_Right"]],
            ("E", "Dates", ">"),
            ("E", "Dates_Right", "<="),
            how="inner",
            keep="last",
        )
        .sort_values(expected.columns.tolist(), ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_dual_conditions_gt_and_ge_dates_last(df, right):
    """Test output for multiple conditions."""

    expected = (
        df[["E"]]
        .reset_index()
        .merge(right[["Dates", "Dates_Right"]], how="cross")
        .query("E>Dates and E>=Dates_Right")
        .groupby("index", as_index=False)
        .tail(1)
        .drop(columns="index")
        .reset_index(drop=True)
    )
    expected = expected.sort_values(expected.columns.tolist(), ignore_index=True)

    actual = (
        df[["E"]]
        .conditional_join(
            right[["Dates", "Dates_Right"]],
            ("E", "Dates", ">"),
            ("E", "Dates_Right", ">="),
            how="inner",
            keep="last",
        )
        .sort_values(expected.columns.tolist(), ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_dual_conditions_gt_and_ge_dates_last_regions(df, right):
    """Test output for multiple conditions."""

    expected = (
        df[["E"]]
        .reset_index()
        .merge(right[["Dates", "Dates_Right"]], how="cross")
        .query("E>Dates and E>=Dates_Right")
        .groupby("index", as_index=False)
        .tail(1)
        .drop(columns="index")
        .reset_index(drop=True)
    )
    expected = expected.sort_values(expected.columns.tolist(), ignore_index=True)

    actual = (
        df[["E"]]
        .conditional_join(
            right[["Dates", "Dates_Right"]],
            ("E", "Dates", ">"),
            ("E", "Dates_Right", ">="),
            how="inner",
            keep="last",
            join_algorithm="regions",
        )
        .sort_values(expected.columns.tolist(), ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_dual_conditions_lt_and_le_dates(df, right):
    """Test output for multiple conditions."""

    expected = (
        df[["E"]]
        .merge(right[["Dates", "Dates_Right"]], how="cross")
        .query("E<=Dates and E<Dates_Right")
        .sort_values(["E", "Dates", "Dates_Right"], ignore_index=True)
    )

    actual = (
        df[["E"]]
        .conditional_join(
            right[["Dates", "Dates_Right"]],
            ("E", "Dates", "<="),
            ("E", "Dates_Right", "<"),
            how="inner",
        )
        .sort_values(["E", "Dates", "Dates_Right"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_dual_conditions_gt_and_lt_dates_keep_first(df, right):
    """Test output for interval conditions."""

    middle, left_on, right_on = ("E", "Dates", "Dates_Right")
    expected = (
        df[["E"]]
        .reset_index(names="index")
        .merge(right[["Dates", "Dates_Right"]], how="cross")
        .loc[lambda df: df.E.between(df.Dates, df.Dates_Right, inclusive="neither")]
        .groupby("index", sort=False)
        .head(1)
        .drop(columns="index")
        .sort_values(["E", "Dates", "Dates_Right"], ignore_index=True)
    )

    actual = (
        df[["E"]]
        .conditional_join(
            right[["Dates", "Dates_Right"]],
            (middle, left_on, ">"),
            (middle, right_on, "<"),
            how="inner",
            keep="first",
        )
        .sort_values(["E", "Dates", "Dates_Right"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_dual_conditions_gt_and_lt_dates_keep_first_regions(df, right):
    """Test output for interval conditions."""

    middle, left_on, right_on = ("E", "Dates", "Dates_Right")
    expected = (
        df[["E"]]
        .reset_index(names="index")
        .merge(right[["Dates", "Dates_Right"]], how="cross")
        .loc[lambda df: df.E.between(df.Dates, df.Dates_Right, inclusive="neither")]
        .groupby("index", sort=False)
        .head(1)
        .drop(columns="index")
        .sort_values(["E", "Dates", "Dates_Right"], ignore_index=True)
    )

    actual = (
        df[["E"]]
        .conditional_join(
            right[["Dates", "Dates_Right"]],
            (middle, left_on, ">"),
            (middle, right_on, "<"),
            how="inner",
            keep="first",
            join_algorithm="regions",
        )
        .sort_values(["E", "Dates", "Dates_Right"], ignore_index=True)
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_dual_conditions_gt_and_lt_dates_keep_last(df, right):
    """Test output for interval conditions."""

    middle, left_on, right_on = ("E", "Dates", "Dates_Right")
    expected = (
        df[["E"]]
        .reset_index(names="index")
        .merge(right[["Dates", "Dates_Right"]], how="cross")
        .loc[lambda df: df.E.between(df.Dates, df.Dates_Right, inclusive="neither")]
        .groupby("index", sort=False)
        .tail(1)
        .drop(columns="index")
        .sort_values(["E", "Dates", "Dates_Right"], ignore_index=True)
    )

    actual = (
        df[["E"]]
        .conditional_join(
            right[["Dates", "Dates_Right"]],
            (middle, left_on, ">"),
            (middle, right_on, "<"),
            how="inner",
            keep="last",
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
        .loc[lambda df: df.E.between(df.Dates, df.Dates_Right, inclusive="neither")]
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
        .loc[lambda df: df.E.between(df.Dates, df.Dates_Right, inclusive="both")]
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
        .loc[lambda df: df.E.between(df.Dates, df.Dates_Right, inclusive="both")]
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
        .loc[lambda df: df.E.between(df.Dates, df.Dates_Right, inclusive="both")]
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
        .loc[lambda df: df.E.between(df.Dates, df.Dates_Right, inclusive="both")]
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
        .loc[lambda df: df.E.between(df.Dates, df.Dates_Right, inclusive="right")]
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
        .loc[lambda df: df.E.between(df.Dates, df.Dates_Right, inclusive="right")]
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
        .loc[lambda df: df.B.between(df.Numeric, df.Floats, inclusive="neither")]
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


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_dual_conditions_gt_and_lt_numbers_regions(df, right):
    """Test output for interval conditions."""

    expected = (
        df[["B"]]
        .merge(right[["Numeric", "Floats"]], how="cross")
        .loc[lambda df: df.B.between(df.Numeric, df.Floats, inclusive="neither")]
        .sort_values(["B", "Numeric", "Floats"], ignore_index=True)
    )

    actual = (
        df[["B"]]
        .conditional_join(
            right[["Numeric", "Floats"]],
            ("B", "Floats", "<"),
            ("B", "Numeric", ">"),
            how="inner",
            join_algorithm="regions",
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
        .loc[lambda df: df.B.between(df.Numeric, df.Floats, inclusive="neither")]
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
        .loc[lambda df: df.B.between(df.Numeric, df.Floats, inclusive="neither")]
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
        .loc[lambda df: df.B.between(df.Numeric, df.Floats, inclusive="neither")]
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
        .loc[lambda df: df.B.between(df.Numeric, df.Floats, inclusive="neither")]
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

    expected = df[["A", "B"]].merge(right[["Integers", "Numeric"]], how="cross")
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
        .loc[lambda df: df.A.ne(df.Integers) & df.E.ne(df.Dates) & df.B.ne(df.Numeric)]
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
        .merge(right.dropna(subset=["Numeric"]), left_on="B", right_on="Numeric")
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
def test_conditions_eq_and_lt_ne(df, right):
    """Test output for equal and not equal conditions."""

    columns = ["B", "Numeric", "E", "Dates", "A", "Integers"]
    expected = (
        df.merge(right, how="cross")
        .loc[
            lambda df: df.E.ne(df.Dates) & df.A.lt(df.Integers) & df.B.eq(df.Numeric),
            columns,
        ]
        .sort_values(columns, ignore_index=True)
    )

    actual = (
        df.conditional_join(
            right,
            ("B", "Numeric", "=="),
            ("E", "Dates", "!="),
            ("A", "Integers", "<"),
            how="inner",
        )
        .sort_values(columns, ignore_index=True)
        .loc[:, columns]
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_conditions_eq_and_lt_ne_numba(df, right):
    """Test output for equal and not equal conditions."""

    columns = ["B", "Numeric", "E", "Dates", "A", "Integers"]
    expected = (
        df.merge(right, how="cross")
        .loc[
            lambda df: df.E.ne(df.Dates) & df.A.lt(df.Integers) & df.B.eq(df.Numeric),
            columns,
        ]
        .sort_values(columns, ignore_index=True)
    )

    actual = (
        df.conditional_join(
            right,
            ("B", "Numeric", "=="),
            ("E", "Dates", "!="),
            ("A", "Integers", "<"),
            how="inner",
            use_numba=True,
        )
        .sort_values(columns, ignore_index=True)
        .loc[:, columns]
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_conditions_eq_and_gt_ne(df, right):
    """Test output for equal and not equal conditions."""

    columns = ["B", "Numeric", "E", "Dates", "A", "Integers"]
    expected = (
        df.merge(right, how="cross")
        .loc[
            lambda df: df.E.ne(df.Dates) & df.A.gt(df.Integers) & df.B.eq(df.Numeric),
            columns,
        ]
        .sort_values(columns, ignore_index=True)
    )

    actual = (
        df.conditional_join(
            right,
            ("B", "Numeric", "=="),
            ("E", "Dates", "!="),
            ("A", "Integers", ">"),
            how="inner",
        )
        .sort_values(columns, ignore_index=True)
        .loc[:, columns]
    )

    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_conditions_eq_and_gt_ne_numba(df, right):
    """Test output for equal and not equal conditions."""

    columns = ["B", "Numeric", "E", "Dates", "A", "Integers"]
    expected = (
        df.merge(right, how="cross")
        .loc[
            lambda df: df.E.ne(df.Dates) & df.A.gt(df.Integers) & df.B.eq(df.Numeric),
            columns,
        ]
        .sort_values(columns, ignore_index=True)
    )

    actual = (
        df.conditional_join(
            right,
            ("B", "Numeric", "=="),
            ("E", "Dates", "!="),
            ("A", "Integers", ">"),
            how="inner",
            use_numba=True,
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
        .dropna(subset="E")
        .merge(
            right[["Integers", "Dates"]].dropna(subset="Dates"),
            left_on="E",
            right_on="Dates",
        )
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
        .loc[lambda df: df.A.gt(df.Integers) & df.B.lt(df.Numeric) & df.E.ne(df.Dates)]
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
def test_gt_lt_ne_conditions_regions(df, right):
    """
    Test output for multiple conditions.
    """

    filters = ["A", "B", "E", "Integers", "Numeric", "Dates"]
    expected = (
        df[["A", "B", "E"]]
        .merge(right[["Integers", "Numeric", "Dates"]], how="cross")
        .loc[lambda df: df.A.gt(df.Integers) & df.B.lt(df.Numeric) & df.E.ne(df.Dates)]
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
            join_algorithm="regions",
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
        .loc[lambda df: df.A.gt(df.Integers) & df.B.lt(df.Numeric) & df.E.ne(df.Dates)]
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
        .loc[lambda df: df.A.gt(df.Integers) & df.B.lt(df.Numeric) & df.E.ne(df.Dates)]
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
        df.merge(right, left_on="B", right_on="Floats", how="inner", sort=False)
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
        df.dropna(subset="B")
        .merge(
            right.dropna(subset="Floats"),
            left_on="B",
            right_on="Floats",
            how="inner",
            sort=False,
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
        df.merge(right, left_on="B", right_on="Floats", how="inner", sort=False)
        .loc[
            lambda df: df.A.ge(df.Integers) & df.E.le(df.Dates) & df.B.gt(df.Numeric),
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
        df.merge(right, left_on="B", right_on="Floats", how="inner", sort=False)
        .loc[
            lambda df: df.A.ge(df.Integers) & df.E.le(df.Dates) & df.B.lt(df.Numeric),
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
        .loc[lambda df: df.A.ge(df.Integers) & df.E.lt(df.Dates_Right), columns]
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
        .loc[lambda df: df.A.ge(df.Integers) & df.E.lt(df.Dates_Right), columns]
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
        df.merge(right, left_on="B", right_on="Floats", how="inner", sort=False)
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
        df.merge(right, left_on="B", right_on="Floats", how="inner", sort=False)
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
        df.dropna(subset="B")
        .merge(
            right.dropna(subset="Floats"),
            left_on="B",
            right_on="Floats",
            how="inner",
            sort=False,
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
        df.merge(right, left_on="B", right_on="Floats", how="inner", sort=False)
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
        df.merge(right, left_on="A", right_on="Integers", how="inner", sort=False)
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
        df.merge(right, left_on="A", right_on="Integers", how="inner", sort=False)
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
        df.merge(right, left_on="A", right_on="Integers", how="inner", sort=False)
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
        df.merge(right, left_on="A", right_on="Integers", how="inner", sort=False)
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
            lambda df: df.B.gt(df.Floats) & df.A.lt(df.Integers) & df.B.ne(df.Numeric),
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
            lambda df: df.A.ge(df.Integers) & df.E.le(df.Dates) & df.B.lt(df.Floats),
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
def test_multiple_non_equi_regions(df, right):
    """Test output for multiple conditions."""

    columns = ["B", "A", "E", "Floats", "Integers", "Dates"]
    expected = (
        df.merge(
            right,
            how="cross",
        )
        .loc[
            lambda df: df.A.ge(df.Integers) & df.E.le(df.Dates) & df.B.lt(df.Floats),
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
            join_algorithm="regions",
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
            lambda df: df.A.ge(df.Integers) & df.E.le(df.Dates) & df.B.lt(df.Floats),
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
            use_numba=True,
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


# --- issue #1641: anchor predicate selection for multi le/ge conditions ---
#
# For `keep='first'`/`keep='last'` with 2+ `<`/`<=`/`>`/`>=` predicates,
# `conditional_join` internally picks one predicate as an anchor to narrow
# candidate pairs via binary search before checking the rest. Output must
# be identical no matter which predicate is chosen, so it must also be
# identical no matter what order the predicates are supplied in. `keep='all'`
# is excluded: which predicate anchors changes `right`'s sort order, which
# changes the row order of the output (not its content), so anchor choice
# must remain fixed (first-supplied) for that case.

_METHOD_NAME = {"<": "lt", "<=": "le", ">": "gt", ">=": "ge"}


def _dual_le_ge_frames(seed, n=500, broad_high=5, selective_high=5000):
    """Left/right frames with one coarse ("broad") and one fine-grained
    ("selective") integer column, sized like the issue's own repro. Left
    and right column names are kept distinct, matching this file's
    convention elsewhere, so there is no join-column collision."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "l_broad": rng.integers(0, broad_high, size=n),
            "l_selective": rng.integers(0, selective_high, size=n),
        }
    )
    right = pd.DataFrame(
        {
            "r_broad": rng.integers(0, broad_high, size=n),
            "r_selective": rng.integers(0, selective_high, size=n),
        }
    )
    return df, right


def _cross_join_first_or_last(df, right, broad_op, selective_op, keep):
    """Independent oracle: cross join, filter, and pick the first/last
    matching right row (by right's default positional index) per left row.
    """
    left = df.assign(index=df.index)
    merged = left.merge(right, how="cross")
    broad_mask = getattr(merged["l_broad"], _METHOD_NAME[broad_op])(merged["r_broad"])
    selective_mask = getattr(merged["l_selective"], _METHOD_NAME[selective_op])(
        merged["r_selective"]
    )
    matches = merged.loc[broad_mask & selective_mask]
    grouped = matches.groupby("index", sort=True)
    picked = grouped.head(1) if keep == "first" else grouped.tail(1)
    return (
        picked.drop(columns="index")
        .loc[:, ["l_broad", "l_selective", "r_broad", "r_selective"]]
        .reset_index(drop=True)
    )


@pytest.mark.parametrize("keep", ["first", "last"])
@pytest.mark.parametrize(
    "broad_op, selective_op",
    [
        ("<", "<"),
        ("<=", "<="),
        (">", ">"),
        (">=", ">="),
        ("<", "<="),
        (">", ">="),
    ],
)
def test_dual_le_ge_anchor_order_invariant(keep, broad_op, selective_op):
    """Output for keep='first'/'last' must not depend on which of two
    le/ge predicates is listed first - see issue #1641."""
    df, right = _dual_le_ge_frames(seed=0)
    broad_cond = ("l_broad", "r_broad", broad_op)
    selective_cond = ("l_selective", "r_selective", selective_op)

    bad_order = df.conditional_join(
        right, broad_cond, selective_cond, keep=keep, how="inner"
    )
    good_order = df.conditional_join(
        right, selective_cond, broad_cond, keep=keep, how="inner"
    )
    assert_frame_equal(bad_order, good_order)

    expected = _cross_join_first_or_last(df, right, broad_op, selective_op, keep)
    assert_frame_equal(expected, bad_order.reset_index(drop=True))


@pytest.mark.parametrize("keep", ["first", "last"])
def test_dual_le_ge_anchor_order_invariant_zero_match(keep):
    """Zero-match case: one predicate alone has no matches anywhere, so
    the whole join is empty regardless of anchor choice."""
    df = pd.DataFrame({"l_broad": [1, 2, 3], "l_selective": [100, 200, 300]})
    right = pd.DataFrame({"r_broad": [1, 2, 3], "r_selective": [1, 2, 3]})
    broad_cond = ("l_broad", "r_broad", "<=")
    # selective_cond can never be satisfied: l_selective is always >> r_selective
    selective_cond = ("l_selective", "r_selective", "<")

    bad_order = df.conditional_join(
        right, broad_cond, selective_cond, keep=keep, how="inner"
    )
    good_order = df.conditional_join(
        right, selective_cond, broad_cond, keep=keep, how="inner"
    )
    assert_frame_equal(bad_order, good_order)
    assert bad_order.empty


@pytest.mark.parametrize("keep", ["first", "last"])
def test_dual_le_ge_anchor_order_invariant_full_match(keep):
    """Full-match case: every left/right pair satisfies both predicates."""
    df = pd.DataFrame({"l_broad": [0, 0, 0], "l_selective": [0, 0, 0]})
    right = pd.DataFrame({"r_broad": [10, 20, 30], "r_selective": [10, 20, 30]})
    broad_cond = ("l_broad", "r_broad", "<")
    selective_cond = ("l_selective", "r_selective", "<")

    bad_order = df.conditional_join(
        right, broad_cond, selective_cond, keep=keep, how="inner"
    )
    good_order = df.conditional_join(
        right, selective_cond, broad_cond, keep=keep, how="inner"
    )
    assert_frame_equal(bad_order, good_order)
    assert len(bad_order) == len(df)


@pytest.mark.parametrize("keep", ["first", "last"])
def test_dual_le_ge_anchor_order_invariant_unsorted_right(keep):
    """Order-invariance must hold whether or not `right`'s columns already
    happen to be sorted (both branches of the monotonic check)."""
    df, right = _dual_le_ge_frames(seed=1)
    right_sorted = right.sort_values(["r_broad", "r_selective"], ignore_index=True)
    broad_cond = ("l_broad", "r_broad", "<")
    selective_cond = ("l_selective", "r_selective", "<=")

    for r in (right, right_sorted):
        bad_order = df.conditional_join(
            r, broad_cond, selective_cond, keep=keep, how="inner"
        )
        good_order = df.conditional_join(
            r, selective_cond, broad_cond, keep=keep, how="inner"
        )
        assert_frame_equal(bad_order, good_order)


@pytest.mark.parametrize("keep", ["first", "last"])
def test_dual_le_ge_anchor_order_invariant_extension_array(keep):
    """Order-invariance must hold for nullable/extension dtypes, including
    when some values are null."""
    df, right = _dual_le_ge_frames(seed=2, n=200)
    df = df.assign(l_selective=df["l_selective"].astype("Int64"))
    right = right.assign(r_selective=right["r_selective"].astype("Int64"))
    # sprinkle some nulls into the extension-array column
    df.loc[df.index % 7 == 0, "l_selective"] = pd.NA
    right.loc[right.index % 11 == 0, "r_selective"] = pd.NA

    broad_cond = ("l_broad", "r_broad", "<")
    selective_cond = ("l_selective", "r_selective", "<=")

    bad_order = df.conditional_join(
        right, broad_cond, selective_cond, keep=keep, how="inner"
    )
    good_order = df.conditional_join(
        right, selective_cond, broad_cond, keep=keep, how="inner"
    )
    assert_frame_equal(bad_order, good_order)


def _triple_le_ge_frames(seed, n=300):
    """Three integer columns with different selectivity profiles, for
    order-invariance coverage with 3+ candidates (not just 2)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "l_a": rng.integers(0, 5, size=n),
            "l_b": rng.integers(0, 500, size=n),
            "l_c": rng.integers(n - 50, n, size=n),
        }
    )
    right = pd.DataFrame(
        {
            "r_a": rng.integers(0, n, size=n),
            "r_b": rng.integers(0, 500, size=n),
            "r_c": rng.integers(0, n, size=n),
        }
    )
    return df, right


@pytest.mark.parametrize("keep", ["first", "last"])
def test_triple_le_ge_anchor_order_invariant(keep):
    """Order-invariance must hold with 3+ le/ge predicates, not just 2 -
    `_select_anchor` picks among every candidate, not just a pair."""
    df, right = _triple_le_ge_frames(seed=4)
    conditions = [
        ("l_a", "r_a", "<"),
        ("l_b", "r_b", "<="),
        ("l_c", "r_c", "<"),
    ]

    results = [
        df.conditional_join(right, *perm, keep=keep, how="inner")
        for perm in permutations(conditions)
    ]
    for result in results[1:]:
        assert_frame_equal(results[0], result)


def test_dual_le_ge_anchor_selection_used_for_keep_all():
    """`_select_anchor` must be invoked for `keep='all'` too (issue #1657) -
    row content is invariant to anchor choice regardless of `keep`, so
    there's no correctness reason to leave `keep='all'` on the unselective
    first-supplied-predicate path."""
    df, right = _dual_le_ge_frames(seed=3, n=50)
    broad_cond = ("l_broad", "r_broad", "<")
    selective_cond = ("l_selective", "r_selective", "<=")

    with mock.patch(
        "janitor.functions._conditional_join._le_ge_1_or_more._select_anchor",
        wraps=_le_ge_1_or_more._select_anchor,
    ) as patched:
        df.conditional_join(right, broad_cond, selective_cond, keep="all", how="inner")
        patched.assert_called_once()


@pytest.mark.parametrize(
    "broad_op, selective_op",
    [("<", "<"), ("<=", "<="), (">", ">"), (">=", ">=")],
)
def test_dual_le_ge_anchor_content_invariant_keep_all(broad_op, selective_op):
    """For `keep='all'`, the *set* of matched rows must not depend on
    argument order, even though row order is no longer guaranteed to
    match between orders (see #1657) - so compare sorted, the same way
    the pre-existing `keep='all'` tests elsewhere in this file do."""
    df, right = _skewed_broad_selective_frames(seed=11)
    broad_cond = ("l_broad", "r_broad", broad_op)
    selective_cond = ("l_selective", "r_selective", selective_op)
    columns = ["l_broad", "l_selective", "r_broad", "r_selective"]

    bad_order = df.conditional_join(
        right, broad_cond, selective_cond, keep="all", how="inner"
    ).sort_values(columns, ignore_index=True)
    good_order = df.conditional_join(
        right, selective_cond, broad_cond, keep="all", how="inner"
    ).sort_values(columns, ignore_index=True)
    assert_frame_equal(bad_order, good_order)


def test_dual_le_ge_anchor_row_order_can_differ_for_keep_all_on_tie():
    """Documents the actual trade-off #1657 accepts: for `keep='all'`,
    which predicate anchors determines which column `right` gets sorted
    by, so unsorted row order can differ between argument orders - but
    only found this to actually happen when both candidates' sampled
    costs tie (verified empirically: 0/30 seeds of genuinely-skewed,
    clearly-one-more-selective data from `_skewed_broad_selective_frames`
    produced a row-order difference, since `_select_anchor` reliably
    picks the same logical predicate regardless of argument order in
    that case - it's the earliest-wins tie-break that reintroduces
    position-dependence, not a general property of anchor selection).
    Content stays identical either way (see the test above); this test
    exists to make the tie-break trade-off visible, not to pin a specific
    order."""
    df = pd.DataFrame({"l_a": [0, 0], "l_b": [0, 0]})
    right = pd.DataFrame({"r_a": [30, 10, 20], "r_b": [1, 3, 2]})
    cond_a = ("l_a", "r_a", "<")
    cond_b = ("l_b", "r_b", "<")

    a_first = df.conditional_join(right, cond_a, cond_b, keep="all", how="inner")
    b_first = df.conditional_join(right, cond_b, cond_a, keep="all", how="inner")

    columns = list(a_first.columns)
    assert_frame_equal(
        a_first.sort_values(columns, ignore_index=True),
        b_first.sort_values(columns, ignore_index=True),
    )
    assert not a_first.reset_index(drop=True).equals(b_first.reset_index(drop=True))


@pytest.mark.turtle
@pytest.mark.parametrize("keep", ["first", "last"])
def test_dual_le_ge_anchor_order_invariant_above_sample_size(keep):
    """Output stays correct once `n` exceeds `_select_anchor`'s fixed sample
    size (1024), where anchor choice is driven by a genuine subsample rather
    than the full column. This only exercises the code path, not whether
    the sample actually favors the selective predicate - see
    `test_select_anchor_picks_the_selective_candidate` for that."""
    df, right = _dual_le_ge_frames(seed=5, n=2000)
    broad_cond = ("l_broad", "r_broad", "<")
    selective_cond = ("l_selective", "r_selective", "<=")

    bad_order = df.conditional_join(
        right, broad_cond, selective_cond, keep=keep, how="inner"
    )
    good_order = df.conditional_join(
        right, selective_cond, broad_cond, keep=keep, how="inner"
    )
    assert_frame_equal(bad_order, good_order)


@pytest.mark.parametrize("keep", ["first", "last"])
def test_dual_le_ge_anchor_order_invariant_duplicate_right_index(keep):
    """Order-invariance must hold when `right` has a non-default, duplicate
    index - `_select_anchor` must not assume `right`'s index is unique or a
    contiguous RangeIndex."""
    df, right = _dual_le_ge_frames(seed=6, n=40)
    right.index = np.repeat(np.arange(len(right) // 2), 2)
    broad_cond = ("l_broad", "r_broad", "<")
    selective_cond = ("l_selective", "r_selective", "<=")

    bad_order = df.conditional_join(
        right, broad_cond, selective_cond, keep=keep, how="inner"
    )
    good_order = df.conditional_join(
        right, selective_cond, broad_cond, keep=keep, how="inner"
    )
    assert_frame_equal(bad_order, good_order)


def test_dual_le_ge_anchor_selection_is_deterministic():
    """Repeated calls with identical inputs must produce byte-identical
    output. This alone doesn't prove the same anchor was picked each time -
    output is invariant to anchor choice by construction - see
    `test_select_anchor_choice_is_deterministic` for a test that inspects
    the actual choice made."""
    df, right = _dual_le_ge_frames(seed=7, n=3000)
    broad_cond = ("l_broad", "r_broad", "<")
    selective_cond = ("l_selective", "r_selective", "<=")

    results = [
        df.conditional_join(
            right, broad_cond, selective_cond, keep="first", how="inner"
        )
        for _ in range(5)
    ]
    for result in results[1:]:
        assert_frame_equal(results[0], result)


def _skewed_broad_selective_frames(seed, n=3000):
    """Deliberately skewed, unlike `_dual_le_ge_frames`: `l_broad` sits near
    the bottom of `r_broad`'s range (matches almost every right row) and
    `l_selective` sits near the top of `r_selective`'s range (matches
    almost none). `_dual_le_ge_frames` draws both sides of each column from
    similar-scale ranges - fine for output-invariance tests, which don't
    care which candidate wins, but not a reliable basis for asserting
    *which* candidate `_select_anchor` should favor - see the discussion on
    PR #1658 (which candidate is genuinely selective is otherwise close to
    a coin flip per seed)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "l_broad": rng.integers(0, 10, size=n),
            "l_selective": rng.integers(n - 10, n, size=n),
        }
    )
    right = pd.DataFrame(
        {
            "r_broad": rng.integers(0, n, size=n),
            "r_selective": rng.integers(0, n, size=n),
        }
    )
    return df, right


def test_select_anchor_picks_the_selective_candidate():
    """`_select_anchor` must actually favor the more selective predicate,
    not merely leave output correct regardless of its choice (which output
    -equality tests alone can't distinguish from a coin flip)."""
    df, right = _skewed_broad_selective_frames(seed=8)
    broad_cond = ("l_broad", "r_broad", "<")
    selective_cond = ("l_selective", "r_selective", "<=")

    best_pos, *_ = _le_ge_1_or_more._select_anchor(
        [broad_cond, selective_cond], df, right
    )
    assert best_pos == 1


def test_select_anchor_picks_same_predicate_regardless_of_order():
    """The same logical predicate must be selected as anchor whichever
    position it's supplied in - not merely "whichever position happens to
    win"."""
    df, right = _skewed_broad_selective_frames(seed=9)
    broad_cond = ("l_broad", "r_broad", "<")
    selective_cond = ("l_selective", "r_selective", "<=")

    best_pos_a, *_ = _le_ge_1_or_more._select_anchor(
        [broad_cond, selective_cond], df, right
    )
    best_pos_b, *_ = _le_ge_1_or_more._select_anchor(
        [selective_cond, broad_cond], df, right
    )
    assert [broad_cond, selective_cond][best_pos_a] == selective_cond
    assert [selective_cond, broad_cond][best_pos_b] == selective_cond


def test_select_anchor_choice_is_deterministic():
    """Repeated calls with identical inputs must pick the *same* candidate
    position every time - inspects the choice directly, rather than relying
    on output equality (which holds regardless of choice)."""
    df, right = _dual_le_ge_frames(seed=10, n=3000)
    broad_cond = ("l_broad", "r_broad", "<")
    selective_cond = ("l_selective", "r_selective", "<=")

    positions = [
        _le_ge_1_or_more._select_anchor([broad_cond, selective_cond], df, right)[0]
        for _ in range(5)
    ]
    assert len(set(positions)) == 1


@pytest.mark.parametrize(
    "op, expected_cost",
    [("<", 4.0), ("<=", 5.0), (">", 5.0), (">=", 6.0)],
)
def test_sample_candidate_cost_matches_expected_for_each_operator(op, expected_cost):
    """`_sample_candidate_cost` must compute the correct window size for
    each operator. Uses fewer rows than the sample size (1024), so the
    "sample" is the full population and the expected cost is exact, not
    approximate."""
    df = pd.DataFrame({"l": [5, 5, 5]})
    right = pd.DataFrame({"r": list(range(10))})  # 0..9, already sorted

    cost = _le_ge_1_or_more._sample_candidate_cost(("l", "r", op), df, right)
    assert cost == expected_cost


def test_select_anchor_can_miss_a_rare_selective_feature_but_stays_correct():
    """Documents a known limitation: a fixed 1024-row sample can miss a
    rare-but-decisive feature. For a feature present in only 0.1% of rows,
    the probability a uniform sample of 1024 misses it entirely is
    `0.999**1024` ~= 36%. Because `_sample_candidate_cost` seeds its RNG
    deterministically (see its docstring), whether a *specific* rare
    feature is captured is fixed by the column length, not re-rolled per
    call - so this constructs a case, by direct inspection of the sampled
    positions, where the rare feature is guaranteed to be missed, and
    confirms output is still correct regardless (only anchor-choice
    quality is ever at risk, per `_select_anchor`'s invariance guarantee)."""
    n = 5000
    # a condition that matches almost nothing, except for a rare block of
    # rows placed at the very end - outside where the fixed-seed sample
    # (drawn from `np.random.default_rng(0).choice(n, ...)`) is likely to
    # land, since the sampled positions are an arbitrary, seed-fixed subset
    # unrelated to where this block sits.
    rare_block = n - 5  # last 5 rows are the rare, highly-selective feature
    l_broad = np.zeros(n, dtype=int)
    l_broad[rare_block:] = 10**9  # unmatched by any r_broad below
    r_broad = np.arange(n)

    l_selective = np.full(n, 10**9, dtype=int)  # unmatched by default
    l_selective[rare_block:] = 0  # only the rare block is selective here
    r_selective = np.arange(n)

    df = pd.DataFrame({"l_broad": l_broad, "l_selective": l_selective})
    right = pd.DataFrame({"r_broad": r_broad, "r_selective": r_selective})
    broad_cond = ("l_broad", "r_broad", "<")
    selective_cond = ("l_selective", "r_selective", "<")

    rng = np.random.default_rng(0)
    sampled_positions = set(rng.choice(n, size=min(n, 1024), replace=False).tolist())
    assert not sampled_positions & set(range(rare_block, n)), (
        "test assumption violated: the fixed-seed sample now reaches the "
        "rare block, so this no longer demonstrates a miss"
    )

    bad_order = df.conditional_join(
        right, broad_cond, selective_cond, keep="first", how="inner"
    )
    good_order = df.conditional_join(
        right, selective_cond, broad_cond, keep="first", how="inner"
    )
    assert_frame_equal(bad_order, good_order)


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
        .loc[lambda df: df.A.ge(df.Integers) & df.E.gt(df.Dates) & df.B.gt(df.Floats)]
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
        .sort_values(["b", "A", "E", "floats", "Integers", "Dates"], ignore_index=True)
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
        .loc[lambda df: df.A.ge(df.Integers) & df.E.gt(df.Dates) & df.B.gt(df.Floats)]
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
        .sort_values(["b", "A", "E", "floats", "Integers", "Dates"], ignore_index=True)
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
        .loc[lambda df: df.B.le(df.Floats) & df.A.lt(df.Integers) & df.E.lt(df.Dates)]
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
        .loc[lambda df: df.B.le(df.Floats) & df.A.lt(df.Integers) & df.E.lt(df.Dates)]
        .groupby("index", sort=False)
        .head(1)
        .drop(columns="index")
        .reset_index(drop=True)
        .sort_values(["B", "A", "E", "Floats", "Integers", "Dates"], ignore_index=True)
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
        .sort_values(["B", "A", "E", "Floats", "Integers", "Dates"], ignore_index=True)
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
        .loc[lambda df: df.B.le(df.Floats) & df.A.gt(df.Integers) & df.E.lt(df.Dates)]
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
        .loc[lambda df: df.B.le(df.Floats) & df.A.gt(df.Integers) & df.E.lt(df.Dates)]
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
        .loc[lambda df: df.B.le(df.Floats) & df.A.gt(df.Integers) & df.E.lt(df.Dates)]
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
        .loc[lambda df: df.B.le(df.Floats) & df.A.gt(df.Integers) & df.E.lt(df.Dates)]
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
        .select_columns(("right", "B"), invert=True)
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


@pytest.mark.skip(reason="Flaky test - needs investigation")
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
    expected = expected.loc[expected.A >= expected.Integers, columns].sort_values(
        columns, ignore_index=True
    )

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
    df1 = pd.DataFrame({"id": [1, 1, 1, 2, 2, 3], "value_1": [2, 5, 7, 1, 3, 4]})
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
    df1 = pd.DataFrame({"id": [1, 1, 1, 2, 2, 3], "value_1": [2, 5, 7, 1, 3, 4]})
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
    df1 = pd.DataFrame({"id": [1, 1, 1, 2, 2, 3], "value_1": [2, 5, 7, 1, 3, 4]})
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
    df1 = pd.DataFrame({"id": [1, 1, 1, 2, 2, 3], "value_1": [2, 5, 7, 1, 3, 4]})
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
    expected = expected.drop(columns=("right", "id")).droplevel(axis=1, level=0)
    actual = (
        df1.merge(df2, on="id")
        .loc[lambda df: df.value_1.gt(df.value_2A) & df.value_1.lt(df.value_2B)]
        .reset_index(drop=True)
    )

    assert_frame_equal(expected, actual)


def test_extension_array_eq_range_numba():
    """Extension arrays when matching on equality."""
    df1 = pd.DataFrame({"id": [1, 1, 1, 2, 2, 3], "value_1": [2, 5, 7, 1, 3, 4]})
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
    expected = expected.drop(columns=("right", "id")).droplevel(axis=1, level=0)
    actual = (
        df1.merge(df2, on="id")
        .loc[lambda df: df.value_1.gt(df.value_2A) & df.value_1.lt(df.value_2B)]
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
    expected = df1.conditional_join(df2, ("A", "A", "=="), ("B", "B", "<=")).drop(
        columns=("right", "A")
    )
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
    expected = df1.conditional_join(df2, ("A", "A", "=="), ("B", "B", "<=")).drop(
        columns=("right", "A")
    )
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
    expected = df1.conditional_join(df2, ("A", "A", "=="), ("B", "B", ">")).drop(
        columns=("right", "A")
    )
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
    df1 = pd.DataFrame({"id": [1, 1, 1, 2, 2, 3], "value_1": [2, 5, 7, 1, 3, 4]})
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


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_single_condition_less_than_dates_agg(df, right):
    """Test output for a single condition. "<"."""
    right = right.sort_values("Dates", ignore_index=True)
    expected = (
        df.reset_index(names="l")
        .merge(right, how="cross")
        .query("E < Dates")
        .groupby("l")
        .agg(
            {
                "Numeric": [
                    "size",
                    "min",
                    "max",
                ],
                "Integers": ["prod", "sum"],
            }
        )
    )
    expected.index.names = [None]

    actual = df.join_agg(
        right,
        ("E", "Dates", "<"),
        aggfunc=[
            ("Numeric", "size"),
            ("Numeric", "min"),
            ("Numeric", "max"),
            ("Integers", "prod"),
            ("Integers", "sum"),
        ],
    )
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_single_condition_greater_than_dates_agg(df, right):
    """Test output for a single condition. ">"."""
    right = right.sort_values("Dates", ignore_index=True)
    expected = (
        df.reset_index(names="l")
        .merge(right, how="cross")
        .query("E > Dates")
        .groupby("l")
        .agg({"Numeric": ["size", "min", "max"], "Integers": ["prod", "sum"]})
    )
    expected.index.names = [None]

    actual = df.join_agg(
        right,
        ("E", "Dates", ">"),
        aggfunc=[
            ("Numeric", "size"),
            ("Numeric", "min"),
            ("Numeric", "max"),
            ("Integers", "prod"),
            ("Integers", "sum"),
        ],
    )
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_gt_ne_agg(df, right):
    """Test output for agg."""
    right = right.sort_values("Dates", ignore_index=True)
    expected = (
        df.reset_index(names="l")
        .merge(right, how="cross")
        .query("E > Dates and B != Numeric")
        .groupby("l")
        .agg({"Numeric": ["size", "min", "max"], "Integers": ["prod", "sum"]})
    )
    expected.index.names = [None]

    actual = df.join_agg(
        right,
        ("E", "Dates", ">"),
        ("B", "Numeric", "!="),
        aggfunc=[
            ("Numeric", "size"),
            ("Numeric", "min"),
            ("Numeric", "max"),
            ("Integers", "prod"),
            ("Integers", "sum"),
        ],
    )
    actual = actual.loc[expected.index]
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_lt_ne_agg(df, right):
    """Test output for agg."""
    right = right.sort_values("Dates", ignore_index=True)
    expected = (
        df.reset_index(names="l")
        .merge(right, how="cross")
        .query("E < Dates and B != Numeric")
        .groupby("l")
        .agg({"Numeric": ["size", "min", "max"], "Integers": ["prod", "sum"]})
    )
    expected.index.names = [None]

    actual = df.join_agg(
        right,
        ("E", "Dates", "<"),
        ("B", "Numeric", "!="),
        aggfunc=[
            ("Numeric", "size"),
            ("Numeric", "min"),
            ("Numeric", "max"),
            ("Integers", "prod"),
            ("Integers", "sum"),
        ],
    )
    actual = actual.loc[expected.index]
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_dual_gt_agg(df, right):
    """Test output for agg."""
    right = right.sort_values("Dates", ignore_index=True)
    expected = (
        df.reset_index(names="l")
        .merge(right, how="cross")
        .query("E > Dates and B > Numeric")
        .groupby("l")
        .agg({"Numeric": ["size", "min", "max"], "Integers": ["prod", "sum"]})
    )
    expected.index.names = [None]
    actual = df.join_agg(
        right,
        ("E", "Dates", ">"),
        ("B", "Numeric", ">"),
        aggfunc=[
            ("Numeric", "size"),
            ("Numeric", "min"),
            ("Numeric", "max"),
            ("Integers", "prod"),
            ("Integers", "sum"),
        ],
    )
    actual = actual.loc[expected.index]
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_dual_lt_agg(df, right):
    """Test output for agg."""
    right = right.sort_values("Dates", ignore_index=True)
    expected = (
        df.reset_index(names="l")
        .merge(right, how="cross")
        .query("E < Dates and B <= Numeric")
        .groupby("l")
        .agg({"Numeric": ["size", "min", "max"], "Integers": ["prod", "sum"]})
    )
    expected.index.names = [None]
    actual = df.join_agg(
        right,
        ("E", "Dates", "<"),
        ("B", "Numeric", "<="),
        aggfunc=[
            ("Numeric", "size"),
            ("Numeric", "min"),
            ("Numeric", "max"),
            ("Integers", "prod"),
            ("Integers", "sum"),
        ],
    )
    actual = actual.loc[expected.index]
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_multiple__ge__agg(df, right):
    """Test output for agg."""
    right = right.sort_values("Dates", ignore_index=True)
    expected = (
        df.reset_index(names="l")
        .merge(right, how="cross")
        .query("E > Dates and B > Numeric and A!=Integers")
        .groupby("l")
        .agg({"Numeric": ["size", "min", "max"], "Integers": ["prod", "sum"]})
    )
    expected.index.names = [None]
    actual = df.join_agg(
        right,
        ("E", "Dates", ">"),
        ("B", "Numeric", ">"),
        ("A", "Integers", "!="),
        aggfunc=[
            ("Numeric", "size"),
            ("Numeric", "min"),
            ("Numeric", "max"),
            ("Integers", "prod"),
            ("Integers", "sum"),
        ],
    )
    actual = actual.loc[expected.index]
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_multiple__le__agg(df, right):
    """Test output for agg."""
    right = right.sort_values("Dates", ignore_index=True)
    expected = (
        df.reset_index(names="l")
        .merge(right, how="cross")
        .query("E < Dates and B <= Numeric and A!=Integers")
        .groupby("l")
        .agg({"Numeric": ["size", "min", "max"], "Integers": ["prod", "sum"]})
    )
    expected.index.names = [None]
    actual = df.join_agg(
        right,
        ("E", "Dates", "<"),
        ("B", "Numeric", "<="),
        ("A", "Integers", "!="),
        aggfunc=[
            ("Numeric", "size"),
            ("Numeric", "min"),
            ("Numeric", "max"),
            ("Integers", "prod"),
            ("Integers", "sum"),
        ],
    )
    actual = actual.loc[expected.index]
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_multiple_range_aggs(df, right):
    """Test output for agg."""
    right = right.sort_values("Dates", ignore_index=True)
    expected = (
        df.reset_index(names="l")
        .merge(right, how="cross")
        .query("E > Dates and B < Numeric and A!=Integers")
        .groupby("l")
        .agg({"Numeric": ["size", "min", "max"], "Integers": ["prod", "sum"]})
    )
    expected.index.names = [None]
    actual = df.join_agg(
        right,
        ("E", "Dates", ">"),
        ("B", "Numeric", "<"),
        ("A", "Integers", "!="),
        aggfunc=[
            ("Numeric", "size"),
            ("Numeric", "min"),
            ("Numeric", "max"),
            ("Integers", "prod"),
            ("Integers", "sum"),
        ],
    )
    actual = actual.loc[expected.index]
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_multiple_range_ne_agg(df, right):
    """Test output for agg."""
    right = right.sort_values("Dates", ignore_index=True)
    expected = (
        df.reset_index(names="l")
        .merge(right, how="cross")
        .query("E > Dates and E <= Dates_Right and B < Numeric and A!=Integers")
        .groupby("l")
        .agg({"Numeric": ["size", "min", "max"], "Integers": ["prod", "sum"]})
    )
    expected.index.names = [None]
    actual = df.join_agg(
        right,
        ("E", "Dates", ">"),
        ("E", "Dates_Right", "<="),
        ("B", "Numeric", "<"),
        ("A", "Integers", "!="),
        aggfunc=[
            ("Numeric", "size"),
            ("Numeric", "min"),
            ("Numeric", "max"),
            ("Integers", "prod"),
            ("Integers", "sum"),
        ],
    )
    actual = actual.loc[expected.index]
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_range_only_agg(df, right):
    """Test output for agg."""
    right = right.sort_values("Dates", ignore_index=True)
    expected = (
        df.reset_index(names="l")
        .merge(right, how="cross")
        .query("E > Dates and B < Numeric")
        .groupby("l")
        .agg({"Numeric": ["size", "min", "max"], "Integers": ["prod", "sum"]})
    )
    expected.index.names = [None]
    actual = df.join_agg(
        right,
        ("E", "Dates", ">"),
        ("B", "Numeric", "<"),
        aggfunc=[
            ("Numeric", "size"),
            ("Numeric", "min"),
            ("Numeric", "max"),
            ("Integers", "prod"),
            ("Integers", "sum"),
        ],
    )
    actual = actual.loc[expected.index]
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_equi_agg(df, right):
    """Test output for agg."""
    right = right.sort_values("Dates", ignore_index=True)
    expected = (
        df.reset_index(names="l")
        .merge(right, how="cross")
        .query("E == Dates and A == Integers")
        .groupby("l")
        .agg({"Numeric": ["size", "min", "max"], "Integers": ["prod", "sum"]})
    )
    expected.index.names = [None]
    actual = df.join_agg(
        right,
        ("E", "Dates", "=="),
        ("A", "Integers", "=="),
        aggfunc=[
            ("Numeric", "size"),
            ("Numeric", "min"),
            ("Numeric", "max"),
            ("Integers", "prod"),
            ("Integers", "sum"),
        ],
    )
    actual = actual.loc[expected.index]
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_equi_only_agg(df, right):
    """Test output for agg."""
    right = right.sort_values("Dates", ignore_index=True)
    expected = (
        df.reset_index(names="l")
        .merge(right, how="cross")
        .query("E == Dates")
        .groupby("l")
        .agg({"Numeric": ["size", "min", "max"], "Integers": ["prod", "sum"]})
    )
    expected.index.names = [None]
    actual = df.join_agg(
        right,
        ("E", "Dates", "=="),
        aggfunc=[
            ("Numeric", "size"),
            ("Numeric", "min"),
            ("Numeric", "max"),
            ("Integers", "prod"),
            ("Integers", "sum"),
        ],
    )
    actual = actual.loc[expected.index]
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_equi_ne_agg(df, right):
    """Test output for agg."""
    right = right.sort_values("Dates", ignore_index=True)
    expected = (
        df.reset_index(names="l")
        .merge(right, how="cross")
        .query("E == Dates and B!=Numeric")
        .groupby("l")
        .agg({"Numeric": ["size", "min", "max"], "Integers": ["prod", "sum"]})
    )
    expected.index.names = [None]
    actual = df.join_agg(
        right,
        ("E", "Dates", "=="),
        ("B", "Numeric", "!="),
        aggfunc=[
            ("Numeric", "size"),
            ("Numeric", "min"),
            ("Numeric", "max"),
            ("Integers", "prod"),
            ("Integers", "sum"),
        ],
    )
    actual = actual.loc[expected.index]
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_equi_le_ne_agg(df, right):
    """Test output for agg."""
    right = right.sort_values("Dates", ignore_index=True)
    expected = (
        df.reset_index(names="l")
        .merge(right, how="cross")
        .query("E == Dates and B <= Numeric and A != Integers")
        .groupby("l")
        .agg({"Numeric": ["size", "min", "max"], "Integers": ["prod", "sum"]})
    )
    expected.index.names = [None]
    actual = df.join_agg(
        right,
        ("E", "Dates", "=="),
        ("B", "Numeric", "<="),
        ("A", "Integers", "!="),
        aggfunc=[
            ("Numeric", "size"),
            ("Numeric", "min"),
            ("Numeric", "max"),
            ("Integers", "prod"),
            ("Integers", "sum"),
        ],
    )
    actual = actual.loc[expected.index]
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_equi_ge_ne_agg(df, right):
    """Test output for agg."""
    right = right.sort_values("Dates", ignore_index=True)
    expected = (
        df.reset_index(names="l")
        .merge(right, how="cross")
        .query("E == Dates and B >= Numeric and A != Integers")
        .groupby("l")
        .agg({"Numeric": ["size", "min", "max"], "Integers": ["prod", "sum"]})
    )
    expected.index.names = [None]
    actual = df.join_agg(
        right,
        ("E", "Dates", "=="),
        ("B", "Numeric", ">="),
        ("A", "Integers", "!="),
        aggfunc=[
            ("Numeric", "size"),
            ("Numeric", "min"),
            ("Numeric", "max"),
            ("Integers", "prod"),
            ("Integers", "sum"),
        ],
    )
    actual = actual.loc[expected.index]
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_equi_le_ge_agg(df, right):
    """Test output for agg."""
    right = right.sort_values("Dates", ignore_index=True)
    expected = (
        df.reset_index(names="l")
        .merge(right, how="cross")
        .query("E == Dates and B <= Numeric and A >= Integers")
        .groupby("l")
        .agg({"Numeric": ["size", "min", "max"], "Integers": ["prod", "sum"]})
    )
    expected.index.names = [None]
    actual = df.join_agg(
        right,
        ("E", "Dates", "=="),
        ("B", "Numeric", "<="),
        ("A", "Integers", ">="),
        aggfunc=[
            ("Numeric", "size"),
            ("Numeric", "min"),
            ("Numeric", "max"),
            ("Integers", "prod"),
            ("Integers", "sum"),
        ],
    )
    actual = actual.loc[expected.index]
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_equi_le_ge_ne_agg(df, right):
    """Test output for agg."""
    right = right.sort_values("Dates", ignore_index=True)
    expected = (
        df.reset_index(names="l")
        .merge(right, how="cross")
        .query("E == Dates and B <= Numeric and A >= Integers and E!=Dates_Right")
        .groupby("l")
        .agg({"Numeric": ["size", "min", "max"], "Integers": ["prod", "sum"]})
    )
    expected.index.names = [None]
    actual = df.join_agg(
        right,
        ("E", "Dates", "=="),
        ("B", "Numeric", "<="),
        ("A", "Integers", ">="),
        ("E", "Dates_Right", "!="),
        aggfunc=[
            ("Numeric", "size"),
            ("Numeric", "min"),
            ("Numeric", "max"),
            ("Integers", "prod"),
            ("Integers", "sum"),
        ],
    )
    actual = actual.loc[expected.index]
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_equi_ge_ge_ne_agg(df, right):
    """Test output for agg."""
    right = right.sort_values("Dates", ignore_index=True)
    expected = (
        df.reset_index(names="l")
        .merge(right, how="cross")
        .query("E == Dates and B > Numeric and A >= Integers and E!=Dates_Right")
        .groupby("l")
        .agg({"Numeric": ["size", "min", "max"], "Integers": ["prod", "sum"]})
    )
    expected.index.names = [None]
    actual = df.join_agg(
        right,
        ("E", "Dates", "=="),
        ("B", "Numeric", ">"),
        ("A", "Integers", ">="),
        ("E", "Dates_Right", "!="),
        aggfunc=[
            ("Numeric", "size"),
            ("Numeric", "min"),
            ("Numeric", "max"),
            ("Integers", "prod"),
            ("Integers", "sum"),
        ],
    )
    actual = actual.loc[expected.index]
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_equi_le_le_ne_agg(df, right):
    """Test output for agg."""
    right = right.sort_values("Dates", ignore_index=True)
    expected = (
        df.reset_index(names="l")
        .merge(right, how="cross")
        .query("E == Dates and B < Numeric and A <= Integers and E!=Dates_Right")
        .groupby("l")
        .agg({"Numeric": ["size", "min", "max"], "Integers": ["prod", "sum"]})
    )
    expected.index.names = [None]
    actual = df.join_agg(
        right,
        ("E", "Dates", "=="),
        ("B", "Numeric", "<"),
        ("A", "Integers", "<="),
        ("E", "Dates_Right", "!="),
        aggfunc=[
            ("Numeric", "size"),
            ("Numeric", "min"),
            ("Numeric", "max"),
            ("Integers", "prod"),
            ("Integers", "sum"),
        ],
    )
    actual = actual.loc[expected.index]
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_equi_le_ge_ge_ne_agg(df, right):
    """Test output for agg."""
    right = right.sort_values("Dates", ignore_index=True)
    expected = (
        df.reset_index(names="l")
        .merge(right, how="cross")
        .query(
            "E == Dates and B <= Numeric "
            "and B > Floats and A >= Integers "
            "and E!=Dates_Right"
        )
        .groupby("l")
        .agg({"Numeric": ["size", "min", "max"], "Integers": ["prod", "sum"]})
    )
    expected.index.names = [None]
    actual = df.join_agg(
        right,
        ("E", "Dates", "=="),
        ("B", "Numeric", "<="),
        ("B", "Floats", ">"),
        ("A", "Integers", ">="),
        ("E", "Dates_Right", "!="),
        aggfunc=[
            ("Numeric", "size"),
            ("Numeric", "min"),
            ("Numeric", "max"),
            ("Integers", "prod"),
            ("Integers", "sum"),
        ],
    )
    actual = actual.loc[expected.index]
    assert_frame_equal(expected, actual)


def test_join_positions():
    """
    Test output for include_join_positions
    """
    df1 = pd.DataFrame({"id": [1, 1, 1, 2, 2, 3], "value_1": [2, 5, 7, 1, 3, 4]})
    df2 = pd.DataFrame(
        {
            "id": [1, 1, 1, 1, 2, 2, 2, 3],
            "value_2A": [0, 3, 7, 12, 0, 2, 3, 1],
            "value_2B": [1, 5, 9, 15, 1, 4, 6, 3],
        }
    )
    actual = df1.conditional_join(
        df2,
        ("value_1", "value_2A", ">"),
        ("value_1", "value_2B", "<"),
        ("id", "id", "=="),
        include_join_positions=True,
        right_columns="value*",
    )
    expected = (
        df1.reset_index(names=["l"])
        .merge(df2.reset_index(names=["r"]), on="id")
        .query("value_2A<value_1<value_2B")
        .set_index(["l", "r"])
    )
    expected.index.names = [None, None]
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_single_condition_less_than_dates_agg_rev(df, right):
    """Test output for a single condition. "<"."""
    right = right.sort_values("Dates", ignore_index=True)
    expected = (
        df.merge(right.reset_index(names="l"), how="cross")
        .query("E < Dates")
        .groupby("l")
        .agg(
            {
                "B": [
                    "size",
                    "min",
                    "max",
                ],
                "A": ["prod", "sum"],
            }
        )
        .sort_index()
    )
    expected.index.names = [None]

    actual = df.join_agg(
        right,
        ("E", "Dates", "<"),
        reverse=True,
        aggfunc=[
            ("B", "size"),
            ("B", "min"),
            ("B", "max"),
            ("A", "prod"),
            ("A", "sum"),
        ],
    ).sort_index()
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_single_condition_greater_than_dates_agg_rev(df, right):
    """Test output for a single condition. ">"."""
    right = right.sort_values("Dates", ignore_index=True)
    expected = (
        df.merge(right.reset_index(names="l"), how="cross")
        .query("E > Dates")
        .groupby("l")
        .agg(
            {
                "B": [
                    "size",
                    "min",
                    "max",
                ],
                "A": ["prod", "sum"],
            }
        )
    ).sort_index()
    expected.index.names = [None]

    actual = df.join_agg(
        right,
        ("E", "Dates", ">"),
        reverse=True,
        aggfunc=[
            ("B", "size"),
            ("B", "min"),
            ("B", "max"),
            ("A", "prod"),
            ("A", "sum"),
        ],
    ).sort_index()
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_gt_ne_agg_rev(df, right):
    """Test output for agg."""
    right = right.sort_values("Dates", ignore_index=True)
    expected = (
        df.merge(right.reset_index(names="l"), how="cross")
        .query("E > Dates and B != Numeric")
        .groupby("l")
        .agg({"B": ["size", "min", "max"], "A": ["prod", "sum"]})
    ).sort_index()
    expected.index.names = [None]

    actual = df.join_agg(
        right,
        ("E", "Dates", ">"),
        ("B", "Numeric", "!="),
        aggfunc=[
            ("B", "size"),
            ("B", "min"),
            ("B", "max"),
            ("A", "prod"),
            ("A", "sum"),
        ],
        reverse=True,
    ).sort_index()
    actual = actual.loc[expected.index]
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_lt_ne_agg_rev(df, right):
    """Test output for agg."""
    right = right.sort_values("Dates", ignore_index=True)
    expected = (
        df.merge(right.reset_index(names="l"), how="cross")
        .query("E < Dates and B != Numeric")
        .groupby("l")
        .agg({"B": ["size", "min", "max"], "A": ["prod", "sum"]})
    ).sort_index()
    expected.index.names = [None]

    actual = df.join_agg(
        right,
        ("E", "Dates", "<"),
        ("B", "Numeric", "!="),
        aggfunc=[
            ("B", "size"),
            ("B", "min"),
            ("B", "max"),
            ("A", "prod"),
            ("A", "sum"),
        ],
        reverse=True,
    ).sort_index()
    actual = actual.loc[expected.index]
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_dual_gt_agg_rev(df, right):
    """Test output for agg."""
    right = right.sort_values("Dates", ignore_index=True)
    expected = (
        df.merge(right.reset_index(names="l"), how="cross")
        .query("E > Dates and B > Numeric")
        .groupby("l")
        .agg({"B": ["size", "min", "max"], "A": ["prod", "sum"]})
    ).sort_index()
    expected.index.names = [None]
    actual = df.join_agg(
        right,
        ("E", "Dates", ">"),
        ("B", "Numeric", ">"),
        aggfunc=[
            ("B", "size"),
            ("B", "min"),
            ("B", "max"),
            ("A", "prod"),
            ("A", "sum"),
        ],
        reverse=True,
    ).sort_index()
    actual = actual.loc[expected.index]
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_dual_lt_agg_rev(df, right):
    """Test output for agg."""
    right = right.sort_values("Dates", ignore_index=True)
    expected = (
        df.merge(right.reset_index(names="l"), how="cross")
        .query("E < Dates and B <= Numeric")
        .groupby("l")
        .agg({"B": ["size", "min", "max"], "A": ["prod", "sum"]})
    ).sort_index()
    expected.index.names = [None]
    actual = df.join_agg(
        right,
        ("E", "Dates", "<"),
        ("B", "Numeric", "<="),
        aggfunc=[
            ("B", "size"),
            ("B", "min"),
            ("B", "max"),
            ("A", "prod"),
            ("A", "sum"),
        ],
        reverse=True,
    ).sort_index()
    actual = actual.loc[expected.index]
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_multiple__ge__agg_rev(df, right):
    """Test output for agg."""
    right = right.sort_values("Dates", ignore_index=True)
    expected = (
        df.merge(right.reset_index(names="l"), how="cross")
        .query("E > Dates and B > Numeric and A!=Integers")
        .groupby("l")
        .agg({"B": ["size", "min", "max"], "A": ["prod", "sum"]})
    ).sort_index()
    expected.index.names = [None]
    actual = df.join_agg(
        right,
        ("E", "Dates", ">"),
        ("B", "Numeric", ">"),
        ("A", "Integers", "!="),
        aggfunc=[
            ("B", "size"),
            ("B", "min"),
            ("B", "max"),
            ("A", "prod"),
            ("A", "sum"),
        ],
        reverse=True,
    ).sort_index()
    actual = actual.loc[expected.index]
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_multiple__le__agg_rev(df, right):
    """Test output for agg."""
    right = right.sort_values("Dates", ignore_index=True)
    expected = (
        df.merge(right.reset_index(names="l"), how="cross")
        .query("E < Dates and B <= Numeric and A!=Integers")
        .groupby("l")
        .agg({"B": ["size", "min", "max"], "A": ["prod", "sum"]})
    ).sort_index()
    expected.index.names = [None]
    actual = df.join_agg(
        right,
        ("E", "Dates", "<"),
        ("B", "Numeric", "<="),
        ("A", "Integers", "!="),
        aggfunc=[
            ("B", "size"),
            ("B", "min"),
            ("B", "max"),
            ("A", "prod"),
            ("A", "sum"),
        ],
        reverse=True,
    ).sort_index()
    actual = actual.loc[expected.index]
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_multiple_range_aggs_rev(df, right):
    """Test output for agg."""
    right = right.sort_values("Dates", ignore_index=True)
    expected = (
        df.merge(right.reset_index(names="l"), how="cross")
        .query("E > Dates and B < Numeric and A!=Integers")
        .groupby("l")
        .agg({"B": ["size", "min", "max"], "A": ["prod", "sum"]})
    ).sort_index()
    expected.index.names = [None]
    actual = df.join_agg(
        right,
        ("E", "Dates", ">"),
        ("B", "Numeric", "<"),
        ("A", "Integers", "!="),
        aggfunc=[
            ("B", "size"),
            ("B", "min"),
            ("B", "max"),
            ("A", "prod"),
            ("A", "sum"),
        ],
        reverse=True,
    ).sort_index()
    actual = actual.loc[expected.index]
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_multiple_range_ne_agg_rev(df, right):
    """Test output for agg."""
    right = right.sort_values("Dates", ignore_index=True)
    expected = (
        df.merge(right.reset_index(names="l"), how="cross")
        .query("E > Dates and E <= Dates_Right and B < Numeric and A!=Integers")
        .groupby("l")
        .agg({"B": ["size", "min", "max"], "A": ["prod", "sum"]})
    ).sort_index()
    expected.index.names = [None]
    actual = df.join_agg(
        right,
        ("E", "Dates", ">"),
        ("E", "Dates_Right", "<="),
        ("B", "Numeric", "<"),
        ("A", "Integers", "!="),
        aggfunc=[
            ("B", "size"),
            ("B", "min"),
            ("B", "max"),
            ("A", "prod"),
            ("A", "sum"),
        ],
        reverse=True,
    ).sort_index()
    actual = actual.loc[expected.index]
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_range_only_agg_rev(df, right):
    """Test output for agg."""
    right = right.sort_values("Dates", ignore_index=True)
    expected = (
        df.merge(right.reset_index(names="l"), how="cross")
        .query("E > Dates and B < Numeric")
        .groupby("l")
        .agg({"B": ["size", "min", "max"], "A": ["prod", "sum"]})
    ).sort_index()
    expected.index.names = [None]
    actual = df.join_agg(
        right,
        ("E", "Dates", ">"),
        ("B", "Numeric", "<"),
        aggfunc=[
            ("B", "size"),
            ("B", "min"),
            ("B", "max"),
            ("A", "prod"),
            ("A", "sum"),
        ],
        reverse=True,
    ).sort_index()
    actual = actual.loc[expected.index]
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_equi_agg_rev(df, right):
    """Test output for agg."""
    right = right.sort_values("Dates", ignore_index=True)
    expected = (
        df.merge(right.reset_index(names="l"), how="cross")
        .query("E == Dates and A == Integers")
        .groupby("l")
        .agg({"B": ["size", "min", "max"], "A": ["prod", "sum"]})
    ).sort_index()
    expected.index.names = [None]
    actual = df.join_agg(
        right,
        ("E", "Dates", "=="),
        ("A", "Integers", "=="),
        aggfunc=[
            ("B", "size"),
            ("B", "min"),
            ("B", "max"),
            ("A", "prod"),
            ("A", "sum"),
        ],
        reverse=True,
    ).sort_index()
    actual = actual.loc[expected.index]
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_equi_only_agg_rev(df, right):
    """Test output for agg."""
    right = right.sort_values("Dates", ignore_index=True)
    expected = (
        df.merge(right.reset_index(names="l"), how="cross")
        .query("E == Dates")
        .groupby("l")
        .agg({"B": ["size", "min", "max"], "A": ["prod", "sum"]})
    ).sort_index()
    expected.index.names = [None]
    actual = df.join_agg(
        right,
        ("E", "Dates", "=="),
        aggfunc=[
            ("B", "size"),
            ("B", "min"),
            ("B", "max"),
            ("A", "prod"),
            ("A", "sum"),
        ],
        reverse=True,
    ).sort_index()
    actual = actual.loc[expected.index]
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_equi_ne_agg_rev(df, right):
    """Test output for agg."""
    right = right.sort_values("Dates", ignore_index=True)
    expected = (
        df.merge(right.reset_index(names="l"), how="cross")
        .query("E == Dates and B!=Numeric")
        .groupby("l")
        .agg({"B": ["size", "min", "max"], "A": ["prod", "sum"]})
    ).sort_index()
    expected.index.names = [None]
    actual = df.join_agg(
        right,
        ("E", "Dates", "=="),
        ("B", "Numeric", "!="),
        aggfunc=[
            ("B", "size"),
            ("B", "min"),
            ("B", "max"),
            ("A", "prod"),
            ("A", "sum"),
        ],
        reverse=True,
    ).sort_index()
    actual = actual.loc[expected.index]
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_equi_le_ne_agg_rev(df, right):
    """Test output for agg."""
    right = right.sort_values("Dates", ignore_index=True)
    expected = (
        df.merge(right.reset_index(names="l"), how="cross")
        .query("E == Dates and B <= Numeric and A != Integers")
        .groupby("l")
        .agg({"B": ["size", "min", "max"], "A": ["prod", "sum"]})
    ).sort_index()
    expected.index.names = [None]
    actual = df.join_agg(
        right,
        ("E", "Dates", "=="),
        ("B", "Numeric", "<="),
        ("A", "Integers", "!="),
        aggfunc=[
            ("B", "size"),
            ("B", "min"),
            ("B", "max"),
            ("A", "prod"),
            ("A", "sum"),
        ],
        reverse=True,
    ).sort_index()
    actual = actual.loc[expected.index]
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_equi_ge_ne_agg_rev(df, right):
    """Test output for agg."""
    right = right.sort_values("Dates", ignore_index=True)
    expected = (
        df.merge(right.reset_index(names="l"), how="cross")
        .query("E == Dates and B >= Numeric and A != Integers")
        .groupby("l")
        .agg({"B": ["size", "min", "max"], "A": ["prod", "sum"]})
    ).sort_index()
    expected.index.names = [None]
    actual = df.join_agg(
        right,
        ("E", "Dates", "=="),
        ("B", "Numeric", ">="),
        ("A", "Integers", "!="),
        aggfunc=[
            ("B", "size"),
            ("B", "min"),
            ("B", "max"),
            ("A", "prod"),
            ("A", "sum"),
        ],
        reverse=True,
    ).sort_index()
    actual = actual.loc[expected.index]
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_equi_le_ge_agg_rev(df, right):
    """Test output for agg."""
    right = right.sort_values("Dates", ignore_index=True)
    expected = (
        df.merge(right.reset_index(names="l"), how="cross")
        .query("E == Dates and B <= Numeric and A >= Integers")
        .groupby("l")
        .agg({"B": ["size", "min", "max"], "A": ["prod", "sum"]})
    ).sort_index()
    expected.index.names = [None]
    actual = df.join_agg(
        right,
        ("E", "Dates", "=="),
        ("B", "Numeric", "<="),
        ("A", "Integers", ">="),
        aggfunc=[
            ("B", "size"),
            ("B", "min"),
            ("B", "max"),
            ("A", "prod"),
            ("A", "sum"),
        ],
        reverse=True,
    ).sort_index()
    actual = actual.loc[expected.index]
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_equi_le_ge_ne_agg_rev(df, right):
    """Test output for agg."""
    right = right.sort_values("Dates", ignore_index=True)
    expected = (
        df.merge(right.reset_index(names="l"), how="cross")
        .query("E == Dates and B <= Numeric and A >= Integers and E!=Dates_Right")
        .groupby("l")
        .agg({"B": ["size", "min", "max"], "A": ["prod", "sum"]})
    ).sort_index()
    expected.index.names = [None]
    actual = df.join_agg(
        right,
        ("E", "Dates", "=="),
        ("B", "Numeric", "<="),
        ("A", "Integers", ">="),
        ("E", "Dates_Right", "!="),
        aggfunc=[
            ("B", "size"),
            ("B", "min"),
            ("B", "max"),
            ("A", "prod"),
            ("A", "sum"),
        ],
        reverse=True,
    ).sort_index()
    actual = actual.loc[expected.index]
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_equi_ge_ge_ne_agg_rev(df, right):
    """Test output for agg."""
    right = right.sort_values("Dates", ignore_index=True)
    expected = (
        df.merge(right.reset_index(names="l"), how="cross")
        .query("E == Dates and B > Numeric and A >= Integers and E!=Dates_Right")
        .groupby("l")
        .agg({"B": ["size", "min", "max"], "A": ["prod", "sum"]})
    ).sort_index()
    expected.index.names = [None]
    actual = df.join_agg(
        right,
        ("E", "Dates", "=="),
        ("B", "Numeric", ">"),
        ("A", "Integers", ">="),
        ("E", "Dates_Right", "!="),
        aggfunc=[
            ("B", "size"),
            ("B", "min"),
            ("B", "max"),
            ("A", "prod"),
            ("A", "sum"),
        ],
        reverse=True,
    ).sort_index()
    actual = actual.loc[expected.index]
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_equi_le_le_ne_agg_rev(df, right):
    """Test output for agg."""
    right = right.sort_values("Dates", ignore_index=True)
    expected = (
        df.merge(right.reset_index(names="l"), how="cross")
        .query("E == Dates and B < Numeric and A <= Integers and E!=Dates_Right")
        .groupby("l")
        .agg({"B": ["size", "min", "max"], "A": ["prod", "sum"]})
    ).sort_index()
    expected.index.names = [None]
    actual = df.join_agg(
        right,
        ("E", "Dates", "=="),
        ("B", "Numeric", "<"),
        ("A", "Integers", "<="),
        ("E", "Dates_Right", "!="),
        aggfunc=[
            ("B", "size"),
            ("B", "min"),
            ("B", "max"),
            ("A", "prod"),
            ("A", "sum"),
        ],
        reverse=True,
    ).sort_index()
    actual = actual.loc[expected.index]
    assert_frame_equal(expected, actual)


@pytest.mark.turtle
@settings(deadline=None, max_examples=10)
@given(df=conditional_df(), right=conditional_right())
def test_equi_le_ge_ge_ne_agg_rev(df, right):
    """Test output for agg."""
    right = right.sort_values("Dates", ignore_index=True)
    expected = (
        df.merge(right.reset_index(names="l"), how="cross")
        .query(
            "E == Dates and B <= Numeric "
            "and B > Floats and A >= Integers "
            "and E!=Dates_Right"
        )
        .groupby("l")
        .agg({"B": ["size", "min", "max"], "A": ["prod", "sum"]})
    ).sort_index()
    expected.index.names = [None]
    actual = df.join_agg(
        right,
        ("E", "Dates", "=="),
        ("B", "Numeric", "<="),
        ("B", "Floats", ">"),
        ("A", "Integers", ">="),
        ("E", "Dates_Right", "!="),
        aggfunc=[
            ("B", "size"),
            ("B", "min"),
            ("B", "max"),
            ("A", "prod"),
            ("A", "sum"),
        ],
        reverse=True,
    ).sort_index()
    actual = actual.loc[expected.index]
    assert_frame_equal(expected, actual)
