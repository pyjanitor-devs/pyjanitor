import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given
from pandas.testing import assert_frame_equal
from janitor.testing_utils.strategies import (
    conditional_df,
    conditional_right,
    conditional_series,
)


@pytest.mark.xfail(reason="empty object will pass thru")
@given(s=conditional_series())
def test_df_empty(s):
    """Raise ValueError if `df` is empty."""
    df = pd.DataFrame([], dtype="int", columns=["A"])
    with pytest.raises(ValueError):
        df.conditional_join(s, ("A", "non", "=="))


@pytest.mark.xfail(reason="empty object will pass thru")
@given(df=conditional_df())
def test_right_empty(df):
    """Raise ValueError if `right` is empty."""
    s = pd.Series([], dtype="int", name="A")
    with pytest.raises(ValueError):
        df.conditional_join(s, ("A", "non", "=="))


@given(df=conditional_df())
def test_right_df(df):
    """Raise TypeError if `right` is not a Series/DataFrame."""
    with pytest.raises(TypeError):
        df.conditional_join({"non": [2, 3, 4]}, ("A", "non", "=="))


@given(df=conditional_df(), s=conditional_series())
def test_right_series(df, s):
    """Raise ValueError if `right` is not a named Series."""
    with pytest.raises(ValueError):
        df.conditional_join(s, ("A", "non", "=="))


@given(df=conditional_df())
def test_df_MultiIndex(df):
    """Raise ValueError if `df` columns is a MultiIndex."""

    with pytest.raises(ValueError):
        df.columns = [list("ABCDE"), list("FGHIJ")]
        df.conditional_join(
            pd.Series([2, 3, 4], name="A"), (("A", "F"), "non", "==")
        )


@given(df=conditional_df())
def test_right_MultiIndex(df):
    """Raise ValueError if `right` columns is a MultiIndex."""

    with pytest.raises(ValueError):
        right = df.copy()
        right.columns = [list("ABCDE"), list("FGHIJ")]
        df.conditional_join(right, (("A", "F"), "non", ">="))


@given(df=conditional_df(), s=conditional_series())
def test_check_conditions_exist(df, s):
    """Raise ValueError if no condition is provided."""

    with pytest.raises(ValueError):
        s.name = "B"
        df.conditional_join(s)


@given(df=conditional_df(), s=conditional_series())
def test_check_condition_type(df, s):
    """Raise TypeError if any condition in conditions is not a tuple."""

    with pytest.raises(TypeError):
        s.name = "B"
        df.conditional_join(s, ("A", "B", ""), ["A", "B"])


@given(df=conditional_df(), s=conditional_series())
def test_check_condition_length(df, s):
    """Raise ValueError if any condition is not length 3."""

    with pytest.raises(ValueError):
        s.name = "B"
        df.conditional_join(s, ("A", "B", "C", "<"))
        df.conditional_join(s, ("A", "B", ""), ("A", "B"))


@given(df=conditional_df(), s=conditional_series())
def test_check_left_on_type(df, s):
    """Raise TypeError if left_on is not a string."""

    with pytest.raises(TypeError):
        s.name = "B"
        df.conditional_join(s, (1, "B", "<"))


@given(df=conditional_df(), s=conditional_series())
def test_check_right_on_type(df, s):
    """Raise TypeError if right_on is not a string."""

    with pytest.raises(TypeError):
        s.name = "B"
        df.conditional_join(s, ("B", 1, "<"))


@given(df=conditional_df(), s=conditional_series())
def test_check_op_type(df, s):
    """Raise TypeError if the operator is not a string."""

    with pytest.raises(TypeError):
        s.name = "B"
        df.conditional_join(s, ("B", "B", 1))


@given(df=conditional_df(), s=conditional_series())
def test_check_column_exists_df(df, s):
    """
    Raise ValueError if `left_on`
    can not be found in `df`.
    """

    with pytest.raises(ValueError):
        s.name = "B"
        df.conditional_join(s, ("C", "B", "<"))


@given(df=conditional_df(), s=conditional_series())
def test_check_column_exists_right(df, s):
    """
    Raise ValueError if `right_on`
    can not be found in `right`.
    """

    with pytest.raises(ValueError):
        s.name = "B"
        df.conditional_join(s, ("B", "A", ">="))


@given(df=conditional_df(), s=conditional_series())
def test_check_op_correct(df, s):
    """
    Raise ValueError if `op` is not any of
     `!=`, `<`, `>`, `>=`, `<=`.
    """

    with pytest.raises(ValueError):
        s.name = "B"
        df.conditional_join(s, ("B", "B", "=!"))


@given(df=conditional_df(), s=conditional_series())
def test_check_how_type(df, s):
    """
    Raise TypeError if `how` is not a string.
    """

    with pytest.raises(TypeError):
        s.name = "B"
        df.conditional_join(s, ("B", "B", "<"), how=1)


@given(df=conditional_df(), s=conditional_series())
def test_check_how_value(df, s):
    """
    Raise ValueError if `how` is not one of
    `inner`, `left`, or `right`.
    """

    with pytest.raises(ValueError):
        s.name = "B"
        df.conditional_join(s, ("B", "B", "<"), how="INNER")


@given(df=conditional_df(), right=conditional_right())
def test_dtype_strings_non_equi(df, right):
    """
    Raise ValueError if the dtypes are both strings
    on a non-equi operator.
    """
    with pytest.raises(ValueError):
        df.conditional_join(right, ("C", "Strings", "<"))


@given(df=conditional_df(), s=conditional_series())
def test_dtype_not_permitted(df, s):
    """
    Raise ValueError if dtype of column in `df`
    is not an acceptable type.
    """
    df["F"] = pd.Timedelta("1 days")
    with pytest.raises(ValueError):
        s.name = "A"
        df.conditional_join(s, ("F", "A", "<"))


@given(df=conditional_df(), s=conditional_series())
def test_dtype_str(df, s):
    """
    Raise ValueError if dtype of column in `df`
    does not match the dtype of column from `right`.
    """
    with pytest.raises(ValueError):
        s.name = "A"
        df.conditional_join(s, ("C", "A", "<"))


@pytest.mark.xfail(reason="binary search does not support categoricals")
@given(df=conditional_df(), s=conditional_series())
def test_dtype_category_non_equi(df, s):
    """
    Raise ValueError if dtype is category,
    and op is non-equi.
    """
    with pytest.raises(ValueError):
        s.name = "A"
        s = s.astype("category")
        df["C"] = df["C"].astype("category")
        df.conditional_join(s, ("C", "A", "<"))


@given(df=conditional_df(), s=conditional_series())
def test_check_sort_by_appearance_type(df, s):
    """
    Raise TypeError if `sort_by_appearance` is not a boolean.
    """

    with pytest.raises(TypeError):
        s.name = "B"
        df.conditional_join(s, ("B", "B", "<"), sort_by_appearance="True")


@given(df=conditional_df(), right=conditional_right())
def test_single_condition_less_than_floats(df, right):
    """Test output for a single condition. "<"."""

    assume(not right.empty)
    left_on, right_on = ["B", "Numeric"]
    expected = (
        df.assign(t=1)
        .merge(right.assign(t=1), on="t")
        .query(f"{left_on} < {right_on}")
        .reset_index(drop=True)
    )
    expected = expected.filter([left_on, right_on])
    actual = df.conditional_join(
        right, (left_on, right_on, "<"), how="inner", sort_by_appearance=True
    )
    actual = actual.droplevel(level=0, axis=1)
    actual = actual.filter([left_on, right_on])
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_single_condition_less_than_ints(df, right):
    """Test output for a single condition. "<"."""

    assume(not right.empty)
    left_on, right_on = ["A", "Integers"]
    expected = (
        df.assign(t=1)
        .merge(right.assign(t=1), on="t")
        .query(f"{left_on} < {right_on}")
        .reset_index(drop=True)
    )
    expected = expected.filter([left_on, right_on])
    actual = df.conditional_join(
        right, (left_on, right_on, "<"), how="inner", sort_by_appearance=True
    )
    actual = actual.droplevel(level=0, axis=1)
    actual = actual.filter([left_on, right_on])
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_single_condition_less_than_ints_extension_array(df, right):
    """Test output for a single condition. "<"."""

    assume(not right.empty)
    df = df.assign(A=df["A"].astype("Int64"))
    right = right.assign(Integers=right["Integers"].astype(pd.Int64Dtype()))
    left_on, right_on = ["A", "Integers"]
    expected = (
        df.assign(t=1)
        .merge(right.assign(t=1), on="t")
        .query(f"{left_on} < {right_on}")
        .reset_index(drop=True)
    )
    expected = expected.filter([left_on, right_on])
    actual = df.conditional_join(
        right, (left_on, right_on, "<"), how="inner", sort_by_appearance=True
    )
    actual = actual.droplevel(level=0, axis=1)
    actual = actual.filter([left_on, right_on])
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_single_condition_less_than_equal(df, right):
    """Test output for a single condition. "<=". DateTimes"""

    assume(not right.empty)
    left_on, right_on = ["E", "Dates"]
    expected = (
        df.assign(t=1)
        .merge(right.assign(t=1), on="t")
        .query(f"{left_on} <= {right_on}")
        .reset_index(drop=True)
    )
    expected = expected.filter([left_on, right_on])
    actual = df.conditional_join(
        right, (left_on, right_on, "<="), how="inner", sort_by_appearance=True
    )
    actual = actual.droplevel(level=0, axis=1)
    actual = actual.filter([left_on, right_on])
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_single_condition_less_than_date(df, right):
    """Test output for a single condition. "<". Dates"""

    assume(not right.empty)
    left_on, right_on = ["E", "Dates"]
    expected = (
        df.assign(t=1)
        .merge(right.assign(t=1), on="t")
        .query(f"{left_on} < {right_on}")
        .reset_index(drop=True)
    )
    expected = expected.filter([left_on, right_on])
    actual = df.conditional_join(
        right, (left_on, right_on, "<"), how="inner", sort_by_appearance=True
    )
    actual = actual.droplevel(level=0, axis=1)
    actual = actual.filter([left_on, right_on])
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_single_condition_greater_than_datetime(df, right):
    """Test output for a single condition. ">". Datetimes"""

    assume(not right.empty)
    left_on, right_on = ["E", "Dates"]
    expected = (
        df.assign(t=1)
        .merge(right.assign(t=1), on="t")
        .query(f"{left_on} > {right_on}")
        .reset_index(drop=True)
    )
    expected = expected.filter([left_on, right_on])
    actual = df.conditional_join(
        right, (left_on, right_on, ">"), how="inner", sort_by_appearance=True
    )
    actual = actual.droplevel(level=0, axis=1)
    actual = actual.filter([left_on, right_on])
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_single_condition_greater_than_ints(df, right):
    """Test output for a single condition. ">="."""

    assume(not right.empty)
    left_on, right_on = ["A", "Integers"]
    expected = (
        df.assign(t=1)
        .merge(right.assign(t=1), on="t")
        .query(f"{left_on} >= {right_on}")
        .reset_index(drop=True)
    )
    expected = expected.filter([left_on, right_on])
    actual = df.conditional_join(
        right, (left_on, right_on, ">="), how="inner", sort_by_appearance=True
    )
    actual = actual.droplevel(level=0, axis=1)
    actual = actual.filter([left_on, right_on])
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_single_condition_greater_than_floats_floats(df, right):
    """Test output for a single condition. ">"."""

    assume(not right.empty)
    left_on, right_on = ["B", "Numeric"]
    expected = (
        df.assign(t=1)
        .merge(right.assign(t=1), on="t")
        .query(f"{left_on} > {right_on}")
        .reset_index(drop=True)
    )
    expected = expected.filter([left_on, right_on])
    actual = df.conditional_join(
        right, (left_on, right_on, ">"), how="inner", sort_by_appearance=True
    )
    actual = actual.droplevel(level=0, axis=1)
    actual = actual.filter([left_on, right_on])
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_single_condition_greater_than_ints_extension_array(df, right):
    """Test output for a single condition. ">="."""

    assume(not right.empty)
    left_on, right_on = ["A", "Integers"]
    df = df.assign(A=df["A"].astype("Int64"))
    right = right.assign(Integers=right["Integers"].astype(pd.Int64Dtype()))
    expected = (
        df.assign(t=1)
        .merge(right.assign(t=1), on="t")
        .query(f"{left_on} > {right_on}")
        .reset_index(drop=True)
    )
    expected = expected.filter([left_on, right_on])
    actual = df.conditional_join(
        right, (left_on, right_on, ">"), how="inner", sort_by_appearance=True
    )
    actual = actual.droplevel(level=0, axis=1)
    actual = actual.filter([left_on, right_on])
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_single_condition_not_equal_numeric(df, right):
    """Test output for a single condition. "!="."""

    assume(not right.empty)
    left_on, right_on = ["A", "Integers"]
    expected = (
        df.assign(t=1)
        .merge(right.assign(t=1), on="t")
        .dropna(subset=["A", "Integers"])
        .query(f"{left_on} != {right_on}")
        .reset_index(drop=True)
    )
    expected = expected.filter([left_on, right_on])
    actual = df.conditional_join(
        right, (left_on, right_on, "!="), how="inner", sort_by_appearance=True
    )
    actual = actual.droplevel(level=0, axis=1)
    actual = actual.filter([left_on, right_on])
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_single_condition_not_equal_ints_only(df, right):
    """Test output for a single condition. "!="."""

    assume(not right.empty)
    left_on, right_on = ["A", "Integers"]
    expected = (
        df.assign(t=1)
        .merge(right.assign(t=1), on="t")
        .dropna(subset=["A", "Integers"])
        .query(f"{left_on} != {right_on}")
        .reset_index(drop=True)
    )
    expected = expected.filter([left_on, right_on])
    actual = df.conditional_join(
        right, (left_on, right_on, "!="), how="inner", sort_by_appearance=True
    )
    actual = actual.droplevel(level=0, axis=1)
    actual = actual.filter([left_on, right_on])
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_single_condition_not_equal_floats_only(df, right):
    """Test output for a single condition. "!="."""

    assume(not right.empty)
    left_on, right_on = ["B", "Numeric"]
    expected = (
        df.assign(t=1)
        .merge(right.assign(t=1), on="t")
        .dropna(subset=["B", "Numeric"])
        .query(f"{left_on} != {right_on}")
        .reset_index(drop=True)
    )
    expected = expected.filter([left_on, right_on])
    actual = df.conditional_join(
        right, (left_on, right_on, "!="), how="inner", sort_by_appearance=True
    )
    actual = actual.droplevel(level=0, axis=1)
    actual = actual.filter([left_on, right_on])
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_single_condition_not_equal_datetime(df, right):
    """Test output for a single condition. "!="."""

    assume(not right.empty)
    left_on, right_on = ["E", "Dates"]
    expected = (
        df.assign(t=1)
        .merge(right.assign(t=1), on="t")
        .dropna(subset=["E", "Dates"])
        .query(f"{left_on} != {right_on}")
        .reset_index(drop=True)
    )
    expected = expected.filter([left_on, right_on])
    actual = df.conditional_join(
        right, (left_on, right_on, "!="), how="inner", sort_by_appearance=True
    )
    actual = actual.droplevel(level=0, axis=1)
    actual = actual.filter([left_on, right_on])
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_single_condition_equality_string(df, right):
    """Test output for a single condition. "=="."""

    assume(not right.empty)
    left_on, right_on = ["C", "Strings"]
    expected = df.dropna(subset=[left_on]).merge(
        right.dropna(subset=[right_on]), left_on=left_on, right_on=right_on
    )
    expected = expected.reset_index(drop=True)

    expected = expected.filter([left_on, right_on])
    actual = df.conditional_join(
        right, (left_on, right_on, "=="), how="inner", sort_by_appearance=False
    )
    actual = actual.droplevel(level=0, axis=1)
    actual = actual.filter([left_on, right_on])
    assert_frame_equal(expected, actual)


@pytest.mark.xfail(reason="binary search does not support categoricals")
@given(df=conditional_df(), right=conditional_right())
def test_single_condition_equality_category(df, right):
    """Test output for a single condition. "=="."""

    assume(not right.empty)
    left_on, right_on = ["C", "Strings"]
    df = df.assign(C=df["C"].astype("category"))
    right = right.assign(Strings=right["Strings"].astype("category"))
    expected = df.dropna(subset=[left_on]).merge(
        right.dropna(subset=[right_on]), left_on=left_on, right_on=right_on
    )
    expected = expected.reset_index(drop=True)

    expected = expected.filter([left_on, right_on])
    actual = df.conditional_join(
        right, (left_on, right_on, "=="), how="inner", sort_by_appearance=False
    )
    actual = actual.droplevel(level=0, axis=1)
    actual = actual.filter([left_on, right_on])
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_single_condition_equality_numeric(df, right):
    """Test output for a single condition. "=="."""

    assume(not right.empty)
    left_on, right_on = ["A", "Integers"]
    df = df.assign(A=df["A"].astype("Int64"))
    right = right.assign(Integers=right["Integers"].astype(pd.Int64Dtype()))
    df.loc[0, "A"] = pd.NA
    right.loc[0, "Integers"] = pd.NA
    expected = df.dropna(subset=[left_on]).merge(
        right.dropna(subset=[right_on]), left_on=left_on, right_on=right_on
    )
    expected = expected.reset_index(drop=True)

    expected = expected.filter([left_on, right_on])
    actual = df.conditional_join(
        right, (left_on, right_on, "=="), how="inner", sort_by_appearance=False
    )
    actual = actual.droplevel(level=0, axis=1)
    actual = actual.filter([left_on, right_on])
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_single_condition_equality_datetime(df, right):
    """Test output for a single condition. "=="."""

    assume(not right.empty)
    left_on, right_on = ["E", "Dates"]
    expected = df.dropna(subset=[left_on]).merge(
        right.dropna(subset=[right_on]), left_on=left_on, right_on=right_on
    )
    expected = expected.reset_index(drop=True)
    expected = expected.filter([left_on, right_on])
    actual = df.conditional_join(
        right, (left_on, right_on, "=="), how="inner", sort_by_appearance=False
    )
    actual = actual.droplevel(level=0, axis=1)
    actual = actual.filter([left_on, right_on])
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_how_left(df, right):
    """Test output when `how==left`. "<="."""

    assume(not right.empty)
    left_on, right_on = ["A", "Integers"]
    expected = (
        df.assign(t=1, index=np.arange(len(df)))
        .merge(right.assign(t=1), on="t")
        .query(f"{left_on} <= {right_on}")
    )
    expected = expected.set_index("index")
    expected.index.name = None
    expected = df.join(
        expected.filter(right.columns), how="left", sort=False
    ).reset_index(drop=True)
    actual = df.conditional_join(
        right, (left_on, right_on, "<="), how="left", sort_by_appearance=True
    )
    actual = actual.droplevel(level=0, axis=1)
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_how_right(df, right):
    """Test output when `how==right`. ">"."""

    assume(not right.empty)
    left_on, right_on = ["E", "Dates"]
    expected = (
        df.assign(t=1)
        .merge(right.assign(t=1, index=np.arange(len(right))), on="t")
        .query(f"{left_on} > {right_on}")
    )
    expected = expected.set_index("index")
    expected.index.name = None
    expected = (
        expected.filter(df.columns)
        .join(right, how="right", sort=False)
        .reset_index(drop=True)
    )
    actual = df.conditional_join(
        right, (left_on, right_on, ">"), how="right", sort_by_appearance=True
    )
    actual = actual.droplevel(level=0, axis=1)
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_dual_conditions_gt_and_lt_dates(df, right):
    """Test output for interval conditions."""

    assume(not right.empty)
    middle, left_on, right_on = ("E", "Dates", "Dates_Right")
    expected = (
        df.assign(t=1)
        .merge(right.assign(t=1), on="t")
        .query(f"{left_on} < {middle} < {right_on}")
        .reset_index(drop=True)
    )
    expected = expected.filter([left_on, middle, right_on])
    actual = df.conditional_join(
        right,
        (middle, left_on, ">"),
        (middle, right_on, "<"),
        how="inner",
        sort_by_appearance=True,
    )
    actual = actual.droplevel(level=0, axis=1)
    actual = actual.filter([left_on, middle, right_on])
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_dual_conditions_ge_and_le_dates(df, right):
    """Test output for interval conditions."""

    assume(not right.empty)
    middle, left_on, right_on = ("E", "Dates", "Dates_Right")
    expected = (
        df.assign(t=1)
        .merge(right.assign(t=1), on="t")
        .query(f"{left_on} <= {middle} <= {right_on}")
        .reset_index(drop=True)
    )
    expected = expected.filter([left_on, middle, right_on])
    actual = df.conditional_join(
        right,
        (middle, left_on, ">="),
        (middle, right_on, "<="),
        how="inner",
        sort_by_appearance=True,
    )
    actual = actual.droplevel(level=0, axis=1)
    actual = actual.filter([left_on, middle, right_on])
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_dual_conditions_le_and_ge_dates(df, right):
    """Test output for interval conditions."""

    assume(not right.empty)
    middle, left_on, right_on = ("E", "Dates", "Dates_Right")
    expected = (
        df.assign(t=1)
        .merge(right.assign(t=1), on="t")
        .query(f"{left_on} <= {middle} <= {right_on}")
        .reset_index(drop=True)
    )
    expected = expected.filter([left_on, middle, right_on])
    actual = df.conditional_join(
        right,
        (middle, right_on, "<="),
        (middle, left_on, ">="),
        how="inner",
        sort_by_appearance=True,
    )
    actual = actual.droplevel(level=0, axis=1)
    actual = actual.filter([left_on, middle, right_on])
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_dual_conditions_ge_and_le_numbers(df, right):
    """Test output for interval conditions."""

    assume(not right.empty)
    middle, left_on, right_on = ("B", "Numeric", "Floats")
    expected = (
        df.assign(t=1)
        .merge(right.assign(t=1), on="t")
        .query(f"{left_on} <= {middle} <= {right_on}")
        .reset_index(drop=True)
    )
    expected = expected.filter([left_on, middle, right_on])
    actual = df.conditional_join(
        right,
        (middle, left_on, ">="),
        (middle, right_on, "<="),
        how="inner",
        sort_by_appearance=True,
    )
    actual = actual.droplevel(level=0, axis=1)
    actual = actual.filter([left_on, middle, right_on])
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_dual_conditions_le_and_ge_numbers(df, right):
    """Test output for interval conditions."""

    assume(not right.empty)
    middle, left_on, right_on = ("B", "Numeric", "Floats")
    expected = (
        df.assign(t=1)
        .merge(right.assign(t=1), on="t")
        .query(f"{left_on} <= {middle} <= {right_on}")
        .reset_index(drop=True)
    )
    expected = expected.filter([left_on, middle, right_on])
    actual = df.conditional_join(
        right,
        (middle, right_on, "<="),
        (middle, left_on, ">="),
        how="inner",
        sort_by_appearance=True,
    )
    actual = actual.droplevel(level=0, axis=1)
    actual = actual.filter([left_on, middle, right_on])
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_dual_conditions_gt_and_lt_numbers(df, right):
    """Test output for interval conditions."""

    assume(not right.empty)
    middle, left_on, right_on = ("B", "Numeric", "Floats")
    expected = (
        df.assign(t=1)
        .merge(right.assign(t=1), on="t")
        .query(f"{left_on} < {middle} < {right_on}")
        .reset_index(drop=True)
    )
    expected = expected.filter([left_on, middle, right_on])
    actual = df.conditional_join(
        right,
        (middle, left_on, ">"),
        (middle, right_on, "<"),
        how="inner",
        sort_by_appearance=True,
    )
    actual = actual.droplevel(level=0, axis=1)
    actual = actual.filter([left_on, middle, right_on])
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_dual_conditions_gt_and_lt_numbers_(df, right):
    """
    Test output for multiple conditions.
    """

    assume(not right.empty)
    first, second, third = ("Numeric", "Floats", "B")
    expected = (
        right.assign(t=1)
        .merge(df.assign(t=1), on="t")
        .query(f"{first} > {third} and {second} < {third}")
        .reset_index(drop=True)
    )
    expected = expected.filter([first, second, third])
    actual = right.conditional_join(
        df,
        (first, third, ">"),
        (second, third, "<"),
        how="inner",
        sort_by_appearance=True,
    )
    actual = actual.droplevel(level=0, axis=1)
    actual = actual.filter([first, second, third])
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_dual_conditions_gt_and_lt_numbers_left_join(df, right):
    """
    Test output for multiple conditions, and how is `left`.
    """

    assume(not right.empty)
    first, second, third = ("Numeric", "Floats", "B")
    right = right.assign(t=1, check=range(len(right)))
    df = df.assign(t=1)
    expected = right.merge(df, on="t").query(
        f"{first} > {third} and {second} < {third}"
    )
    drop = right.columns.difference(["check"])
    expected = right.merge(
        expected.drop(columns=[*drop]), on="check", how="left", sort=False
    )
    expected = expected.filter([first, second, third])
    actual = right.conditional_join(
        df,
        (first, third, ">"),
        (second, third, "<"),
        how="left",
        sort_by_appearance=True,
    )
    actual = actual.droplevel(level=0, axis=1)
    actual = actual.loc[:, [first, second, third]]
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_dual_conditions_gt_and_lt_numbers_right_join(df, right):
    """
    Test output for multiple conditions, and how is `right`.
    """

    assume(not right.empty)
    first, second, third = ("Numeric", "Floats", "B")
    df = df.assign(t=1, check=range(len(df)))
    right = right.assign(t=1)
    expected = right.merge(df, on="t").query(
        f"{first} > {third} and {second} < {third}"
    )
    drop = df.columns.difference(["check"])
    expected = expected.drop(columns=[*drop]).merge(
        df, on="check", how="right", sort=False
    )
    expected = expected.filter([first, second, third])
    actual = right.conditional_join(
        df,
        (first, third, ">"),
        (second, third, "<"),
        how="right",
        sort_by_appearance=True,
    )
    actual = actual.droplevel(level=0, axis=1)
    actual = actual.loc[:, [first, second, third]]
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_dual_ne(df, right):
    """
    Test output for multiple conditions. `!=`
    """

    assume(not right.empty)
    filters = ["A", "Integers", "B", "Numeric"]
    df = df.assign(A=df["A"].astype("Int64"))
    right = right.assign(Integers=right["Integers"].astype(pd.Int64Dtype()))
    expected = (
        df.assign(t=1)
        .merge(right.assign(t=1), on="t")
        .query("A != Integers and B != Numeric")
        .reset_index(drop=True)
    )
    expected = expected.filter(filters)
    actual = df.conditional_join(
        right,
        ("A", "Integers", "!="),
        ("B", "Numeric", "!="),
        how="inner",
        sort_by_appearance=True,
    )
    actual = actual.droplevel(level=0, axis=1)
    actual = actual.filter(filters)
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_dual_ne_extension(df, right):
    """
    Test output for multiple conditions. `!=`
    """

    assume(not right.empty)
    filters = ["A", "Integers", "B", "Numeric"]
    df = df.assign(A=df["A"].astype("Int64"))
    expected = (
        df.assign(t=1)
        .merge(right.assign(t=1), on="t")
        .query("A != Integers and B != Numeric")
        .reset_index(drop=True)
    )
    expected = expected.filter(filters)
    actual = df.conditional_join(
        right,
        ("A", "Integers", "!="),
        ("B", "Numeric", "!="),
        how="inner",
        sort_by_appearance=True,
    )
    actual = actual.droplevel(level=0, axis=1)
    actual = actual.filter(filters)
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_dual_ne_extension_right(df, right):
    """
    Test output for multiple conditions. `!=`
    """

    assume(not right.empty)
    filters = ["A", "Integers", "B", "Numeric"]
    right = right.assign(Integers=right["Integers"].astype(pd.Int64Dtype()))
    expected = (
        df.assign(t=1)
        .merge(right.assign(t=1), on="t")
        .query("A != Integers and B != Numeric")
        .reset_index(drop=True)
    )
    expected = expected.filter(filters)
    actual = df.conditional_join(
        right,
        ("A", "Integers", "!="),
        ("B", "Numeric", "!="),
        how="inner",
        sort_by_appearance=True,
    )
    actual = actual.droplevel(level=0, axis=1)
    actual = actual.filter(filters)
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_dual_ne_dates(df, right):
    """
    Test output for multiple conditions. `!=`
    """

    assume(not right.empty)
    filters = ["A", "Integers", "E", "Dates"]
    expected = (
        df.assign(t=1)
        .merge(right.assign(t=1), on="t")
        .query("A != Integers and E != Dates")
        .reset_index(drop=True)
    )
    expected = expected.filter(filters)
    actual = df.conditional_join(
        right,
        ("A", "Integers", "!="),
        ("E", "Dates", "!="),
        how="inner",
        sort_by_appearance=True,
    )
    actual = actual.droplevel(level=0, axis=1)
    actual = actual.filter(filters)
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_multiple_ne_dates(df, right):
    """
    Test output for multiple conditions. `!=`
    """

    assume(not right.empty)
    filters = ["A", "Integers", "E", "Dates", "B", "Numeric"]
    expected = (
        df.assign(t=1)
        .merge(right.assign(t=1), on="t")
        .query("A != Integers and E != Dates and B != Numeric")
        .reset_index(drop=True)
    )
    expected = expected.filter(filters)
    actual = df.conditional_join(
        right,
        ("A", "Integers", "!="),
        ("E", "Dates", "!="),
        ("B", "Numeric", "!="),
        how="inner",
        sort_by_appearance=True,
    )
    actual = actual.droplevel(level=0, axis=1)
    actual = actual.filter(filters)
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_dual_conditions_eq_and_ne(df, right):
    """Test output for equal and not equal conditions."""

    assume(not right.empty)
    A, B, C, D = ("B", "Numeric", "E", "Dates")
    expected = (
        df.merge(right, left_on=A, right_on=B)
        .dropna(subset=[A, B])
        .query(f"{C} != {D}")
        .sort_index()
        .reset_index(drop=True)
    )
    expected = expected.filter([A, B, C, D])
    actual = df.conditional_join(
        right,
        (A, B, "=="),
        (C, D, "!="),
        how="inner",
        sort_by_appearance=True,
    )
    actual = actual.droplevel(level=0, axis=1)
    actual = actual.filter([A, B, C, D])
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_dual_conditions_ne_and_eq(df, right):
    """Test output for equal and not equal conditions."""

    assume(not right.empty)

    A, B, C, D = ("A", "Integers", "E", "Dates")
    expected = (
        df.merge(right, left_on=C, right_on=D)
        .dropna(subset=[C, D])
        .query(f"{A} != {B}")
        .reset_index(drop=True)
    )
    expected = expected.filter([A, B, C, D])
    actual = df.conditional_join(
        right,
        (A, B, "!="),
        (C, D, "=="),
        how="inner",
        sort_by_appearance=False,
    )
    actual = actual.droplevel(level=0, axis=1)
    actual = actual.filter([A, B, C, D])
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_gt_lt_ne_conditions(df, right):
    """
    Test output for multiple conditions.
    """

    assume(not right.empty)
    filters = ["A", "Integers", "B", "Numeric", "E", "Dates"]
    expected = (
        df.assign(t=1)
        .merge(right.assign(t=1), on="t")
        .query("A > Integers and B < Numeric and E != Dates")
        .reset_index(drop=True)
    )
    expected = expected.filter(filters)
    actual = df.conditional_join(
        right,
        ("A", "Integers", ">"),
        ("B", "Numeric", "<"),
        ("E", "Dates", "!="),
        how="inner",
        sort_by_appearance=True,
    )
    actual = actual.droplevel(level=0, axis=1)
    actual = actual.filter(filters)
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_gt_lt_ne_start(df, right):
    """
    Test output for multiple conditions.
    """

    assume(not right.empty)
    filters = ["A", "Integers", "B", "Numeric", "E", "Dates"]
    expected = (
        df.assign(t=1)
        .merge(right.assign(t=1), on="t")
        .query("A > Integers and B < Numeric and E != Dates")
        .reset_index(drop=True)
    )
    expected = expected.filter(filters)
    actual = df.conditional_join(
        right,
        ("E", "Dates", "!="),
        ("A", "Integers", ">"),
        ("B", "Numeric", "<"),
        how="inner",
        sort_by_appearance=True,
    )
    actual = actual.droplevel(level=0, axis=1)
    actual = actual.filter(filters)
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_ge_le_ne_extension_array(df, right):
    """
    Test output for multiple conditions.
    """

    assume(not right.empty)
    filters = ["A", "Integers", "B", "Numeric", "E", "Dates"]
    df = df.assign(A=df["A"].astype("Int64"))
    right = right.assign(Integers=right["Integers"].astype(pd.Int64Dtype()))

    expected = (
        df.assign(t=1)
        .merge(right.assign(t=1), on="t")
        .query("A != Integers and B < Numeric and E >= Dates")
        .reset_index(drop=True)
    )
    expected = expected.filter(filters)
    actual = df.conditional_join(
        right,
        ("E", "Dates", ">="),
        ("A", "Integers", "!="),
        ("B", "Numeric", "<"),
        how="inner",
        sort_by_appearance=True,
    )
    actual = actual.droplevel(level=0, axis=1)
    actual = actual.filter(filters)
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_ge_lt_ne_extension(df, right):
    """
    Test output for multiple conditions.
    """

    assume(not right.empty)
    filters = ["A", "Integers", "B", "Numeric", "E", "Dates"]
    df = df.assign(A=df["A"].astype("Int64"))
    right = right.assign(Integers=right["Integers"].astype(pd.Int64Dtype()))

    expected = (
        df.assign(t=1)
        .merge(right.assign(t=1), on="t")
        .query(
            "A != Integers and B < Numeric and E >= Dates and E != Dates_Right"
        )
        .reset_index(drop=True)
    )
    expected = expected.filter(filters)
    actual = df.conditional_join(
        right,
        ("E", "Dates", ">="),
        ("B", "Numeric", "<"),
        ("A", "Integers", "!="),
        ("E", "Dates_Right", "!="),
        how="inner",
        sort_by_appearance=True,
    )
    actual = actual.droplevel(level=0, axis=1)
    actual = actual.filter(filters)
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_eq_ge_and_le_numbers(df, right):
    """Test output for multiple conditions."""

    assume(not right.empty)
    l_eq, l_ge, l_le = ["B", "A", "E"]
    r_eq, r_ge, r_le = ["Floats", "Integers", "Dates"]
    columns = ["B", "A", "E", "Floats", "Integers", "Dates"]
    expected = (
        df.merge(right, left_on=l_eq, right_on=r_eq, how="inner", sort=False)
        .dropna(subset=[l_eq, r_eq])
        .query(f"{l_ge} >= {r_ge} and {l_le} <= {r_le}")
        .reset_index(drop=True)
    )
    expected = expected.filter(columns)
    actual = df.conditional_join(
        right,
        (l_eq, r_eq, "=="),
        (l_ge, r_ge, ">="),
        (l_le, r_le, "<="),
        how="inner",
        sort_by_appearance=False,
    )
    actual = actual.droplevel(level=0, axis=1)
    actual = actual.filter(columns)
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_ge_eq_and_le_numbers(df, right):
    """Test output for multiple conditions."""

    assume(not right.empty)
    l_eq, l_ge, l_le = ["B", "A", "E"]
    r_eq, r_ge, r_le = ["Floats", "Integers", "Dates"]
    columns = ["B", "A", "E", "Floats", "Integers", "Dates"]
    expected = (
        df.merge(right, left_on=l_eq, right_on=r_eq, how="inner", sort=False)
        .dropna(subset=[l_eq, r_eq])
        .query(f"{l_ge} >= {r_ge} and {l_le} <= {r_le}")
        .reset_index(drop=True)
    )
    expected = expected.filter(columns)
    actual = df.conditional_join(
        right,
        (l_ge, r_ge, ">="),
        (l_le, r_le, "<="),
        (l_eq, r_eq, "=="),
        how="inner",
        sort_by_appearance=False,
    )
    actual = actual.droplevel(level=0, axis=1)
    actual = actual.filter(columns)
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_multiple_eqs(df, right):
    """Test output for multiple conditions."""

    assume(not right.empty)
    columns = ["B", "A", "E", "Floats", "Integers", "Dates"]
    expected = (
        df.merge(
            right,
            left_on=["B", "A"],
            right_on=["Floats", "Integers"],
            how="inner",
            sort=False,
        )
        .dropna(subset=["B", "A", "Floats", "Integers"])
        .query("E != Dates")
        .reset_index(drop=True)
    )
    expected = expected.filter(columns)
    actual = df.conditional_join(
        right,
        ("E", "Dates", "!="),
        ("B", "Floats", "=="),
        ("A", "Integers", "=="),
        how="inner",
        sort_by_appearance=False,
    )
    actual = actual.droplevel(level=0, axis=1)
    actual = actual.filter(columns)
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_multiple_eqs_extension_array(df, right):
    """Test output for multiple conditions."""

    assume(not right.empty)
    columns = ["B", "A", "E", "Floats", "Integers", "Dates"]
    df = df.assign(A=df["A"].astype("Int64"))
    right = right.assign(Integers=right["Integers"].astype(pd.Int64Dtype()))
    expected = (
        df.merge(
            right,
            left_on=["B", "A"],
            right_on=["Floats", "Integers"],
            how="inner",
            sort=False,
        )
        .dropna(subset=["B", "A", "Floats", "Integers"])
        .query("E != Dates")
        .reset_index(drop=True)
    )
    expected = expected.filter(columns)
    actual = df.conditional_join(
        right,
        ("E", "Dates", "!="),
        ("B", "Floats", "=="),
        ("A", "Integers", "=="),
        how="inner",
        sort_by_appearance=False,
    )
    actual = actual.droplevel(level=0, axis=1)
    actual = actual.filter(columns)
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_multiple_eqs_only(df, right):
    """Test output for multiple conditions."""

    assume(not right.empty)
    columns = ["B", "A", "E", "Floats", "Integers", "Dates"]
    df = df.assign(A=df["A"].astype("Int64"), C=df["C"].astype("string"))
    right = right.assign(
        Integers=right["Integers"].astype(pd.Int64Dtype()),
        Strings=right["Strings"].astype("string"),
    )
    df.loc[0, "A"] = pd.NA
    right.loc[0, "Integers"] = pd.NA
    expected = (
        df.merge(
            right,
            left_on=["B", "A", "E"],
            right_on=["Floats", "Integers", "Dates"],
            how="inner",
            sort=False,
        )
        .dropna(subset=columns)
        .reset_index(drop=True)
    )
    expected = expected.filter(columns)
    actual = df.conditional_join(
        right,
        ("E", "Dates", "=="),
        ("B", "Floats", "=="),
        ("A", "Integers", "=="),
        how="inner",
        sort_by_appearance=False,
    )
    actual = actual.droplevel(level=0, axis=1)
    actual = actual.filter(columns)
    assert_frame_equal(expected, actual)
