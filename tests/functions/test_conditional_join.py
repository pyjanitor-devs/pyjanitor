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


@given(s=conditional_series())
def test_df_empty(s):
    """Raise ValueError if `df` is empty."""
    df = pd.DataFrame([], dtype="object")
    with pytest.raises(ValueError):
        df.conditional_join(s, ("A", "non", "=="))


@given(df=conditional_df())
def test_right_df(df):
    """Raise TypeError if `right` is not a Series/DataFrame."""
    assume(not df.empty)
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

    assume(not df.empty)
    assume(not s.empty)
    with pytest.raises(ValueError):
        s.name = "B"
        df.conditional_join(s)


@given(df=conditional_df(), s=conditional_series())
def test_check_condition_type(df, s):
    """Raise TypeError if any condition in conditions is not a tuple."""

    assume(not df.empty)
    assume(not s.empty)
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

    assume(not df.empty)
    assume(not s.empty)
    with pytest.raises(TypeError):
        s.name = "B"
        df.conditional_join(s, (1, "B", "<"))


@given(df=conditional_df(), s=conditional_series())
def test_check_right_on_type(df, s):
    """Raise TypeError if right_on is not a string."""

    assume(not df.empty)
    assume(not s.empty)
    with pytest.raises(TypeError):
        s.name = "B"
        df.conditional_join(s, ("B", 1, "<"))


@given(df=conditional_df(), s=conditional_series())
def test_check_op_type(df, s):
    """Raise TypeError if the operator is not a string."""

    assume(not df.empty)
    assume(not s.empty)
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
    `==`, `!=`, `<`, `>`, `>=`, `<=`.
    """

    with pytest.raises(ValueError):
        s.name = "B"
        df.conditional_join(s, ("B", "B", "=!"))


@given(df=conditional_df(), s=conditional_series())
def test_check_how_type(df, s):
    """
    Raise TypeError if `how` is not a string.
    """

    assume(not df.empty)
    assume(not s.empty)
    with pytest.raises(TypeError):
        s.name = "B"
        df.conditional_join(s, ("B", "B", "<"), how=1)


@given(df=conditional_df(), s=conditional_series())
def test_check_how_value(df, s):
    """
    Raise ValueError if `how` is not one of `inner`, `outer`,
    `left`, or `right`.
    """

    with pytest.raises(ValueError):
        s.name = "B"
        df.conditional_join(s, ("B", "B", "<"), how="INNER")


@given(df=conditional_df(), s=conditional_series())
def test_check_sort_by_appearance_type(df, s):
    """
    Raise TypeError if `sort_by_appearance` is not a boolean.
    """

    assume(not df.empty)
    assume(not s.empty)
    with pytest.raises(TypeError):
        s.name = "B"
        df.conditional_join(s, ("B", "B", "<"), sort_by_appearance="True")


@given(df=conditional_df(), s=conditional_series())
def test_check_suffixes_type(df, s):
    """
    Raise TypeError if `suffixes` is not a tuple.
    """

    assume(not df.empty)
    assume(not s.empty)
    with pytest.raises(TypeError):
        s.name = "B"
        df.conditional_join(s, ("B", "B", "<"), suffixes=["_x", "_y"])


@given(df=conditional_df(), s=conditional_series())
def test_check_suffixes_length(df, s):
    """
    Raise ValueError if `suffixes` is not a tuple of length 2.
    """
    with pytest.raises(ValueError):
        s.name = "B"
        df.conditional_join(s, ("B", "B", "<"), suffixes=("_x",))
        df.conditional_join(s, ("B", "B", "<"), suffixes=("_x", "_y", None))


@given(df=conditional_df(), s=conditional_series())
def test_check_suffixes_None(df, s):
    """
    Raise ValueError if `suffixes` is (None, None).
    """
    with pytest.raises(ValueError):
        s.name = "B"
        df.conditional_join(s, ("B", "B", "<"), suffixes=(None, None))


@given(df=conditional_df(), s=conditional_series())
def test_check_suffixes_subtype(df, s):
    """
    Raise TypeError if any entry `suffixes`
    is not either None or a string type.
    """

    assume(not df.empty)
    assume(not s.empty)
    with pytest.raises(TypeError):
        s.name = "B"
        df.conditional_join(s, ("B", "B", "<"), suffixes=(1, None))


@given(df=conditional_df())
def test_check_suffixes_exists_df(df):
    """
    Raise ValueError if suffix already exists in `df`.
    """
    with pytest.raises(ValueError):
        right = df.copy()
        df.columns = ["A", "A_x", "B", "F", "H"]
        df.conditional_join(right, ("A", "A", "<"), suffixes=("_x", "_y"))


@given(df=conditional_df())
def test_check_suffixes_exists_right(df):
    """
    Raise ValueError if suffix already exists in `right`.
    """
    with pytest.raises(ValueError):
        right = df.copy()
        right.columns = ["A", "B", "B_y", "C", "E"]
        df.conditional_join(right, ("B", "B", "<"), suffixes=("_x", "_y"))


@given(df=conditional_df(), right=conditional_right())
def test_dtype_strings_non_equi(df, right):
    """
    Raise ValueError if the dtypes are both strings,
    and the join operator is not the equal operator.
    """
    with pytest.raises(ValueError):
        df.conditional_join(
            right, ("C", "Strings", "<"), suffixes=("_x", "_y")
        )


@given(df=conditional_df(), s=conditional_series())
def test_dtype_not_permitted(df, s):
    """
    Raise ValueError if dtype of column in `df`
    is not an acceptable type.
    """
    df["C"] = df["C"].astype("category")
    with pytest.raises(ValueError):
        s.name = "A"
        df.conditional_join(s, ("C", "A", "<"), suffixes=("_x", "_y"))


@given(df=conditional_df(), s=conditional_series())
def test_dtype_Series(df, s):
    """
    Raise ValueError if dtype of column in `df`
    does not match the dtype of column from `right`.
    """
    with pytest.raises(ValueError):
        s.name = "A"
        df.conditional_join(s, ("C", "A", "<"), suffixes=("_x", "_y"))


@given(df=conditional_df(), s=conditional_right())
def test_dtype_int(df, s):
    """
    Raise ValueError if dtype of column in `df`
    does not match the dtype of column from `right`.
    """
    with pytest.raises(ValueError):
        df.conditional_join(s, ("A", "Strings", "<"), suffixes=("_x", "_y"))


@given(df=conditional_df(), s=conditional_right())
def test_dtype_dates(df, s):
    """
    Raise ValueError if dtype of column in `df`
    does not match the dtype of column from `right`.
    """
    with pytest.raises(ValueError):
        df.conditional_join(s, ("E", "Integers", "<"), suffixes=("_x", "_y"))


@given(df=conditional_df(), right=conditional_right())
def test_single_condition_equality_numeric(df, right):
    """Test output for a single condition. "=="."""
    assume(not df.empty)
    assume(not right.empty)
    left_on, right_on = ["A", "Integers"]
    expected = pd.merge(
        df[left_on],
        right[right_on],
        left_on=left_on,
        right_on=right_on,
        how="inner",
        sort=False,
    )

    actual = df.conditional_join(
        right, (left_on, right_on, "=="), how="inner", sort_by_appearance=True
    )
    actual = actual.filter([left_on, right_on])
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_suffixes(df, right):
    """Test output for suffixes."""
    assume(not df.empty)
    assume(not right.empty)
    right = right.rename(columns={"Integers": "A"})
    left_on, right_on = ["A", "A"]
    left_c = df[left_on].rename("A_x")
    right_c = right[right_on].rename("A_y")
    expected = pd.merge(
        left_c,
        right_c,
        left_on="A_x",
        right_on="A_y",
        how="left",
        sort=False,
    )

    actual = df.conditional_join(right, (left_on, right_on, "=="), how="left")
    actual = actual.filter(["A_x", "A_y"])
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_single_condition_equality_datetime(df, right):
    """Test output for a single condition. "=="."""
    assume(not df.empty)
    assume(not right.empty)
    left_on, right_on = ["E", "Dates"]
    expected = pd.merge(
        df[left_on],
        right[right_on],
        left_on=left_on,
        right_on=right_on,
        how="right",
        sort=False,
    )

    actual = df.conditional_join(
        right, (left_on, right_on, "=="), how="right", sort_by_appearance=True
    )
    actual = actual.filter([left_on, right_on])
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_single_condition_equality_strings(df, right):
    """Test output for a single condition. "=="."""
    assume(not df.empty)
    assume(not right.empty)
    left_on, right_on = ["C", "Strings"]
    expected = pd.merge(
        df[left_on],
        right[right_on],
        left_on=left_on,
        right_on=right_on,
        how="inner",
        sort=False,
    )

    actual = df.conditional_join(
        right, (left_on, right_on, "=="), how="inner", sort_by_appearance=True
    )
    actual = actual.filter([left_on, right_on])
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_single_condition_not_equal_numeric(df, right):
    """Test output for a single condition. "!="."""
    assume(not df.empty)
    assume(not right.empty)
    left_on, right_on = ["A", "Integers"]
    expected = (
        df.assign(t=1)
        .dropna(subset=["A"])
        .merge(right.assign(t=1).dropna(subset=["Integers"]), on="t")
        .query(f"{left_on} != {right_on}")
        .reset_index(drop=True)
    )
    expected = expected.filter([left_on, right_on])
    actual = df.conditional_join(
        right, (left_on, right_on, "!="), how="inner", sort_by_appearance=True
    )
    actual = actual.filter([left_on, right_on])
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_single_condition_not_equal_ints_only(df, right):
    """Test output for a single condition. "!="."""
    assume(not df.empty)
    assume(not right.empty)
    left_on, right_on = ["A", "Integers"]
    expected = (
        df.assign(t=1)
        .merge(right.assign(t=1), on="t")
        .query(f"{left_on} != {right_on}")
        .reset_index(drop=True)
    )
    expected = expected.filter([left_on, right_on])
    actual = df.conditional_join(
        right, (left_on, right_on, "!="), how="inner", sort_by_appearance=True
    )
    actual = actual.filter([left_on, right_on])
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_single_condition_not_equal_floats_only(df, right):
    """Test output for a single condition. "!="."""
    assume(not df.empty)
    assume(not right.empty)
    # simulate output as it would be in SQL
    left_on, right_on = ["B", "Numeric"]
    expected = (
        df.assign(t=1)
        .merge(right.assign(t=1), on="t")
        .query(f"{left_on} != {right_on}")
        .reset_index(drop=True)
    )
    expected = expected.filter([left_on, right_on])
    actual = df.conditional_join(
        right, (left_on, right_on, "!="), how="inner", sort_by_appearance=True
    )
    actual = actual.filter([left_on, right_on])
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_single_condition_not_equal_datetime(df, right):
    """Test output for a single condition. "!="."""
    assume(not df.empty)
    assume(not right.empty)
    # simulate output as it would be in SQL
    left_on, right_on = ["E", "Dates"]
    expected = (
        df.assign(t=1)
        .merge(right.assign(t=1), on="t")
        .query(f"{left_on} != {right_on}")
        .reset_index(drop=True)
    )
    expected = expected.filter([left_on, right_on])
    actual = df.conditional_join(
        right, (left_on, right_on, "!="), how="inner", sort_by_appearance=True
    )
    actual = actual.filter([left_on, right_on])
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_how_left(df, right):
    """Test output when `how==left`. "<="."""
    assume(not df.empty)
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
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_how_right(df, right):
    """Test output when `how==right`. ">"."""
    assume(not df.empty)
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
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_single_condition_less_than_floats(df, right):
    """Test output for a single condition. "<"."""
    assume(not df.empty)
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
    actual = actual.filter([left_on, right_on])
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_single_condition_less_than_ints(df, right):
    """Test output for a single condition. "<"."""
    assume(not df.empty)
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
    actual = actual.filter([left_on, right_on])
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_single_condition_less_than_equal(df, right):
    """Test output for a single condition. "<=". DateTimes"""
    assume(not df.empty)
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
    actual = actual.filter([left_on, right_on])
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_single_condition_less_than_date(df, right):
    """Test output for a single condition. "<". Dates"""
    assume(not df.empty)
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
    actual = actual.filter([left_on, right_on])
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_single_condition_greater_than_datetime(df, right):
    """Test output for a single condition. ">". Datetimes"""
    assume(not df.empty)
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
    actual = actual.filter([left_on, right_on])
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_single_condition_greater_than_ints(df, right):
    """Test output for a single condition. ">="."""
    assume(not df.empty)
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
    actual = actual.filter([left_on, right_on])
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_single_condition_greater_than_floats_floats(df, right):
    """Test output for a single condition. ">"."""
    assume(not df.empty)
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
    actual = actual.filter([left_on, right_on])
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_dual_conditions_gt_and_lt_dates(df, right):
    """Test output for interval conditions."""
    assume(not df.empty)
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
    actual = actual.filter([left_on, middle, right_on])
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_dual_conditions_ge_and_le_dates(df, right):
    """Test output for interval conditions."""
    assume(not df.empty)
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
    actual = actual.filter([left_on, middle, right_on])
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_dual_conditions_le_and_ge_dates(df, right):
    """Test output for interval conditions."""
    assume(not df.empty)
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
    actual = actual.filter([left_on, middle, right_on])
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_dual_conditions_ge_and_le_numbers(df, right):
    """Test output for interval conditions."""
    assume(not df.empty)
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
    actual = actual.filter([left_on, middle, right_on])
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_dual_conditions_le_and_ge_numbers(df, right):
    """Test output for interval conditions."""
    assume(not df.empty)
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
    actual = actual.filter([left_on, middle, right_on])
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_dual_conditions_gt_and_lt_numbers(df, right):
    """Test output for interval conditions."""
    assume(not df.empty)
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
    actual = actual.filter([left_on, middle, right_on])
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_dual_conditions_gt_and_lt_numbers_(df, right):
    """
    Test output for multiple columns from the left
    and single column from right.
    """
    assume(not df.empty)
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
    actual = actual.filter([first, second, third])
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_dual_conditions_eq_and_ne(df, right):
    """Test output for equal and not equal conditions."""
    assume(not df.empty)
    assume(not right.empty)
    eq_A, eq_B, ne_A, ne_B = ("B", "Numeric", "E", "Dates")
    expected = (
        df.assign(t=1)
        .merge(right.assign(t=1), on="t")
        .query(f"{eq_A} == {eq_B} and {ne_A} != {ne_B}")
        .reset_index(drop=True)
    )
    expected = expected.filter([eq_A, eq_B, ne_A, ne_B])
    actual = df.conditional_join(
        right,
        (eq_A, eq_B, "=="),
        (ne_A, ne_B, "!="),
        how="inner",
        sort_by_appearance=True,
    )
    actual = actual.filter([eq_A, eq_B, ne_A, ne_B])
    assert_frame_equal(actual, actual)


@given(df=conditional_df(), right=conditional_right())
def test_dual_conditions_ne_and_eq(df, right):
    """Test output for equal and not equal conditions."""

    assume(not df.empty)
    assume(not right.empty)

    eq_A, eq_B, ne_A, ne_B = ("B", "Numeric", "E", "Dates")
    expected = (
        df.assign(t=1)
        .merge(right.assign(t=1), on="t")
        .query(f"{eq_A} != {eq_B} and {ne_A} == {ne_B}")
        .reset_index(drop=True)
    )
    expected = expected.filter([eq_A, eq_B, ne_A, ne_B])
    actual = df.conditional_join(
        right,
        (eq_A, eq_B, "!="),
        (ne_A, ne_B, "=="),
        how="inner",
        sort_by_appearance=True,
    )
    actual = actual.filter([eq_A, eq_B, ne_A, ne_B])
    assert_frame_equal(expected, actual)


@given(df=conditional_df(), right=conditional_right())
def test_eq_ge_and_le_numbers(df, right):
    """Test output for multiple conditions."""
    assume(not df.empty)
    assume(not right.empty)
    l_eq, l_ge, l_le = ["C", "A", "E"]
    r_eq, r_ge, r_le = ["Strings", "Integers", "Dates"]
    columns = ["C", "A", "E", "Strings", "Integers", "Dates"]
    expected = (
        df.assign(t=1)
        .merge(right.assign(t=1), on="t")
        .query(f"{l_eq} == {r_eq} and {l_ge} >= {r_ge} and {l_le} <= {r_le}")
        .reset_index(drop=True)
    )
    expected = expected.filter(columns)
    actual = df.conditional_join(
        right,
        (l_eq, r_eq, "=="),
        (l_ge, r_ge, ">="),
        (l_le, r_le, "<="),
        how="inner",
        sort_by_appearance=True,
    )
    actual = actual.filter(columns)
    assert_frame_equal(expected, actual)
