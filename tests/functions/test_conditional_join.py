# import numpy as np
import pandas as pd
import pytest

# from hypothesis import given, settings
# from pandas.testing import assert_frame_equal
# from janitor.testing_utils.strategies import (
#     conditional_df,
#     conditional_right,
#     conditional_series,
# )


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


def test_df_multiindex(dummy, series):
    """Raise ValueError if `df` columns is a MultiIndex."""
    with pytest.raises(
        ValueError,
        match="MultiIndex columns are not supported for conditional joins.",
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


def test_right_multiindex(dummy):
    """Raise ValueError if `right` columns is a MultiIndex."""
    with pytest.raises(
        ValueError,
        match="MultiIndex columns are not supported for conditional joins.",
    ):
        right = dummy.copy()
        right.columns = [list("ABC"), list("FGH")]
        dummy.conditional_join(right, ("id", ("A", "F"), ">="))


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


def test_check_condition_length(dummy, series):
    """Raise ValueError if any condition is not length 3."""
    with pytest.raises(
        ValueError, match="condition should have only three elements;.+"
    ):
        dummy.conditional_join(series, ("id", "B", "C", "<"))


def test_check_left_on_type(dummy, series):
    """Raise TypeError if left_on is not a string."""
    with pytest.raises(TypeError, match="left_on should be one of.+"):
        dummy.conditional_join(series, (1, "B", "<"))


def test_check_right_on_type(dummy, series):
    """Raise TypeError if right_on is not a string."""
    with pytest.raises(TypeError, match="right_on should be one of.+"):
        dummy.conditional_join(series, ("id", 1, "<"))


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
