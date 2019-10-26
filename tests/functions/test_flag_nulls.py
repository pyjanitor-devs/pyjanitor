import pytest
from pandas.testing import assert_frame_equal

from janitor.functions import flag_nulls


@pytest.mark.functions
def test_functional_on_all_columns(missingdata_df):
    expected = missingdata_df.copy()
    expected["null_flag"] = [0, 1, 1] * 3

    df = missingdata_df.flag_nulls()

    assert_frame_equal(df, expected, check_dtype=False)

    # Should also be the same for explicit columns
    df = missingdata_df.flag_nulls(columns=["a", "Bell__Chart"])

    assert_frame_equal(df, expected, check_dtype=False)


@pytest.mark.functions
def test_non_method_functional(missingdata_df):
    expected = missingdata_df.copy()
    expected["null_flag"] = [0, 1, 1] * 3

    df = flag_nulls(missingdata_df)

    assert_frame_equal(df, expected, check_dtype=False)


@pytest.mark.functions
def test_functional_on_some_columns(missingdata_df):
    expected = missingdata_df.copy()
    expected["null_flag"] = [0, 0, 1] * 3

    df = missingdata_df.flag_nulls(columns=["a"])

    assert_frame_equal(df, expected, check_dtype=False)

    # Testing when we provide the direct name
    df = missingdata_df.flag_nulls(columns="a")

    assert_frame_equal(df, expected, check_dtype=False)


@pytest.mark.functions
def test_rename_output_column(missingdata_df):
    expected = missingdata_df.copy()
    expected["flag"] = [0, 1, 1] * 3

    df = missingdata_df.flag_nulls(column_name="flag")

    assert_frame_equal(df, expected, check_dtype=False)


@pytest.mark.functions
def test_fail_column_name_in_columns(missingdata_df):
    with pytest.raises(ValueError):
        missingdata_df.flag_nulls(column_name="a")


@pytest.mark.functions
def test_fail_column_val_not_in_columns(missingdata_df):
    with pytest.raises(ValueError):
        missingdata_df.flag_nulls(columns=["c"])
