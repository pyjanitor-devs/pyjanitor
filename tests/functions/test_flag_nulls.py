import pandas as pd
import pytest
from janitor.functions import flag_nulls
from pandas.testing import assert_frame_equal

TEST_DF = pd.DataFrame({"a": [1, 2, None, 4], "b": [5.0, None, 7.0, 8.0]})


@pytest.mark.functions
def test_functional_on_all_columns():
    expected = pd.DataFrame(
        {
            "a": [1, 2, None, 4],
            "b": [5.0, None, 7.0, 8.0],
            "null_flag": [0, 1, 1, 0],
        }
    )

    df = TEST_DF.flag_nulls()

    assert_frame_equal(df, expected)

    # Should also be the same for explicit columns
    df = TEST_DF.flag_nulls(columns=["a", "b"])

    assert_frame_equal(df, expected)


@pytest.mark.functions
def test_non_method_functional():
    expected = pd.DataFrame(
        {
            "a": [1, 2, None, 4],
            "b": [5.0, None, 7.0, 8.0],
            "null_flag": [0, 1, 1, 0],
        }
    )

    df = flag_nulls(TEST_DF)

    assert_frame_equal(df, expected)


@pytest.mark.functions
def test_functional_on_some_columns():
    expected = pd.DataFrame(
        {
            "a": [1, 2, None, 4],
            "b": [5.0, None, 7.0, 8.0],
            "null_flag": [0, 0, 1, 0],
        }
    )

    df = TEST_DF.flag_nulls(columns=["a"])

    assert_frame_equal(df, expected)

    # Testing when we provide the direct name
    df = TEST_DF.flag_nulls(columns="a")

    assert_frame_equal(df, expected)


@pytest.mark.functions
def test_rename_output_column():
    expected = pd.DataFrame(
        {
            "a": [1, 2, None, 4],
            "b": [5.0, None, 7.0, 8.0],
            "flag": [0, 1, 1, 0],
        }
    )

    df = TEST_DF.flag_nulls(column_name="flag")

    assert_frame_equal(df, expected)


@pytest.mark.functions
def test_fail_column_name_in_columns():
    with pytest.raises(ValueError):
        TEST_DF.flag_nulls(column_name="b")


@pytest.mark.functions
def test_fail_column_val_not_in_columns():
    with pytest.raises(ValueError):
        TEST_DF.flag_nulls(columns=["c"])
