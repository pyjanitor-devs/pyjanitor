import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import janitor  # noqa: F401


@pytest.mark.functions
def test_drop_duplicate_columns(df_duplicated_columns):
    # df_duplicated_columns contains columns 'a', duplicated three times
    clean_df = df_duplicated_columns.drop_duplicate_columns(column_name="a")
    assert clean_df.columns.to_list() == ["b", "a", "a"]
    expected_df = pd.DataFrame(
        {"b": range(10), "a": range(10, 20), "a*": range(20, 30)}
    ).clean_names(remove_special=True)
    assert_frame_equal(clean_df, expected_df)


@pytest.mark.functions
def test_drop_duplicate_columns_for_second_duplicated_column(
    df_duplicated_columns,
):
    clean_df = df_duplicated_columns.drop_duplicate_columns(
        column_name="a", nth_index=1
    )
    expected_df = pd.DataFrame(
        {"a": range(10), "b": range(10), "a*": range(20, 30)}
    ).clean_names(remove_special=True)
    assert clean_df.columns.to_list() == ["a", "b", "a"]
    assert_frame_equal(clean_df, expected_df)


@pytest.mark.functions
def test_drop_duplicate_columns_for_third_duplicated_column(
    df_duplicated_columns,
):
    clean_df = df_duplicated_columns.drop_duplicate_columns(
        column_name="a", nth_index=2
    )
    expected_df = pd.DataFrame(
        {"a": range(10), "b": range(10), "A": range(10, 20)}
    ).clean_names(remove_special=True)
    assert clean_df.columns.to_list() == ["a", "b", "a"]
    assert_frame_equal(clean_df, expected_df)


@pytest.mark.functions
def test_drop_duplicate_columns_with_error(df_duplicated_columns):
    with pytest.raises(IndexError):
        df_duplicated_columns.drop_duplicate_columns(
            column_name="a", nth_index=3
        )
