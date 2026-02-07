import numpy as np
import pandas as pd
import pytest

import janitor


@pytest.mark.functions
def test_compare_df_cols_named_args():
    df1 = pd.DataFrame({"A": [1, 2], "B": ["x", "y"]})
    df2 = pd.DataFrame(
        {"A": [3.0, 4.0], "B": ["z", "w"], "C": [True, False]}
    )

    result = janitor.compare_df_cols(train=df1, test=df2)

    expected = pd.DataFrame(
        {
            "column_name": ["A", "B", "C"],
            "train": ["int64", "object", np.nan],
            "test": ["float64", "object", "bool"],
        }
    )

    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.functions
def test_compare_df_cols_mismatch_bind_rows_return_alias():
    df1 = pd.DataFrame({"A": [1, 2], "B": ["x", "y"]})
    df2 = pd.DataFrame(
        {"A": [3.0, 4.0], "B": ["z", "w"], "C": [True, False]}
    )

    result = janitor.compare_df_cols(
        train=df1, test=df2, return="mismatch"
    )

    expected = pd.DataFrame(
        {
            "column_name": ["A"],
            "train": ["int64"],
            "test": ["float64"],
        }
    )

    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.functions
def test_compare_df_cols_rbind_missing_column():
    df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    df2 = pd.DataFrame({"A": [5, 6], "B": [7, 8], "C": [9, 10]})

    result = janitor.compare_df_cols(
        left=df1, right=df2, return="mismatch", bind_method="rbind"
    )

    expected = pd.DataFrame(
        {
            "column_name": ["C"],
            "left": [np.nan],
            "right": ["int64"],
        }
    )

    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.functions
def test_compare_df_cols_list_input():
    df1 = pd.DataFrame({"A": [1]})
    df2 = pd.DataFrame({"A": [2.0]})

    result = janitor.compare_df_cols(group=[df1, df2])

    expected = pd.DataFrame(
        {
            "column_name": ["A"],
            "group_1": ["int64"],
            "group_2": ["float64"],
        }
    )

    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.functions
def test_compare_df_cols_same_verbose(capsys):
    df1 = pd.DataFrame({"A": [1, 2]})
    df2 = pd.DataFrame({"A": [3.0, 4.0]})

    result = janitor.compare_df_cols_same(df1, df2, verbose=True)

    captured = capsys.readouterr()

    assert result is False
    assert "column_name" in captured.out


@pytest.mark.functions
def test_describe_class_categorical():
    series = pd.Series(pd.Categorical(["a", "b", "a"]))

    assert (
        janitor.describe_class(series) == 'category(levels=["a", "b"])'
    )
    assert (
        janitor.describe_class(series, strict_description=False) == "category"
    )
