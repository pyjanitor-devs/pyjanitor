import numpy as np
import pandas as pd
import pytest


@pytest.mark.functions
def test_groupby_agg():

    df = pd.DataFrame(
        {
            "date": [
                "20190101",
                "20190101",
                "20190102",
                "20190102",
                "20190304",
                "20190304",
            ],
            "values": [1, 1, 2, 2, 3, 3],
        }
    )

    df_new = df.groupby_agg(
        by="date",
        new_column_name="date_average",
        agg_column_name="values",
        agg=pd.np.mean,
    )
    assert df.shape[0] == df_new.shape[0]
    assert "date_average" in df_new.columns
    assert df_new["date_average"].iloc[0] == 1


@pytest.mark.functions
def test_groupby_agg_multi():

    df = pd.DataFrame(
        {
            "date": [
                "20190101",
                "20190101",
                "20190102",
                "20190102",
                "20190304",
                "20190304",
            ],
            "user_id": [1, 2, 1, 2, 1, 2],
            "values": [1, 2, 3, 4, 5, 6],
        }
    )

    df_new = df.groupby_agg(
        by=["date", "user_id"],
        new_column_name="date_average",
        agg_column_name="values",
        agg=pd.np.count_nonzero,
    )

    expected_agg = np.array([1, 1, 1, 1, 1, 1])

    np.testing.assert_equal(df_new["date_average"], expected_agg)


@pytest.mark.functions
def test_groupby_agg_multi_column():
    """Test for the case when we want to groupby one column and agg on another, while leaving out other columns."""

    df = pd.DataFrame(
        {
            "date": [
                "20190101",
                "20190101",
                "20190102",
                "20190102",
                "20190304",
                "20190304",
            ],
            "user_id": [1, 2, 1, 2, 1, 2],
            "values": [1, 2, 3, 4, 5, 6],
        }
    )

    df_new = df.groupby_agg(
        by=["date"],
        new_column_name="values_avg",
        agg_column_name="values",
        agg="mean",
    )

    expected_agg = np.array([1.5, 1.5, 3.5, 3.5, 5.5, 5.5])
    np.testing.assert_equal(df_new["values_avg"], expected_agg)
