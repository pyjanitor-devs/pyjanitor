import pytest

from janitor.functions import count_cumulative_unique


@pytest.mark.functions
def test_case_sensitive(dataframe):
    dataframe["ccu"] = ["a", "b", "c", "A", "B", "C", "a", "b", "c"]
    df = dataframe.count_cumulative_unique(
        "ccu", dest_column_name="ccu_count", case_sensitive=True
    )
    # column ccu_count should contain [1,2,3,4,5,6,6,6,6]
    assert all(df["ccu_count"] == [1, 2, 3, 4, 5, 6, 6, 6, 6])


@pytest.mark.functions
def test_not_case_sensitive(dataframe):
    dataframe["ccu"] = ["a", "b", "c", "A", "B", "C", "a", "b", "c"]
    df = dataframe.count_cumulative_unique(
        "ccu", dest_column_name="ccu_count", case_sensitive=False
    )
    # column ccu_count should contain [1,2,3,3,3,3,3,3,3]
    assert all(df["ccu_count"] == [1, 2, 3, 3, 3, 3, 3, 3, 3])
