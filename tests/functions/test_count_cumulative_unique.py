import pytest
import numpy as np


@pytest.mark.functions
def test_column_name_not_exists(dataframe):
    """Checks that an error is raised if the column to be counted does not exist
    in df.
    """
    with pytest.raises(
        ValueError,
        match="_foo_ not present in dataframe columns",
    ):
        dataframe.count_cumulative_unique(
            "_foo_",
            dest_column_name="foo_count",
        )


@pytest.mark.functions
@pytest.mark.parametrize("case_sensitive", [True, False])
def test_original_column_values_not_altered(dataframe, case_sensitive):
    """Checks that the column to be counted is not altered by the case-switching
    logic implemented by `count_cumulative_unique`.
    """
    before = np.array(["a", "b", "c", "A", "B", "C", "a", "b", "c"])
    dataframe["ccu"] = before

    df = dataframe.count_cumulative_unique(
        "ccu",
        dest_column_name="ccu_count",
        case_sensitive=case_sensitive,
    )
    after = df["ccu"].to_numpy()

    assert (before == after).all()


@pytest.mark.functions
def test_case_sensitive(dataframe):
    """Checks a new column is added containing the correct distinct count,
    when case sensitive is True.
    """
    dataframe["ccu"] = ["a", "b", "c", "A", "B", "C", "a", "b", "c"]
    df = dataframe.count_cumulative_unique(
        "ccu", dest_column_name="ccu_count", case_sensitive=True
    )

    assert all(df["ccu_count"] == [1, 2, 3, 4, 5, 6, 6, 6, 6])


@pytest.mark.functions
def test_not_case_sensitive(dataframe):
    """Checks a new column is added containing the correct distinct count,
    when case sensitive is False.
    """
    dataframe["ccu"] = ["a", "b", "c", "A", "B", "C", "a", "b", "c"]
    df = dataframe.count_cumulative_unique(
        "ccu", dest_column_name="ccu_count", case_sensitive=False
    )

    assert all(df["ccu_count"] == [1, 2, 3, 3, 3, 3, 3, 3, 3])
