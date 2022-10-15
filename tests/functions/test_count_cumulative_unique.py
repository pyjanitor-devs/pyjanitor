import numpy as np
import pandas as pd
import pytest


@pytest.mark.functions
def test_column_name_not_exists(dataframe):
    """Checks that an error is raised if the column to be counted does not
    exist in df.
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
    """Checks that the column to be counted is not altered by the case-
    switching logic implemented by `count_cumulative_unique`.
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
        "ccu",
        dest_column_name="ccu_count",
        case_sensitive=True,
    )

    assert all(df["ccu_count"] == [1, 2, 3, 4, 5, 6, 6, 6, 6])


@pytest.mark.functions
def test_not_case_sensitive(dataframe):
    """Checks a new column is added containing the correct distinct count,
    when case sensitive is False.
    """
    dataframe["ccu"] = ["a", "b", "c", "A", "B", "C", "a", "b", "c"]
    df = dataframe.count_cumulative_unique(
        "ccu",
        dest_column_name="ccu_count",
        case_sensitive=False,
    )

    assert all(df["ccu_count"] == [1, 2, 3, 3, 3, 3, 3, 3, 3])


@pytest.mark.functions
def test_not_case_sensitive_but_nonstring():
    """Checks TypeError is raised if case sensitive is explicitly set to
    False but the column being counted does not support `.str.lower()`.
    """
    df = pd.DataFrame(
        {
            "ok1": ["ABC", None, "zzz"],
            "ok2": pd.Categorical(["A", "b", "A"], ordered=False),
            "notok1": [1, 2, 3],
            "notok2": [b"ABC", None, b"zzz"],
        }
    )

    # acceptable string types
    for okcol in ["ok1", "ok2"]:
        _ = df.count_cumulative_unique(
            okcol,
            dest_column_name="ok_count",
            case_sensitive=False,
        )

    # forbidden data types
    msg = "case_sensitive=False can only be used with a string-like type.*"
    for notokcol in ["notok1", "notok2"]:
        with pytest.raises(TypeError, match=msg):
            _ = df.count_cumulative_unique(
                notokcol,
                dest_column_name="notok_count",
                case_sensitive=False,
            )
