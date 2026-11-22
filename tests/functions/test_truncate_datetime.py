from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal


@pytest.mark.functions
def test_truncate_datetime_dataframe_invalid_datepart():
    """Checks if a ValueError is appropriately raised when datepart is
    not a valid enumeration.
    """
    with pytest.raises(ValueError, match=r"invalid `datepart`"):
        pd.DataFrame().truncate_datetime_dataframe("INVALID")


@pytest.mark.functions
def test_truncate_datetime_dataframe_all_parts():
    """Test for truncate_datetime_dataframe, for all valid dateparts.
    Also only passes if `truncate_datetime_dataframe` method is idempotent.
    """
    x = datetime(2022, 3, 21, 9, 1, 15, 666)
    df = pd.DataFrame({"dt": [x], "foo": [np.nan]}, copy=False)

    result = df.truncate_datetime_dataframe("second")
    assert result.loc[0, "dt"] == datetime(2022, 3, 21, 9, 1, 15, 0)
    result = df.truncate_datetime_dataframe("minute")
    assert result.loc[0, "dt"] == datetime(2022, 3, 21, 9, 1)
    result = df.truncate_datetime_dataframe("HOUR")
    assert result.loc[0, "dt"] == datetime(2022, 3, 21, 9)
    result = df.truncate_datetime_dataframe("Day")
    assert result.loc[0, "dt"] == datetime(2022, 3, 21)
    result = df.truncate_datetime_dataframe("month")
    assert result.loc[0, "dt"] == datetime(2022, 3, 1)
    result = df.truncate_datetime_dataframe("yeaR")
    assert result.loc[0, "dt"] == datetime(2022, 1, 1)


# bad data
@pytest.mark.functions
def test_truncate_datetime_dataframe_do_nothing():
    """Ensure nothing changes (and no errors raised) if there are no datetime-
    compatible columns.
    """
    in_data = {
        "a": [1, 0],
        "b": ["foo", ""],
        "c": [np.nan, 3.0],
        "d": [True, False],
    }

    result = pd.DataFrame(in_data).truncate_datetime_dataframe("year")
    expected = pd.DataFrame(in_data)

    assert_frame_equal(result, expected)


@pytest.mark.functions
def test_truncate_datetime_containing_NaT():
    """Ensure NaT is ignored safely (no-op) and no TypeError is thrown."""
    x = datetime(2022, 3, 21, 9, 1, 15, 666)
    df = pd.DataFrame({"dt": [x, pd.NaT], "foo": [np.nan, 3]})
    # df['dt']=df['dt'].dt.as_unit('ns')
    expected = pd.DataFrame(
        {"dt": [x.replace(microsecond=0), pd.NaT], "foo": [np.nan, 3]}
    ).assign(dt=lambda df: df["dt"].dt.as_unit("ns"))

    result = df.truncate_datetime_dataframe("second").assign(
        dt=lambda df: df["dt"].dt.as_unit("ns")
    )
    assert_frame_equal(result, expected)


@pytest.mark.functions
def test_truncate_datetime_preserves_index():
    df = pd.DataFrame(
        {"dt": pd.to_datetime(["2022-03-21 09:01", "2023-07-04 18:30"])},
        index=["x", "y"],
    )
    result = df.truncate_datetime_dataframe("day")
    expected = pd.DataFrame(
        {"dt": pd.to_datetime(["2022-03-21", "2023-07-04"])},
        index=["x", "y"],
    )
    assert_frame_equal(result, expected, check_dtype=False)


@pytest.mark.functions
def test_truncate_datetime_timezone_aware():
    df = pd.DataFrame(
        {
            "dt": pd.to_datetime(["2022-03-21 03:00", "2022-11-06 01:30"]).tz_localize(
                "America/New_York", ambiguous=[True, False]
            ),
            "foo": [1, 2],
        }
    )
    result = df.truncate_datetime_dataframe("day")
    expected = pd.DataFrame(
        {
            "dt": pd.to_datetime(["2022-03-21", "2022-11-06"]).tz_localize(
                "America/New_York"
            ),
            "foo": [1, 2],
        }
    )
    assert_frame_equal(result, expected, check_dtype=False)
    hourly = df.truncate_datetime_dataframe("hour")
    assert hourly["dt"].tolist() == [
        pd.Timestamp("2022-03-21 03:00", tz="America/New_York"),
        pd.Timestamp("2022-11-06 06:00", tz="UTC").tz_convert("America/New_York"),
    ]


@pytest.mark.functions
def test_truncate_datetime_preserves_multiindex():
    index = pd.MultiIndex.from_tuples([("a", 1), ("b", 2)], names=["key", "num"])
    df = pd.DataFrame(
        {"dt": pd.to_datetime(["2022-03-21 09:01", "2023-07-04 18:30"])},
        index=index,
    )
    result = df.truncate_datetime_dataframe("day")
    expected = pd.DataFrame(
        {"dt": pd.to_datetime(["2022-03-21", "2023-07-04"])},
        index=index,
    )
    assert_frame_equal(result, expected, check_dtype=False)


@pytest.mark.functions
def test_truncate_datetime_timezone_aware_with_NaT():
    df = pd.DataFrame(
        {
            "dt": pd.to_datetime(
                ["2022-03-21 09:01", None, "2022-07-04 18:30"]
            ).tz_localize("Europe/Berlin"),
        },
        index=[10, 20, 30],
    )
    result = df.truncate_datetime_dataframe("day").assign(
        dt=lambda df: df["dt"].dt.as_unit("ns")
    )
    expected = pd.DataFrame(
        {
            "dt": pd.to_datetime(["2022-03-21", None, "2022-07-04"])
            .tz_localize("Europe/Berlin")
            .as_unit("ns"),
        },
        index=[10, 20, 30],
    )
    assert_frame_equal(result, expected)
