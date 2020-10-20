"""
Unit tests for `.flag_jumps()`.
"""

import numpy as np
import pandas as pd
import pytest

import janitor.timeseries  # noqa: F401
from janitor.errors import JanitorError
from janitor.timeseries import _flag_jumps_single_col


@pytest.fixture
def timeseries_dataframe() -> pd.DataFrame:
    """Returns a time series dataframe."""
    ts_index = pd.date_range("1/1/2019", periods=10, freq="1H")
    c1 = [*range(10)]
    c2 = [*range(100, 110)]
    c3 = c1[::-1]
    c4 = c2[::-1]
    c5 = [-2.0, -1.0, 0, 1.0, 2.0, 1.0, 0, -1.0, -2.0, -7.5]
    test_df = pd.DataFrame(
        {"col1": c1, "col2": c2, "col3": c3, "col4": c4, "col5": c5},
        index=ts_index,
    )
    return test_df


@pytest.mark.timeseries
def test__flag_jumps_single_col_raises_error_for_bad_scale_type(
    timeseries_dataframe,
):
    """Test that invalid scale argument raises a JanitorError."""
    # Setup
    df = timeseries_dataframe
    expected_error_msg = (
        "Unrecognized scale: 'bad_scale'. "
        + "Must be one of: ['absolute', 'percentage']."
    )

    # Exercise
    with pytest.raises(JanitorError) as error_info:
        _flag_jumps_single_col(
            df, col="col1", scale="bad_scale", direction="any", threshold=1
        )

    # Verify
    assert str(error_info.value) == expected_error_msg

    # Cleanup - none necessary


@pytest.mark.timeseries
def test__flag_jumps_single_col_raises_error_for_bad_direction_type(
    timeseries_dataframe,
):
    """Test that invalid direction argument raises a JanitorError."""
    # Setup
    df = timeseries_dataframe
    expected_error_msg = (
        "Unrecognized direction: 'bad_direction'. "
        + "Must be one of: ['increasing', 'decreasing', 'any']."
    )

    # Exercise
    with pytest.raises(JanitorError) as error_info:
        _flag_jumps_single_col(
            df,
            col="col1",
            scale="absolute",
            direction="bad_direction",
            threshold=1,
        )

    # Verify
    assert str(error_info.value) == expected_error_msg

    # Cleanup - none necessary


@pytest.mark.timeseries
def test__flag_jumps_single_col_raises_error_for_bad_threshold_value(
    timeseries_dataframe,
):
    """Test that invalid threshold argument raises a JanitorError."""
    # Setup
    df = timeseries_dataframe
    expected_error_msg = (
        "Unrecognized threshold: -1. This value must be >= 0.0. "
        + "Use 'direction' to specify positive or negative intent."
    )

    # Exercise
    with pytest.raises(JanitorError) as error_info:
        _flag_jumps_single_col(
            df, col="col1", scale="absolute", direction="any", threshold=-1
        )

    # Verify
    assert str(error_info.value) == expected_error_msg

    # Cleanup - none necessary


@pytest.mark.timeseries
@pytest.mark.parametrize(
    "col, direction, expected",
    [
        ("col1", "increasing", [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        ("col2", "increasing", [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        ("col3", "decreasing", [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        ("col4", "decreasing", [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        ("col5", "increasing", [0, 1, 1, 1, 1, 0, 0, 0, 0, 0]),
        ("col5", "decreasing", [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
    ],
)
def test__flag_jumps_single_col_absolute_scale_correct_direction(
    timeseries_dataframe, col, direction, expected
):
    """
    Test utility function for flagging jumps with absolute scale.
    Here, the correct/anticipated `direction` is provided
        (i.e. increasing when the df column is truly increasing
        and decreasing when the df column is truly decreasing)
    """
    # Setup
    df = timeseries_dataframe

    # Exercise
    result = _flag_jumps_single_col(
        df, col, scale="absolute", direction=direction, threshold=0.5
    )

    # Verify
    np.testing.assert_array_equal(result.array, expected)

    # Cleanup - none necessary


@pytest.mark.timeseries
@pytest.mark.parametrize(
    "col, direction, expected",
    [
        ("col1", "decreasing", [0] * 10),
        ("col2", "decreasing", [0] * 10),
        ("col3", "increasing", [0] * 10),
        ("col4", "increasing", [0] * 10),
        ("col5", "decreasing", [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
        ("col5", "increasing", [0, 1, 1, 1, 1, 0, 0, 0, 0, 0]),
    ],
)
def test__flag_jumps_single_col_absolute_scale_inverse_direction(
    timeseries_dataframe, col, direction, expected
):
    """
    Test utility function for flagging jumps with absolute scale.
    Here, the inverse `direction` is provided so should not flag
    anything (i.e. increasing when the df column is truly decreasing
    and increasing when the df column is truly increasing)
    """
    # Setup
    df = timeseries_dataframe

    # Exercise
    result = _flag_jumps_single_col(
        df, col, scale="absolute", direction=direction, threshold=0.5
    )

    # Verify
    np.testing.assert_array_equal(result.array, expected)

    # Cleanup - none necessary


@pytest.mark.timeseries
@pytest.mark.parametrize("col", ("col1", "col2", "col3", "col4", "col5"))
def test__flag_jumps_single_col_absolute_scale_any_direction(
    timeseries_dataframe, col
):
    """
    Test utility function for flagging jumps with absolute scale.
    Here, the any `direction` is provided so should flag everything.
    """
    # Setup
    df = timeseries_dataframe

    # Exercise
    result = _flag_jumps_single_col(
        df, col, scale="absolute", direction="any", threshold=0.5
    )

    # Verify
    np.testing.assert_array_equal(result.array, [0, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    # Cleanup - none necessary


@pytest.mark.timeseries
def test__flag_jumps_single_col_absolute_scale_flags_large_jump(
    timeseries_dataframe,
):
    """
    Test utility function for flagging jumps with absolute scale.
    Here, a large threshold is used to verify only one row is flagged.
    """

    # Setup
    df = timeseries_dataframe

    # Exercise
    result_incr = _flag_jumps_single_col(
        df, "col5", scale="absolute", direction="increasing", threshold=5
    )
    result_decr = _flag_jumps_single_col(
        df, "col5", scale="absolute", direction="decreasing", threshold=5
    )
    result_any = _flag_jumps_single_col(
        df, "col5", scale="absolute", direction="any", threshold=5
    )

    # Verify
    np.testing.assert_array_equal(result_incr.array, [0] * 10)
    np.testing.assert_array_equal(
        result_decr.array, [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    )
    np.testing.assert_array_equal(
        result_any.array, [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    )

    # Cleanup - none necessary


@pytest.mark.timeseries
@pytest.mark.parametrize(
    "col, direction, expected",
    [
        ("col1", "increasing", [0, 1, 1, 1, 1, 0, 0, 0, 0, 0]),
        ("col2", "increasing", [0] * 10),
        ("col3", "decreasing", [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]),
        ("col4", "decreasing", [0] * 10),
        ("col5", "increasing", [0, 1, 1, 1, 1, 0, 0, 0, 0, 0]),
        ("col5", "decreasing", [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
    ],
)
def test__flag_jumps_single_col_percentage_scale_correct_direction(
    timeseries_dataframe, col, direction, expected
):
    """
    Test utility function for flagging jumps with percentage scale and
    a 25% jump. Here, the correct/anticipated `direction` is provided
    (i.e. increasing when the df column is truly increasing and
    decreasing when the df column is truly decreasing).
    """
    # Setup
    df = timeseries_dataframe

    # Exercise
    result = _flag_jumps_single_col(
        df, col, scale="percentage", direction=direction, threshold=0.25
    )

    # Verify
    np.testing.assert_array_equal(result.array, expected)

    # Cleanup - none necessary


@pytest.mark.timeseries
@pytest.mark.parametrize(
    "col, direction, expected",
    [
        ("col1", "decreasing", [0] * 10),
        ("col2", "decreasing", [0] * 10),
        ("col3", "increasing", [0] * 10),
        ("col4", "increasing", [0] * 10),
        ("col5", "decreasing", [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
        ("col5", "increasing", [0, 1, 1, 1, 1, 0, 0, 0, 0, 0]),
    ],
)
def test__flag_jumps_single_col_percentage_scale_inverse_direction(
    timeseries_dataframe, col, direction, expected
):
    """
    Test utility function for flagging jumps with percentage scale and
    a 25% jump. Here, the inverse `direction` is provided so should not
    flag anything (i.e. increasing when the df column is truly
    decreasing and increasing when the df column is truly increasing).
    """
    # Setup
    df = timeseries_dataframe

    # Exercise
    result = _flag_jumps_single_col(
        df, col, scale="percentage", direction=direction, threshold=0.25
    )

    # Verify
    np.testing.assert_array_equal(result.array, expected)

    # Cleanup - none necessary


@pytest.mark.timeseries
@pytest.mark.parametrize(
    "col, expected",
    [
        ("col1", [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        ("col2", [0] * 10),
        ("col3", [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        ("col4", [0] * 10),
        ("col5", [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    ],
)
def test__flag_jumps_single_col_percentage_scale_any_direction(
    timeseries_dataframe, col, expected
):
    """
    Test utility function for flagging jumps with percentage scale and
    a 10% jump. Here, the any direction is provided so should flag
    everything.
    """
    # Setup
    df = timeseries_dataframe

    # Exercise
    result = _flag_jumps_single_col(
        df, col, scale="percentage", direction="any", threshold=0.10
    )

    # Verify
    np.testing.assert_array_equal(result.array, expected)

    # Cleanup - none necessary


@pytest.mark.timeseries
def test__flag_jumps_single_col_percentage_scale_flags_large_jump(
    timeseries_dataframe,
):
    """
    Test utility function for flagging jumps with percentage scale and
    a 100% jump. Here, a large threshold is used to verify only
    drastically changed rows are flagged.
    """
    # Setup
    df = timeseries_dataframe

    # Exercise
    result_incr = _flag_jumps_single_col(
        df, "col5", scale="percentage", direction="increasing", threshold=1.0
    )
    result_decr = _flag_jumps_single_col(
        df, "col5", scale="percentage", direction="decreasing", threshold=1.0
    )
    result_any = _flag_jumps_single_col(
        df, "col5", scale="percentage", direction="any", threshold=1.0
    )

    # Verify
    np.testing.assert_array_equal(
        result_incr.array, [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    )
    np.testing.assert_array_equal(
        result_decr.array, [0, 0, 0, 0, 0, 0, 0, 1, 0, 1]
    )
    np.testing.assert_array_equal(
        result_any.array, [0, 0, 0, 1, 0, 0, 0, 1, 0, 1]
    )

    # Cleanup - none necessary


@pytest.mark.timeseries
def test_flag_jumps_raises_error_for_strict_no_arg_dicts(timeseries_dataframe):
    """
    Test that an error is raised when `strict=True`
    and no input arguments are of type dict.
    """
    # Setup
    df = timeseries_dataframe

    # Exercise
    expected_error_msg = (
        "When enacting 'strict=True', 'scale', 'direction', "
        + "or 'threshold' must be a dictionary."
    )

    # Exercise
    with pytest.raises(JanitorError) as error_info:
        df.flag_jumps(
            scale="absolute", direction="any", threshold=0, strict=True
        )

    # Verify
    assert str(error_info.value) == expected_error_msg

    # Cleanup - none necessary


@pytest.mark.timeseries
def test_flag_jumps_default_args(timeseries_dataframe):
    """
    Test the default values behave as expected.
    Namely, `scale=percentage`, `direction=any`, `threshold=0.0` and
    `strict=False`.
    """
    # Setup
    df = timeseries_dataframe
    orig_cols = df.columns

    expected = np.ones((10, 5), dtype=int)
    expected[0, :] = 0
    expected_cols = [f"{c}_jump_flag" for c in orig_cols]
    expected_df = pd.DataFrame(expected, columns=expected_cols, index=df.index)

    # Exercise
    df = df.flag_jumps()

    # Verify
    assert list(df.columns) == list(orig_cols) + expected_cols
    assert df.filter(regex="flag").equals(expected_df)

    # Cleanup - none necessary


@pytest.mark.timeseries
def test_flag_jumps_all_args_specifed_as_non_dict(timeseries_dataframe):
    """Test provided kwargs (not of type dict) behave as expected."""
    # Setup
    df = timeseries_dataframe
    orig_cols = df.columns

    expected = np.ones((10, 5), dtype=int)
    expected[0, :] = 0
    expected[:, 2:4] = 0
    expected[5:10, 4] = 0
    expected_cols = [f"{c}_jump_flag" for c in orig_cols]
    expected_df = pd.DataFrame(expected, columns=expected_cols, index=df.index)

    # Exercise
    df = df.flag_jumps(
        scale="absolute", direction="increasing", threshold=0, strict=False
    )

    # Verify
    assert list(df.columns) == list(orig_cols) + expected_cols
    assert df.filter(regex="flag").equals(expected_df)

    # Cleanup - none necessary


@pytest.mark.timeseries
def test_flag_jumps_all_args_specified_as_dict(timeseries_dataframe):
    """
    Test provided kwargs (of type dict) behaves as expected.
    Since strict defaults to `False`, col3, col4, and col5 will be
    flagged and will use default args (`scale=percentage`,
    `direction=any`, and `threshold=0.0`).
    """
    df = timeseries_dataframe
    orig_cols = df.columns

    expected = np.ones((10, 5), dtype=int)
    expected[0, :] = 0
    expected[:, 0:2] = 0
    expected_cols = [f"{c}_jump_flag" for c in orig_cols]
    expected_df = pd.DataFrame(expected, columns=expected_cols, index=df.index)

    # Exercise
    df = df.flag_jumps(
        scale=dict(col1="absolute", col2="percentage"),
        direction=dict(col1="increasing", col2="any"),
        threshold=dict(col1=1, col2=2),
    )

    # Verify
    assert list(df.columns) == list(orig_cols) + expected_cols
    assert df.filter(regex="flag").equals(expected_df)

    # Cleanup - none necessary


@pytest.mark.timeseries
def test_flag_jumps_strict_with_both_cols_in_all_args(timeseries_dataframe):
    """
    Test provided strict behaves as expected
    (only col1 and col2 flagged).
    """
    df = timeseries_dataframe
    orig_cols = df.columns

    expected = np.zeros((10, 2), dtype=int)
    expected_cols = ["col1_jump_flag", "col2_jump_flag"]
    expected_df = pd.DataFrame(expected, columns=expected_cols, index=df.index)

    # Exercise
    df = df.flag_jumps(
        scale=dict(col1="absolute", col2="percentage"),
        direction=dict(col1="increasing", col2="any"),
        threshold=dict(col1=1, col2=2),
        strict=True,
    )

    # Verify
    assert list(df.columns) == list(orig_cols) + expected_cols
    assert df.filter(regex="flag").equals(expected_df)

    # Cleanup - none necessary


@pytest.mark.timeseries
def test_flag_jumps_strict_with_both_cols_in_at_least_one_args(
    timeseries_dataframe,
):
    """
    Test provided strict behaves as expected
    (col4 not provided in any input arg dict thus not flagged)
    When left unspecified, a column will be flagged based on defaults
    (`scale=percentage`, `direction=any`, `threshold=0.0`).
    """
    df = timeseries_dataframe
    orig_cols = df.columns

    expected = np.ones((10, 4), dtype=int)
    expected[0, :] = 0
    expected[:, 3] = 0
    expected[3, 3] = 1
    expected[7, 3] = 1
    expected[9, 3] = 1
    expected_cols = [f"col{i}_jump_flag" for i in [1, 2, 3, 5]]
    expected_df = pd.DataFrame(expected, columns=expected_cols, index=df.index)

    # Exercise
    df = df.flag_jumps(
        scale=dict(col1="absolute", col3="absolute"),
        direction=dict(col2="increasing"),
        threshold=dict(col5=2),
        strict=True,
    )

    # Verify
    assert list(df.columns) == list(orig_cols) + expected_cols
    assert df.filter(regex="flag").equals(expected_df)

    # Cleanup - none necessary


@pytest.mark.timeseries
def test_flag_jumps_for_one_column(timeseries_dataframe):
    """
    Test provided strict behaves as expected for a single column.
    """
    df = timeseries_dataframe
    orig_cols = df.columns
    expected = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    # Exercise
    df = df.flag_jumps(scale=dict(col1="absolute"), strict=True)

    # Verify
    assert list(df.columns) == list(orig_cols) + ["col1_jump_flag"]
    np.testing.assert_array_equal(df["col1_jump_flag"].array, expected)

    # Cleanup - none necessary


@pytest.mark.timeseries
def test_flag_jumps_on_issue_provided_use_case():
    """
    Test example provided in issue is solved with `flag_jumps()`
    See issue # 711
    """
    # Setup
    df = pd.DataFrame(
        data=[
            ["2015-01-01 00:00:00", -0.76, 2, 2, 1.2],
            ["2015-01-01 01:00:00", -0.73, 2, 4, 1.1],
            ["2015-01-01 02:00:00", -0.71, 2, 4, 1.1],
            ["2015-01-01 03:00:00", -0.68, 2, 32, 1.1],
            ["2015-01-01 04:00:00", -0.65, 2, 2, 1.0],
            ["2015-01-01 05:00:00", -0.76, 2, 2, 1.2],
            ["2015-01-01 06:00:00", -0.73, 2, 4, 1.1],
            ["2015-01-01 07:00:00", -0.71, 2, 4, 1.1],
            ["2015-01-01 08:00:00", -0.68, 2, 32, 1.1],
            ["2015-01-01 09:00:00", -0.65, 2, 2, 1.0],
            ["2015-01-01 10:00:00", -0.76, 2, 2, 1.2],
            ["2015-01-01 11:00:00", -0.73, 2, 4, 1.1],
            ["2015-01-01 12:00:00", -0.71, 2, 4, 1.1],
            ["2015-01-01 13:00:00", -0.68, 2, 32, 1.1],
            ["2015-01-01 14:00:00", -0.65, 2, 2, 1.0],
            ["2015-01-01 15:00:00", -0.76, 2, 2, 1.2],
            ["2015-01-01 16:00:00", -0.73, 2, 4, 1.1],
            ["2015-01-01 17:00:00", -0.71, 2, 4, 1.1],
            ["2015-01-01 18:00:00", -0.68, 2, 32, 1.1],
            ["2015-01-01 19:00:00", -0.65, 2, 2, 1.0],
            ["2015-01-01 20:00:00", -0.76, 2, 2, 1.2],
            ["2015-01-01 21:00:00", -0.73, 2, 4, 1.1],
            ["2015-01-01 22:00:00", -0.71, 2, 4, 1.1],
            ["2015-01-01 23:00:00", -0.68, 2, 32, 1.1],
            ["2015-01-02 00:00:00", -0.65, 2, 2, 1.0],
        ],
        columns=["DateTime", "column1", "column2", "column3", "column4"],
    )

    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df = df.set_index("DateTime")
    orig_cols = df.columns

    expected = np.zeros((25, 4), dtype=int)
    expected[3, 2] = 1
    expected[8, 2] = 1
    expected[13, 2] = 1
    expected[18, 2] = 1
    expected[23, 2] = 1
    expected_cols = [f"{c}_jump_flag" for c in orig_cols]
    expected_df = pd.DataFrame(expected, columns=expected_cols, index=df.index)

    # Exercise
    result = df.flag_jumps(
        scale="absolute", direction="increasing", threshold=2
    )

    # Verify
    assert list(result.columns) == list(orig_cols) + expected_cols
    assert result.filter(regex="flag").equals(expected_df)

    # Cleanup - none necessary
