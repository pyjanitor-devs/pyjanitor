"""Tests for `flag_outliers`."""

import warnings

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from janitor.functions import flag_outliers


def _frame(values, dtype=None, index=None):
    """Build a one-column frame for outlier tests."""
    series = pd.Series(values, dtype=dtype, index=index)
    return pd.DataFrame({"values": series})


@pytest.mark.functions
def test_iqr_flags_only_the_extreme_value():
    """IQR with the default threshold flags a clear high outlier."""
    df = _frame([10, 12, 11, 13, 100, 9, 11])

    result = df.flag_outliers("values")

    assert result["values_outlier_flag"].tolist() == [
        False,
        False,
        False,
        False,
        True,
        False,
        False,
    ]
    assert result["values_outlier_flag"].dtype == np.dtype("bool")


@pytest.mark.functions
def test_zscore_flags_only_the_extreme_value():
    """Z-score with the default threshold flags a clear high outlier."""
    df = _frame([10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 50])

    result = df.flag_outliers("values", method="zscore")

    expected = [False] * 11 + [True]
    assert result["values_outlier_flag"].tolist() == expected


@pytest.mark.functions
def test_iqr_default_threshold_matches_1_5():
    """The IQR default cutoff is 1.5, not the Z-score cutoff of 3."""
    # Q1=2.25, Q3=4.75, IQR=2.5. 10 is outside a 1.5 fence and inside a 3 fence.
    df = _frame([1, 2, 3, 4, 5, 10])

    default = df.flag_outliers("values")
    explicit = df.flag_outliers("values", method="iqr", threshold=1.5)
    wide = df.flag_outliers("values", method="iqr", threshold=3)

    assert default["values_outlier_flag"].tolist() == [
        False,
        False,
        False,
        False,
        False,
        True,
    ]
    assert_series_equal(
        default["values_outlier_flag"],
        explicit["values_outlier_flag"],
        check_names=False,
    )
    assert wide["values_outlier_flag"].tolist() == [False] * 6


@pytest.mark.functions
def test_zscore_default_threshold_matches_3():
    """The Z-score default cutoff is 3, not the IQR cutoff of 1.5."""
    # The last value has a sample Z-score of about 1.79.
    df = _frame([0, 0, 0, 0, 1])

    default = df.flag_outliers("values", method="zscore")
    explicit = df.flag_outliers("values", method="zscore", threshold=3)
    tight = df.flag_outliers("values", method="zscore", threshold=1.5)

    assert default["values_outlier_flag"].tolist() == [False] * 5
    assert_series_equal(
        default["values_outlier_flag"],
        explicit["values_outlier_flag"],
        check_names=False,
    )
    assert tight["values_outlier_flag"].tolist() == [False, False, False, False, True]


@pytest.mark.functions
def test_zscore_default_does_not_use_the_iqr_threshold():
    """A modest outlier is inside 3 sample SDs even when IQR flags it."""
    df = _frame([10, 12, 11, 13, 100, 9, 11])

    iqr_flags = df.flag_outliers("values")["values_outlier_flag"]
    zscore_flags = df.flag_outliers("values", method="zscore")["values_outlier_flag"]
    zscore_at_two = df.flag_outliers("values", method="zscore", threshold=2)[
        "values_outlier_flag"
    ]

    assert iqr_flags.tolist()[4]
    assert zscore_flags.tolist() == [False] * 7
    assert zscore_at_two.tolist() == [False, False, False, False, True, False, False]


@pytest.mark.functions
def test_iqr_fence_is_exclusive():
    """A value exactly on the IQR fence is kept; a value beyond it is flagged."""
    # Q1=2, Q3=4, IQR=2, so the 1.5 fence is [-1, 7].
    on_fence = _frame([1, 2, 3, 4, 7])
    beyond_fence = _frame([1, 2, 3, 4, 8])

    assert (
        on_fence.flag_outliers("values")["values_outlier_flag"].tolist() == [False] * 5
    )
    assert beyond_fence.flag_outliers("values")["values_outlier_flag"].tolist() == [
        False,
        False,
        False,
        False,
        True,
    ]


@pytest.mark.functions
def test_zscore_cutoff_is_exclusive():
    """A value whose absolute Z-score equals the threshold is not flagged."""
    values = pd.Series([0, 0, 0, 0, 1])
    zscore = abs((values.iloc[-1] - values.mean()) / values.std())
    df = _frame(values)

    on_cutoff = df.flag_outliers("values", method="zscore", threshold=zscore)
    beyond_cutoff = df.flag_outliers(
        "values",
        method="zscore",
        threshold=float(zscore) - 1e-9,
    )

    assert on_cutoff["values_outlier_flag"].tolist() == [False] * 5
    assert beyond_cutoff["values_outlier_flag"].tolist() == [
        False,
        False,
        False,
        False,
        True,
    ]


@pytest.mark.functions
@pytest.mark.parametrize("method", ["iqr", "zscore"])
def test_nan_is_not_flagged_and_does_not_change_the_cutoff(method):
    """Missing values are skipped in the statistics and are never flagged."""
    clean = _frame([1, 2, 3, 4, 5, 10])
    dirty = _frame([1, np.nan, 2, 3, 4, 5, np.nan, 10])
    threshold = 1.5 if method == "iqr" else 1.0

    clean_flags = clean.flag_outliers("values", method=method, threshold=threshold)[
        "values_outlier_flag"
    ].tolist()
    dirty_flags = dirty.flag_outliers("values", method=method, threshold=threshold)[
        "values_outlier_flag"
    ].tolist()

    assert dirty_flags[1] is False
    assert dirty_flags[6] is False
    assert [dirty_flags[i] for i in (0, 2, 3, 4, 5, 7)] == clean_flags
    assert clean_flags[-1] is True


@pytest.mark.functions
@pytest.mark.parametrize("method", ["iqr", "zscore"])
def test_nullable_missing_values_are_not_flagged(method):
    """`pd.NA` in nullable numeric columns is ignored and left unflagged."""
    int_frame = _frame([1, 2, pd.NA, 3, 4, 5, 10], dtype="Int64")
    float_frame = _frame([1, 2, pd.NA, 3, 4, 5, 10], dtype="Float64")
    threshold = 1.5 if method == "iqr" else 1.0
    expected = [False, False, False, False, False, False, True]

    for df in (int_frame, float_frame):
        result = df.flag_outliers("values", method=method, threshold=threshold)
        assert_series_equal(result["values"], df["values"])
        assert result["values_outlier_flag"].dtype == np.dtype("bool")
        assert result["values_outlier_flag"].tolist() == expected


@pytest.mark.functions
@pytest.mark.parametrize("method", ["iqr", "zscore"])
def test_constant_column_has_no_outliers(method):
    """A constant column has IQR 0 and sample SD 0, so nothing is flagged."""
    frames = (
        _frame([5, 5, 5, 5, 5]),
        _frame([5.0, 5.0, np.nan, 5.0]),
        _frame([5, 5, pd.NA, 5], dtype="Int64"),
    )

    for df in frames:
        result = df.flag_outliers("values", method=method)
        assert result["values_outlier_flag"].tolist() == [False] * len(df)


@pytest.mark.functions
def test_zero_iqr_still_flags_a_distinct_value():
    """IQR 0 does not mean 'no outliers' when one value differs."""
    df = _frame([10, 10, 10, 10, 20])

    result = df.flag_outliers("values")

    assert result["values_outlier_flag"].tolist() == [
        False,
        False,
        False,
        False,
        True,
    ]


@pytest.mark.functions
@pytest.mark.parametrize(
    "dtype",
    ["int64", "int32", "float64", "float32", "Int64", "Float64"],
)
def test_numeric_dtypes_share_flags_and_keep_their_dtype(dtype):
    """Integer, float, and nullable numeric columns use the same flags."""
    df = _frame([10, 12, 11, 13, 100, 9, 11], dtype=dtype)

    result = df.flag_outliers("values")

    assert_series_equal(result["values"], df["values"])
    assert result["values_outlier_flag"].tolist() == [
        False,
        False,
        False,
        False,
        True,
        False,
        False,
    ]


@pytest.mark.functions
def test_custom_flag_column_name():
    """`flag_column_name` replaces the default `<column>_outlier_flag` name."""
    df = _frame([10, 12, 11, 100])

    result = df.flag_outliers("values", flag_column_name="is_outlier")

    assert result.columns.tolist() == ["values", "is_outlier"]
    assert result["is_outlier"].tolist() == [False, False, False, True]


@pytest.mark.functions
def test_does_not_mutate_original():
    """The input frame, including its index and other columns, is unchanged."""
    df = pd.DataFrame(
        {"values": [10, 12, 11, 100], "label": ["a", "b", "c", "d"]},
        index=pd.Index([3, 1, 4, 2], name="row"),
    )
    original = df.copy()

    result = df.flag_outliers("values")

    assert result is not df
    assert_frame_equal(df, original)
    assert result.index.equals(original.index)
    assert result.columns.tolist() == ["values", "label", "values_outlier_flag"]
    assert result["label"].tolist() == ["a", "b", "c", "d"]
    assert result["values_outlier_flag"].tolist() == [False, False, False, True]


@pytest.mark.functions
def test_duplicate_index_flags_by_position():
    """Duplicate index labels do not realign the flag onto the wrong row."""
    df = _frame([1, 100, 2, 3], index=[0, 0, 1, 2])

    result = df.flag_outliers("values")

    assert result.index.tolist() == [0, 0, 1, 2]
    assert result["values_outlier_flag"].tolist() == [False, True, False, False]


@pytest.mark.functions
def test_standalone_function_call():
    """`flag_outliers` can be called as a function, not only as a method."""
    df = _frame([10, 12, 11, 100])

    result = flag_outliers(df, column_name="values")

    assert result["values_outlier_flag"].tolist() == [False, False, False, True]
    assert_frame_equal(df, _frame([10, 12, 11, 100]))


@pytest.mark.functions
@pytest.mark.parametrize("method", ["iqr", "zscore"])
def test_all_missing_and_empty_columns_have_no_outliers(method):
    """Undefined statistics produce an all-False flag and do not raise."""
    all_missing = _frame([np.nan, np.nan])
    empty = _frame(pd.Series(dtype="float64"))

    missing_result = all_missing.flag_outliers("values", method=method)
    empty_result = empty.flag_outliers("values", method=method)

    assert missing_result["values_outlier_flag"].tolist() == [False, False]
    assert empty_result["values_outlier_flag"].tolist() == []
    assert empty_result["values_outlier_flag"].dtype == np.dtype("bool")


@pytest.mark.functions
@pytest.mark.parametrize("method", ["iqr", "zscore"])
def test_no_warnings_for_constant_or_missing_columns(method):
    """Constant, single-observation, and all-missing columns do not warn."""
    frames = (
        _frame([5, 5, 5, 5]),
        _frame([1.0, np.nan]),
        _frame([np.nan, np.nan]),
        _frame(pd.Series(dtype="Float64")),
    )

    for df in frames:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            df.flag_outliers("values", method=method)


@pytest.mark.functions
@pytest.mark.parametrize("method", ["mad", "IQR", "Zscore", ""])
def test_invalid_method_raises(method):
    """Unknown methods raise ValueError and leave the frame unchanged."""
    df = _frame([10, 12, 11, 100])
    original = df.copy()

    with pytest.raises(ValueError, match="Invalid method"):
        df.flag_outliers("values", method=method)

    assert_frame_equal(df, original)


@pytest.mark.functions
@pytest.mark.parametrize("threshold", [0, -1, -0.1, float("nan"), float("inf")])
def test_non_positive_threshold_raises(threshold):
    """Zero, negative, and non-finite thresholds raise ValueError."""
    df = _frame([10, 12, 11, 100])

    with pytest.raises(ValueError, match="positive number"):
        df.flag_outliers("values", threshold=threshold)


@pytest.mark.functions
@pytest.mark.parametrize("threshold", ["1.5", True, False])
def test_non_numeric_threshold_raises(threshold):
    """Boolean and non-numeric thresholds raise TypeError."""
    df = _frame([10, 12, 11, 100])

    with pytest.raises(TypeError, match="threshold"):
        df.flag_outliers("values", threshold=threshold)


@pytest.mark.functions
@pytest.mark.parametrize(
    "values",
    [
        ["alice", "bob", "carol"],
        pd.date_range("2020-01-01", periods=3),
    ],
)
def test_non_numeric_column_raises(values):
    """Text and datetime columns raise TypeError."""
    df = pd.DataFrame({"values": values})

    with pytest.raises(TypeError, match="must be numeric"):
        df.flag_outliers("values")


@pytest.mark.functions
def test_missing_column_raises():
    """A column that is not in the frame raises ValueError."""
    df = _frame([10, 12, 11, 100])

    with pytest.raises(ValueError, match="nonexistent not present"):
        df.flag_outliers("nonexistent")


@pytest.mark.functions
def test_existing_flag_column_raises():
    """An existing default or custom flag column raises ValueError."""
    df = _frame([10, 12, 11, 100])
    df["values_outlier_flag"] = False
    custom = _frame([10, 12, 11, 100])
    custom["is_outlier"] = False

    with pytest.raises(ValueError, match="values_outlier_flag already present"):
        df.flag_outliers("values")
    with pytest.raises(ValueError, match="is_outlier already present"):
        custom.flag_outliers("values", flag_column_name="is_outlier")
