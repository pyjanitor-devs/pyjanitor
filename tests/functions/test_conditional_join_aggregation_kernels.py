"""Coverage for dtype-specific fused conditional-join aggregation kernels."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

NUMERIC_DTYPES = [
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float32",
    "float64",
]

EXTENSION_DTYPES = [
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
    "Float32",
    "Float64",
]


def _with_matched_level(expected, output_length, matched):
    """Expand expected aggregations to the complete output domain."""
    if output_length == 0:
        return expected
    original_dtypes = {column: expected[column].dtype for column in expected.columns}
    expected = expected.reindex(range(output_length))
    for column_name, operation in expected.columns:
        if operation in {"size", "count", "sum"}:
            expected[(column_name, operation)] = expected[
                (column_name, operation)
            ].fillna(0)
            if operation in {"size", "count"}:
                expected[(column_name, operation)] = expected[
                    (column_name, operation)
                ].astype("int64")
            elif operation == "sum":
                expected[(column_name, operation)] = expected[
                    (column_name, operation)
                ].astype(_reduction_dtype(original_dtypes[(column_name, operation)]))
        elif operation == "prod":
            expected[(column_name, operation)] = expected[
                (column_name, operation)
            ].fillna(1)
            expected[(column_name, operation)] = expected[
                (column_name, operation)
            ].astype(_reduction_dtype(original_dtypes[(column_name, operation)]))
    expected.index = pd.MultiIndex.from_arrays(
        [range(output_length), matched],
        names=[None, "matched"],
    )
    return expected


def _numeric_frames(dtype):
    """Build small, sorted frames for a numeric dtype kernel test."""
    left = pd.DataFrame(
        {
            "key": pd.Series([1, 2, 3], dtype=dtype),
            "left_value": pd.Series([1, 2, 3], dtype=dtype),
            "residual": pd.Series([0, 1, 2], dtype=dtype),
        }
    )
    right = pd.DataFrame(
        {
            "key": pd.Series([2, 3, 4], dtype=dtype),
            "value": pd.Series([10, 20, 30], dtype=dtype),
            "residual": pd.Series([1, 1, 3], dtype=dtype),
        }
    )
    return left, right


def _apply_rust_integer_contract(expected, source, output_column):
    """Keep pandas' promoted sum/product baseline unchanged.

    Pandas/NumPy reductions promote signed integers to ``int64`` and unsigned
    integers to ``uint64``. The helper remains as a named compatibility point
    for the shared expected-result builders.
    """
    return expected


def _reduction_dtype(dtype):
    """Return the pandas/NumPy dtype for a sum or product reduction."""
    dtype = pd.api.types.pandas_dtype(dtype)
    if pd.api.types.is_unsigned_integer_dtype(dtype):
        return "UInt64" if pd.api.types.is_extension_array_dtype(dtype) else "uint64"
    if pd.api.types.is_integer_dtype(dtype):
        return "Int64" if pd.api.types.is_extension_array_dtype(dtype) else "int64"
    if pd.api.types.is_float_dtype(dtype):
        return dtype
    return dtype


def _expected_single(left, right, reverse):
    """Compute the single-range expectation using an explicit cross join."""
    pairs = left.assign(_left=np.arange(len(left))).merge(
        right.assign(_right=np.arange(len(right))), how="cross"
    )
    pairs = pairs.loc[pairs["key_x"] < pairs["key_y"]]
    if reverse:
        expected = pairs.groupby("_right", sort=True)["left_value"].agg(
            ["size", "sum", "prod", "min", "max"]
        )
    else:
        expected = pairs.groupby("_left", sort=True)["value"].agg(
            ["size", "sum", "prod", "min", "max"]
        )
    output_column = "left_value" if reverse else "value"
    expected.columns = pd.MultiIndex.from_tuples(
        [(output_column, operation) for operation in expected.columns]
    )
    source = left["left_value"] if reverse else right["value"]
    expected = _apply_rust_integer_contract(expected, source, output_column)
    output_length = len(right) if reverse else len(left)
    matched = expected.reindex(range(output_length))[(output_column, "size")].notna()
    return _with_matched_level(expected, output_length, matched.to_numpy())


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
@pytest.mark.parametrize("reverse", [False, True])
def test_anchor_non_equi_join_aggregation_dispatches_all_numeric_dtypes(dtype, reverse):
    """Every numeric dtype reaches the correct single Rust kernel."""
    left, right = _numeric_frames(dtype)
    actual = left.join_agg(
        right,
        ("key", "key", "<"),
        reverse=reverse,
        aggfunc=[
            ("value", "size") if not reverse else ("left_value", "size"),
            ("value", "sum") if not reverse else ("left_value", "sum"),
            ("value", "prod") if not reverse else ("left_value", "prod"),
            ("value", "min") if not reverse else ("left_value", "min"),
            ("value", "max") if not reverse else ("left_value", "max"),
        ],
    )
    expected = _expected_single(left, right, reverse)
    assert_frame_equal(expected, actual)


def test_count_and_size_support_non_numeric_aggregation_columns():
    """Count uses only its mask and size counts every matched pair."""
    left = pd.DataFrame({"limit": [3, 6, 10]})
    right = pd.DataFrame(
        {
            "value": [1, 4, 7, 12],
            "label": pd.Series(["a", "b", None, "d"], dtype="string"),
        }
    )

    actual = left.join_agg(
        right,
        ("limit", "value", "<"),
        aggfunc=[("label", "count"), ("label", "size")],
    )

    expected = pd.DataFrame(
        {
            ("label", "count"): [2, 1, 1],
            ("label", "size"): [3, 2, 1],
        },
        index=pd.MultiIndex.from_arrays(
            [[0, 1, 2], [True, True, True]],
            names=[None, "matched"],
        ),
    )
    assert_frame_equal(expected, actual)


def test_join_agg_can_omit_matched_level():
    """A caller that does not need matched metadata receives a plain index."""
    left = pd.DataFrame({"limit": [3, 6, 10]})
    right = pd.DataFrame({"value": [1, 4, 7, 12]})

    actual = left.join_agg(
        right,
        ("limit", "value", "<"),
        aggfunc=[("value", "size")],
        return_matched=False,
    )

    expected = pd.DataFrame(
        {("value", "size"): [3, 2, 1]},
        index=pd.RangeIndex(3),
    )
    assert_frame_equal(expected, actual)


def test_extended_join_agg_can_omit_matched_level():
    """The extended fused path uses the same plain-index contract."""
    left = pd.DataFrame({"limit": [3, 6, 10], "left_filter": [0, 1, 2]})
    right = pd.DataFrame({"value": [1, 4, 7, 12], "right_filter": [1, 0, 3, 2]})

    actual = left.join_agg(
        right,
        ("limit", "value", "<"),
        ("left_filter", "right_filter", "!="),
        aggfunc=[("value", "size")],
        return_matched=False,
    )

    expected = pd.DataFrame(
        {("value", "size"): [2, 2, 0]},
        index=pd.RangeIndex(3),
    )
    assert_frame_equal(expected, actual)


def test_join_agg_no_match_returns_empty_plain_index_without_matched():
    """No-match aggregation returns an empty frame without matched metadata."""
    left = pd.DataFrame({"key": [1]})
    right = pd.DataFrame({"key": [1], "value": [10]})

    actual = left.join_agg(
        right,
        ("key", "key", "<"),
        aggfunc=[("value", "size")],
        return_matched=False,
    )

    assert actual.empty
    assert isinstance(actual.index, pd.RangeIndex)
    assert list(actual.columns) == [("value", "size")]


def test_single_not_equal_aggregation_counts_duplicate_right_candidates():
    """Each duplicate right row contributes to a non-equal aggregation."""
    left = pd.DataFrame({"key": [1]})
    right = pd.DataFrame({"key": [2, 2], "value": [10, 20]})

    actual = left.join_agg(
        right,
        ("key", "key", "!="),
        aggfunc=[("value", "size"), ("value", "sum")],
    )

    expected = pd.DataFrame(
        {
            ("value", "size"): pd.Series([2], dtype="int64"),
            ("value", "sum"): pd.Series([30], dtype="int64"),
        },
        index=pd.Index([0]),
    )
    expected = _with_matched_level(expected, 1, np.array([True]))
    assert_frame_equal(expected, actual)


def test_reverse_not_equal_aggregation_preserves_reordered_right_positions():
    """Reverse ``!=`` aggregation stays aligned to physical right rows."""
    left = pd.DataFrame({"key": [1, 2], "left_value": [10, 20]})
    right = pd.DataFrame({"key": [3, 1, 2]})

    actual = left.join_agg(
        right,
        ("key", "key", "!="),
        reverse=True,
        aggfunc=[("left_value", "size"), ("left_value", "sum")],
    )

    expected = pd.DataFrame(
        {
            ("left_value", "size"): pd.array([1, 1, 2], dtype="int64"),
            ("left_value", "sum"): pd.array([20, 10, 30], dtype="int64"),
        },
        index=pd.MultiIndex.from_arrays(
            [[1, 2, 0], [True, True, True]], names=[None, "matched"]
        ),
    )
    assert_frame_equal(expected, actual)


def test_reverse_not_equal_aggregation_marks_only_matching_slots():
    """Reverse ``!=`` matched metadata identifies the surviving right row."""
    left = pd.DataFrame({"key": [1], "left_value": [10]})
    right = pd.DataFrame({"key": [1, 2]})

    actual = left.join_agg(
        right,
        ("key", "key", "!="),
        reverse=True,
        aggfunc=[("left_value", "size"), ("left_value", "sum")],
    )

    expected = pd.DataFrame(
        {
            ("left_value", "size"): pd.array([0, 1], dtype="int64"),
            ("left_value", "sum"): pd.array([0, 10], dtype="int64"),
        },
        index=pd.MultiIndex.from_arrays(
            [[0, 1], [False, True]], names=[None, "matched"]
        ),
    )
    assert_frame_equal(expected, actual)


def _expected_extended(left, right, reverse):
    """Compute the range-plus-residual expectation with a cross join."""
    pairs = left.assign(_left=np.arange(len(left))).merge(
        right.assign(_right=np.arange(len(right))), how="cross"
    )
    pairs = pairs.loc[
        (pairs["key_x"] < pairs["key_y"]) & (pairs["residual_x"] != pairs["residual_y"])
    ]
    if reverse:
        expected = pairs.groupby("_right", sort=True)["left_value"].agg(
            ["size", "sum", "prod", "min", "max"]
        )
    else:
        expected = pairs.groupby("_left", sort=True)["value"].agg(
            ["size", "sum", "prod", "min", "max"]
        )
    output_column = "left_value" if reverse else "value"
    expected.columns = pd.MultiIndex.from_tuples(
        [(output_column, operation) for operation in expected.columns]
    )
    source = left["left_value"] if reverse else right["value"]
    expected = _apply_rust_integer_contract(expected, source, output_column)
    output_length = len(right) if reverse else len(left)
    matched = expected.reindex(range(output_length))[(output_column, "size")].notna()
    return _with_matched_level(expected, output_length, matched.to_numpy())


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
@pytest.mark.parametrize("reverse", [False, True])
def test_extended_join_aggregation_dispatches_all_numeric_dtypes(dtype, reverse):
    """Every numeric dtype reaches the correct extended Rust kernel."""
    left, right = _numeric_frames(dtype)
    if reverse:
        aggfunc = [
            ("left_value", operation)
            for operation in ("size", "sum", "prod", "min", "max")
        ]
    else:
        aggfunc = [
            ("value", operation) for operation in ("size", "sum", "prod", "min", "max")
        ]
    actual = left.join_agg(
        right,
        ("key", "key", "<"),
        ("residual", "residual", "!="),
        reverse=reverse,
        aggfunc=aggfunc,
    )
    expected = _expected_extended(left, right, reverse)
    assert_frame_equal(expected, actual)


@pytest.mark.parametrize("dtype", EXTENSION_DTYPES)
def test_single_not_equal_aggregation_preserves_nullable_dtypes(dtype):
    """Nullable numeric arrays use extension null semantics and dtypes."""
    left = pd.DataFrame(
        {
            "key": pd.array([1, pd.NA], dtype=dtype),
            "value": pd.array([1, pd.NA], dtype=dtype),
        }
    )
    right = pd.DataFrame(
        {
            "key": pd.array([2, pd.NA], dtype=dtype),
            "value": pd.array([10, pd.NA], dtype=dtype),
        }
    )
    actual = left.join_agg(
        right,
        ("key", "key", "!="),
        aggfunc=[
            ("value", operation) for operation in ("size", "sum", "prod", "min", "max")
        ],
    )
    expected = pd.DataFrame(
        {
            ("value", "size"): pd.Series([1], dtype="int64"),
            ("value", "sum"): pd.Series(pd.array([10], dtype=_reduction_dtype(dtype))),
            ("value", "prod"): pd.Series(pd.array([10], dtype=_reduction_dtype(dtype))),
            ("value", "min"): pd.Series(pd.array([10], dtype=dtype)),
            ("value", "max"): pd.Series(pd.array([10], dtype=dtype)),
        },
        index=pd.Index([0]),
    )
    expected = _with_matched_level(
        expected,
        len(actual),
        np.isin(np.arange(len(actual)), expected.index),
    )
    assert_frame_equal(expected, actual)


@pytest.mark.parametrize("dtype", EXTENSION_DTYPES)
def test_extended_not_equal_aggregation_preserves_nullable_dtypes(dtype):
    """Extended all-``!=`` joins preserve nullable aggregation dtypes."""
    left = pd.DataFrame(
        {
            "key": pd.array([1, pd.NA], dtype=dtype),
            "residual": pd.array([0, 1], dtype=dtype),
            "left_value": pd.array([1, pd.NA], dtype=dtype),
        }
    )
    right = pd.DataFrame(
        {
            "key": pd.array([2, pd.NA], dtype=dtype),
            "residual": pd.array([1, 1], dtype=dtype),
            "value": pd.array([10, pd.NA], dtype=dtype),
        }
    )
    actual = left.join_agg(
        right,
        ("key", "key", "!="),
        ("residual", "residual", "!="),
        aggfunc=[
            ("value", operation) for operation in ("size", "sum", "prod", "min", "max")
        ],
    )
    expected = pd.DataFrame(
        {
            ("value", "size"): pd.Series([1], dtype="int64"),
            ("value", "sum"): pd.Series(pd.array([10], dtype=_reduction_dtype(dtype))),
            ("value", "prod"): pd.Series(pd.array([10], dtype=_reduction_dtype(dtype))),
            ("value", "min"): pd.Series(pd.array([10], dtype=dtype)),
            ("value", "max"): pd.Series(pd.array([10], dtype=dtype)),
        },
        index=pd.Index([0]),
    )
    expected = _with_matched_level(
        expected,
        len(actual),
        np.isin(np.arange(len(actual)), expected.index),
    )
    assert_frame_equal(expected, actual)


def test_single_not_equal_numpy_nulls_match_nulls():
    """NumPy-style ``!=`` treats a null pair as unequal."""
    left = pd.DataFrame({"key": [np.nan], "value": [1.0]})
    right = pd.DataFrame({"key": [np.nan], "value": [10.0]})
    actual = left.join_agg(
        right,
        ("key", "key", "!="),
        aggfunc=[("value", "size"), ("value", "sum")],
    )
    expected = pd.DataFrame(
        {
            ("value", "size"): pd.Series([1], dtype="int64"),
            ("value", "sum"): pd.Series([10.0], dtype="float64"),
        },
        index=pd.Index([0]),
    )
    expected = _with_matched_level(
        expected,
        len(actual),
        np.isin(np.arange(len(actual)), expected.index),
    )
    assert_frame_equal(expected, actual)


def test_single_not_equal_extension_nulls_do_not_match():
    """Pandas extension ``!=`` treats null comparisons as false filters."""
    dtype = "Float64"
    left = pd.DataFrame(
        {"key": pd.array([pd.NA], dtype=dtype), "value": pd.array([1], dtype=dtype)}
    )
    right = pd.DataFrame(
        {"key": pd.array([pd.NA], dtype=dtype), "value": pd.array([10], dtype=dtype)}
    )
    actual = left.join_agg(
        right,
        ("key", "key", "!="),
        aggfunc=[("value", "size"), ("value", "sum")],
    )
    expected = pd.DataFrame(
        {
            ("value", "size"): pd.Series([], dtype="int64"),
            ("value", "sum"): pd.Series([], dtype=_reduction_dtype(dtype)),
        },
        index=pd.Index([], dtype="int64"),
    )
    expected = _with_matched_level(
        expected,
        len(actual),
        np.isin(np.arange(len(actual)), expected.index),
    )
    assert_frame_equal(expected, actual)


def test_single_range_aggregation_preserves_match_for_null_value():
    """A null aggregation value does not erase an otherwise valid match."""
    left = pd.DataFrame({"key": [1]})
    right = pd.DataFrame({"key": [2], "value": pd.array([pd.NA], dtype="Int64")})
    actual = left.join_agg(
        right,
        ("key", "key", "<"),
        aggfunc=[("value", "size"), ("value", "min"), ("value", "max")],
    )
    expected = pd.DataFrame(
        {
            ("value", "size"): pd.Series([1], dtype="int64"),
            ("value", "min"): pd.Series(pd.array([pd.NA], dtype="Int64")),
            ("value", "max"): pd.Series(pd.array([pd.NA], dtype="Int64")),
        },
        index=pd.Index([0]),
    )
    expected = _with_matched_level(
        expected,
        len(actual),
        np.isin(np.arange(len(actual)), expected.index),
    )
    assert_frame_equal(expected, actual)


def test_single_reverse_aggregation_preserves_trimmed_right_order():
    """Reverse output follows the trimmed, value-sorted right layout."""
    left = pd.DataFrame({"key": [1], "value": [10]})
    right = pd.DataFrame({"key": [2, 1], "payload": [20, 30]})
    actual = left.join_agg(
        right,
        ("key", "key", "<"),
        reverse=True,
        aggfunc=[("value", "size"), ("value", "sum")],
    )
    expected = pd.DataFrame(
        {
            ("value", "size"): np.array([0, 1], dtype="int64"),
            ("value", "sum"): np.array([0, 10], dtype="int64"),
        },
        index=pd.MultiIndex.from_arrays(
            [[1, 0], [False, True]],
            names=[None, "matched"],
        ),
    )
    assert_frame_equal(expected, actual)


def test_single_range_aggregation_counts_duplicate_right_values():
    """Duplicate right values produce separate aggregation candidates."""
    left = pd.DataFrame({"key": [1]})
    right = pd.DataFrame({"key": [2, 2], "value": [20, 21]})
    actual = left.join_agg(
        right,
        ("key", "key", "<"),
        aggfunc=[
            ("value", "size"),
            ("value", "sum"),
            ("value", "prod"),
            ("value", "min"),
            ("value", "max"),
        ],
    )
    expected = pd.DataFrame(
        {
            ("value", "size"): pd.Series([2], dtype="int64"),
            ("value", "sum"): pd.Series([41], dtype="int64"),
            ("value", "prod"): pd.Series([420], dtype="int64"),
            ("value", "min"): pd.Series([20], dtype="int64"),
            ("value", "max"): pd.Series([21], dtype="int64"),
        },
        index=pd.Index([0]),
    )
    expected = _with_matched_level(
        expected,
        len(actual),
        np.isin(np.arange(len(actual)), expected.index),
    )
    assert_frame_equal(expected, actual)


def test_extended_aggregation_returns_empty_when_residual_rejects_all():
    """Residual filtering can remove every candidate from a range window."""
    left = pd.DataFrame({"key": [1], "residual": [5]})
    right = pd.DataFrame({"key": [2], "residual": [6], "value": [10]})
    actual = left.join_agg(
        right,
        ("key", "key", "<"),
        ("residual", "residual", "=="),
        aggfunc=[("value", "size"), ("value", "sum")],
    )
    expected = pd.DataFrame(
        {
            ("value", "size"): pd.Series([], dtype="int64"),
            ("value", "sum"): pd.Series([], dtype="int64"),
        },
        index=pd.Index([], dtype="int64"),
    )
    expected = _with_matched_level(
        expected,
        len(actual),
        np.isin(np.arange(len(actual)), expected.index),
    )
    assert_frame_equal(expected, actual)


def test_extended_aggregation_intersects_sorted_range_residuals():
    """Sorted residual ranges narrow candidates before aggregation."""
    left = pd.DataFrame({"first": [4], "second": [4]})
    right = pd.DataFrame(
        {
            "first": [1, 3, 5, 7],
            "second": [0, 1, 2, 6],
            "value": [10, 20, 30, 40],
        }
    )

    actual = left.join_agg(
        right,
        ("first", "first", "<"),
        ("second", "second", "<"),
        aggfunc=[("value", "size"), ("value", "sum")],
    )

    expected = pd.DataFrame(
        {
            ("value", "size"): pd.Series([1], dtype="int64"),
            ("value", "sum"): pd.Series([40], dtype="int64"),
        },
        index=pd.Index([0]),
    )
    expected = _with_matched_level(expected, len(actual), np.array([True]))
    assert_frame_equal(expected, actual)


def test_all_not_equal_aggregation_rejects_regions_algorithm():
    """Regions does not support fused all-``!=`` aggregation."""
    left = pd.DataFrame({"left_key": [1, 2]})
    right = pd.DataFrame({"right_key": [1, 2], "value": [10, 20]})
    with pytest.raises(
        NotImplementedError,
        match="aggfunc is not supported for all-!= joins with the regions algorithm",
    ):
        left.join_agg(
            right,
            ("left_key", "right_key", "!="),
            join_algorithm="regions",
            aggfunc=[("value", "size")],
        )


def test_single_reverse_aggregation_tracks_unsorted_right_positions():
    """Reverse aggregation preserves values when the right values are sorted."""
    left = pd.DataFrame({"key": [1, 2], "value": [10, 20]})
    right = pd.DataFrame({"key": [3, 1, 2], "payload": [30, 40, 50]})
    actual = left.join_agg(
        right,
        ("key", "key", "<"),
        reverse=True,
        aggfunc=[("value", "size"), ("value", "sum")],
    ).sort_index()
    expected = pd.DataFrame(
        {
            ("value", "size"): pd.Series([2, 1], index=[0, 2], dtype="int64"),
            ("value", "sum"): pd.Series([30, 10], index=[0, 2], dtype="int64"),
        },
        index=pd.Index([0, 2]),
    )
    expected = _with_matched_level(
        expected,
        len(actual),
        np.isin(np.arange(len(actual)), expected.index),
    )
    assert_frame_equal(expected, actual)


def test_extended_numpy_all_null_not_equal_aggregation_matches():
    """All-null NumPy ``!=`` candidates remain valid in extended joins."""
    left = pd.DataFrame({"key": [np.nan], "residual": [1.0], "value": [1.0]})
    right = pd.DataFrame({"key": [np.nan], "residual": [2.0], "value": [10.0]})
    actual = left.join_agg(
        right,
        ("key", "key", "!="),
        ("residual", "residual", "!="),
        aggfunc=[("value", "size"), ("value", "sum")],
    )
    expected = pd.DataFrame(
        {
            ("value", "size"): pd.Series([1], dtype="int64"),
            ("value", "sum"): pd.Series([10.0], dtype="float64"),
        },
        index=pd.Index([0]),
    )
    expected = _with_matched_level(
        expected,
        len(actual),
        np.isin(np.arange(len(actual)), expected.index),
    )
    assert_frame_equal(expected, actual)


def test_extended_extension_all_null_not_equal_aggregation_is_empty():
    """All-null extension ``!=`` candidates are filtered out as false."""
    dtype = "Int64"
    left = pd.DataFrame(
        {
            "key": pd.array([pd.NA], dtype=dtype),
            "residual": pd.array([1], dtype=dtype),
        }
    )
    right = pd.DataFrame(
        {
            "key": pd.array([pd.NA], dtype=dtype),
            "residual": pd.array([2], dtype=dtype),
            "value": pd.array([10], dtype=dtype),
        }
    )
    actual = left.join_agg(
        right,
        ("key", "key", "!="),
        ("residual", "residual", "!="),
        aggfunc=[("value", "size"), ("value", "sum")],
    )
    expected = pd.DataFrame(
        {
            ("value", "size"): pd.Series([], dtype="int64"),
            ("value", "sum"): pd.Series([], dtype=_reduction_dtype(dtype)),
        },
        index=pd.Index([], dtype="int64"),
    )
    expected = _with_matched_level(
        expected,
        len(actual),
        np.isin(np.arange(len(actual)), expected.index),
    )
    assert_frame_equal(expected, actual)


def test_single_reverse_extension_aggregation_preserves_dtype():
    """Reverse aggregation keeps nullable source values and labels aligned."""
    dtype = "Int64"
    left = pd.DataFrame(
        {
            "key": pd.array([1, 2], dtype=dtype),
            "value": pd.array([10, 20], dtype=dtype),
        }
    )
    right = pd.DataFrame({"key": pd.array([2, 3], dtype=dtype)})
    actual = left.join_agg(
        right,
        ("key", "key", "<"),
        reverse=True,
        aggfunc=[("value", "size"), ("value", "sum")],
    ).sort_index()
    expected = pd.DataFrame(
        {
            ("value", "size"): pd.Series([1, 2], index=[0, 1], dtype="int64"),
            ("value", "sum"): pd.Series(
                pd.array([10, 30], dtype=_reduction_dtype(dtype)), index=[0, 1]
            ),
        },
        index=pd.Index([0, 1]),
    )
    expected = _with_matched_level(
        expected,
        len(actual),
        np.isin(np.arange(len(actual)), expected.index),
    )
    assert_frame_equal(expected, actual)


@pytest.mark.parametrize(
    ("left_key", "right_key", "right_value", "expected_size", "expected_sum"),
    [
        ([np.nan], [1.0, 2.0], [10.0, 20.0], 2, 30.0),
        ([1.0], [np.nan], [10.0], 1, 10.0),
    ],
)
def test_single_not_equal_numpy_one_sided_nulls(
    left_key, right_key, right_value, expected_size, expected_sum
):
    """NumPy nulls compare unequal to every non-null value."""
    left = pd.DataFrame({"key": left_key, "value": [1.0]})
    right = pd.DataFrame({"key": right_key, "value": right_value})
    actual = left.join_agg(
        right,
        ("key", "key", "!="),
        aggfunc=[("value", "size"), ("value", "sum")],
    )
    expected = pd.DataFrame(
        {
            ("value", "size"): pd.Series([expected_size], dtype="int64"),
            ("value", "sum"): pd.Series([expected_sum], dtype="float64"),
        },
        index=pd.Index([0]),
    )
    expected = _with_matched_level(
        expected,
        len(actual),
        np.isin(np.arange(len(actual)), expected.index),
    )
    assert_frame_equal(expected, actual)


@pytest.mark.parametrize(
    ("left_key", "right_key"),
    [([np.nan], [1.0]), ([1.0], [np.nan])],
)
def test_single_range_all_null_filtered_side_returns_empty(left_key, right_key):
    """Range aggregation returns an empty result when one side is all null."""
    left = pd.DataFrame({"key": left_key})
    right = pd.DataFrame(
        {"key": right_key, "value": pd.Series([10.0] * len(right_key))}
    )
    actual = left.join_agg(
        right,
        ("key", "key", "<"),
        aggfunc=[("value", "size"), ("value", "sum")],
    )
    expected = pd.DataFrame(
        {
            ("value", "size"): pd.Series([], dtype="int64"),
            ("value", "sum"): pd.Series([], dtype="float64"),
        },
        index=pd.Index([], dtype="int64"),
    )
    expected = _with_matched_level(
        expected,
        len(actual),
        np.isin(np.arange(len(actual)), expected.index),
    )
    assert_frame_equal(expected, actual)


def test_single_range_all_null_extension_value_handles_sum_and_product():
    """All-null aggregation values retain a valid match and dtype."""
    left = pd.DataFrame({"key": [1]})
    right = pd.DataFrame({"key": [2], "value": pd.array([pd.NA], dtype="Int64")})
    actual = left.join_agg(
        right,
        ("key", "key", "<"),
        aggfunc=[
            ("value", "size"),
            ("value", "sum"),
            ("value", "prod"),
        ],
    )
    expected = pd.DataFrame(
        {
            ("value", "size"): pd.Series([1], dtype="int64"),
            ("value", "sum"): pd.Series(pd.array([0], dtype="Int64")),
            ("value", "prod"): pd.Series(pd.array([1], dtype="Int64")),
        },
        index=pd.Index([0]),
    )
    expected = _with_matched_level(
        expected,
        len(actual),
        np.isin(np.arange(len(actual)), expected.index),
    )
    assert_frame_equal(expected, actual)


def test_single_reverse_aggregation_keeps_unsorted_duplicate_right_rows():
    """Unsorted duplicate right values retain both physical output rows."""
    left = pd.DataFrame({"key": [1], "value": [10]})
    right = pd.DataFrame({"key": [2, 1, 2], "payload": [20, 10, 21]})
    actual = left.join_agg(
        right,
        ("key", "key", "<"),
        reverse=True,
        aggfunc=[("value", "size"), ("value", "sum")],
    ).sort_index()
    expected = pd.DataFrame(
        {
            ("value", "size"): pd.Series([1, 1], index=[0, 2], dtype="int64"),
            ("value", "sum"): pd.Series([10, 10], index=[0, 2], dtype="int64"),
        },
        index=pd.Index([0, 2]),
    )
    expected = _with_matched_level(
        expected,
        len(actual),
        np.isin(np.arange(len(actual)), expected.index),
    )
    assert_frame_equal(expected, actual)


def test_single_range_aggregation_handles_float_infinities():
    """Valid infinite float values participate in range comparisons."""
    left = pd.DataFrame({"key": [0.0, np.inf, -np.inf], "value": [1.0, 2.0, 3.0]})
    right = pd.DataFrame({"key": [0.0, np.inf], "value": [10.0, 20.0]})
    actual = left.join_agg(
        right,
        ("key", "key", "<"),
        aggfunc=[("value", "size"), ("value", "sum")],
    )
    expected = pd.DataFrame(
        {
            ("value", "size"): pd.Series([1, 2], index=[0, 2], dtype="int64"),
            ("value", "sum"): pd.Series([20.0, 30.0], index=[0, 2]),
        },
        index=pd.Index([0, 2]),
    )
    expected = _with_matched_level(
        expected,
        len(actual),
        np.isin(np.arange(len(actual)), expected.index),
    )
    assert_frame_equal(expected, actual)


def test_single_range_aggregation_treats_nan_as_null():
    """NaN is filtered from range comparisons as a null value."""
    left = pd.DataFrame({"key": [1.0, np.nan], "value": [1.0, 2.0]})
    right = pd.DataFrame({"key": [2.0, 3.0], "value": [10.0, 20.0]})
    actual = left.join_agg(
        right,
        ("key", "key", "<"),
        aggfunc=[("value", "size"), ("value", "sum")],
    )
    expected = pd.DataFrame(
        {
            ("value", "size"): pd.Series([2], index=[0], dtype="int64"),
            ("value", "sum"): pd.Series([30.0], index=[0]),
        },
        index=pd.Index([0]),
    )
    expected = _with_matched_level(
        expected,
        len(actual),
        np.isin(np.arange(len(actual)), expected.index),
    )
    assert_frame_equal(expected, actual)


def test_extended_reverse_extension_not_equal_aggregation_preserves_dtype():
    """Reverse all-``!=`` aggregation handles nullable extension values."""
    dtype = "Int64"
    left = pd.DataFrame(
        {
            "key": pd.array([1, 2, pd.NA], dtype=dtype),
            "residual": pd.array([0, 1, 2], dtype=dtype),
            "value": pd.array([10, 20, pd.NA], dtype=dtype),
        }
    )
    right = pd.DataFrame(
        {
            "key": pd.array([2, 3, pd.NA], dtype=dtype),
            "residual": pd.array([1, 1, 3], dtype=dtype),
        }
    )
    actual = left.join_agg(
        right,
        ("key", "key", "!="),
        ("residual", "residual", "!="),
        reverse=True,
        aggfunc=[("value", "size"), ("value", "sum")],
    ).sort_index()
    expected = pd.DataFrame(
        {
            ("value", "size"): pd.Series([1, 1], index=[0, 1], dtype="int64"),
            ("value", "sum"): pd.Series(
                pd.array([10, 10], dtype=_reduction_dtype(dtype)), index=[0, 1]
            ),
        },
        index=pd.Index([0, 1]),
    )
    expected = _with_matched_level(
        expected,
        len(actual),
        np.isin(np.arange(len(actual)), expected.index),
    )
    assert_frame_equal(expected, actual)


@pytest.mark.parametrize("dtype", ["int64", "uint64"])
def test_single_range_aggregation_handles_integer_boundaries(dtype):
    """Signed and unsigned 64-bit boundary values remain comparable."""
    if dtype == "int64":
        left_values = [np.iinfo(np.int64).min, np.iinfo(np.int64).max - 1]
        right_values = [np.iinfo(np.int64).min + 1, np.iinfo(np.int64).max]
    else:
        left_values = [0, np.iinfo(np.uint64).max - 1]
        right_values = [1, np.iinfo(np.uint64).max]

    left = pd.DataFrame(
        {
            "key": pd.Series(left_values, dtype=dtype),
            "value": pd.Series([1, 2], dtype=dtype),
        }
    )
    right = pd.DataFrame(
        {
            "key": pd.Series(right_values, dtype=dtype),
            "value": pd.Series([10, 20], dtype=dtype),
        }
    )
    actual = left.join_agg(
        right,
        ("key", "key", "<"),
        aggfunc=[("value", "size"), ("value", "sum")],
    )
    expected = pd.DataFrame(
        {
            ("value", "size"): pd.Series([2, 1], index=[0, 1], dtype="int64"),
            ("value", "sum"): pd.Series(
                [30, 20], index=[0, 1], dtype=_reduction_dtype(dtype)
            ),
        },
        index=pd.Index([0, 1]),
    )
    expected = _with_matched_level(
        expected,
        len(actual),
        np.isin(np.arange(len(actual)), expected.index),
    )
    assert_frame_equal(expected, actual)


@pytest.mark.parametrize(
    ("dtype", "expected_sum", "expected_prod", "output_dtype"),
    [
        ("int8", 129, 254, "int64"),
        ("int16", 32769, 65534, "int64"),
        ("int32", 2147483649, 4294967294, "int64"),
        ("uint8", 257, 510, "uint64"),
        ("uint16", 65537, 131070, "uint64"),
        ("uint32", 4294967297, 8589934590, "uint64"),
    ],
)
def test_single_range_aggregation_uses_pandas_integer_promotion(
    dtype, expected_sum, expected_prod, output_dtype
):
    """Integer reductions use pandas-style signed/unsigned promotion."""
    maximum = np.iinfo(dtype).max
    left = pd.DataFrame({"key": pd.Series([1], dtype=dtype)})
    right = pd.DataFrame(
        {
            "key": pd.Series([2, 3], dtype=dtype),
            "value": pd.Series([maximum, 2], dtype=dtype),
        }
    )
    actual = left.join_agg(
        right,
        ("key", "key", "<"),
        aggfunc=[("value", "sum"), ("value", "prod")],
    )
    expected = pd.DataFrame(
        {
            ("value", "sum"): pd.Series([expected_sum], dtype=output_dtype),
            ("value", "prod"): pd.Series([expected_prod], dtype=output_dtype),
        },
        index=pd.Index([0]),
    )
    expected = _with_matched_level(
        expected,
        len(actual),
        np.isin(np.arange(len(actual)), expected.index),
    )
    assert_frame_equal(expected, actual)


def test_single_range_float32_aggregation_preserves_float32():
    """Float32 sum follows pandas and retains the float32 result dtype."""
    left = pd.DataFrame({"key": pd.Series([1], dtype="float32")})
    right = pd.DataFrame(
        {
            "key": pd.Series([2, 3], dtype="float32"),
            "value": pd.Series([0.1, 0.2], dtype="float32"),
        }
    )
    actual = left.join_agg(
        right,
        ("key", "key", "<"),
        aggfunc=[("value", "sum")],
    )
    expected_value = right["value"].sum()
    expected = pd.DataFrame(
        {("value", "sum"): pd.Series([expected_value], dtype="float32")},
        index=pd.Index([0]),
    )
    expected = _with_matched_level(
        expected,
        len(actual),
        np.isin(np.arange(len(actual)), expected.index),
    )
    assert_frame_equal(expected, actual)
