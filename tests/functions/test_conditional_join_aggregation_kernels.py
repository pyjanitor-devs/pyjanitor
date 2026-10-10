"""Coverage for dtype-specific fused conditional-join aggregation kernels."""

import operator

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
        expected.index = pd.MultiIndex.from_arrays(
            [np.array([], dtype=np.intp), np.array([], dtype=bool)],
            names=[None, "matched"],
        )
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
        elif operation in {"min", "max"} and pd.api.types.is_integer_dtype(
            original_dtypes[(column_name, operation)]
        ):
            dtype = original_dtypes[(column_name, operation)]
            if expected[(column_name, operation)].isna().any():
                if not pd.api.types.is_extension_array_dtype(dtype):
                    dtype = (
                        "UInt64"
                        if pd.api.types.is_unsigned_integer_dtype(dtype)
                        else "Int64"
                    )
                expected[(column_name, operation)] = pd.array(
                    expected[(column_name, operation)], dtype=dtype
                )
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


def _expected_single(left, right, reverse, operator="<"):
    """Compute the single-range expectation using an explicit cross join."""
    pairs = left.assign(_left=np.arange(len(left))).merge(
        right.assign(_right=np.arange(len(right))), how="cross"
    )
    comparisons = {
        "<": pairs["key_x"] < pairs["key_y"],
        "<=": pairs["key_x"] <= pairs["key_y"],
        ">": pairs["key_x"] > pairs["key_y"],
        ">=": pairs["key_x"] >= pairs["key_y"],
    }
    pairs = pairs.loc[comparisons[operator]]
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


def test_join_agg_no_match_retains_plain_output_index_without_matched():
    """No-match aggregation retains eligible rows without matched metadata."""
    left = pd.DataFrame({"key": [1]})
    right = pd.DataFrame({"key": [1], "value": [10]})

    actual = left.join_agg(
        right,
        ("key", "key", "<"),
        aggfunc=[("value", "size")],
        return_matched=False,
    )

    expected = pd.DataFrame(
        {("value", "size"): [0]},
        index=pd.RangeIndex(1),
    )
    assert_frame_equal(expected, actual)
    assert isinstance(actual.index, pd.RangeIndex)


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


def test_reverse_not_equal_aggregation_uses_public_right_order():
    """Reverse ``!=`` aggregation is returned in physical right-row order."""
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
            ("left_value", "size"): pd.array([2, 1, 1], dtype="int64"),
            ("left_value", "sum"): pd.array([30, 20, 10], dtype="int64"),
        },
        index=pd.MultiIndex.from_arrays(
            [[0, 1, 2], [True, True, True]], names=[None, "matched"]
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


@pytest.mark.parametrize("operator", ["<", "<=", ">", ">="])
@pytest.mark.parametrize("reverse", [False, True])
def test_single_range_aggregation_supports_every_range_operator(operator, reverse):
    """Every single-range operator uses the correct optimized window."""
    left = pd.DataFrame(
        {
            "key": [1, 4],
            "left_value": [10, 40],
        }
    )
    right = pd.DataFrame(
        {
            "key": [1, 3, 5],
            "value": [10, 30, 50],
        }
    )
    column = "left_value" if reverse else "value"
    actual = left.join_agg(
        right,
        ("key", "key", operator),
        reverse=reverse,
        aggfunc=[
            (column, operation) for operation in ("size", "sum", "prod", "min", "max")
        ],
    )
    expected = _expected_single(left, right, reverse, operator)
    assert_frame_equal(expected, actual)


@pytest.mark.parametrize("reverse", [False, True])
def test_single_range_aggregation_aligns_unsorted_right_source(reverse):
    """Aggregation values follow the sorted predicate layout, not input order."""
    left = pd.DataFrame(
        {
            "key": [4],
            "left_value": [40],
        }
    )
    right = pd.DataFrame(
        {
            "key": [5, 1, 3],
            "value": [50, 10, 30],
        }
    )
    column = "left_value" if reverse else "value"
    actual = left.join_agg(
        right,
        ("key", "key", "<"),
        reverse=reverse,
        aggfunc=[
            (column, operation) for operation in ("size", "sum", "prod", "min", "max")
        ],
    )

    if reverse:
        expected = pd.DataFrame(
            {
                (column, "size"): [0, 0, 1],
                (column, "sum"): [0, 0, 40],
                (column, "prod"): [1, 1, 40],
                (column, "min"): pd.array([pd.NA, pd.NA, 40], dtype="Int64"),
                (column, "max"): pd.array([pd.NA, pd.NA, 40], dtype="Int64"),
            },
            index=pd.MultiIndex.from_arrays(
                [[1, 2, 0], [False, False, True]],
                names=[None, "matched"],
            ),
        )
    else:
        expected = pd.DataFrame(
            {
                (column, "size"): [1],
                (column, "sum"): [50],
                (column, "prod"): [50],
                (column, "min"): [50],
                (column, "max"): [50],
            },
            index=pd.MultiIndex.from_arrays([[0], [True]], names=[None, "matched"]),
        )
    assert_frame_equal(expected, actual)


@pytest.mark.parametrize("reverse", [False, True])
def test_single_range_aggregation_all_null_predicate_side_is_empty(reverse):
    """Null range anchors produce an empty aggregation schema."""
    left = pd.DataFrame(
        {
            "key": pd.Series([pd.NA], dtype="Int64"),
            "left_value": [10],
        }
    )
    right = pd.DataFrame(
        {
            "key": pd.Series([pd.NA], dtype="Int64"),
            "value": [20],
        }
    )
    column = "left_value" if reverse else "value"
    actual = left.join_agg(
        right,
        ("key", "key", "<"),
        reverse=reverse,
        aggfunc=[(column, "size"), (column, "sum")],
        return_matched=False,
    )
    assert actual.empty
    assert list(actual.columns) == [(column, "size"), (column, "sum")]
    assert isinstance(actual.index, pd.Index)


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
    if len(actual) == 0:
        expected.index = pd.MultiIndex.from_arrays(
            [np.array([], dtype="int64"), np.array([], dtype=bool)],
            names=[None, "matched"],
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


def test_dual_range_join_dispatches_each_anchor_dtype_independently():
    """Different numeric anchor dtypes still use the dual-range Rust path."""
    left = pd.DataFrame({"left_int": [2], "left_float": [6.0]})
    right = pd.DataFrame(
        {
            "right_int": [1, 3, 5, 7],
            "right_float": pd.Series([0.0, 2.0, 4.0, 6.0], dtype="float64"),
        }
    )

    actual = left.conditional_join(
        right,
        ("left_int", "right_int", "<"),
        ("left_float", "right_float", ">"),
        keep="all",
    )

    expected = pd.DataFrame(
        {
            "left_int": [2, 2],
            "left_float": [6.0, 6.0],
            "right_int": [3, 5],
            "right_float": [2.0, 4.0],
        },
        index=pd.RangeIndex(2),
    )
    assert_frame_equal(expected, actual)


def test_dual_range_aggregation_dispatches_each_anchor_dtype_independently():
    """Dual-range aggregation accepts independently typed range anchors."""
    left = pd.DataFrame({"left_int": [2], "left_float": [6.0]})
    right = pd.DataFrame(
        {
            "right_int": [1, 3, 5, 7],
            "right_float": pd.Series([0.0, 2.0, 4.0, 6.0], dtype="float64"),
            "value": [10, 20, 30, 40],
        }
    )

    actual = left.join_agg(
        right,
        ("left_int", "right_int", "<"),
        ("left_float", "right_float", ">"),
        aggfunc=[("value", "sum"), ("value", "size")],
        return_matched=True,
    )

    assert actual["value", "sum"].tolist() == [50]
    assert actual["value", "size"].tolist() == [2]
    assert actual.index.get_level_values("matched").tolist() == [True]


def test_duplicate_equi_aggregation_reverse_updates_each_right_slot():
    """Reverse equi aggregation visits every duplicate-right equi match."""
    left = pd.DataFrame({"key": ["a", "a"], "value": [2, 3]})
    right = pd.DataFrame({"key": ["a", "a", "b"], "payload": [10, 20, 30]})

    actual = left.join_agg(
        right,
        ("key", "key", "=="),
        aggfunc=[("value", "sum")],
        reverse=True,
        return_matched=True,
    )

    expected = pd.DataFrame(
        {("value", "sum"): [5, 5, 0]},
        index=pd.MultiIndex.from_tuples(
            [(0, True), (1, True), (2, False)],
            names=[None, "matched"],
        ),
    )
    assert_frame_equal(expected, actual)


def test_duplicate_equi_aggregation_applies_range_and_residual_filters():
    """Duplicate equi candidates are narrowed by range and residual filters."""
    left = pd.DataFrame({"key": ["a"], "bound": [2], "residual": [2]})
    right = pd.DataFrame(
        {
            "key": ["a", "a", "b"],
            "bound": [1, 3, 5],
            "residual": [0, 2, 4],
            "value": [10, 20, 30],
        }
    )

    actual = left.join_agg(
        right,
        ("key", "key", "=="),
        ("bound", "bound", "<"),
        ("residual", "residual", "=="),
        aggfunc=[("value", "sum"), ("value", "size")],
        return_matched=True,
    )

    expected = pd.DataFrame(
        {("value", "sum"): [20], ("value", "size"): [1]},
        index=pd.MultiIndex.from_tuples([(0, True)], names=[None, "matched"]),
    )
    assert_frame_equal(expected, actual)


def test_unique_equi_aggregation_covers_forward_and_reverse_output_domains():
    """Unique equi keys produce complete forward and reverse output domains."""
    left = pd.DataFrame(
        {"key": ["b", "a", "c"], "left_value": [2, 3, 4]},
        index=pd.Index([10, 11, 12], name="left_id"),
    )
    right = pd.DataFrame(
        {"key": ["a", "b", "d"], "value": [10, 20, 30]},
        index=pd.Index([20, 21, 22], name="right_id"),
    )
    aggfunc = [
        ("value", "sum"),
        ("value", "prod"),
        ("value", "min"),
        ("value", "max"),
        ("value", "count"),
        ("value", "size"),
    ]

    forward = left.join_agg(
        right,
        ("key", "key", "=="),
        aggfunc=aggfunc,
    )
    expected_forward = pd.DataFrame(
        {
            ("value", "sum"): [20, 10, 0],
            ("value", "prod"): [20, 10, 1],
            ("value", "min"): pd.array([20, 10, pd.NA], dtype="Int64"),
            ("value", "max"): pd.array([20, 10, pd.NA], dtype="Int64"),
            ("value", "count"): [1, 1, 0],
            ("value", "size"): [1, 1, 0],
        },
        index=pd.MultiIndex.from_tuples(
            [(0, True), (1, True), (2, False)],
            names=[None, "matched"],
        ),
    )
    assert_frame_equal(expected_forward, forward)

    reverse = left.join_agg(
        right,
        ("key", "key", "=="),
        aggfunc=[
            ("left_value", operation)
            for operation in [
                "sum",
                "prod",
                "min",
                "max",
                "count",
                "size",
            ]
        ],
        reverse=True,
    )
    expected_reverse = pd.DataFrame(
        {
            ("left_value", "sum"): [3, 2, 0],
            ("left_value", "prod"): [3, 2, 1],
            ("left_value", "min"): pd.array([3, 2, pd.NA], dtype="Int64"),
            ("left_value", "max"): pd.array([3, 2, pd.NA], dtype="Int64"),
            ("left_value", "count"): [1, 1, 0],
            ("left_value", "size"): [1, 1, 0],
        },
        index=pd.MultiIndex.from_tuples(
            [(0, True), (1, True), (2, False)],
            names=[None, "matched"],
        ),
    )
    assert_frame_equal(expected_reverse, reverse)


def test_duplicate_equi_aggregation_forward_covers_all_operations_without_matched():
    """Duplicate-right forward aggregation covers every operation and shape."""
    left = pd.DataFrame({"key": ["a", "b"]})
    right = pd.DataFrame(
        {"key": ["a", "b", "a"], "value": [2, 3, 4]},
        index=pd.Index([20, 21, 22], name="right_id"),
    )

    actual = left.join_agg(
        right,
        ("key", "key", "=="),
        aggfunc=[
            ("value", "sum"),
            ("value", "prod"),
            ("value", "min"),
            ("value", "max"),
            ("value", "count"),
            ("value", "size"),
        ],
        return_matched=False,
    )

    expected = pd.DataFrame(
        {
            ("value", "sum"): [6, 3],
            ("value", "prod"): [8, 3],
            ("value", "min"): [2, 3],
            ("value", "max"): [4, 3],
            ("value", "count"): [2, 1],
            ("value", "size"): [2, 1],
        },
        index=pd.RangeIndex(2),
    )
    assert_frame_equal(expected, actual)


def test_duplicate_equi_reverse_aggregation_intersects_two_sorted_ranges():
    """Reverse duplicate equi aggregation intersects two compatible ranges."""
    left = pd.DataFrame({"key": ["a"], "lower": [2], "upper": [3], "value": [9]})
    right = pd.DataFrame(
        {
            "key": ["a", "a", "a", "a"],
            "lower": [1, 3, 5, 7],
            "upper": [0, 2, 4, 6],
        },
        index=pd.Index([20, 21, 22, 23], name="right_id"),
    )

    actual = left.join_agg(
        right,
        ("key", "key", "=="),
        ("lower", "lower", "<"),
        ("upper", "upper", "<="),
        aggfunc=[("value", "sum"), ("value", "size")],
        reverse=True,
    )

    expected = pd.DataFrame(
        {
            ("value", "sum"): [0, 0, 9, 9],
            ("value", "size"): [0, 0, 1, 1],
        },
        index=pd.MultiIndex.from_tuples(
            [(0, False), (1, False), (2, True), (3, True)],
            names=[None, "matched"],
        ),
    )
    assert_frame_equal(expected, actual)


def test_duplicate_equi_aggregation_filters_incompatible_second_range_as_residual():
    """A differently ordered second range is evaluated as a residual."""
    left = pd.DataFrame({"key": ["a"], "first": [2], "second": [25]})
    right = pd.DataFrame(
        {
            "key": ["a", "a", "a", "a"],
            "first": [1, 3, 5, 7],
            "second": [40, 10, 30, 20],
            "value": [10, 20, 30, 40],
        }
    )

    actual = left.join_agg(
        right,
        ("key", "key", "=="),
        ("first", "first", "<"),
        ("second", "second", "<"),
        aggfunc=[("value", "sum"), ("value", "size")],
    )

    expected = pd.DataFrame(
        {("value", "sum"): [30], ("value", "size"): [1]},
        index=pd.MultiIndex.from_tuples([(0, True)], names=[None, "matched"]),
    )
    assert_frame_equal(expected, actual)


@pytest.mark.parametrize("second_op", ["<", "<=", ">", ">="])
@pytest.mark.parametrize("reverse", [False, True])
def test_duplicate_equi_aggregation_unordered_second_range_matches_reference(
    second_op, reverse
):
    """Unordered second ranges preserve aggregation semantics for every operator."""
    left = pd.DataFrame(
        {
            "key": ["a", "a"],
            "first": [3, 6],
            "second": [5, 1],
            "value": [100, 200],
        }
    )
    right = pd.DataFrame(
        {
            "key": ["a"] * 5,
            "first": [1, 2, 4, 5, 7],
            "second": [4, 1, 6, 0, 2],
            "value": [10, 20, 30, 40, 50],
        }
    )
    compare = {
        "<": operator.lt,
        "<=": operator.le,
        ">": operator.gt,
        ">=": operator.ge,
    }[second_op]

    expected_sums = []
    expected_sizes = []
    output = right if reverse else left
    for output_position in range(len(output)):
        matching_values = []
        for left_position, left_row in left.iterrows():
            for right_position, right_row in right.iterrows():
                if (
                    left_row["key"] == right_row["key"]
                    and left_row["first"] < right_row["first"]
                    and compare(left_row["second"], right_row["second"])
                    and (right_position if reverse else left_position)
                    == output_position
                ):
                    matching_values.append(
                        left_row["value"] if reverse else right_row["value"]
                    )
        expected_sums.append(sum(matching_values))
        expected_sizes.append(len(matching_values))

    actual = left.join_agg(
        right,
        ("key", "key", "=="),
        ("first", "first", "<"),
        ("second", "second", second_op),
        aggfunc=[
            ("value", "sum"),
            ("value", "size"),
        ],
        reverse=reverse,
    )
    matched = np.array(expected_sizes, dtype=bool)
    expected = pd.DataFrame(
        {
            ("value", "sum"): expected_sums,
            ("value", "size"): expected_sizes,
        },
        index=pd.MultiIndex.from_arrays(
            [range(len(output)), matched], names=[None, "matched"]
        ),
    )

    assert_frame_equal(expected, actual)


def test_equi_aggregation_preserves_not_equal_residual_semantics():
    """An equi candidate can be narrowed by the existing ``!=`` residual."""
    left = pd.DataFrame({"key": ["a"], "filter": [1]})
    right = pd.DataFrame({"key": ["a", "a"], "filter": [1, 2], "value": [10, 20]})

    actual = left.join_agg(
        right,
        ("key", "key", "=="),
        ("filter", "filter", "!="),
        aggfunc=[("value", "sum"), ("value", "size")],
    )

    expected = pd.DataFrame(
        {("value", "sum"): [20], ("value", "size"): [1]},
        index=pd.MultiIndex.from_tuples([(0, True)], names=[None, "matched"]),
    )
    assert_frame_equal(expected, actual)


def test_equi_aggregation_supports_multiple_equi_columns():
    """MultiIndex equi keys are mapped before aggregation."""
    left = pd.DataFrame({"key_a": ["a", "a"], "key_b": [1, 2]})
    right = pd.DataFrame(
        {
            "key_a": ["a", "a", "a"],
            "key_b": [1, 1, 3],
            "value": [10, 20, 30],
        }
    )

    actual = left.join_agg(
        right,
        ("key_a", "key_a", "=="),
        ("key_b", "key_b", "=="),
        aggfunc=[("value", "sum"), ("value", "size")],
    )

    expected = pd.DataFrame(
        {
            ("value", "sum"): [30, 0],
            ("value", "size"): [2, 0],
        },
        index=pd.MultiIndex.from_tuples(
            [(0, True), (1, False)], names=[None, "matched"]
        ),
    )
    assert_frame_equal(expected, actual)


@pytest.mark.parametrize("return_matched", [True, False])
def test_equi_aggregation_with_no_matches_retains_requested_output_shape(
    return_matched,
):
    """No equi match retains eligible rows in the requested index shape."""
    left = pd.DataFrame({"key": [1]})
    right = pd.DataFrame({"key": [2], "value": [10]})

    actual = left.join_agg(
        right,
        ("key", "key", "=="),
        aggfunc=[("value", "sum"), ("value", "size")],
        return_matched=return_matched,
    )

    expected = pd.DataFrame(
        {
            ("value", "sum"): [0],
            ("value", "size"): [0],
        },
        index=pd.RangeIndex(1),
    )
    if return_matched:
        expected = _with_matched_level(expected, 1, np.array([False]))
    else:
        assert isinstance(actual.index, pd.RangeIndex)
    assert_frame_equal(expected, actual)
