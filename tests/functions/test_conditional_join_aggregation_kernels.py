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
    if source.dtype == np.dtype("float32"):
        for operation in ("sum", "prod"):
            expected[(output_column, operation)] = expected[
                (output_column, operation)
            ].astype("float64")
    expected.index.name = None
    return expected


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
@pytest.mark.parametrize("reverse", [False, True])
def test_single_join_aggregation_dispatches_all_numeric_dtypes(dtype, reverse):
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
    if source.dtype == np.dtype("float32"):
        for operation in ("sum", "prod"):
            expected[(output_column, operation)] = expected[
                (output_column, operation)
            ].astype("float64")
    expected.index.name = None
    return expected


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
            ("value", "sum"): pd.Series(pd.array([10], dtype=dtype)),
            ("value", "prod"): pd.Series(pd.array([10], dtype=dtype)),
            ("value", "min"): pd.Series(pd.array([10], dtype=dtype)),
            ("value", "max"): pd.Series(pd.array([10], dtype=dtype)),
        },
        index=pd.Index([0]),
    )
    expected.index.name = None
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
            ("value", "sum"): pd.Series(pd.array([10], dtype=dtype)),
            ("value", "prod"): pd.Series(pd.array([10], dtype=dtype)),
            ("value", "min"): pd.Series(pd.array([10], dtype=dtype)),
            ("value", "max"): pd.Series(pd.array([10], dtype=dtype)),
        },
        index=pd.Index([0]),
    )
    expected.index.name = None
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
    expected.index.name = None
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
            ("value", "sum"): pd.Series([], dtype=dtype),
        },
        index=pd.Index([], dtype="int64"),
    )
    expected.index.name = None
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
    expected.index.name = None
    assert_frame_equal(expected, actual)


def test_single_reverse_aggregation_omits_unmatched_right_rows():
    """Reverse output contains only right rows with at least one match."""
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
            ("value", "size"): pd.Series([1], dtype="int64"),
            ("value", "sum"): pd.Series([10], dtype="int64"),
        },
        index=pd.Index([0]),
    )
    expected.index.name = None
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
    expected.index.name = None
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
    expected.index.name = None
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
    expected.index.name = None
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
    expected.index.name = None
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
            ("value", "sum"): pd.Series([], dtype=dtype),
        },
        index=pd.Index([], dtype="int64"),
    )
    expected.index.name = None
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
            ("value", "sum"): pd.Series(pd.array([10, 30], dtype=dtype), index=[0, 1]),
        },
        index=pd.Index([0, 1]),
    )
    expected.index.name = None
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
    expected.index.name = None
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
    expected.index.name = None
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
    expected.index.name = None
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
    expected.index.name = None
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
    expected.index.name = None
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
    expected.index.name = None
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
            ("value", "sum"): pd.Series(pd.array([10, 10], dtype=dtype), index=[0, 1]),
        },
        index=pd.Index([0, 1]),
    )
    expected.index.name = None
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
            ("value", "sum"): pd.Series([30, 20], index=[0, 1], dtype=dtype),
        },
        index=pd.Index([0, 1]),
    )
    expected.index.name = None
    assert_frame_equal(expected, actual)


@pytest.mark.parametrize(
    ("dtype", "expected_sum", "expected_prod"),
    [
        ("int8", -127, -2),
        ("int16", -32767, -2),
        ("int32", -2147483647, -2),
        ("uint8", 1, 254),
        ("uint16", 1, 65534),
        ("uint32", 1, 4294967294),
    ],
)
def test_single_range_aggregation_wraps_at_source_integer_width(
    dtype, expected_sum, expected_prod
):
    """Integer sum and product wrap at the source dtype width."""
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
            ("value", "sum"): pd.Series([expected_sum], dtype="int64"),
            ("value", "prod"): pd.Series([expected_prod], dtype="int64"),
        },
        index=pd.Index([0]),
    )
    expected.index.name = None
    assert_frame_equal(expected, actual)


def test_single_range_float32_aggregation_returns_float64():
    """Float32 inputs use the documented float64 aggregation contract."""
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
    expected_value = float(right["value"].iloc[0]) + float(right["value"].iloc[1])
    expected = pd.DataFrame(
        {("value", "sum"): pd.Series([expected_value], dtype="float64")},
        index=pd.Index([0]),
    )
    expected.index.name = None
    assert_frame_equal(expected, actual)
