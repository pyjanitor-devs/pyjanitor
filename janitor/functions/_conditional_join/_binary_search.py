import janitor_rs
import numpy as np


def _binary_search_lt(
    left: np.ndarray,
    right: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> tuple:
    """
    Get starts for < joins
    """
    mapping = {
        "int64": janitor_rs.binary_search_lt_int64,
        "int32": janitor_rs.binary_search_lt_int32,
        "int16": janitor_rs.binary_search_lt_int16,
        "int8": janitor_rs.binary_search_lt_int8,
        "uint64": janitor_rs.binary_search_lt_uint64,
        "uint32": janitor_rs.binary_search_lt_uint32,
        "uint16": janitor_rs.binary_search_lt_uint16,
        "uint8": janitor_rs.binary_search_lt_uint8,
        "float64": janitor_rs.binary_search_lt_f64,
        "float32": janitor_rs.binary_search_lt_f32,
    }
    dtype_name = left.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(left, right, starts, ends)


def _binary_search_le(
    left: np.ndarray,
    right: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> tuple:
    """
    Get starts for <= joins
    """
    mapping = {
        "int64": janitor_rs.binary_search_le_int64,
        "int32": janitor_rs.binary_search_le_int32,
        "int16": janitor_rs.binary_search_le_int16,
        "int8": janitor_rs.binary_search_le_int8,
        "uint64": janitor_rs.binary_search_le_uint64,
        "uint32": janitor_rs.binary_search_le_uint32,
        "uint16": janitor_rs.binary_search_le_uint16,
        "uint8": janitor_rs.binary_search_le_uint8,
        "float64": janitor_rs.binary_search_le_f64,
        "float32": janitor_rs.binary_search_le_f32,
    }
    dtype_name = left.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(left, right, starts, ends)


def _binary_search_gt(
    left: np.ndarray,
    right: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> tuple:
    """
    Get ends for > joins
    """
    mapping = {
        "int64": janitor_rs.binary_search_gt_int64,
        "int32": janitor_rs.binary_search_gt_int32,
        "int16": janitor_rs.binary_search_gt_int16,
        "int8": janitor_rs.binary_search_gt_int8,
        "uint64": janitor_rs.binary_search_gt_uint64,
        "uint32": janitor_rs.binary_search_gt_uint32,
        "uint16": janitor_rs.binary_search_gt_uint16,
        "uint8": janitor_rs.binary_search_gt_uint8,
        "float64": janitor_rs.binary_search_gt_f64,
        "float32": janitor_rs.binary_search_gt_f32,
    }
    dtype_name = left.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(left, right, starts, ends)


def _binary_search_ge(
    left: np.ndarray,
    right: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> tuple:
    """
    Get ends for >= joins
    """
    mapping = {
        "int64": janitor_rs.binary_search_ge_int64,
        "int32": janitor_rs.binary_search_ge_int32,
        "int16": janitor_rs.binary_search_ge_int16,
        "int8": janitor_rs.binary_search_ge_int8,
        "uint64": janitor_rs.binary_search_ge_uint64,
        "uint32": janitor_rs.binary_search_ge_uint32,
        "uint16": janitor_rs.binary_search_ge_uint16,
        "uint8": janitor_rs.binary_search_ge_uint8,
        "float64": janitor_rs.binary_search_ge_f64,
        "float32": janitor_rs.binary_search_ge_f32,
    }
    dtype_name = left.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(left, right, starts, ends)
