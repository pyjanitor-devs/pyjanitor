import numpy as np

from ._dtype_dispatch import _rs_func


def _binary_search_lt(
    left: np.ndarray,
    right: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> tuple:
    """
    Get starts for < joins
    """
    func = _rs_func("binary_search_lt", left.dtype.name)
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
    func = _rs_func("binary_search_le", left.dtype.name)
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
    func = _rs_func("binary_search_gt", left.dtype.name)
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
    func = _rs_func("binary_search_ge", left.dtype.name)
    return func(left, right, starts, ends)


def _binary_search_lt_first(
    left: np.ndarray,
    right: np.ndarray,
    left_index: np.ndarray,
) -> tuple:
    """
    Get starts for < joins
    """
    func = _rs_func("binary_search_lt_first", left.dtype.name)
    search_indices, left_index, total = func(left, right, left_index)
    if not total:
        return None
    return left_index, search_indices


def _binary_search_le_first(
    left: np.ndarray, right: np.ndarray, left_index: np.ndarray
) -> tuple:
    """
    Get starts for <= joins
    """
    func = _rs_func("binary_search_le_first", left.dtype.name)
    search_indices, left_index, total = func(left, right, left_index)
    if not total:
        return None
    return left_index, search_indices


def _binary_search_gt_first(
    left: np.ndarray, right: np.ndarray, left_index: np.ndarray
) -> tuple:
    """
    Get ends for > joins
    """
    func = _rs_func("binary_search_gt_first", left.dtype.name)
    search_indices, left_index, total = func(left, right, left_index)
    if not total:
        return None
    return left_index, search_indices


def _binary_search_ge_first(
    left: np.ndarray, right: np.ndarray, left_index: np.ndarray
) -> tuple:
    """
    Get ends for >= joins
    """
    func = _rs_func("binary_search_ge_first", left.dtype.name)
    search_indices, left_index, total = func(left, right, left_index)
    if not total:
        return None
    return left_index, search_indices


def _binary_search_gt_regions(
    left: np.ndarray, right: np.ndarray, left_index: np.ndarray
) -> tuple:
    """
    Get ends for > joins
    """
    func = _rs_func("binary_search_gt_regions", left.dtype.name)
    search_indices, left_index, total = func(left, right, left_index)
    if not total:
        return None
    return left_index, search_indices


def _binary_search_ge_regions(
    left: np.ndarray, right: np.ndarray, left_index: np.ndarray
) -> tuple:
    """
    Get ends for >= joins
    """
    func = _rs_func("binary_search_ge_regions", left.dtype.name)
    search_indices, left_index, total = func(left, right, left_index)
    if not total:
        return None
    return left_index, search_indices
