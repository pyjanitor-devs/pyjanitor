import numpy as np

from ._dtype_dispatch import _rs_func


def _compare_ne_no_ranges(
    left: np.ndarray,
    right: np.ndarray,
    positions: np.ndarray,
    left_booleans: np.ndarray,
    right_booleans: np.ndarray,
    is_extension_array: bool,
    op: int,
) -> tuple:
    """
    Compute for no ranges (no starts/ends) for != operator
    """
    func = _rs_func("compare_no_range_ne", left.dtype.name)
    return func(
        left,
        right,
        positions,
        left_booleans,
        right_booleans,
        is_extension_array,
        op,
    )


def _compare_no_ranges(
    left: np.ndarray,
    right: np.ndarray,
    positions: np.ndarray,
    op: int,
) -> tuple:
    """
    Compute for no ranges (no starts/ends)
    """
    func = _rs_func("compare_no_range", left.dtype.name)
    return func(
        left,
        right,
        positions,
        op,
    )


def _compare_ne_first_run_starts_only(
    left: np.ndarray,
    right: np.ndarray,
    starts: np.ndarray,
    left_booleans: np.ndarray,
    right_booleans: np.ndarray,
    is_extension_array: bool,
    op: int,
) -> tuple:
    """
    compute for first run
    """
    func = _rs_func("compare_start_ne_1st", left.dtype.name)
    return func(
        left,
        right,
        starts,
        left_booleans,
        right_booleans,
        is_extension_array,
        op,
    )


def _compare_ne_starts_only(
    left: np.ndarray,
    right: np.ndarray,
    starts: np.ndarray,
    left_booleans: np.ndarray,
    right_booleans: np.ndarray,
    is_extension_array: bool,
    counts_array: np.ndarray,
    matches: np.ndarray,
    op: int,
) -> tuple:
    """
    compute for starts
    """
    func = _rs_func("compare_start_ne", left.dtype.name)
    return func(
        left,
        right,
        starts,
        left_booleans,
        right_booleans,
        counts_array,
        matches,
        is_extension_array,
        op,
    )


def _compare_ne_first_run_ends_only(
    left: np.ndarray,
    right: np.ndarray,
    ends: np.ndarray,
    left_booleans: np.ndarray,
    right_booleans: np.ndarray,
    is_extension_array: bool,
    op: int,
) -> tuple:
    """
    compute for first run
    """
    func = _rs_func("compare_end_ne_1st", left.dtype.name)
    return func(
        left,
        right,
        ends,
        left_booleans,
        right_booleans,
        is_extension_array,
        op,
    )


def _compare_ne_ends_only(
    left: np.ndarray,
    right: np.ndarray,
    ends: np.ndarray,
    left_booleans: np.ndarray,
    right_booleans: np.ndarray,
    is_extension_array: bool,
    counts_array: np.ndarray,
    matches: np.ndarray,
    op: int,
) -> tuple:
    """
    compute for ends
    """
    func = _rs_func("compare_end_ne", left.dtype.name)
    return func(
        left,
        right,
        ends,
        left_booleans,
        right_booleans,
        counts_array,
        matches,
        is_extension_array,
        op,
    )


def _compare_ne_first_run_starts_ends(
    left: np.ndarray,
    right: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    left_booleans: np.ndarray,
    right_booleans: np.ndarray,
    is_extension_array: bool,
    op: int,
) -> tuple:
    """
    compute for first run
    """
    func = _rs_func("compare_start_end_ne_1st", left.dtype.name)
    return func(
        left,
        right,
        starts,
        ends,
        left_booleans,
        right_booleans,
        is_extension_array,
        op,
    )


def _compare_ne_starts_ends(
    left: np.ndarray,
    right: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    left_booleans: np.ndarray,
    right_booleans: np.ndarray,
    is_extension_array: bool,
    matches: np.ndarray,
    op: int,
) -> tuple:
    """
    compute for starts and ends
    """
    func = _rs_func("compare_start_end_ne", left.dtype.name)
    return func(
        left,
        right,
        starts,
        ends,
        left_booleans,
        right_booleans,
        matches,
        is_extension_array,
        op,
    )


def _compare_first_run_starts_only(
    left: np.ndarray,
    right: np.ndarray,
    starts: np.ndarray,
    op: int,
) -> tuple:
    """
    compute for first run
    """
    func = _rs_func("compare_first_start", left.dtype.name)
    return func(left, right, starts, op)


def _compare_starts_only(
    left: np.ndarray,
    right: np.ndarray,
    starts: np.ndarray,
    counts_array: np.ndarray,
    matches: np.ndarray,
    op: int,
) -> tuple:
    """
    compute for starts
    """
    func = _rs_func("compare_start", left.dtype.name)
    return func(left, right, starts, counts_array, matches, op)


def _compare_first_run_ends_only(
    left: np.ndarray, right: np.ndarray, ends: np.ndarray, op: int
) -> tuple:
    """
    compute for first run
    """
    func = _rs_func("compare_first_end", left.dtype.name)
    return func(left, right, ends, op)


def _compare_ends_only(
    left: np.ndarray,
    right: np.ndarray,
    ends: np.ndarray,
    counts_array: np.ndarray,
    matches: np.ndarray,
    op: int,
) -> tuple:
    """
    compute for ends
    """
    func = _rs_func("compare_end", left.dtype.name)
    return func(left, right, ends, counts_array, matches, op)


def _compare_first_run_starts_ends(
    left: np.ndarray,
    right: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    op: int,
) -> tuple:
    """
    compute for first run
    """
    func = _rs_func("compare_first_start_end", left.dtype.name)
    return func(left, right, starts, ends, op)


def _compare_starts_ends(
    left: np.ndarray,
    right: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    matches: np.ndarray,
    op: int,
) -> tuple:
    """
    compute for starts and ends
    """
    func = _rs_func("compare_start_end", left.dtype.name)
    return func(left, right, starts, ends, matches, op)


def _compare_positions(
    left: np.ndarray,
    right: np.ndarray,
    positions: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    op: int,
) -> tuple:
    """
    compute for first run
    """
    func = _rs_func("compare_posns", left.dtype.name)
    return func(
        left=left,
        right=right,
        positions=positions,
        starts=starts,
        ends=ends,
        op=op,
    )


def _compare_positions_ne(
    left: np.ndarray,
    right: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    positions: np.ndarray,
    left_booleans: np.ndarray | None,
    right_booleans: np.ndarray | None,
    is_extension_array: bool,
    op: int,
) -> tuple:
    """
    compute for first run
    """
    func = _rs_func("compare_posns_ne", left.dtype.name)
    return func(
        left=left,
        right=right,
        starts=starts,
        ends=ends,
        positions=positions,
        left_booleans=left_booleans,
        right_booleans=right_booleans,
        is_extension_array=is_extension_array,
        op=op,
    )
