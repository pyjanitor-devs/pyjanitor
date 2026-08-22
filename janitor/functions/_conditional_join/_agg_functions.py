import janitor_rs
import numpy as np

# float32/float64 sum ranges are computed with a Neumaier-compensated
# prefix sum instead of a Rust round-trip - see _compensated_prefix_sum.
_FLOAT_PREFIX_DTYPES = frozenset(("float64", "float32"))


def _compensated_prefix_sum(
    arr: np.ndarray,
    booleans: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a Neumaier-compensated running-sum prefix, upcast to float64.

    ELI5: a plain running total loses precision when a tiny number gets
    added to a much bigger one - the tiny part just gets rounded away.
    This keeps a second "leftover" running total alongside the main one,
    tracking what got rounded away at each step, so that a later
    `prefix[end] - prefix[start]` subtraction stays close to summing that
    slice directly instead of drifting the way a naive running total would.
    `prefix[0]` is `(0.0, 0.0)`; `prefix[i]` holds the compensated sum of
    the first `i` elements (with null positions treated as `0.0`).
    """
    values = arr.astype(np.float64, copy=False)
    n = values.size
    hi = np.empty(n + 1, dtype=np.float64)
    lo = np.empty(n + 1, dtype=np.float64)
    hi[0] = 0.0
    lo[0] = 0.0
    total = 0.0
    compensation = 0.0
    # A running total near +/-inf can legitimately overflow, and the
    # caller (_build_prefix_if_safe) already detects and handles that -
    # silence the transient overflow/NaN warnings that would otherwise
    # surface for a case this code correctly falls back on.
    with np.errstate(over="ignore", invalid="ignore"):
        for pos in range(n):
            value = 0.0 if booleans[pos] else values[pos]
            new_total = total + value
            if abs(total) >= abs(value):
                compensation += (total - new_total) + value
            else:
                compensation += (value - new_total) + total
            total = new_total
            hi[pos + 1] = total
            lo[pos + 1] = compensation
    return hi, lo


def _range_sum_from_prefix(
    hi: np.ndarray,
    lo: np.ndarray,
    starts,
    ends,
) -> np.ndarray:
    """
    Answer [start, end) range-sum queries from a compensated prefix.
    """
    return (hi[ends] - hi[starts]) + (lo[ends] - lo[starts])


# Neumaier compensation is itself just a float64, so once it has absorbed
# a moderate-magnitude correction (say ~1e12) it can no longer represent a
# *later* tiny correction (say ~1e-6) - that increment underflows below
# the compensation term's own ULP. Empirically (see PR discussion), a big
# excursion followed by another huge, opposite-signed excursion followed
# by tiny values can silently lose 100% of a small window's true value,
# well before the running total itself ever overflows. Random dynamic
# range alone stays safe under the scale-aware accuracy contract exercised
# against `math.fsum` in the tests. A result-relative error is deliberately
# not used: cancellation can make the true result arbitrarily close to zero
# even when both summation methods are accurate. The 1e15 cutoff remains
# several orders of magnitude below the observed catastrophic regime.
_MAX_SAFE_DYNAMIC_RANGE = 1e15


def _has_safe_dynamic_range(arr: np.ndarray, booleans: np.ndarray) -> bool:
    """
    Whether the non-null magnitudes in `arr` are too spread out to trust.
    """
    non_null = arr if not booleans.any() else arr[~booleans]
    magnitudes = np.abs(non_null.astype(np.float64, copy=False))
    magnitudes = magnitudes[magnitudes > 0]
    if magnitudes.size <= 1:
        return True
    with np.errstate(over="ignore"):
        threshold = magnitudes.min() * _MAX_SAFE_DYNAMIC_RANGE
    return bool(magnitudes.max() <= threshold)


def _build_prefix_if_safe(
    arr: np.ndarray,
    booleans: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Build the compensated prefix, or signal it isn't safe to use.

    Two distinct ways the fast path can silently go wrong, both handled
    here by falling back to Rust (which recomputes each range from
    scratch and so isn't exposed to either failure mode):

    1. A real +/-inf already in `arr` (which then poisons every later
       `hi` entry, since finite + inf stays infinite forever), or two
       ordinary finite values whose partial sum genuinely overflows
       float64 range. Either way, once `hi` holds a +/-inf, subtracting
       two such entries for a range that never touched the offending
       value(s) produces `inf - inf = NaN`. Checked by requiring every
       `hi` entry beyond the leading zero to be finite.
    2. Extreme dynamic range within `arr` - see `_MAX_SAFE_DYNAMIC_RANGE`.
    """
    if not _has_safe_dynamic_range(arr=arr, booleans=booleans):
        return None
    hi, lo = _compensated_prefix_sum(arr=arr, booleans=booleans)
    if not np.isfinite(hi).all():
        return None
    return hi, lo


# Building the prefix costs ~250ns/element in a pure Python loop, vs.
# ~2ns/element for Rust's native per-range scan (measured directly) -
# roughly 120x slower per element. The one-time O(n) build only pays off
# once the *total* width summed across every query range exceeds that
# same ratio times the array size; below that, Rust doing direct O(width)
# work per range is cheaper overall, even though it re-scans overlapping
# regions. A margin above the measured ~120x ratio keeps this conservative
# (i.e. biased toward Rust when it's a close call).
_MIN_TOTAL_WIDTH_RATIO = 150


def _prefix_sum_is_worthwhile(array_size: int, total_width: int) -> bool:
    """
    Whether the one-time O(n) prefix build is expected to pay off overall.
    """
    return total_width > _MIN_TOTAL_WIDTH_RATIO * array_size


def _sum_starts(
    arr: np.ndarray,
    starts: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute sum
    """
    dtype_name = arr.dtype.name
    if dtype_name in _FLOAT_PREFIX_DTYPES:
        total_width = starts.size * arr.size - int(starts.sum())
        if _prefix_sum_is_worthwhile(arr.size, total_width):
            prefix = _build_prefix_if_safe(arr=arr, booleans=booleans)
            if prefix is not None:
                hi, lo = prefix
                return _range_sum_from_prefix(
                    hi=hi, lo=lo, starts=starts, ends=arr.size
                )
    mapping = {
        "int64": janitor_rs.compute_sum_start_int64,
        "int32": janitor_rs.compute_sum_start_int32,
        "int16": janitor_rs.compute_sum_start_int16,
        "int8": janitor_rs.compute_sum_start_int8,
        "uint64": janitor_rs.compute_sum_start_uint64,
        "uint32": janitor_rs.compute_sum_start_uint32,
        "uint16": janitor_rs.compute_sum_start_uint16,
        "uint8": janitor_rs.compute_sum_start_uint8,
        "float64": janitor_rs.compute_sum_start_f64,
        "float32": janitor_rs.compute_sum_start_f32,
    }
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(arr=arr, starts=starts, booleans=booleans)


def _sum_ends(
    arr: np.ndarray,
    ends: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute sum
    """
    dtype_name = arr.dtype.name
    if dtype_name in _FLOAT_PREFIX_DTYPES:
        total_width = int(ends.sum())
        if _prefix_sum_is_worthwhile(arr.size, total_width):
            prefix = _build_prefix_if_safe(arr=arr, booleans=booleans)
            if prefix is not None:
                hi, lo = prefix
                return _range_sum_from_prefix(hi=hi, lo=lo, starts=0, ends=ends)
    mapping = {
        "int64": janitor_rs.compute_sum_end_int64,
        "int32": janitor_rs.compute_sum_end_int32,
        "int16": janitor_rs.compute_sum_end_int16,
        "int8": janitor_rs.compute_sum_end_int8,
        "uint64": janitor_rs.compute_sum_end_uint64,
        "uint32": janitor_rs.compute_sum_end_uint32,
        "uint16": janitor_rs.compute_sum_end_uint16,
        "uint8": janitor_rs.compute_sum_end_uint8,
        "float64": janitor_rs.compute_sum_end_f64,
        "float32": janitor_rs.compute_sum_end_f32,
    }
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(arr=arr, ends=ends, booleans=booleans)


def _size_rev_starts(
    starts: np.ndarray,
    index: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute size_rev
    """
    return janitor_rs.compute_size_rev_start(starts=starts, index=index, length=length)


def _size_rev_ends(
    ends: np.ndarray,
    index: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute size_rev
    """
    return janitor_rs.compute_size_rev_end(ends=ends, index=index, length=length)


def _size_rev_starts_ends(
    starts: np.ndarray,
    ends: np.ndarray,
    index: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute size_rev
    """
    return janitor_rs.compute_size_rev_start_end(
        starts=starts, ends=ends, index=index, length=length
    )


def _size_rev_ends_matches(
    ends: np.ndarray,
    index: np.ndarray,
    matches: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute size_rev
    """
    return janitor_rs.compute_size_rev_end_matches(
        ends=ends, index=index, matches=matches, length=length
    )


def _size_rev_starts_matches(
    starts: np.ndarray,
    index: np.ndarray,
    matches: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute size_rev
    """
    return janitor_rs.compute_size_rev_start_matches(
        starts=starts, index=index, matches=matches, length=length
    )


def _size_rev_starts_ends_matches(
    starts: np.ndarray,
    ends: np.ndarray,
    index: np.ndarray,
    matches: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute size_rev
    """
    return janitor_rs.compute_size_rev_start_end_matches(
        starts=starts, ends=ends, index=index, matches=matches, length=length
    )


def _size_rev_positions(
    starts: np.ndarray,
    ends: np.ndarray,
    index: np.ndarray,
    positions: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute size_rev
    """
    return janitor_rs.compute_size_rev_positions(
        starts=starts,
        ends=ends,
        index=index,
        positions=positions,
        length=length,
    )


def _min_starts(
    arr: np.ndarray,
    starts: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute min
    """
    mapping = {
        "int64": janitor_rs.compute_min_start_int64,
        "int32": janitor_rs.compute_min_start_int32,
        "int16": janitor_rs.compute_min_start_int16,
        "int8": janitor_rs.compute_min_start_int8,
        "uint64": janitor_rs.compute_min_start_uint64,
        "uint32": janitor_rs.compute_min_start_uint32,
        "uint16": janitor_rs.compute_min_start_uint16,
        "uint8": janitor_rs.compute_min_start_uint8,
        "float64": janitor_rs.compute_min_start_f64,
        "float32": janitor_rs.compute_min_start_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(arr=arr, starts=starts, booleans=booleans)


def _min_ends(
    arr: np.ndarray,
    ends: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute min
    """
    mapping = {
        "int64": janitor_rs.compute_min_end_int64,
        "int32": janitor_rs.compute_min_end_int32,
        "int16": janitor_rs.compute_min_end_int16,
        "int8": janitor_rs.compute_min_end_int8,
        "uint64": janitor_rs.compute_min_end_uint64,
        "uint32": janitor_rs.compute_min_end_uint32,
        "uint16": janitor_rs.compute_min_end_uint16,
        "uint8": janitor_rs.compute_min_end_uint8,
        "float64": janitor_rs.compute_min_end_f64,
        "float32": janitor_rs.compute_min_end_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(arr=arr, ends=ends, booleans=booleans)


def _max_starts(
    arr: np.ndarray,
    starts: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute max
    """
    mapping = {
        "int64": janitor_rs.compute_max_start_int64,
        "int32": janitor_rs.compute_max_start_int32,
        "int16": janitor_rs.compute_max_start_int16,
        "int8": janitor_rs.compute_max_start_int8,
        "uint64": janitor_rs.compute_max_start_uint64,
        "uint32": janitor_rs.compute_max_start_uint32,
        "uint16": janitor_rs.compute_max_start_uint16,
        "uint8": janitor_rs.compute_max_start_uint8,
        "float64": janitor_rs.compute_max_start_f64,
        "float32": janitor_rs.compute_max_start_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(arr=arr, starts=starts, booleans=booleans)


def _max_ends(
    arr: np.ndarray,
    ends: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute max
    """
    mapping = {
        "int64": janitor_rs.compute_max_end_int64,
        "int32": janitor_rs.compute_max_end_int32,
        "int16": janitor_rs.compute_max_end_int16,
        "int8": janitor_rs.compute_max_end_int8,
        "uint64": janitor_rs.compute_max_end_uint64,
        "uint32": janitor_rs.compute_max_end_uint32,
        "uint16": janitor_rs.compute_max_end_uint16,
        "uint8": janitor_rs.compute_max_end_uint8,
        "float64": janitor_rs.compute_max_end_f64,
        "float32": janitor_rs.compute_max_end_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(arr=arr, ends=ends, booleans=booleans)


def _prod_starts(
    arr: np.ndarray,
    starts: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute prod
    """
    mapping = {
        "int64": janitor_rs.compute_prod_start_int64,
        "int32": janitor_rs.compute_prod_start_int32,
        "int16": janitor_rs.compute_prod_start_int16,
        "int8": janitor_rs.compute_prod_start_int8,
        "uint64": janitor_rs.compute_prod_start_uint64,
        "uint32": janitor_rs.compute_prod_start_uint32,
        "uint16": janitor_rs.compute_prod_start_uint16,
        "uint8": janitor_rs.compute_prod_start_uint8,
        "float64": janitor_rs.compute_prod_start_f64,
        "float32": janitor_rs.compute_prod_start_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(arr=arr, starts=starts, booleans=booleans)


def _prod_ends(
    arr: np.ndarray,
    ends: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute prod
    """
    mapping = {
        "int64": janitor_rs.compute_prod_end_int64,
        "int32": janitor_rs.compute_prod_end_int32,
        "int16": janitor_rs.compute_prod_end_int16,
        "int8": janitor_rs.compute_prod_end_int8,
        "uint64": janitor_rs.compute_prod_end_uint64,
        "uint32": janitor_rs.compute_prod_end_uint32,
        "uint16": janitor_rs.compute_prod_end_uint16,
        "uint8": janitor_rs.compute_prod_end_uint8,
        "float64": janitor_rs.compute_prod_end_f64,
        "float32": janitor_rs.compute_prod_end_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(arr=arr, ends=ends, booleans=booleans)


def _sum_starts_matches(
    arr: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
    matches: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute sum
    """
    mapping = {
        "int64": janitor_rs.compute_sum_start_match_int64,
        "int32": janitor_rs.compute_sum_start_match_int32,
        "int16": janitor_rs.compute_sum_start_match_int16,
        "int8": janitor_rs.compute_sum_start_match_int8,
        "uint64": janitor_rs.compute_sum_start_match_uint64,
        "uint32": janitor_rs.compute_sum_start_match_uint32,
        "uint16": janitor_rs.compute_sum_start_match_uint16,
        "uint8": janitor_rs.compute_sum_start_match_uint8,
        "float64": janitor_rs.compute_sum_start_match_f64,
        "float32": janitor_rs.compute_sum_start_match_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(
        arr=arr,
        starts=starts,
        counts=counts,
        matches=matches,
        booleans=booleans,
    )


def _sum_ends_matches(
    arr: np.ndarray,
    ends: np.ndarray,
    counts: np.ndarray,
    matches: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute sum
    """
    mapping = {
        "int64": janitor_rs.compute_sum_end_match_int64,
        "int32": janitor_rs.compute_sum_end_match_int32,
        "int16": janitor_rs.compute_sum_end_match_int16,
        "int8": janitor_rs.compute_sum_end_match_int8,
        "uint64": janitor_rs.compute_sum_end_match_uint64,
        "uint32": janitor_rs.compute_sum_end_match_uint32,
        "uint16": janitor_rs.compute_sum_end_match_uint16,
        "uint8": janitor_rs.compute_sum_end_match_uint8,
        "float64": janitor_rs.compute_sum_end_match_f64,
        "float32": janitor_rs.compute_sum_end_match_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(arr=arr, ends=ends, counts=counts, matches=matches, booleans=booleans)


def _max_starts_matches(
    arr: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
    matches: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute max
    """
    mapping = {
        "int64": janitor_rs.compute_max_start_match_int64,
        "int32": janitor_rs.compute_max_start_match_int32,
        "int16": janitor_rs.compute_max_start_match_int16,
        "int8": janitor_rs.compute_max_start_match_int8,
        "uint64": janitor_rs.compute_max_start_match_uint64,
        "uint32": janitor_rs.compute_max_start_match_uint32,
        "uint16": janitor_rs.compute_max_start_match_uint16,
        "uint8": janitor_rs.compute_max_start_match_uint8,
        "float64": janitor_rs.compute_max_start_match_f64,
        "float32": janitor_rs.compute_max_start_match_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(
        arr=arr,
        starts=starts,
        counts=counts,
        matches=matches,
        booleans=booleans,
    )


def _max_ends_matches(
    arr: np.ndarray,
    ends: np.ndarray,
    counts: np.ndarray,
    matches: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute max
    """
    mapping = {
        "int64": janitor_rs.compute_max_end_match_int64,
        "int32": janitor_rs.compute_max_end_match_int32,
        "int16": janitor_rs.compute_max_end_match_int16,
        "int8": janitor_rs.compute_max_end_match_int8,
        "uint64": janitor_rs.compute_max_end_match_uint64,
        "uint32": janitor_rs.compute_max_end_match_uint32,
        "uint16": janitor_rs.compute_max_end_match_uint16,
        "uint8": janitor_rs.compute_max_end_match_uint8,
        "float64": janitor_rs.compute_max_end_match_f64,
        "float32": janitor_rs.compute_max_end_match_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(arr=arr, ends=ends, counts=counts, matches=matches, booleans=booleans)


def _min_starts_matches(
    arr: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
    matches: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute min
    """
    mapping = {
        "int64": janitor_rs.compute_min_start_match_int64,
        "int32": janitor_rs.compute_min_start_match_int32,
        "int16": janitor_rs.compute_min_start_match_int16,
        "int8": janitor_rs.compute_min_start_match_int8,
        "uint64": janitor_rs.compute_min_start_match_uint64,
        "uint32": janitor_rs.compute_min_start_match_uint32,
        "uint16": janitor_rs.compute_min_start_match_uint16,
        "uint8": janitor_rs.compute_min_start_match_uint8,
        "float64": janitor_rs.compute_min_start_match_f64,
        "float32": janitor_rs.compute_min_start_match_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(
        arr=arr,
        starts=starts,
        counts=counts,
        matches=matches,
        booleans=booleans,
    )


def _min_ends_matches(
    arr: np.ndarray,
    ends: np.ndarray,
    counts: np.ndarray,
    matches: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute min
    """
    mapping = {
        "int64": janitor_rs.compute_min_end_match_int64,
        "int32": janitor_rs.compute_min_end_match_int32,
        "int16": janitor_rs.compute_min_end_match_int16,
        "int8": janitor_rs.compute_min_end_match_int8,
        "uint64": janitor_rs.compute_min_end_match_uint64,
        "uint32": janitor_rs.compute_min_end_match_uint32,
        "uint16": janitor_rs.compute_min_end_match_uint16,
        "uint8": janitor_rs.compute_min_end_match_uint8,
        "float64": janitor_rs.compute_min_end_match_f64,
        "float32": janitor_rs.compute_min_end_match_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(arr=arr, ends=ends, counts=counts, matches=matches, booleans=booleans)


def _sum_positions(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    positions: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute sum
    """
    mapping = {
        "int64": janitor_rs.compute_sum_positions_int64,
        "int32": janitor_rs.compute_sum_positions_int32,
        "int16": janitor_rs.compute_sum_positions_int16,
        "int8": janitor_rs.compute_sum_positions_int8,
        "uint64": janitor_rs.compute_sum_positions_uint64,
        "uint32": janitor_rs.compute_sum_positions_uint32,
        "uint16": janitor_rs.compute_sum_positions_uint16,
        "uint8": janitor_rs.compute_sum_positions_uint8,
        "float64": janitor_rs.compute_sum_positions_f64,
        "float32": janitor_rs.compute_sum_positions_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(
        arr=arr,
        starts=starts,
        ends=ends,
        positions=positions,
        booleans=booleans,
    )


def _prod_starts_matches(
    arr: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
    matches: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute prod
    """
    mapping = {
        "int64": janitor_rs.compute_prod_start_match_int64,
        "int32": janitor_rs.compute_prod_start_match_int32,
        "int16": janitor_rs.compute_prod_start_match_int16,
        "int8": janitor_rs.compute_prod_start_match_int8,
        "uint64": janitor_rs.compute_prod_start_match_uint64,
        "uint32": janitor_rs.compute_prod_start_match_uint32,
        "uint16": janitor_rs.compute_prod_start_match_uint16,
        "uint8": janitor_rs.compute_prod_start_match_uint8,
        "float64": janitor_rs.compute_prod_start_match_f64,
        "float32": janitor_rs.compute_prod_start_match_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(
        arr=arr,
        starts=starts,
        counts=counts,
        matches=matches,
        booleans=booleans,
    )


def _prod_ends_matches(
    arr: np.ndarray,
    ends: np.ndarray,
    counts: np.ndarray,
    matches: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute prod
    """
    mapping = {
        "int64": janitor_rs.compute_prod_end_match_int64,
        "int32": janitor_rs.compute_prod_end_match_int32,
        "int16": janitor_rs.compute_prod_end_match_int16,
        "int8": janitor_rs.compute_prod_end_match_int8,
        "uint64": janitor_rs.compute_prod_end_match_uint64,
        "uint32": janitor_rs.compute_prod_end_match_uint32,
        "uint16": janitor_rs.compute_prod_end_match_uint16,
        "uint8": janitor_rs.compute_prod_end_match_uint8,
        "float64": janitor_rs.compute_prod_end_match_f64,
        "float32": janitor_rs.compute_prod_end_match_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(arr=arr, ends=ends, counts=counts, matches=matches, booleans=booleans)


def _prod_positions(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    positions: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute prod
    """
    mapping = {
        "int64": janitor_rs.compute_prod_positions_int64,
        "int32": janitor_rs.compute_prod_positions_int32,
        "int16": janitor_rs.compute_prod_positions_int16,
        "int8": janitor_rs.compute_prod_positions_int8,
        "uint64": janitor_rs.compute_prod_positions_uint64,
        "uint32": janitor_rs.compute_prod_positions_uint32,
        "uint16": janitor_rs.compute_prod_positions_uint16,
        "uint8": janitor_rs.compute_prod_positions_uint8,
        "float64": janitor_rs.compute_prod_positions_f64,
        "float32": janitor_rs.compute_prod_positions_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(
        arr=arr,
        starts=starts,
        ends=ends,
        positions=positions,
        booleans=booleans,
    )


def _min_positions(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    positions: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute min
    """
    mapping = {
        "int64": janitor_rs.compute_min_positions_int64,
        "int32": janitor_rs.compute_min_positions_int32,
        "int16": janitor_rs.compute_min_positions_int16,
        "int8": janitor_rs.compute_min_positions_int8,
        "uint64": janitor_rs.compute_min_positions_uint64,
        "uint32": janitor_rs.compute_min_positions_uint32,
        "uint16": janitor_rs.compute_min_positions_uint16,
        "uint8": janitor_rs.compute_min_positions_uint8,
        "float64": janitor_rs.compute_min_positions_f64,
        "float32": janitor_rs.compute_min_positions_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(
        arr=arr,
        starts=starts,
        ends=ends,
        positions=positions,
        booleans=booleans,
    )


def _max_positions(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    positions: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute max
    """
    mapping = {
        "int64": janitor_rs.compute_max_positions_int64,
        "int32": janitor_rs.compute_max_positions_int32,
        "int16": janitor_rs.compute_max_positions_int16,
        "int8": janitor_rs.compute_max_positions_int8,
        "uint64": janitor_rs.compute_max_positions_uint64,
        "uint32": janitor_rs.compute_max_positions_uint32,
        "uint16": janitor_rs.compute_max_positions_uint16,
        "uint8": janitor_rs.compute_max_positions_uint8,
        "float64": janitor_rs.compute_max_positions_f64,
        "float32": janitor_rs.compute_max_positions_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(
        arr=arr,
        starts=starts,
        ends=ends,
        positions=positions,
        booleans=booleans,
    )


def _max_starts_ends(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute max
    """
    mapping = {
        "int64": janitor_rs.compute_max_start_end_int64,
        "int32": janitor_rs.compute_max_start_end_int32,
        "int16": janitor_rs.compute_max_start_end_int16,
        "int8": janitor_rs.compute_max_start_end_int8,
        "uint64": janitor_rs.compute_max_start_end_uint64,
        "uint32": janitor_rs.compute_max_start_end_uint32,
        "uint16": janitor_rs.compute_max_start_end_uint16,
        "uint8": janitor_rs.compute_max_start_end_uint8,
        "float64": janitor_rs.compute_max_start_end_f64,
        "float32": janitor_rs.compute_max_start_end_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(arr=arr, starts=starts, ends=ends, booleans=booleans)


def _min_starts_ends(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute min
    """
    mapping = {
        "int64": janitor_rs.compute_min_start_end_int64,
        "int32": janitor_rs.compute_min_start_end_int32,
        "int16": janitor_rs.compute_min_start_end_int16,
        "int8": janitor_rs.compute_min_start_end_int8,
        "uint64": janitor_rs.compute_min_start_end_uint64,
        "uint32": janitor_rs.compute_min_start_end_uint32,
        "uint16": janitor_rs.compute_min_start_end_uint16,
        "uint8": janitor_rs.compute_min_start_end_uint8,
        "float64": janitor_rs.compute_min_start_end_f64,
        "float32": janitor_rs.compute_min_start_end_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(arr=arr, starts=starts, ends=ends, booleans=booleans)


def _sum_starts_ends(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute sum
    """
    dtype_name = arr.dtype.name
    if dtype_name in _FLOAT_PREFIX_DTYPES:
        total_width = int((ends - starts).sum())
        if _prefix_sum_is_worthwhile(arr.size, total_width):
            prefix = _build_prefix_if_safe(arr=arr, booleans=booleans)
            if prefix is not None:
                hi, lo = prefix
                return _range_sum_from_prefix(hi=hi, lo=lo, starts=starts, ends=ends)
    mapping = {
        "int64": janitor_rs.compute_sum_start_end_int64,
        "int32": janitor_rs.compute_sum_start_end_int32,
        "int16": janitor_rs.compute_sum_start_end_int16,
        "int8": janitor_rs.compute_sum_start_end_int8,
        "uint64": janitor_rs.compute_sum_start_end_uint64,
        "uint32": janitor_rs.compute_sum_start_end_uint32,
        "uint16": janitor_rs.compute_sum_start_end_uint16,
        "uint8": janitor_rs.compute_sum_start_end_uint8,
        "float64": janitor_rs.compute_sum_start_end_f64,
        "float32": janitor_rs.compute_sum_start_end_f32,
    }
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(arr=arr, starts=starts, ends=ends, booleans=booleans)


def _prod_starts_ends(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute prod
    """
    mapping = {
        "int64": janitor_rs.compute_prod_start_end_int64,
        "int32": janitor_rs.compute_prod_start_end_int32,
        "int16": janitor_rs.compute_prod_start_end_int16,
        "int8": janitor_rs.compute_prod_start_end_int8,
        "uint64": janitor_rs.compute_prod_start_end_uint64,
        "uint32": janitor_rs.compute_prod_start_end_uint32,
        "uint16": janitor_rs.compute_prod_start_end_uint16,
        "uint8": janitor_rs.compute_prod_start_end_uint8,
        "float64": janitor_rs.compute_prod_start_end_f64,
        "float32": janitor_rs.compute_prod_start_end_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(arr=arr, starts=starts, ends=ends, booleans=booleans)


def _prod_starts_ends_matches(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    counts: np.ndarray,
    matches: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute prod
    """
    mapping = {
        "int64": janitor_rs.compute_prod_start_end_match_int64,
        "int32": janitor_rs.compute_prod_start_end_match_int32,
        "int16": janitor_rs.compute_prod_start_end_match_int16,
        "int8": janitor_rs.compute_prod_start_end_match_int8,
        "uint64": janitor_rs.compute_prod_start_end_match_uint64,
        "uint32": janitor_rs.compute_prod_start_end_match_uint32,
        "uint16": janitor_rs.compute_prod_start_end_match_uint16,
        "uint8": janitor_rs.compute_prod_start_end_match_uint8,
        "float64": janitor_rs.compute_prod_start_end_match_f64,
        "float32": janitor_rs.compute_prod_start_end_match_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(
        arr=arr,
        starts=starts,
        ends=ends,
        counts=counts,
        matches=matches,
        booleans=booleans,
    )


def _sum_starts_ends_matches(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    counts: np.ndarray,
    matches: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute sum
    """
    mapping = {
        "int64": janitor_rs.compute_sum_start_end_match_int64,
        "int32": janitor_rs.compute_sum_start_end_match_int32,
        "int16": janitor_rs.compute_sum_start_end_match_int16,
        "int8": janitor_rs.compute_sum_start_end_match_int8,
        "uint64": janitor_rs.compute_sum_start_end_match_uint64,
        "uint32": janitor_rs.compute_sum_start_end_match_uint32,
        "uint16": janitor_rs.compute_sum_start_end_match_uint16,
        "uint8": janitor_rs.compute_sum_start_end_match_uint8,
        "float64": janitor_rs.compute_sum_start_end_match_f64,
        "float32": janitor_rs.compute_sum_start_end_match_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(
        arr=arr,
        starts=starts,
        ends=ends,
        counts=counts,
        matches=matches,
        booleans=booleans,
    )


def _min_starts_ends_matches(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    counts: np.ndarray,
    matches: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute min
    """
    mapping = {
        "int64": janitor_rs.compute_min_start_end_match_int64,
        "int32": janitor_rs.compute_min_start_end_match_int32,
        "int16": janitor_rs.compute_min_start_end_match_int16,
        "int8": janitor_rs.compute_min_start_end_match_int8,
        "uint64": janitor_rs.compute_min_start_end_match_uint64,
        "uint32": janitor_rs.compute_min_start_end_match_uint32,
        "uint16": janitor_rs.compute_min_start_end_match_uint16,
        "uint8": janitor_rs.compute_min_start_end_match_uint8,
        "float64": janitor_rs.compute_min_start_end_match_f64,
        "float32": janitor_rs.compute_min_start_end_match_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(
        arr=arr,
        starts=starts,
        ends=ends,
        counts=counts,
        matches=matches,
        booleans=booleans,
    )


def _max_starts_ends_matches(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    counts: np.ndarray,
    matches: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute max
    """
    mapping = {
        "int64": janitor_rs.compute_max_start_end_match_int64,
        "int32": janitor_rs.compute_max_start_end_match_int32,
        "int16": janitor_rs.compute_max_start_end_match_int16,
        "int8": janitor_rs.compute_max_start_end_match_int8,
        "uint64": janitor_rs.compute_max_start_end_match_uint64,
        "uint32": janitor_rs.compute_max_start_end_match_uint32,
        "uint16": janitor_rs.compute_max_start_end_match_uint16,
        "uint8": janitor_rs.compute_max_start_end_match_uint8,
        "float64": janitor_rs.compute_max_start_end_match_f64,
        "float32": janitor_rs.compute_max_start_end_match_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(
        arr=arr,
        starts=starts,
        ends=ends,
        counts=counts,
        matches=matches,
        booleans=booleans,
    )


def _prod_rev_starts(
    arr: np.ndarray,
    starts: np.ndarray,
    index: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute prod
    """
    mapping = {
        "int64": janitor_rs.compute_prod_rev_start_int64,
        "int32": janitor_rs.compute_prod_rev_start_int32,
        "int16": janitor_rs.compute_prod_rev_start_int16,
        "int8": janitor_rs.compute_prod_rev_start_int8,
        "uint64": janitor_rs.compute_prod_rev_start_uint64,
        "uint32": janitor_rs.compute_prod_rev_start_uint32,
        "uint16": janitor_rs.compute_prod_rev_start_uint16,
        "uint8": janitor_rs.compute_prod_rev_start_uint8,
        "float64": janitor_rs.compute_prod_rev_start_f64,
        "float32": janitor_rs.compute_prod_rev_start_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(arr=arr, starts=starts, index=index, booleans=booleans, length=length)


def _prod_rev_ends(
    arr: np.ndarray,
    ends: np.ndarray,
    index: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute prod
    """
    mapping = {
        "int64": janitor_rs.compute_prod_rev_end_int64,
        "int32": janitor_rs.compute_prod_rev_end_int32,
        "int16": janitor_rs.compute_prod_rev_end_int16,
        "int8": janitor_rs.compute_prod_rev_end_int8,
        "uint64": janitor_rs.compute_prod_rev_end_uint64,
        "uint32": janitor_rs.compute_prod_rev_end_uint32,
        "uint16": janitor_rs.compute_prod_rev_end_uint16,
        "uint8": janitor_rs.compute_prod_rev_end_uint8,
        "float64": janitor_rs.compute_prod_rev_end_f64,
        "float32": janitor_rs.compute_prod_rev_end_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(arr=arr, ends=ends, index=index, booleans=booleans, length=length)


def _prod_rev_starts_matches(
    arr: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
    index: np.ndarray,
    matches: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute prod
    """
    mapping = {
        "int64": janitor_rs.compute_prod_rev_start_match_int64,
        "int32": janitor_rs.compute_prod_rev_start_match_int32,
        "int16": janitor_rs.compute_prod_rev_start_match_int16,
        "int8": janitor_rs.compute_prod_rev_start_match_int8,
        "uint64": janitor_rs.compute_prod_rev_start_match_uint64,
        "uint32": janitor_rs.compute_prod_rev_start_match_uint32,
        "uint16": janitor_rs.compute_prod_rev_start_match_uint16,
        "uint8": janitor_rs.compute_prod_rev_start_match_uint8,
        "float64": janitor_rs.compute_prod_rev_start_match_f64,
        "float32": janitor_rs.compute_prod_rev_start_match_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(
        arr=arr,
        starts=starts,
        counts=counts,
        index=index,
        matches=matches,
        booleans=booleans,
        length=length,
    )


def _prod_rev_ends_matches(
    arr: np.ndarray,
    index: np.ndarray,
    ends: np.ndarray,
    counts: np.ndarray,
    matches: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute prod
    """
    mapping = {
        "int64": janitor_rs.compute_prod_rev_end_match_int64,
        "int32": janitor_rs.compute_prod_rev_end_match_int32,
        "int16": janitor_rs.compute_prod_rev_end_match_int16,
        "int8": janitor_rs.compute_prod_rev_end_match_int8,
        "uint64": janitor_rs.compute_prod_rev_end_match_uint64,
        "uint32": janitor_rs.compute_prod_rev_end_match_uint32,
        "uint16": janitor_rs.compute_prod_rev_end_match_uint16,
        "uint8": janitor_rs.compute_prod_rev_end_match_uint8,
        "float64": janitor_rs.compute_prod_rev_end_match_f64,
        "float32": janitor_rs.compute_prod_rev_end_match_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(
        arr=arr,
        index=index,
        ends=ends,
        counts=counts,
        matches=matches,
        booleans=booleans,
        length=length,
    )


def _prod_rev_positions(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    index: np.ndarray,
    positions: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute prod
    """
    mapping = {
        "int64": janitor_rs.compute_prod_rev_positions_int64,
        "int32": janitor_rs.compute_prod_rev_positions_int32,
        "int16": janitor_rs.compute_prod_rev_positions_int16,
        "int8": janitor_rs.compute_prod_rev_positions_int8,
        "uint64": janitor_rs.compute_prod_rev_positions_uint64,
        "uint32": janitor_rs.compute_prod_rev_positions_uint32,
        "uint16": janitor_rs.compute_prod_rev_positions_uint16,
        "uint8": janitor_rs.compute_prod_rev_positions_uint8,
        "float64": janitor_rs.compute_prod_rev_positions_f64,
        "float32": janitor_rs.compute_prod_rev_positions_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(
        arr=arr,
        starts=starts,
        ends=ends,
        index=index,
        positions=positions,
        booleans=booleans,
        length=length,
    )


def _prod_rev_starts_ends(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    index: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute prod
    """
    mapping = {
        "int64": janitor_rs.compute_prod_rev_start_end_int64,
        "int32": janitor_rs.compute_prod_rev_start_end_int32,
        "int16": janitor_rs.compute_prod_rev_start_end_int16,
        "int8": janitor_rs.compute_prod_rev_start_end_int8,
        "uint64": janitor_rs.compute_prod_rev_start_end_uint64,
        "uint32": janitor_rs.compute_prod_rev_start_end_uint32,
        "uint16": janitor_rs.compute_prod_rev_start_end_uint16,
        "uint8": janitor_rs.compute_prod_rev_start_end_uint8,
        "float64": janitor_rs.compute_prod_rev_start_end_f64,
        "float32": janitor_rs.compute_prod_rev_start_end_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(
        arr=arr,
        starts=starts,
        ends=ends,
        index=index,
        booleans=booleans,
        length=length,
    )


def _prod_rev_starts_ends_matches(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    index: np.ndarray,
    counts: np.ndarray,
    matches: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute prod
    """
    mapping = {
        "int64": janitor_rs.compute_prod_rev_start_end_match_int64,
        "int32": janitor_rs.compute_prod_rev_start_end_match_int32,
        "int16": janitor_rs.compute_prod_rev_start_end_match_int16,
        "int8": janitor_rs.compute_prod_rev_start_end_match_int8,
        "uint64": janitor_rs.compute_prod_rev_start_end_match_uint64,
        "uint32": janitor_rs.compute_prod_rev_start_end_match_uint32,
        "uint16": janitor_rs.compute_prod_rev_start_end_match_uint16,
        "uint8": janitor_rs.compute_prod_rev_start_end_match_uint8,
        "float64": janitor_rs.compute_prod_rev_start_end_match_f64,
        "float32": janitor_rs.compute_prod_rev_start_end_match_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(
        arr=arr,
        starts=starts,
        ends=ends,
        index=index,
        counts=counts,
        matches=matches,
        booleans=booleans,
        length=length,
    )


def _min_rev_starts(
    arr: np.ndarray,
    starts: np.ndarray,
    index: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute min
    """
    mapping = {
        "int64": janitor_rs.compute_min_rev_start_int64,
        "int32": janitor_rs.compute_min_rev_start_int32,
        "int16": janitor_rs.compute_min_rev_start_int16,
        "int8": janitor_rs.compute_min_rev_start_int8,
        "uint64": janitor_rs.compute_min_rev_start_uint64,
        "uint32": janitor_rs.compute_min_rev_start_uint32,
        "uint16": janitor_rs.compute_min_rev_start_uint16,
        "uint8": janitor_rs.compute_min_rev_start_uint8,
        "float64": janitor_rs.compute_min_rev_start_f64,
        "float32": janitor_rs.compute_min_rev_start_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(arr=arr, starts=starts, index=index, booleans=booleans, length=length)


def _min_rev_ends(
    arr: np.ndarray,
    ends: np.ndarray,
    index: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute min
    """
    mapping = {
        "int64": janitor_rs.compute_min_rev_end_int64,
        "int32": janitor_rs.compute_min_rev_end_int32,
        "int16": janitor_rs.compute_min_rev_end_int16,
        "int8": janitor_rs.compute_min_rev_end_int8,
        "uint64": janitor_rs.compute_min_rev_end_uint64,
        "uint32": janitor_rs.compute_min_rev_end_uint32,
        "uint16": janitor_rs.compute_min_rev_end_uint16,
        "uint8": janitor_rs.compute_min_rev_end_uint8,
        "float64": janitor_rs.compute_min_rev_end_f64,
        "float32": janitor_rs.compute_min_rev_end_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(arr=arr, ends=ends, index=index, booleans=booleans, length=length)


def _min_rev_starts_matches(
    arr: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
    index: np.ndarray,
    matches: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute min
    """
    mapping = {
        "int64": janitor_rs.compute_min_rev_start_match_int64,
        "int32": janitor_rs.compute_min_rev_start_match_int32,
        "int16": janitor_rs.compute_min_rev_start_match_int16,
        "int8": janitor_rs.compute_min_rev_start_match_int8,
        "uint64": janitor_rs.compute_min_rev_start_match_uint64,
        "uint32": janitor_rs.compute_min_rev_start_match_uint32,
        "uint16": janitor_rs.compute_min_rev_start_match_uint16,
        "uint8": janitor_rs.compute_min_rev_start_match_uint8,
        "float64": janitor_rs.compute_min_rev_start_match_f64,
        "float32": janitor_rs.compute_min_rev_start_match_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(
        arr=arr,
        starts=starts,
        counts=counts,
        index=index,
        matches=matches,
        booleans=booleans,
        length=length,
    )


def _min_rev_ends_matches(
    arr: np.ndarray,
    index: np.ndarray,
    ends: np.ndarray,
    counts: np.ndarray,
    matches: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute min
    """
    mapping = {
        "int64": janitor_rs.compute_min_rev_end_match_int64,
        "int32": janitor_rs.compute_min_rev_end_match_int32,
        "int16": janitor_rs.compute_min_rev_end_match_int16,
        "int8": janitor_rs.compute_min_rev_end_match_int8,
        "uint64": janitor_rs.compute_min_rev_end_match_uint64,
        "uint32": janitor_rs.compute_min_rev_end_match_uint32,
        "uint16": janitor_rs.compute_min_rev_end_match_uint16,
        "uint8": janitor_rs.compute_min_rev_end_match_uint8,
        "float64": janitor_rs.compute_min_rev_end_match_f64,
        "float32": janitor_rs.compute_min_rev_end_match_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(
        arr=arr,
        index=index,
        ends=ends,
        counts=counts,
        matches=matches,
        booleans=booleans,
        length=length,
    )


def _min_rev_positions(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    index: np.ndarray,
    positions: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute min
    """
    mapping = {
        "int64": janitor_rs.compute_min_rev_positions_int64,
        "int32": janitor_rs.compute_min_rev_positions_int32,
        "int16": janitor_rs.compute_min_rev_positions_int16,
        "int8": janitor_rs.compute_min_rev_positions_int8,
        "uint64": janitor_rs.compute_min_rev_positions_uint64,
        "uint32": janitor_rs.compute_min_rev_positions_uint32,
        "uint16": janitor_rs.compute_min_rev_positions_uint16,
        "uint8": janitor_rs.compute_min_rev_positions_uint8,
        "float64": janitor_rs.compute_min_rev_positions_f64,
        "float32": janitor_rs.compute_min_rev_positions_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(
        arr=arr,
        starts=starts,
        ends=ends,
        index=index,
        positions=positions,
        booleans=booleans,
        length=length,
    )


def _min_rev_starts_ends(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    index: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute min
    """
    mapping = {
        "int64": janitor_rs.compute_min_rev_start_end_int64,
        "int32": janitor_rs.compute_min_rev_start_end_int32,
        "int16": janitor_rs.compute_min_rev_start_end_int16,
        "int8": janitor_rs.compute_min_rev_start_end_int8,
        "uint64": janitor_rs.compute_min_rev_start_end_uint64,
        "uint32": janitor_rs.compute_min_rev_start_end_uint32,
        "uint16": janitor_rs.compute_min_rev_start_end_uint16,
        "uint8": janitor_rs.compute_min_rev_start_end_uint8,
        "float64": janitor_rs.compute_min_rev_start_end_f64,
        "float32": janitor_rs.compute_min_rev_start_end_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(
        arr=arr,
        starts=starts,
        ends=ends,
        index=index,
        booleans=booleans,
        length=length,
    )


def _min_rev_starts_ends_matches(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    index: np.ndarray,
    counts: np.ndarray,
    matches: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute min
    """
    mapping = {
        "int64": janitor_rs.compute_min_rev_start_end_match_int64,
        "int32": janitor_rs.compute_min_rev_start_end_match_int32,
        "int16": janitor_rs.compute_min_rev_start_end_match_int16,
        "int8": janitor_rs.compute_min_rev_start_end_match_int8,
        "uint64": janitor_rs.compute_min_rev_start_end_match_uint64,
        "uint32": janitor_rs.compute_min_rev_start_end_match_uint32,
        "uint16": janitor_rs.compute_min_rev_start_end_match_uint16,
        "uint8": janitor_rs.compute_min_rev_start_end_match_uint8,
        "float64": janitor_rs.compute_min_rev_start_end_match_f64,
        "float32": janitor_rs.compute_min_rev_start_end_match_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(
        arr=arr,
        starts=starts,
        ends=ends,
        index=index,
        counts=counts,
        matches=matches,
        booleans=booleans,
        length=length,
    )


def _max_rev_starts(
    arr: np.ndarray,
    starts: np.ndarray,
    index: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute max
    """
    mapping = {
        "int64": janitor_rs.compute_max_rev_start_int64,
        "int32": janitor_rs.compute_max_rev_start_int32,
        "int16": janitor_rs.compute_max_rev_start_int16,
        "int8": janitor_rs.compute_max_rev_start_int8,
        "uint64": janitor_rs.compute_max_rev_start_uint64,
        "uint32": janitor_rs.compute_max_rev_start_uint32,
        "uint16": janitor_rs.compute_max_rev_start_uint16,
        "uint8": janitor_rs.compute_max_rev_start_uint8,
        "float64": janitor_rs.compute_max_rev_start_f64,
        "float32": janitor_rs.compute_max_rev_start_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(arr=arr, starts=starts, index=index, booleans=booleans, length=length)


def _max_rev_ends(
    arr: np.ndarray,
    ends: np.ndarray,
    index: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute max
    """
    mapping = {
        "int64": janitor_rs.compute_max_rev_end_int64,
        "int32": janitor_rs.compute_max_rev_end_int32,
        "int16": janitor_rs.compute_max_rev_end_int16,
        "int8": janitor_rs.compute_max_rev_end_int8,
        "uint64": janitor_rs.compute_max_rev_end_uint64,
        "uint32": janitor_rs.compute_max_rev_end_uint32,
        "uint16": janitor_rs.compute_max_rev_end_uint16,
        "uint8": janitor_rs.compute_max_rev_end_uint8,
        "float64": janitor_rs.compute_max_rev_end_f64,
        "float32": janitor_rs.compute_max_rev_end_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(arr=arr, ends=ends, index=index, booleans=booleans, length=length)


def _max_rev_starts_matches(
    arr: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
    index: np.ndarray,
    matches: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute max
    """
    mapping = {
        "int64": janitor_rs.compute_max_rev_start_match_int64,
        "int32": janitor_rs.compute_max_rev_start_match_int32,
        "int16": janitor_rs.compute_max_rev_start_match_int16,
        "int8": janitor_rs.compute_max_rev_start_match_int8,
        "uint64": janitor_rs.compute_max_rev_start_match_uint64,
        "uint32": janitor_rs.compute_max_rev_start_match_uint32,
        "uint16": janitor_rs.compute_max_rev_start_match_uint16,
        "uint8": janitor_rs.compute_max_rev_start_match_uint8,
        "float64": janitor_rs.compute_max_rev_start_match_f64,
        "float32": janitor_rs.compute_max_rev_start_match_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(
        arr=arr,
        starts=starts,
        counts=counts,
        index=index,
        matches=matches,
        booleans=booleans,
        length=length,
    )


def _max_rev_ends_matches(
    arr: np.ndarray,
    index: np.ndarray,
    ends: np.ndarray,
    counts: np.ndarray,
    matches: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute max
    """
    mapping = {
        "int64": janitor_rs.compute_max_rev_end_match_int64,
        "int32": janitor_rs.compute_max_rev_end_match_int32,
        "int16": janitor_rs.compute_max_rev_end_match_int16,
        "int8": janitor_rs.compute_max_rev_end_match_int8,
        "uint64": janitor_rs.compute_max_rev_end_match_uint64,
        "uint32": janitor_rs.compute_max_rev_end_match_uint32,
        "uint16": janitor_rs.compute_max_rev_end_match_uint16,
        "uint8": janitor_rs.compute_max_rev_end_match_uint8,
        "float64": janitor_rs.compute_max_rev_end_match_f64,
        "float32": janitor_rs.compute_max_rev_end_match_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(
        arr=arr,
        index=index,
        ends=ends,
        counts=counts,
        matches=matches,
        booleans=booleans,
        length=length,
    )


def _max_rev_positions(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    index: np.ndarray,
    positions: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute max
    """
    mapping = {
        "int64": janitor_rs.compute_max_rev_positions_int64,
        "int32": janitor_rs.compute_max_rev_positions_int32,
        "int16": janitor_rs.compute_max_rev_positions_int16,
        "int8": janitor_rs.compute_max_rev_positions_int8,
        "uint64": janitor_rs.compute_max_rev_positions_uint64,
        "uint32": janitor_rs.compute_max_rev_positions_uint32,
        "uint16": janitor_rs.compute_max_rev_positions_uint16,
        "uint8": janitor_rs.compute_max_rev_positions_uint8,
        "float64": janitor_rs.compute_max_rev_positions_f64,
        "float32": janitor_rs.compute_max_rev_positions_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(
        arr=arr,
        starts=starts,
        ends=ends,
        index=index,
        positions=positions,
        booleans=booleans,
        length=length,
    )


def _max_rev_starts_ends(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    index: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute max
    """
    mapping = {
        "int64": janitor_rs.compute_max_rev_start_end_int64,
        "int32": janitor_rs.compute_max_rev_start_end_int32,
        "int16": janitor_rs.compute_max_rev_start_end_int16,
        "int8": janitor_rs.compute_max_rev_start_end_int8,
        "uint64": janitor_rs.compute_max_rev_start_end_uint64,
        "uint32": janitor_rs.compute_max_rev_start_end_uint32,
        "uint16": janitor_rs.compute_max_rev_start_end_uint16,
        "uint8": janitor_rs.compute_max_rev_start_end_uint8,
        "float64": janitor_rs.compute_max_rev_start_end_f64,
        "float32": janitor_rs.compute_max_rev_start_end_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(
        arr=arr,
        starts=starts,
        ends=ends,
        index=index,
        booleans=booleans,
        length=length,
    )


def _max_rev_starts_ends_matches(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    index: np.ndarray,
    counts: np.ndarray,
    matches: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute max
    """
    mapping = {
        "int64": janitor_rs.compute_max_rev_start_end_match_int64,
        "int32": janitor_rs.compute_max_rev_start_end_match_int32,
        "int16": janitor_rs.compute_max_rev_start_end_match_int16,
        "int8": janitor_rs.compute_max_rev_start_end_match_int8,
        "uint64": janitor_rs.compute_max_rev_start_end_match_uint64,
        "uint32": janitor_rs.compute_max_rev_start_end_match_uint32,
        "uint16": janitor_rs.compute_max_rev_start_end_match_uint16,
        "uint8": janitor_rs.compute_max_rev_start_end_match_uint8,
        "float64": janitor_rs.compute_max_rev_start_end_match_f64,
        "float32": janitor_rs.compute_max_rev_start_end_match_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(
        arr=arr,
        starts=starts,
        ends=ends,
        index=index,
        counts=counts,
        matches=matches,
        booleans=booleans,
        length=length,
    )


def _prod_rev_no_ranges(
    arr: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute prod
    """
    mapping = {
        "int64": janitor_rs.compute_prod_rev_no_range_int64,
        "int32": janitor_rs.compute_prod_rev_no_range_int32,
        "int16": janitor_rs.compute_prod_rev_no_range_int16,
        "int8": janitor_rs.compute_prod_rev_no_range_int8,
        "uint64": janitor_rs.compute_prod_rev_no_range_uint64,
        "uint32": janitor_rs.compute_prod_rev_no_range_uint32,
        "uint16": janitor_rs.compute_prod_rev_no_range_uint16,
        "uint8": janitor_rs.compute_prod_rev_no_range_uint8,
        "float64": janitor_rs.compute_prod_rev_no_range_f64,
        "float32": janitor_rs.compute_prod_rev_no_range_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(
        arr=arr,
        left_index=left_index,
        right_index=right_index,
        booleans=booleans,
        length=length,
    )


def _max_rev_no_ranges(
    arr: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute max
    """
    mapping = {
        "int64": janitor_rs.compute_max_rev_no_range_int64,
        "int32": janitor_rs.compute_max_rev_no_range_int32,
        "int16": janitor_rs.compute_max_rev_no_range_int16,
        "int8": janitor_rs.compute_max_rev_no_range_int8,
        "uint64": janitor_rs.compute_max_rev_no_range_uint64,
        "uint32": janitor_rs.compute_max_rev_no_range_uint32,
        "uint16": janitor_rs.compute_max_rev_no_range_uint16,
        "uint8": janitor_rs.compute_max_rev_no_range_uint8,
        "float64": janitor_rs.compute_max_rev_no_range_f64,
        "float32": janitor_rs.compute_max_rev_no_range_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(
        arr=arr,
        left_index=left_index,
        right_index=right_index,
        booleans=booleans,
        length=length,
    )


def _min_rev_no_ranges(
    arr: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute min
    """
    mapping = {
        "int64": janitor_rs.compute_min_rev_no_range_int64,
        "int32": janitor_rs.compute_min_rev_no_range_int32,
        "int16": janitor_rs.compute_min_rev_no_range_int16,
        "int8": janitor_rs.compute_min_rev_no_range_int8,
        "uint64": janitor_rs.compute_min_rev_no_range_uint64,
        "uint32": janitor_rs.compute_min_rev_no_range_uint32,
        "uint16": janitor_rs.compute_min_rev_no_range_uint16,
        "uint8": janitor_rs.compute_min_rev_no_range_uint8,
        "float64": janitor_rs.compute_min_rev_no_range_f64,
        "float32": janitor_rs.compute_min_rev_no_range_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(
        arr=arr,
        left_index=left_index,
        right_index=right_index,
        booleans=booleans,
        length=length,
    )


def _sum_rev_no_ranges(
    arr: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute sum
    """
    mapping = {
        "int64": janitor_rs.compute_sum_rev_no_range_int64,
        "int32": janitor_rs.compute_sum_rev_no_range_int32,
        "int16": janitor_rs.compute_sum_rev_no_range_int16,
        "int8": janitor_rs.compute_sum_rev_no_range_int8,
        "uint64": janitor_rs.compute_sum_rev_no_range_uint64,
        "uint32": janitor_rs.compute_sum_rev_no_range_uint32,
        "uint16": janitor_rs.compute_sum_rev_no_range_uint16,
        "uint8": janitor_rs.compute_sum_rev_no_range_uint8,
        "float64": janitor_rs.compute_sum_rev_no_range_f64,
        "float32": janitor_rs.compute_sum_rev_no_range_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(
        arr=arr,
        left_index=left_index,
        right_index=right_index,
        booleans=booleans,
        length=length,
    )


def _sum_rev_starts(
    arr: np.ndarray,
    starts: np.ndarray,
    index: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute sum
    """
    mapping = {
        "int64": janitor_rs.compute_sum_rev_start_int64,
        "int32": janitor_rs.compute_sum_rev_start_int32,
        "int16": janitor_rs.compute_sum_rev_start_int16,
        "int8": janitor_rs.compute_sum_rev_start_int8,
        "uint64": janitor_rs.compute_sum_rev_start_uint64,
        "uint32": janitor_rs.compute_sum_rev_start_uint32,
        "uint16": janitor_rs.compute_sum_rev_start_uint16,
        "uint8": janitor_rs.compute_sum_rev_start_uint8,
        "float64": janitor_rs.compute_sum_rev_start_f64,
        "float32": janitor_rs.compute_sum_rev_start_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(arr=arr, starts=starts, index=index, booleans=booleans, length=length)


def _sum_rev_ends(
    arr: np.ndarray,
    ends: np.ndarray,
    index: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute sum
    """
    mapping = {
        "int64": janitor_rs.compute_sum_rev_end_int64,
        "int32": janitor_rs.compute_sum_rev_end_int32,
        "int16": janitor_rs.compute_sum_rev_end_int16,
        "int8": janitor_rs.compute_sum_rev_end_int8,
        "uint64": janitor_rs.compute_sum_rev_end_uint64,
        "uint32": janitor_rs.compute_sum_rev_end_uint32,
        "uint16": janitor_rs.compute_sum_rev_end_uint16,
        "uint8": janitor_rs.compute_sum_rev_end_uint8,
        "float64": janitor_rs.compute_sum_rev_end_f64,
        "float32": janitor_rs.compute_sum_rev_end_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(arr=arr, ends=ends, index=index, booleans=booleans, length=length)


def _sum_rev_starts_matches(
    arr: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
    index: np.ndarray,
    matches: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute sum
    """
    mapping = {
        "int64": janitor_rs.compute_sum_rev_start_match_int64,
        "int32": janitor_rs.compute_sum_rev_start_match_int32,
        "int16": janitor_rs.compute_sum_rev_start_match_int16,
        "int8": janitor_rs.compute_sum_rev_start_match_int8,
        "uint64": janitor_rs.compute_sum_rev_start_match_uint64,
        "uint32": janitor_rs.compute_sum_rev_start_match_uint32,
        "uint16": janitor_rs.compute_sum_rev_start_match_uint16,
        "uint8": janitor_rs.compute_sum_rev_start_match_uint8,
        "float64": janitor_rs.compute_sum_rev_start_match_f64,
        "float32": janitor_rs.compute_sum_rev_start_match_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(
        arr=arr,
        starts=starts,
        counts=counts,
        index=index,
        matches=matches,
        booleans=booleans,
        length=length,
    )


def _sum_rev_ends_matches(
    arr: np.ndarray,
    index: np.ndarray,
    ends: np.ndarray,
    counts: np.ndarray,
    matches: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute sum
    """
    mapping = {
        "int64": janitor_rs.compute_sum_rev_end_match_int64,
        "int32": janitor_rs.compute_sum_rev_end_match_int32,
        "int16": janitor_rs.compute_sum_rev_end_match_int16,
        "int8": janitor_rs.compute_sum_rev_end_match_int8,
        "uint64": janitor_rs.compute_sum_rev_end_match_uint64,
        "uint32": janitor_rs.compute_sum_rev_end_match_uint32,
        "uint16": janitor_rs.compute_sum_rev_end_match_uint16,
        "uint8": janitor_rs.compute_sum_rev_end_match_uint8,
        "float64": janitor_rs.compute_sum_rev_end_match_f64,
        "float32": janitor_rs.compute_sum_rev_end_match_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(
        arr=arr,
        index=index,
        ends=ends,
        counts=counts,
        matches=matches,
        booleans=booleans,
        length=length,
    )


def _sum_rev_positions(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    index: np.ndarray,
    positions: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute sum
    """
    mapping = {
        "int64": janitor_rs.compute_sum_rev_positions_int64,
        "int32": janitor_rs.compute_sum_rev_positions_int32,
        "int16": janitor_rs.compute_sum_rev_positions_int16,
        "int8": janitor_rs.compute_sum_rev_positions_int8,
        "uint64": janitor_rs.compute_sum_rev_positions_uint64,
        "uint32": janitor_rs.compute_sum_rev_positions_uint32,
        "uint16": janitor_rs.compute_sum_rev_positions_uint16,
        "uint8": janitor_rs.compute_sum_rev_positions_uint8,
        "float64": janitor_rs.compute_sum_rev_positions_f64,
        "float32": janitor_rs.compute_sum_rev_positions_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(
        arr=arr,
        starts=starts,
        ends=ends,
        index=index,
        positions=positions,
        booleans=booleans,
        length=length,
    )


def _sum_rev_starts_ends(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    index: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute sum
    """
    mapping = {
        "int64": janitor_rs.compute_sum_rev_start_end_int64,
        "int32": janitor_rs.compute_sum_rev_start_end_int32,
        "int16": janitor_rs.compute_sum_rev_start_end_int16,
        "int8": janitor_rs.compute_sum_rev_start_end_int8,
        "uint64": janitor_rs.compute_sum_rev_start_end_uint64,
        "uint32": janitor_rs.compute_sum_rev_start_end_uint32,
        "uint16": janitor_rs.compute_sum_rev_start_end_uint16,
        "uint8": janitor_rs.compute_sum_rev_start_end_uint8,
        "float64": janitor_rs.compute_sum_rev_start_end_f64,
        "float32": janitor_rs.compute_sum_rev_start_end_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(
        arr=arr,
        starts=starts,
        ends=ends,
        index=index,
        booleans=booleans,
        length=length,
    )


def _sum_rev_starts_ends_matches(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    index: np.ndarray,
    counts: np.ndarray,
    matches: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute sum
    """
    mapping = {
        "int64": janitor_rs.compute_sum_rev_start_end_match_int64,
        "int32": janitor_rs.compute_sum_rev_start_end_match_int32,
        "int16": janitor_rs.compute_sum_rev_start_end_match_int16,
        "int8": janitor_rs.compute_sum_rev_start_end_match_int8,
        "uint64": janitor_rs.compute_sum_rev_start_end_match_uint64,
        "uint32": janitor_rs.compute_sum_rev_start_end_match_uint32,
        "uint16": janitor_rs.compute_sum_rev_start_end_match_uint16,
        "uint8": janitor_rs.compute_sum_rev_start_end_match_uint8,
        "float64": janitor_rs.compute_sum_rev_start_end_match_f64,
        "float32": janitor_rs.compute_sum_rev_start_end_match_f32,
    }
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(
        arr=arr,
        starts=starts,
        ends=ends,
        index=index,
        counts=counts,
        matches=matches,
        booleans=booleans,
        length=length,
    )
