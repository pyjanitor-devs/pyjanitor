import janitor_rs
import numpy as np

_ARGEXT_WORK_FACTOR = 20
# ELI5: the NumPy path (`_prefix_argext`/`_suffix_argext`) doesn't check
# dtype itself -- it just runs generic NumPy ops that would happily
# "succeed" on a dtype the Rust kernels don't actually support (e.g.
# bool), silently skipping the `KeyError` the mapping dict below is
# supposed to raise. Gating the NumPy path on this set first means an
# unsupported dtype always falls through to the mapping dict and its
# error, no matter how many/wide the requested ranges are.
_ARGEXT_DTYPE_NAMES = frozenset(
    {
        "int64",
        "int32",
        "int16",
        "int8",
        "uint64",
        "uint32",
        "uint16",
        "uint8",
        "float64",
        "float32",
    }
)


def _use_argext(arr_size: int, total_width: int) -> bool:
    """
    Use the O(n) prefix/suffix precompute only when repeated Rust range
    scans would cost more.

    ELI5: same idea as the prefix-sum work-factor check for range sums --
    only pay for building the running-best array once the Rust kernel
    would otherwise re-scan the array more than `_ARGEXT_WORK_FACTOR`
    times over. Benchmarked empirically (see PR description): this
    kernel does more per-element work than a plain running sum, so its
    break-even point is higher.
    """
    return total_width > (_ARGEXT_WORK_FACTOR * arr_size)


def _running_argext_compact(vals: np.ndarray, is_max: bool, strict: bool) -> np.ndarray:
    """
    Positions (into `vals`) of the running best value, scanning `vals`
    left to right. `vals` must already have every null/NaN entry removed
    -- that's what makes this safe to vectorize with no sentinel value:
    there's nothing to accidentally collide with a real data point at a
    narrow dtype's extreme (e.g. 255 for uint8).

    ELI5: walk once, remembering the position of the best value seen so
    far. `strict=True` means a tie does NOT replace the earlier record
    (used for prefixes); `strict=False` means a tie DOES replace it
    (used for suffixes, scanning a reversed array, so "replace on tie"
    ends up keeping the smaller/earlier original index).
    """
    m = vals.size
    # ELI5: `running_val[i]` = the best value seen anywhere in vals[:i+1].
    # `fmax`/`fmin` are the same as `maximum`/`minimum` for these already-
    # cleaned values, computed once for the whole array in one call
    # (that's the "O(n) instead of O(n) per row" trick).
    running_val = (np.fmax if is_max else np.fmin).accumulate(vals)

    # ELI5: a position is a "new record" exactly when it's the reason
    # `running_val` changed (or didn't change but ties count -- see
    # `strict` below). Index 0 is always a record: there's nothing
    # before it to compare against, so it's automatically the best seen
    # so far.
    is_new_record = np.empty(m, dtype=bool)
    is_new_record[0] = True
    if strict:
        # Prefix rule: only a STRICT improvement counts as a new record,
        # so an earlier tie keeps its spot -- `running_val` only changes
        # on a genuine improvement, so comparing consecutive running
        # values directly tells us exactly where that happened.
        is_new_record[1:] = (
            (running_val[1:] > running_val[:-1])
            if is_max
            else (running_val[1:] < running_val[:-1])
        )
    else:
        # Suffix rule: a TIE also counts as a new record (see the
        # `strict=False` note above -- this scans a reversed array, so
        # "record on tie" is what makes the earlier original index win).
        # `vals[i] == running_val[i]` is true both when vals[i] set a new
        # best AND when it merely matched the existing best.
        is_new_record[1:] = vals[1:] == running_val[1:]

    # ELI5: turn "is this position a record?" into "what's the latest
    # record position seen so far?" -- put each record's own index at
    # its position, -1 everywhere else, then let `maximum.accumulate`
    # carry the largest (i.e. most recent) index forward over the -1s.
    candidate = np.where(is_new_record, np.arange(m), -1)
    return np.maximum.accumulate(candidate)


def _isnan_if_float(arr: np.ndarray) -> np.ndarray:
    """`np.isnan` raises on integer dtypes, which never have NaN anyway."""
    if np.issubdtype(arr.dtype, np.floating):
        return np.isnan(arr)
    return np.zeros(arr.shape, dtype=bool)


def _prefix_argext(arr: np.ndarray, booleans: np.ndarray, is_max: bool) -> np.ndarray:
    """
    result[e] = the Rust `compute_{min,max}_end_*` answer for that `e`,
    for every e in 0..n, computed once in O(n) instead of once per row.

    ELI5: the Rust kernel restarts its scan from index 0 every single
    time, no matter what `end` is, so every row is just a different-length
    prefix of the *same* walk. Walk it once and remember the running best
    position; every row can then just read off the answer for its own
    `end`.

    Floats get one extra wrinkle: the Rust kernel's `<`/`>` comparisons
    are IEEE754, where anything compared with NaN is always false. So
    once the scan's very first non-null value is a NaN, nothing --
    real or NaN -- can ever replace it (`0 < NaN` and `NaN < 0` are both
    false), and that NaN's position "freezes" as the answer for every
    later `end` too. A NaN encountered *after* a real anchor, on the
    other hand, is just silently never selected, exactly like a null.
    """
    n = arr.size
    # result[e] is the answer for range [0, e); -1 is "no valid element
    # yet" (covers e == 0, and the whole array being null).
    result = np.full(n + 1, -1, dtype=np.int64)
    if n == 0:
        return result
    not_null = ~booleans
    if not not_null.any():
        return result  # every position is null -> every answer is -1

    # ELI5: the very first non-null element is special -- it's the one
    # every row's scan eventually reaches first (see the docstring: all
    # prefixes share the same walk from index 0). `np.argmax` on a bool
    # array returns the index of the first True.
    first_valid = int(np.argmax(not_null))
    isnan = _isnan_if_float(arr)

    if isnan[first_valid]:
        # Frozen case: nothing can ever beat/tie a NaN anchor (IEEE754),
        # so every `end` that reaches past `first_valid` is stuck
        # reporting that NaN's position; every `end` before it hasn't
        # found any real value yet, so it's still -1 (already the
        # default fill above).
        result[first_valid + 1 :] = first_valid
        return result

    # Anchor is real: NaN can now never win (it can't beat a real value,
    # and a real value can't lose to one either), so treat NaN exactly
    # like a null for the rest of this scan and compact both away.
    extended_null = booleans | isnan
    # original positions of the real, non-null values
    valid_idx = np.flatnonzero(~extended_null)
    valid_vals = arr[valid_idx]  # ...and their values, same order
    # Run the O(m) compact scan (m = count of real values), then map its
    # answers (positions *within* `valid_vals`) back to real array
    # positions via `valid_idx`.
    running_pos = valid_idx[_running_argext_compact(valid_vals, is_max, strict=True)]

    # ELI5: `result[e]` should reuse the compact scan's answer for
    # whichever real value was the LAST one inside [0, e). Count how
    # many real values exist in [0, e) (that's `valid_count_prefix[e]`,
    # a running count with a leading 0 so index 0 means "none yet"), and
    # that count doubles as a 1-indexed position into `running_pos` --
    # subtract 1 to make it a normal 0-indexed lookup.
    # dtype=np.int64 pins the accumulator explicitly -- np.cumsum's
    # platform-default int (int32 on 64-bit Windows) would silently
    # overflow past ~2.1 billion elements otherwise.
    valid_count_prefix = np.concatenate(
        ([0], np.cumsum(~extended_null, dtype=np.int64))
    )
    has_any = valid_count_prefix > 0
    k = np.clip(valid_count_prefix - 1, 0, max(running_pos.size - 1, 0))
    result[:] = np.where(has_any, running_pos[k], -1)
    return result


def _suffix_argext(arr: np.ndarray, booleans: np.ndarray, is_max: bool) -> np.ndarray:
    """
    result[s] = the Rust `compute_{min,max}_start_*` answer for that `s`,
    for every s in 0..n, computed once in O(n) instead of once per row.

    ELI5: unlike the prefix case, each row's `start` restarts its own
    scan from a different position, so which value gets "frozen" by the
    NaN quirk (see `_prefix_argext`) can differ row to row -- a single
    shared backward scan over the whole array isn't enough on its own.
    So this splits the work: `next_valid` finds each `s`'s own anchor
    position in O(n); if that anchor is a NaN the answer is just that
    position (frozen); otherwise NaN behaves like null for the rest of
    that row's range, and the answer comes from one shared "clean"
    backward scan that skips nulls and NaNs alike.
    """
    n = arr.size
    # result[s] is the answer for range [s, n); -1 is "no valid element
    # from s onward" (also serves as the s == n empty-range answer).
    result = np.full(n + 1, -1, dtype=np.int64)
    if n == 0:
        return result
    idx = np.arange(n)
    isnan = _isnan_if_float(arr)

    # ELI5: `next_valid[s]` = the smallest index >= s that isn't null,
    # or n if there isn't one. Built with a classic "fill backward" trick:
    # put each non-null position's own index in `raw`, n (a value larger
    # than any real index) everywhere else, then `minimum.accumulate` on
    # the REVERSED array carries the smallest real index found so far
    # back toward the front; reversing again undoes the flip so `raw[s]`
    # lines up with position s again. This is the "each row re-anchors
    # at its own start" position from the docstring, found for every s
    # at once.
    raw = np.where(~booleans, idx, n)
    next_valid = np.minimum.accumulate(raw[::-1])[::-1]

    # Once we know a row's anchor isn't a NaN (checked below via
    # `next_valid`), NaN behaves exactly like null for the rest of that
    # row's scan -- so compact both away together, same as the prefix
    # case, just scanning right-to-left.
    extended_null = booleans | isnan
    # original positions of the real, non-null values
    valid_idx = np.flatnonzero(~extended_null)
    if valid_idx.size:
        valid_vals = arr[valid_idx]
        # Reverse both arrays so "left to right on the reversed array"
        # is the same walk as "right to left on the real array" --
        # `_running_argext_compact(..., strict=False)` then applies the
        # tie-keeps-the-later-record rule, which (because everything is
        # reversed) works out to "ties keep the smaller original index".
        rev_pos_in_compact = _running_argext_compact(
            valid_vals[::-1], is_max, strict=False
        )
        # Map compact-array positions back to real array positions, in
        # the same reversed order used above.
        running_pos_rev = valid_idx[::-1][rev_pos_in_compact]

        # Same "count real values, use the count as a lookup index"
        # trick as the prefix case, mirrored: count real values from s
        # to the end (a running count from the right, with a trailing 0
        # so index n means "none"), then look that many entries into the
        # *reversed* running-position array.
        # dtype=np.int64: see the matching comment in _prefix_argext.
        valid_count_suffix = np.concatenate(
            (np.cumsum((~extended_null)[::-1], dtype=np.int64)[::-1], [0])
        )
        has_any = valid_count_suffix[:n] > 0
        k = np.clip(valid_count_suffix[:n] - 1, 0, max(running_pos_rev.size - 1, 0))
        clean_suffix = np.where(has_any, running_pos_rev[k], -1)
    else:
        clean_suffix = np.full(n, -1, dtype=np.int64)

    # ELI5: now decide, per row, which answer actually applies.
    # `frozen_mask[s]` = "does [s, n) contain any non-null value at
    # all?" (next_valid[s] < n). If so, check whether THAT row's own
    # anchor (its first non-null value) happens to be NaN -- if it is,
    # the frozen-NaN answer wins; otherwise fall back to the shared
    # "clean" scan computed above.
    frozen_mask = next_valid < n
    anchor_isnan = np.zeros(n, dtype=bool)
    if np.issubdtype(arr.dtype, np.floating):
        valid_next = next_valid[frozen_mask]
        anchor_isnan[frozen_mask] = np.isnan(arr[valid_next])

    result[:n] = np.where(
        ~frozen_mask, -1, np.where(anchor_isnan, next_valid, clean_suffix)
    )
    return result


def _sum_starts(
    arr: np.ndarray,
    starts: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute sum
    """
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
    dtype_name = arr.dtype.name
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
    dtype_name = arr.dtype.name
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
    Compute min.

    ELI5: for a handful of narrow ranges, just ask Rust to scan each one
    directly. For lots of wide/overlapping ranges, it's cheaper to walk
    the whole array once (`_suffix_argext`) and look up every row's answer.
    """
    dtype_name = arr.dtype.name
    if dtype_name in _ARGEXT_DTYPE_NAMES and starts.size > _ARGEXT_WORK_FACTOR:
        # Each row's Rust scan covers (arr.size - start) elements; sum
        # that over every row without a Python loop by distributing:
        # sum(n - start) == n*len(starts) - sum(starts).
        total_width = (arr.size * starts.size) - starts.sum(dtype=np.int64)
        if _use_argext(arr_size=arr.size, total_width=total_width):
            return _suffix_argext(arr=arr, booleans=booleans, is_max=False)[starts]
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
    Compute min.

    ELI5: same trade-off as `_min_starts`, mirrored for prefixes -- Rust
    for a few narrow ranges, `_prefix_argext`'s one-pass precompute once
    there are enough wide/overlapping ones to make it worth it.
    """
    dtype_name = arr.dtype.name
    if dtype_name in _ARGEXT_DTYPE_NAMES and ends.size > _ARGEXT_WORK_FACTOR:
        # Each row's Rust scan covers `end` elements (range [0, end)),
        # so the total is just the sum of all the requested ends.
        total_width = int(ends.sum(dtype=np.int64))
        if _use_argext(arr_size=arr.size, total_width=total_width):
            return _prefix_argext(arr=arr, booleans=booleans, is_max=False)[ends]
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
    Compute max.

    ELI5: same trade-off as `_min_starts`, just tracking the running
    maximum instead of the minimum.
    """
    dtype_name = arr.dtype.name
    if dtype_name in _ARGEXT_DTYPE_NAMES and starts.size > _ARGEXT_WORK_FACTOR:
        # sum(n - start) == n*len(starts) - sum(starts); see _min_starts.
        total_width = (arr.size * starts.size) - starts.sum(dtype=np.int64)
        if _use_argext(arr_size=arr.size, total_width=total_width):
            return _suffix_argext(arr=arr, booleans=booleans, is_max=True)[starts]
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
    Compute max.

    ELI5: same trade-off as `_min_ends`, just tracking the running
    maximum instead of the minimum.
    """
    dtype_name = arr.dtype.name
    if dtype_name in _ARGEXT_DTYPE_NAMES and ends.size > _ARGEXT_WORK_FACTOR:
        # sum of the requested ends; see _min_ends.
        total_width = int(ends.sum(dtype=np.int64))
        if _use_argext(arr_size=arr.size, total_width=total_width):
            return _prefix_argext(arr=arr, booleans=booleans, is_max=True)[ends]
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
    dtype_name = arr.dtype.name
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


def _sum_starts_ends(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute sum
    """
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
    dtype_name = arr.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(arr=arr, starts=starts, ends=ends, booleans=booleans)


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
