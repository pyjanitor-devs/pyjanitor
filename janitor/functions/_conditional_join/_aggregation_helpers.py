"""Shared Python-side helpers for Rust conditional-join aggregations.

The Rust aggregation kernels operate on positional NumPy arrays and return a
common result shape::

    ``(output_positions, matched, aggregation_arrays)`` when matched metadata
    is requested, or ``(output_positions, aggregation_arrays)`` otherwise.

``output_positions`` and ``matched`` have one slot per trimmed output row. The
aggregation arrays have the same trimmed output length. These helpers bridge that
positional representation back to pandas while preserving column labels,
extension dtypes, null behavior, the trimmed output domain, and the matched
flag as a ``MultiIndex`` level.

All range-family callers pass aggregation arrays in the prepared calculation
layout used by their anchor values. That layout may be filtered and, for a
sorted right anchor, value-sorted. Rust uses compact window offsets to read
these aligned arrays and returns physical output positions separately. The
not-equals family has its own split non-null/null layout contract, documented
at its caller, but uses the same result materialization shape.

Numerical contract:
    Signed integer ``sum`` and ``prod`` results use ``int64``; unsigned
    integer results use ``uint64``; ``float32`` results remain ``float32``;
    and ``float64`` results remain ``float64``. ``min`` and ``max`` preserve
    the source dtype where possible; integer results use nullable ``Int64`` or
    ``UInt64`` when an unmatched output requires ``pd.NA``. Position, length,
    and allocation calculations remain checked in Rust.

Layout contract:
    ``indexer`` selects the exact calculation layout sent to Rust. It is not
    an output reorder instruction. For a forward aggregation, source values
    are read from the right-side layout and written to left output slots. For
    a reverse aggregation, source values are read from the left-side layout
    and written to right output slots. The caller must build ``output_index``
    from the same anchor layout so returned positions and pandas rows remain
    aligned.

Sentinel contract:
    Rust extrema kernels return physical source positions and use ``-1`` for
    an output slot with no valid value. This module resolves those positions
    back to values and converts the sentinel to a pandas missing value; it
    must never pass ``-1`` directly to ``Series.iloc``.
"""

from collections.abc import Callable
from typing import Hashable

import numpy as np
import pandas as pd

from janitor.functions._conditional_join._helpers import _convert_array_to_numpy


def _nullable_integer_dtype(dtype):
    """Return the pandas nullable dtype corresponding to an integer dtype.

    NumPy integer dtypes cannot represent ``pd.NA``, so missing integer
    extrema use pandas' nullable ``Int64``/``UInt64`` dtypes. Existing pandas
    integer extension dtypes are preserved, including narrower widths.
    """
    if pd.api.types.is_extension_array_dtype(dtype):
        return dtype
    return "UInt64" if pd.api.types.is_unsigned_integer_dtype(dtype) else "Int64"


def _integer_reduction_dtype(dtype):
    """Return the promoted dtype used by integer sum/product reductions."""
    if pd.api.types.is_extension_array_dtype(dtype):
        return "UInt64" if pd.api.types.is_unsigned_integer_dtype(dtype) else "Int64"
    return "uint64" if pd.api.types.is_unsigned_integer_dtype(dtype) else "int64"


def _build_agg_label(column_name: Hashable, agg_name: str) -> tuple:
    """Build the output label for one aggregation request.

    Tuple-valued column labels retain their levels and receive the aggregation
    name as one additional level. Scalar labels become a two-level tuple.

    Args:
        column_name: Source column label, scalar or tuple-valued.
        agg_name: Aggregation name such as ``"sum"`` or ``"size"``.

    Returns:
        A tuple suitable for use as a pandas column label.

    Examples:
        >>> _build_agg_label("amount", "sum")
        ('amount', 'sum')
        >>> _build_agg_label(("sales", "amount"), "max")
        ('sales', 'amount', 'max')
    """
    if isinstance(column_name, tuple):
        return (*column_name, agg_name)
    return (f"{column_name}", agg_name)


def _select_aggregation_kernel(
    registry: dict[str, tuple[Callable, Callable]],
    dtype: str,
    reverse: bool,
) -> Callable:
    """Select a forward or reverse Rust aggregation kernel.

    Single-predicate and extended-predicate aggregations use separate kernel
    registries because their PyO3 functions accept different predicate
    contracts. Their lookup and direction-selection rules are nevertheless
    identical, so this helper centralizes only that shared dispatch logic.

    Args:
        registry: Mapping from NumPy dtype names to ``(forward, reverse)``
            Rust callables.
        dtype: NumPy dtype name for the anchor predicate, such as ``"int64"``.
        reverse: Select the reverse aggregation kernel when true; otherwise
            select the forward kernel.

    Returns:
        The registered Rust aggregation callable for ``dtype`` and direction.

    Raises:
        TypeError: If no Rust aggregation kernel is registered for ``dtype``.
    """
    try:
        forward_kernel, reverse_kernel = registry[dtype]
    except KeyError as error:
        raise TypeError(f"Rust aggregation does not support dtype {dtype}") from error
    return reverse_kernel if reverse else forward_kernel


def _aggregation_inputs(
    source: pd.DataFrame, aggfunc: list[tuple], indexer: pd.Index | slice = slice(None)
) -> list[tuple]:
    """Prepare aggregation requests for the Rust input contract.

    Numeric value reductions are converted to
    ``(values, null_mask, operation)``. ``count`` is converted to
    ``("*", null_mask, "count")`` because its result depends only on the
    authoritative mask; ``size`` is converted to ``("*", "size")``.
    Arrays and masks use the layout selected by ``indexer``. For ordinary
    single-range and range-first multi-predicate aggregation this is the
    filtered/sorted compact layout. Rust receives physical position maps for
    output labels, but its window offsets index the aligned arrays directly.

    Before crossing the Rust boundary, integer ``sum`` and ``prod`` values
    are promoted to the pandas/NumPy reduction dtype: signed integers become
    ``int64`` and unsigned integers become ``uint64``. Floating values retain
    their input dtype through the Rust kernel. Count and size do not pass
    source values at all.

    Args:
        source: Dataframe containing aggregation columns in the calculation
            layout used by Rust. For ``!=``, this is non-null positions
            followed by null positions; for range joins, it is the filtered
            and, on the right, value-sorted layout.
        aggfunc: Non-empty ``(column_name, operation)`` requests. Supported
            operations are interpreted by the Rust aggregation parser, for
            example ``"sum"``, ``"count"``, ``"size"``, ``"prod"``,
            ``"min"``, and ``"max"``.
        indexer: Pandas indexer selecting the layout exposed to Rust. It must
            be a positional/label selection that produces arrays whose
            positions match the Rust predicate contract.

    Returns:
        A list of Rust-facing aggregation tuples in the same order as
        ``aggfunc``. Numeric value reductions use values and masks; count and
        size use their dtype-independent wildcard forms.

    Raises:
        KeyError: If a requested column is not present in ``source``.
    """
    result = []
    for column_name, operation in aggfunc:
        series = source.loc[indexer, column_name]
        null_mask = series.isna().to_numpy(dtype=bool)
        if operation == "size":
            result.append(("*", "size"))
        elif operation == "count":
            result.append(("*", null_mask, "count"))
        else:
            # Keep values and masks in the same compact layout as the source
            # dataframe. Rust indexes these arrays with local aggregation
            # positions after translating physical != candidates.
            values = _convert_array_to_numpy(array=series._values)
            if operation in {"sum", "prod"}:
                if pd.api.types.is_signed_integer_dtype(values.dtype):
                    values = values.astype(np.int64, copy=False)
                elif pd.api.types.is_unsigned_integer_dtype(values.dtype):
                    values = values.astype(np.uint64, copy=False)
            result.append(
                (
                    values,
                    null_mask,
                    operation,
                )
            )
    return result


def _empty_aggregation_result(
    source: pd.DataFrame, aggfunc: list[tuple], return_matched: bool = False
) -> pd.DataFrame:
    """Build the pandas result for a join with no surviving pairs.

    This path is used before calling Rust when a filtered predicate side is
    empty and no eligible output domain exists. It returns an empty dataframe
    with the requested output schema. A prepared non-empty output domain with
    no surviving pairs uses :func:`_unmatched_aggregation_result` instead.

    Args:
        source: Dataframe supplying the requested columns and their dtypes.
        aggfunc: ``(column_name, operation)`` requests whose output columns
            must be represented in the empty result.
        return_matched: Preserve the standard ``matched`` MultiIndex level
            even though the result contains no rows.

    Returns:
        An empty dataframe with one column per aggregation request.

    Raises:
        KeyError: If a requested column is not present in ``source``.
        TypeError: If pandas cannot construct an empty array for a requested
            source dtype.
    """
    result = {}
    for column_name, operation in aggfunc:
        series = source[column_name]
        if operation in {"count", "size"}:
            dtype = "int64"
        elif operation in {"sum", "prod"} and pd.api.types.is_integer_dtype(
            series.dtype
        ):
            dtype = _integer_reduction_dtype(series.dtype)
        elif operation in {"sum", "prod"} and pd.api.types.is_float_dtype(series.dtype):
            dtype = series.dtype
        else:
            dtype = series.dtype
        result[_build_agg_label(column_name, operation)] = pd.array([], dtype=dtype)
    if return_matched:
        index = pd.MultiIndex.from_arrays(
            [np.array([], dtype=np.intp), np.array([], dtype=bool)],
            names=[None, "matched"],
        )
    else:
        index = pd.RangeIndex(0)
    return pd.DataFrame(result, copy=False, index=index)


def _unmatched_aggregation_result(
    output_index: pd.Index,
    source: pd.DataFrame,
    aggfunc: list[tuple],
    return_matched: bool = False,
) -> pd.DataFrame:
    """Build identity-valued output rows when no eligible pair matches.

    Unlike :func:`_empty_aggregation_result`, this path has a real prepared
    output domain. Rows with null join keys have already been removed by the
    caller; every remaining output row is retained with ``matched=False`` and
    the identity for its requested reduction. ``sum`` and ``size`` use zero,
    ``prod`` uses one, and ``min``/``max`` use missing values.
    """
    length = len(output_index)
    result = {}
    for column_name, operation in aggfunc:
        series = source[column_name]
        if operation in {"count", "size"}:
            values = np.zeros(length, dtype=np.int64)
        elif operation in {"sum", "prod"}:
            identity = 0 if operation == "sum" else 1
            if pd.api.types.is_extension_array_dtype(series.dtype):
                values = pd.array([identity] * length, dtype=series.dtype)
            elif pd.api.types.is_integer_dtype(series.dtype):
                values = np.full(
                    length,
                    identity,
                    dtype=_integer_reduction_dtype(series.dtype),
                )
            else:
                values = np.full(length, identity, dtype=series.dtype)
        elif pd.api.types.is_integer_dtype(series.dtype):
            values = pd.array(
                [pd.NA] * length,
                dtype=_nullable_integer_dtype(series.dtype),
            )
        elif pd.api.types.is_datetime64_any_dtype(
            series.dtype
        ) or pd.api.types.is_timedelta64_dtype(series.dtype):
            values = pd.array([pd.NaT] * length, dtype=series.dtype)
        elif pd.api.types.is_float_dtype(series.dtype):
            values = np.full(length, np.nan, dtype=series.dtype)
        elif pd.api.types.is_extension_array_dtype(series.dtype):
            values = pd.array([pd.NA] * length, dtype=series.dtype)
        else:
            values = pd.array([pd.NA] * length, dtype=series.dtype)
        result[_build_agg_label(column_name, operation)] = values

    if return_matched:
        index = pd.MultiIndex.from_arrays(
            [output_index, np.zeros(length, dtype=bool)],
            names=[output_index.name, "matched"],
        )
    else:
        index = output_index
    return pd.DataFrame(result, copy=False, index=index)


def _materialize_aggregation_result(
    result,
    output_index: pd.Index,
    source: pd.DataFrame,
    aggfunc: list[tuple],
    return_matched: bool,
    source_index: pd.Index | slice = slice(None),
) -> pd.DataFrame:
    """Convert the common Rust aggregation result into a pandas dataframe.

    Rust returns one accumulator slot per output position, including output
    rows that never matched. The output uses the prepared compact index for
    the range-family call. The returned positions identify that exact output
    layout and are validated; they are never used to reorder the aggregation
    arrays.

    ``min`` and ``max`` use ``-1`` as Rust's internal no-value sentinel. The
    sentinel cannot be passed directly to ``Series.iloc`` because it would
    select the final row, so this function first replaces it with a safe
    position and then restores those entries to ``pd.NA``. Integer extrema
    use nullable ``Int64``/``UInt64`` arrays when necessary; datetime and
    timedelta extrema retain their pandas dtypes. ``sum`` and ``prod`` keep
    integer identities as integers rather than promoting them to float.

    Args:
        result: Rust return value, either ``None`` or a tuple containing
            ``output_positions`` and aggregation arrays, with ``matched``
            between them when requested. Positions are ``int64`` labels in
            the prepared trimmed layout and must be in the same order as
            ``output_index``. The arrays must contain one result per request
            in ``aggfunc``.
        output_index: Already-prepared output index. It is left-aligned for
            forward aggregation and right-aligned for reverse aggregation.
        source: Dataframe containing the source columns and their original
            pandas dtypes. This is also used to resolve ``min``/``max`` row
            positions back to values.
        aggfunc: Requests in the same order used to create ``result[2]``.
        return_matched: Whether ``result`` contains the matched mask between
            output positions and aggregation arrays.
        source_index: Selection used to recover source dtype and values for
            ``min``/``max``. It matches the source layout sent to Rust.

    Returns:
        A pandas dataframe containing the trimmed output domain. Its index is
        a two-level ``MultiIndex`` when ``return_matched`` is true; otherwise
        it is the original plain output index. Columns use the labels produced
        by :func:`_build_agg_label`.

    Raises:
        IndexError: If Rust returns fewer aggregation arrays than requests or
            an invalid ``min``/``max`` position.
        KeyError: If a requested source column is missing.
        TypeError: If a returned array cannot be converted to NumPy or an
            extension dtype cannot be reconstructed.
    """
    if result is None:
        return _unmatched_aggregation_result(
            output_index=output_index,
            source=source,
            aggfunc=aggfunc,
            return_matched=return_matched,
        )

    # Rust has calculated directly in the trimmed layout. Positions are an
    # explicit cross-language alignment check; they must match the prepared
    # index values exactly and must not be used to reorder the result arrays.
    output_positions = np.asarray(result[0], dtype=np.int64)
    arrays_position = 2 if return_matched else 1
    matched = np.asarray(result[1], dtype=bool) if return_matched else None
    arrays = result[arrays_position]
    if len(output_positions) != len(output_index):
        raise ValueError("aggregation output positions do not match output index")
    expected_positions = np.asarray(output_index, dtype=np.int64)
    if not np.array_equal(output_positions, expected_positions):
        raise ValueError("aggregation output positions drifted from output index")
    if any(len(values) != len(output_positions) for values in arrays):
        raise ValueError("aggregation arrays do not match output positions")
    if return_matched:
        if len(matched) != len(output_index):
            raise ValueError("aggregation matched mask does not match output index")
        index = pd.MultiIndex.from_arrays(
            [output_index, matched],
            names=[output_index.name, "matched"],
        )
    else:
        index = output_index
    output = {}
    for position, (column_name, operation) in enumerate(aggfunc):
        values = np.asarray(arrays[position])
        if operation in {"count", "size"}:
            output[_build_agg_label(column_name, operation)] = values
            continue
        series = source.loc[source_index, column_name]
        if operation in {"sum", "prod"}:
            # Extension arrays are not limited to nullable floats; nullable
            # integer dtypes also enter this path. Integer reductions already
            # arrive promoted to int64/uint64, so preserve that promoted
            # result rather than narrowing back to Int8, Int16, etc. Float
            # reductions retain their source float dtype and need an explicit
            # extension-array construction only for nullable float columns.
            if pd.api.types.is_extension_array_dtype(series.dtype):
                if pd.api.types.is_float_dtype(series.dtype):
                    values = pd.array(values, dtype=series.dtype)
                elif pd.api.types.is_unsigned_integer_dtype(series.dtype):
                    values = pd.array(values, dtype="UInt64")
                elif pd.api.types.is_integer_dtype(series.dtype):
                    values = pd.array(values, dtype="Int64")
            elif pd.api.types.is_float_dtype(series.dtype):
                values = values.astype(series.dtype, copy=False)
        if operation in {"min", "max"}:
            # Rust stores a physical source position for extrema and -1 when
            # an output row contains no non-null value. Replace the sentinel
            # only for the positional lookup, then apply one full-height
            # `where` mask so the result remains aligned with `output_index`.
            invalid = values == -1
            if series.empty:
                # There is no safe physical position to use when the source
                # side itself is empty. Integer dtypes need pandas' nullable
                # representation because NumPy integer arrays cannot hold
                # the missing extrema marker.
                if pd.api.types.is_integer_dtype(series.dtype):
                    values = pd.array(
                        [pd.NA] * len(values),
                        dtype=_nullable_integer_dtype(series.dtype),
                    )
                elif pd.api.types.is_extension_array_dtype(series.dtype):
                    values = pd.array([pd.NA] * len(values), dtype=series.dtype)
                else:
                    values = np.full(len(values), np.nan)
            else:
                safe_positions = np.where(invalid, 0, values)
                selected = series.iloc[safe_positions]
                if pd.api.types.is_integer_dtype(series.dtype):
                    if invalid.any():
                        values = pd.array(
                            selected, dtype=_nullable_integer_dtype(series.dtype)
                        )
                        values[invalid] = pd.NA
                    elif pd.api.types.is_extension_array_dtype(series.dtype):
                        values = pd.array(selected, dtype=series.dtype)
                    else:
                        values = selected.to_numpy(copy=False)
                else:
                    values = selected.where(~invalid, pd.NA).array
        output[_build_agg_label(column_name, operation)] = values
    return pd.DataFrame(output, copy=False, index=index)
