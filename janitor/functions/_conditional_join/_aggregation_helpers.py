"""Shared Python-side helpers for Rust conditional-join aggregations.

The Rust aggregation kernels operate on positional NumPy arrays and return a
common result shape::

    (matched, aggregation_arrays)

``matched`` is a boolean array with one slot per output row. The aggregation
arrays have the same physical output length, but callers should use
``matched`` to discard slots that received no comparison. These helpers bridge
that positional representation back to pandas while preserving column labels,
extension dtypes, null behavior, and the requested output index.

The helpers deliberately receive complete physical-layout aggregation arrays.
Predicate arrays may be filtered or sorted before they reach Rust, but the
aggregation source arrays must remain in the original layout so Rust position
updates address the correct rows.
"""

from typing import Hashable

import numpy as np
import pandas as pd

from janitor.functions._conditional_join._helpers import _convert_array_to_numpy


def _build_agg_label(column_name: Hashable, agg_name: str):
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


def _aggregation_inputs(source: pd.DataFrame, aggfunc: list[tuple]) -> list[tuple]:
    """Prepare aggregation requests for the Rust input contract.

    Each Python request is converted to ``(values, null_mask, operation)``.
    The values and masks retain the complete physical row layout. The null
    mask is authoritative: Rust does not infer nullness from values such as
    ``NaN``. This is important for pandas extension arrays, whose missing
    values may not have the same representation as NumPy missing values.

    Args:
        source: Dataframe containing the aggregation columns in their original
            physical row order.
        aggfunc: Non-empty ``(column_name, operation)`` requests. Supported
            operations are interpreted by the Rust aggregation parser, for
            example ``"sum"``, ``"count"``, ``"size"``, ``"prod"``,
            ``"min"``, and ``"max"``.

    Returns:
        A list of Rust-facing ``(values, null_mask, operation)`` tuples in the
        same order as ``aggfunc``. Values are NumPy-compatible arrays and
        masks are boolean NumPy arrays with matching lengths.

    Raises:
        KeyError: If a requested column is not present in ``source``.
    """
    result = []
    for column_name, operation in aggfunc:
        series = source[column_name]
        # Keep both arrays in the full physical layout. The Rust kernel uses
        # the position supplied by the predicate traversal to index them.
        result.append(
            (
                _convert_array_to_numpy(array=series._values),
                series.isna().to_numpy(dtype=bool),
                operation,
            )
        )
    return result


def _empty_aggregation_result(
    source: pd.DataFrame, aggfunc: list[tuple]
) -> pd.DataFrame:
    """Build the pandas result for a join with no surviving pairs.

    This path is used before calling Rust when a filtered predicate side is
    empty, and when Rust returns ``None`` because no candidate survives. It
    still creates every requested output column so callers receive a stable
    schema rather than an untyped or column-less dataframe.

    Args:
        source: Dataframe supplying the requested columns and their dtypes.
        aggfunc: ``(column_name, operation)`` requests whose output columns
            must be represented in the empty result.

    Returns:
        An empty dataframe with one column per aggregation request. ``size``
        uses ``int64``; other operations preserve the source column dtype.

    Raises:
        KeyError: If a requested column is not present in ``source``.
        TypeError: If pandas cannot construct an empty array for a requested
            source dtype.
    """
    result = {}
    for column_name, operation in aggfunc:
        dtype = "int64" if operation == "size" else source[column_name].dtype
        result[_build_agg_label(column_name, operation)] = pd.array([], dtype=dtype)
    return pd.DataFrame(result, copy=False)


def _materialize_aggregation_result(
    result,
    output_index: pd.Index,
    source: pd.DataFrame,
    aggfunc: list[tuple],
) -> pd.DataFrame:
    """Convert the common Rust aggregation result into a pandas dataframe.

    Rust returns one accumulator slot per output position, including rows that
    never matched. The first tuple item is therefore used as a filter before
    constructing the dataframe index and output columns.

    ``min`` and ``max`` use ``-1`` as Rust's internal no-value sentinel. The
    sentinel cannot be passed directly to ``Series.iloc`` because it would
    select the final row, so this function first replaces it with a safe
    position and then restores those entries to ``pd.NA``. ``sum`` and
    ``prod`` are explicitly reconstructed with an extension dtype when the
    source column uses one.

    Args:
        result: Rust return value, either ``None`` or ``(matched, arrays)``.
            ``matched`` must be boolean-compatible, and ``arrays`` must have
            one array per request in ``aggfunc``.
        output_index: Full output-domain index. It is left-aligned for
            forward aggregation and right-aligned for reverse aggregation.
        source: Dataframe containing the source columns and their original
            pandas dtypes. This is also used to resolve ``min``/``max`` row
            positions back to values.
        aggfunc: Requests in the same order used to create ``result[1]``.

    Returns:
        A pandas dataframe containing only matched output rows, indexed by
        ``output_index[matched]``. Columns use the labels produced by
        :func:`_build_agg_label`.

    Raises:
        IndexError: If Rust returns fewer aggregation arrays than requests or
            an invalid ``min``/``max`` position.
        KeyError: If a requested source column is missing.
        TypeError: If a returned array cannot be converted to NumPy or an
            extension dtype cannot be reconstructed.
    """
    if result is None:
        return _empty_aggregation_result(source, aggfunc)

    # The matched mask is the authoritative indication of which accumulator
    # slots correspond to actual join results. Aggregation values in unmatched
    # slots may be initialization values and must never reach pandas output.
    matched = np.asarray(result[0], dtype=bool)
    index = output_index[matched]
    arrays = result[1]
    output = {}
    for position, (column_name, operation) in enumerate(aggfunc):
        values = np.asarray(arrays[position])
        if operation == "size":
            output[_build_agg_label(column_name, operation)] = values[matched]
            continue

        series = source[column_name]
        if operation in {"min", "max"}:
            # Rust stores a physical source position for extrema and -1 when
            # a matched group contains no non-null value. Guard the sentinel
            # before positional indexing, then restore it as pandas missing.
            invalid = values == -1
            safe_values = values.copy()
            safe_values[invalid] = 0
            values = series.iloc[safe_values].copy()
            values.iloc[invalid] = pd.NA
            values = values.array
        elif operation in {"sum", "prod"} and pd.api.types.is_extension_array_dtype(
            series.dtype
        ):
            # NumPy conversion can erase pandas' nullable dtype. Rebuild it
            # here so the public aggregation result follows the source dtype.
            values = pd.array(values, dtype=series.dtype)
        output[_build_agg_label(column_name, operation)] = values[matched]
    return pd.DataFrame(output, copy=False, index=index)


def _aggregation_kernel(name: str):
    """Resolve a dtype-specific Rust aggregation entry point.

    Args:
        name: Exact PyO3 function name, such as
            ``"single_join_aggregate_int64"`` or
            ``"single_join_extended_aggregate_reverse_f64"``.

    Returns:
        The callable exported by the installed ``janitor_rs`` extension.

    Raises:
        TypeError: If the extension does not expose an aggregation kernel with
            the requested name. The error is normalized so callers do not
            expose a raw ``AttributeError`` from the extension boundary.
    """
    try:
        import janitor_rs

        return getattr(janitor_rs, name)
    except AttributeError as error:
        raise TypeError(f"Rust aggregation does not support dtype {name}") from error
