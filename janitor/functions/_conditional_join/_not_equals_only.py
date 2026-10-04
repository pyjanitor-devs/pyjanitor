"""Python boundary for joins whose predicates are all ``!=``.

This module deliberately owns the complete Python-side boundary for the
dedicated all-not-equal implementation in ``janitor-rs/src/not_equals_only.rs``.
It does not use the mixed non-equi kernels.  The split is important because
``!=`` has null semantics that cannot be represented by an ordinary binary
search window:

* non-null values are compacted for the sorted right-side search;
* every compact value retains its original physical position;
* null positions are kept in a separate partition;
* NumPy-backed nulls and pandas extension-array nulls have different matching
  behavior; and
* residual ``!=`` predicates must address the complete physical layouts.

The first predicate is the anchor.  Its Python tuple is kept in the natural
call order used by this module::

    (
        left_values,
        left_full_positions,
        right_values,
        right_full_positions,
        comparator,
        left_non_null_positions,
        left_null_positions,
        right_non_null_positions,
        right_null_positions,
        is_extension_array,
    )

The extended Rust ABI uses a different historical order for the ten-element
anchor.  The call sites below adapt the natural Python tuple explicitly before
crossing that ABI.  This keeps the Python code readable while making the Rust
boundary visible and auditable.

``anchor_dtype`` selects the dtype-specialized Rust kernel for the comparison
anchor.  It does not select individual aggregation columns: one selected Rust
kernel processes the complete aggregation request list.

The two anchor columns must convert to the same NumPy dtype because the Rust
entry point is monomorphized over one comparison type. The extension-array
flag is read from the left anchor column because the two anchor columns share
the same dtype.

## Python-to-Rust handoff map

The two repositories intentionally divide responsibilities at the PyO3
boundary:

* Python owns pandas concerns: extracting columns, resetting the public index
  to physical row positions, detecting nulls, compacting non-null values,
  sorting the right anchor, and preparing aggregation requests.
* Rust owns typed traversal: validating position partitions, binary-searching
  the sorted right values, applying null-aware ``!=`` semantics, evaluating
  residual predicates, applying ``keep``, and updating aggregation state.

The direct index call maps the natural anchor fields to the Rust function
arguments individually. The extended index and aggregation calls pass a list
whose first element is adapted to Rust's ABI order. This explicit adaptation
is preferable to relying on undocumented tuple positions because the Python
tuple is optimized for readability while the Rust tuple is optimized for
stable parsing.

The Rust function selected by ``anchor_dtype`` is specialized only for the
anchor comparison type. A call may still contain many aggregation requests;
Rust parses and executes that request list inside one typed traversal.
"""

from __future__ import annotations

from dataclasses import dataclass

import janitor_rs
import numpy as np
import pandas as pd

from janitor.functions._conditional_join import _helpers
from janitor.functions._conditional_join._aggregation_helpers import (
    _aggregation_inputs,
    _empty_aggregation_result,
    _materialize_aggregation_result,
    _unmatched_aggregation_result,
)


@dataclass(frozen=True)
class _NotEqualsAnchor:
    """Named Python representation of the prepared ``!=`` anchor.

    The named fields are used throughout Python so a change to the Rust ABI
    cannot silently change the meaning of a numeric tuple offset. The
    ``to_rust_tuple`` method is the only place that constructs the historical
    ten-field tuple consumed by Rust's extended APIs.
    """

    left_values: np.ndarray | None
    left_full_positions: np.ndarray
    right_values: np.ndarray | None
    right_full_positions: np.ndarray
    comparator: str
    left_non_null_positions: np.ndarray | None
    left_null_positions: np.ndarray | None
    right_non_null_positions: np.ndarray | None
    right_null_positions: np.ndarray | None
    is_extension_array: bool

    def to_rust_tuple(self) -> tuple:
        """Return the stable ten-field extended Rust anchor ABI."""
        return (
            self.left_values,
            self.left_full_positions,
            self.left_non_null_positions,
            self.left_null_positions,
            self.right_values,
            self.right_full_positions,
            self.right_non_null_positions,
            self.right_null_positions,
            self.is_extension_array,
            self.comparator,
        )


_NOT_EQUALS_INDEX_FUNCTIONS = {
    "int64": janitor_rs.single_join_indices_int64,
    "int32": janitor_rs.single_join_indices_int32,
    "int16": janitor_rs.single_join_indices_int16,
    "int8": janitor_rs.single_join_indices_int8,
    "uint64": janitor_rs.single_join_indices_uint64,
    "uint32": janitor_rs.single_join_indices_uint32,
    "uint16": janitor_rs.single_join_indices_uint16,
    "uint8": janitor_rs.single_join_indices_uint8,
    "float64": janitor_rs.single_join_indices_f64,
    "float32": janitor_rs.single_join_indices_f32,
}

_NOT_EQUALS_EXTENDED_INDEX_FUNCTIONS = {
    "int64": janitor_rs.single_join_extended_indices_int64,
    "int32": janitor_rs.single_join_extended_indices_int32,
    "int16": janitor_rs.single_join_extended_indices_int16,
    "int8": janitor_rs.single_join_extended_indices_int8,
    "uint64": janitor_rs.single_join_extended_indices_uint64,
    "uint32": janitor_rs.single_join_extended_indices_uint32,
    "uint16": janitor_rs.single_join_extended_indices_uint16,
    "uint8": janitor_rs.single_join_extended_indices_uint8,
    "float64": janitor_rs.single_join_extended_indices_f64,
    "float32": janitor_rs.single_join_extended_indices_f32,
}

_NOT_EQUALS_AGGREGATE_FUNCTIONS = {
    "int64": janitor_rs.not_equals_aggregate_int64,
    "int32": janitor_rs.not_equals_aggregate_int32,
    "int16": janitor_rs.not_equals_aggregate_int16,
    "int8": janitor_rs.not_equals_aggregate_int8,
    "uint64": janitor_rs.not_equals_aggregate_uint64,
    "uint32": janitor_rs.not_equals_aggregate_uint32,
    "uint16": janitor_rs.not_equals_aggregate_uint16,
    "uint8": janitor_rs.not_equals_aggregate_uint8,
    "float64": janitor_rs.not_equals_aggregate_f64,
    "float32": janitor_rs.not_equals_aggregate_f32,
}

_NOT_EQUALS_REVERSE_AGGREGATE_FUNCTIONS = {
    "int64": janitor_rs.not_equals_aggregate_reverse_int64,
    "int32": janitor_rs.not_equals_aggregate_reverse_int32,
    "int16": janitor_rs.not_equals_aggregate_reverse_int16,
    "int8": janitor_rs.not_equals_aggregate_reverse_int8,
    "uint64": janitor_rs.not_equals_aggregate_reverse_uint64,
    "uint32": janitor_rs.not_equals_aggregate_reverse_uint32,
    "uint16": janitor_rs.not_equals_aggregate_reverse_uint16,
    "uint8": janitor_rs.not_equals_aggregate_reverse_uint8,
    "float64": janitor_rs.not_equals_aggregate_reverse_f64,
    "float32": janitor_rs.not_equals_aggregate_reverse_f32,
}

_NOT_EQUALS_EXTENDED_AGGREGATE_FUNCTIONS = {
    "int64": janitor_rs.not_equals_extended_aggregate_int64,
    "int32": janitor_rs.not_equals_extended_aggregate_int32,
    "int16": janitor_rs.not_equals_extended_aggregate_int16,
    "int8": janitor_rs.not_equals_extended_aggregate_int8,
    "uint64": janitor_rs.not_equals_extended_aggregate_uint64,
    "uint32": janitor_rs.not_equals_extended_aggregate_uint32,
    "uint16": janitor_rs.not_equals_extended_aggregate_uint16,
    "uint8": janitor_rs.not_equals_extended_aggregate_uint8,
    "float64": janitor_rs.not_equals_extended_aggregate_f64,
    "float32": janitor_rs.not_equals_extended_aggregate_f32,
}

_NOT_EQUALS_REVERSE_EXTENDED_AGGREGATE_FUNCTIONS = {
    "int64": janitor_rs.not_equals_extended_aggregate_reverse_int64,
    "int32": janitor_rs.not_equals_extended_aggregate_reverse_int32,
    "int16": janitor_rs.not_equals_extended_aggregate_reverse_int16,
    "int8": janitor_rs.not_equals_extended_aggregate_reverse_int8,
    "uint64": janitor_rs.not_equals_extended_aggregate_reverse_uint64,
    "uint32": janitor_rs.not_equals_extended_aggregate_reverse_uint32,
    "uint16": janitor_rs.not_equals_extended_aggregate_reverse_uint16,
    "uint8": janitor_rs.not_equals_extended_aggregate_reverse_uint8,
    "float64": janitor_rs.not_equals_extended_aggregate_reverse_f64,
    "float32": janitor_rs.not_equals_extended_aggregate_reverse_f32,
}


def _get_not_equals_function(
    functions: dict[str, object],
    dtype: np.dtype,
) -> object:
    """Return one dtype-specialized Rust ``!=`` function.

    Args:
        functions: One of the explicit registries declared in this module.
        dtype: NumPy dtype of the non-null anchor values.  The Rust extension
            exposes one PyO3 function per supported dtype because its generic
            comparison type is monomorphized at compile time.

    Returns:
        The Rust callable for ``dtype``.

    Raises:
        KeyError: If the dtype has no corresponding ``not_equals_only.rs``
            export.
    """
    dtype_name = dtype.name
    try:
        return functions[dtype_name]
    except KeyError as err:
        raise KeyError(
            f"Unsupported not-equals function or dtype: {dtype_name!r}"
        ) from err


def _preparatory_work(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list[tuple[str, str, str]],
):
    """Prepare physical and compact layouts for the all-``!=`` Rust API.

    The dataframe indexes have already been reset by ``conditional_join`` and
    therefore represent physical row positions.  The full-position arrays are
    retained before null filtering.  Non-null values are compacted, and the
    corresponding non-null position arrays map each compact offset back to a
    full physical position.  Only the right non-null values are sorted because
    Rust performs two partition-point searches over that array.

    Residual predicates are built from the full, unsorted dataframe layouts.
    They therefore continue to index physical positions rather than compact or
    sorted offsets. The first predicate is the named Python anchor object; it
    is converted to the ten-field Rust tuple only at the ABI boundary.
    Residual predicates follow in their original order.

    Args:
        df: Reset-index left dataframe.
        right: Reset-index right dataframe.
        conditions: One or more validated ``(left_column, right_column,
            operator)`` predicates.  The caller routes only all-``!=`` sets
            here; Rust validates the comparator again at the boundary.

    Returns:
        ``None`` when either input has no rows.  Otherwise returns the two
        dataframes, the anchor-plus-residual predicate list, and the converted
        dtype used to select the Rust anchor kernel.
    """
    if df.empty or right.empty:
        return None

    anchor_condition, *rest = conditions
    left_column = anchor_condition.left
    right_column = anchor_condition.right
    op = anchor_condition.op

    left_column = df[left_column]
    right_column = right[right_column]
    anchor_dtype = _helpers._convert_array_to_numpy(array=right_column._values).dtype
    # The same-dtype contract lets Rust use one extension flag for the pair;
    # read it from the left anchor consistently.
    is_extension_array = pd.api.types.is_extension_array_dtype(left_column.dtype)
    left_index = _helpers._convert_array_to_numpy(array=left_column.index._values)
    booleans = left_column.isna()
    left_non_null_positions = None
    if booleans.all():
        # An all-null side has no compact value array. Its null partition is
        # still required so Rust can validate the complete physical layout.
        left_null_positions = _helpers._convert_array_to_numpy(
            array=left_column.index._values
        )
        left_column = None
    elif booleans.any():
        # Keep the physical positions before compacting the values. The
        # compact offset and physical position are intentionally different.
        left_null_positions = left_column.index[booleans]._values
        left_null_positions = _helpers._convert_array_to_numpy(
            array=left_null_positions
        )
        left_column = left_column[~booleans]
        left_non_null_positions = _helpers._convert_array_to_numpy(
            array=left_column.index._values
        )
        left_column = _helpers._convert_array_to_numpy(array=left_column.values)
    else:
        left_null_positions = None
        left_non_null_positions = _helpers._convert_array_to_numpy(
            array=left_column.index._values
        )
        left_column = _helpers._convert_array_to_numpy(array=left_column.values)

    right_index = _helpers._convert_array_to_numpy(array=right_column.index._values)
    booleans = right_column.isna()
    right_non_null_positions = None
    if booleans.all():
        # Preserve the all-null right partition; the extension flag remains
        # true when this is a pandas nullable dtype.
        right_null_positions = _helpers._convert_array_to_numpy(
            array=right_column.index._values
        )
        right_column = None
    elif booleans.any():
        right_null_positions = right_column.index[booleans]._values
        right_null_positions = _helpers._convert_array_to_numpy(
            array=right_null_positions
        )
        right_column = right_column[~booleans]
    else:
        right_null_positions = None
    if right_column is not None:
        # Only the right anchor is sorted. Its index remains the original
        # physical position map, so residual arrays never see sorted offsets.
        right_column, _ = _helpers._sort_if_not_monotonic(series=right_column)
        right_non_null_positions = _helpers._convert_array_to_numpy(
            array=right_column.index._values
        )
        right_column = _helpers._convert_array_to_numpy(array=right_column.values)
    booleans = None
    anchor_predicate = _NotEqualsAnchor(
        left_values=left_column,
        left_full_positions=left_index,
        right_values=right_column,
        right_full_positions=right_index,
        comparator=op,
        left_non_null_positions=left_non_null_positions,
        left_null_positions=left_null_positions,
        right_non_null_positions=right_non_null_positions,
        right_null_positions=right_null_positions,
        is_extension_array=is_extension_array,
    )

    residual_predicates = []
    for condition in rest:
        residual_predicate = _helpers._build_residual_predicate(
            left=df[condition.left],
            right=right[condition.right],
            operation=condition.op,
            left_index=slice(None),
            right_index=slice(None),
        )
        residual_predicates.append(residual_predicate)

    return anchor_predicate, residual_predicates, anchor_dtype


def _compute_not_equals_join(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list[tuple[str, str, str]],
    keep: str,
    *,
    how: str = "inner",
    df_columns=slice(None),
    right_columns=slice(None),
    indicator: bool | str = False,
    include_join_positions: bool = False,
    return_matching_indices: bool = True,
    return_building_blocks: bool = False,
) -> dict[str, np.ndarray] | pd.DataFrame:
    """Generate all-``!=`` join indices through the Rust boundary.

    A single predicate uses the direct index kernel.  Two or more predicates
    use the extended kernel, which traverses anchor candidates and evaluates
    every residual predicate before applying ``keep``.  For the extended call,
    the natural Python anchor is converted to Rust's ten-field ABI order.

    Args:
        df: Reset-index left dataframe.
        right: Reset-index right dataframe.
        conditions: Complete all-``!=`` predicate list.
        keep: ``"all"``, ``"any"``, ``"first"``, or ``"last"``.
        how: Join shape used when materializing dataframe output.
        df_columns: Left columns to retain when materializing output.
        right_columns: Right columns to retain when materializing output.
        indicator: Whether to add the merge indicator column.
        include_join_positions: Whether to include physical pair positions in
            the materialized result index.
        return_matching_indices: Return physical index arrays instead of a
            materialized dataframe.
        return_building_blocks: Preserve index/building-block output.

    Returns:
        A physical-index dictionary for index callers, or a materialized
        dataframe when ``return_matching_indices`` is false. A no-match result
        is normalized to the corresponding empty form.
    """

    outcome = _preparatory_work(df=df, right=right, conditions=conditions)
    if outcome is None:
        result = _helpers._empty_indices()
        return _helpers._materialize_or_return_indices(
            result=result,
            df=df,
            right=right,
            how=how,
            df_columns=df_columns,
            right_columns=right_columns,
            indicator=indicator,
            include_join_positions=include_join_positions,
            return_matching_indices=return_matching_indices,
            return_building_blocks=return_building_blocks,
        )

    anchor_predicate, residual_predicates, anchor_dtype = outcome

    if len(residual_predicates) == 0:
        function = _get_not_equals_function(
            _NOT_EQUALS_INDEX_FUNCTIONS,
            anchor_dtype,
        )
        result = function(
            left=anchor_predicate.left_values,
            left_full_positions=anchor_predicate.left_full_positions,
            right=anchor_predicate.right_values,
            right_full_positions=anchor_predicate.right_full_positions,
            comparator=anchor_predicate.comparator,
            keep=keep,
            left_non_null_positions=anchor_predicate.left_non_null_positions,
            left_null_positions=anchor_predicate.left_null_positions,
            right_non_null_positions=anchor_predicate.right_non_null_positions,
            right_null_positions=anchor_predicate.right_null_positions,
            is_extension_array=anchor_predicate.is_extension_array,
        )
    else:
        function = _get_not_equals_function(
            _NOT_EQUALS_EXTENDED_INDEX_FUNCTIONS,
            anchor_dtype,
        )
        # The Python tuple is readable/natural; the Rust extended ABI keeps
        # its historical fields grouped by side and partition.
        rust_anchor = anchor_predicate.to_rust_tuple()
        result = function([rust_anchor, *residual_predicates], keep)

    result = _helpers._empty_indices() if result is None else result
    return _helpers._materialize_or_return_indices(
        result=result,
        df=df,
        right=right,
        how=how,
        df_columns=df_columns,
        right_columns=right_columns,
        indicator=indicator,
        include_join_positions=include_join_positions,
        return_matching_indices=return_matching_indices,
        return_building_blocks=return_building_blocks,
    )


def _aggregate(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list[tuple[str, str, str]],
    aggfunc: list[tuple],
    return_matched: bool,
    reverse: bool,
) -> pd.DataFrame:
    """Aggregate all-``!=`` matches without materializing candidate pairs.

    Single-predicate aggregation calls the direct forward or reverse Rust
    aggregation export.  Multiple all-``!=`` predicates call the extended
    export, which applies residual predicates inside the same traversal.
    Aggregation inputs are prepared once for the complete request list; the
    anchor dtype only selects the comparison kernel.

    Args:
        df: Reset-index left dataframe.
        right: Reset-index right dataframe.
        conditions: Complete all-``!=`` predicate list.
        aggfunc: ``(column, operation)`` aggregation requests.
        return_matched: Include the matched level in the materialized output.
            A no-match all-``!=`` result uses the shared empty-result schema.
        reverse: Aggregate left-side source values into right-side output
            rows when true; otherwise aggregate right-side values into left
            output rows.

    Returns:
        A pandas dataframe using the shared conditional-join aggregation
        contract. If Rust finds no surviving all-``!=`` pair, every eligible
        output row is retained with identity values and a false matched flag
        when ``return_matched`` is true.
    """

    aggregation_source = df if reverse else right
    outcome = _preparatory_work(df=df, right=right, conditions=conditions)
    if outcome is None:
        return _empty_aggregation_result(
            source=aggregation_source,
            aggfunc=aggfunc,
            return_matched=return_matched,
        )

    anchor_predicate, residual_predicates, anchor_dtype = outcome
    output_index = right.index if reverse else df.index

    aggregation_inputs = _aggregation_inputs(
        source=aggregation_source,
        aggfunc=aggfunc,
    )
    if len(residual_predicates) == 0:
        registry = (
            _NOT_EQUALS_REVERSE_AGGREGATE_FUNCTIONS
            if reverse
            else _NOT_EQUALS_AGGREGATE_FUNCTIONS
        )
        function = _get_not_equals_function(registry, anchor_dtype)
        result = function(
            left=anchor_predicate.left_values,
            left_full_positions=anchor_predicate.left_full_positions,
            right=anchor_predicate.right_values,
            right_full_positions=anchor_predicate.right_full_positions,
            comparator=anchor_predicate.comparator,
            left_non_null_positions=anchor_predicate.left_non_null_positions,
            left_null_positions=anchor_predicate.left_null_positions,
            right_non_null_positions=anchor_predicate.right_non_null_positions,
            right_null_positions=anchor_predicate.right_null_positions,
            is_extension_array=anchor_predicate.is_extension_array,
            aggregations=aggregation_inputs,
            return_matched=return_matched,
        )
    else:
        registry = (
            _NOT_EQUALS_REVERSE_EXTENDED_AGGREGATE_FUNCTIONS
            if reverse
            else _NOT_EQUALS_EXTENDED_AGGREGATE_FUNCTIONS
        )
        function = _get_not_equals_function(registry, anchor_dtype)
        result = function(
            [anchor_predicate.to_rust_tuple(), *residual_predicates],
            aggregation_inputs,
            return_matched,
        )

    if result is None:
        return _unmatched_aggregation_result(
            output_index=output_index,
            source=aggregation_source,
            aggfunc=aggfunc,
            return_matched=return_matched,
        )
    # Rust can represent an all-unmatched result as a zero-length result
    # carrying empty accumulator arrays. The prepared output domain still
    # exists, so retain one identity-valued row per eligible output row.
    if len(result[0]) == 0:
        return _unmatched_aggregation_result(
            output_index=output_index,
            source=aggregation_source,
            aggfunc=aggfunc,
            return_matched=return_matched,
        )
    # Rust returns output positions and every result array in public physical
    # row order.  Reverse aggregation therefore needs no Python-side
    # permutation: the right dataframe index already has that same order.
    return _materialize_aggregation_result(
        result=result,
        output_index=output_index,
        source=aggregation_source,
        aggfunc=aggfunc,
        return_matched=return_matched,
    )
