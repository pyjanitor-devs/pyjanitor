"""Preparation and dispatch for dual-range conditional joins.

This module owns every path whose first two predicates are range predicates:

* exactly two range predicates use the basic Rust range kernel;
* two compatible range predicates followed by residual predicates use the
  range-extended Rust kernel;
* range-led aggregations use the corresponding range aggregation kernels.

The first two range predicates are called the *dual-range anchors*. PyJanitor
removes null rows from non-``!=`` columns, chooses an anchor pair, and places
both right-hand arrays in one ascending physical layout. Rust then performs
the binary searches, intersects the two half-open windows, and evaluates any
remaining predicates. This module never sorts inside Rust and never treats a
second range predicate as an ordinary residual when the dual-range contract
has been selected.

The neighboring ``_single_non_equi_join_extended`` module handles a single
range anchor plus residual predicates and all-``!=`` candidate streams. Keeping
these responsibilities separate is important: a single-anchor residual path
may use an arbitrary later predicate, while this module may use two sorted
right arrays to build one intersected window.
"""

from __future__ import annotations

import janitor_rs
import numpy as np
import pandas as pd

from janitor.functions._conditional_join._aggregation_helpers import (
    _aggregation_inputs,
    _empty_aggregation_result,
    _materialize_aggregation_result,
)
from janitor.functions._conditional_join._helpers import (
    JoinCondition,
    _build_residual_predicate,
    _convert_array_to_numpy,
    _maybe_remove_nulls_from_dataframe,
    _prepare_range_anchor,
    _RangeAnchor,
)

_RANGE_PAIR_PRIORITY = (
    (">", "<"),
    (">", "<="),
    (">=", "<"),
    (">=", "<="),
    (">", ">"),
    (">", ">="),
    (">=", ">"),
    (">=", ">="),
    ("<", "<"),
    ("<", "<="),
    ("<=", "<"),
    ("<=", "<="),
)


def _select_range_pair(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list[JoinCondition],
) -> tuple[int, int, _RangeAnchor] | None:
    """Select the first compatible pair of range predicates.

    This is the decision point for the dual-range optimization. The priority
    table first prefers complementary operators such as ``(>, <)`` and
    ``(>=, <=)`` because those commonly produce bounded interior windows. It
    then considers compatible same-direction pairs. For each candidate anchor,
    the right values are prepared once; the second right column is reordered
    through the anchor's physical index and must be monotonically increasing
    in that same layout.

    Args:
        df: Working left dataframe. Nulls in columns used by non-``!=``
            predicates must already have been removed.
        right: Working right dataframe with the same physical row layout used
            by all right-side predicates. This function does not mutate it.
        conditions: User-ordered ``(left_column, right_column, operator)``
            triples. The operators considered here must be range operators.

    Returns:
        ``(anchor_position, second_position, anchor)``. ``anchor_position``
        and ``second_position`` are positions in ``conditions``; ``anchor``
        contains the filtered left values and the sorted right values for the
        selected first predicate. ``None`` means that no pair can establish a
        shared ascending right layout, so the caller must use the single-
        anchor fallback.
    """

    for anchor_op, second_op in _RANGE_PAIR_PRIORITY:
        for anchor_position, condition in enumerate(conditions):
            if condition.op != anchor_op:
                continue
            anchor = _prepare_range_anchor(df[condition.left], right[condition.right])
            if anchor is None:
                continue
            for second_position, second_condition in enumerate(conditions):
                if (
                    second_position == anchor_position
                    or second_condition.op != second_op
                ):
                    continue
                second_right = right.loc[anchor.right_index, second_condition.right]
                if second_right.is_monotonic_increasing:
                    return anchor_position, second_position, anchor
    return None


def _filtered_range_frames(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list[JoinCondition],
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Return null-filtered working frames for range preparation.

    A null cannot satisfy ``<``, ``<=``, ``>``, or ``>=``. Filtering is done
    once across all non-``!=`` columns so every candidate array retains the
    same physical row order. Columns used only by ``!=`` remain untouched;
    their null masks are handled by the residual predicate builder.

    Args:
        df: Left dataframe in reset physical-row order.
        right: Right dataframe in reset physical-row order.
        conditions: Predicate triples used to identify columns that require
            null filtering.

    Returns:
        A pair ``(filtered_df, filtered_right)``. Either value is ``None``
        when every row is null in at least one required non-``!=`` column.
    """
    left_columns = {condition.left for condition in conditions if condition.op != "!="}
    right_columns = {
        condition.right for condition in conditions if condition.op != "!="
    }
    filtered_df = _maybe_remove_nulls_from_dataframe(df, left_columns)
    if filtered_df is None:
        # No left row can satisfy the required non-`!=` predicates, so the
        # right frame does not need to be inspected or copied.
        return None, None
    filtered_right = _maybe_remove_nulls_from_dataframe(right, right_columns)
    return filtered_df, filtered_right


def _can_use_dual_range(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list[JoinCondition],
) -> bool:
    """Report whether a multi-predicate call has a usable dual-range pair.

    This is a routing probe, not a result-producing operation. It repeats the
    inexpensive preparation needed by the eventual kernel call so the caller
    can distinguish “the range path found no matches” from “the predicates do
    not satisfy the dual-range contract.” The latter must fall back to the
    single-anchor residual path.

    Args:
        df: Left working dataframe with reset physical positions.
        right: Right working dataframe with reset physical positions.
        conditions: Complete user-ordered predicate list.

    Returns:
        ``True`` only when two range predicates can share an ascending right
        layout; otherwise ``False``.
    """
    filtered_df, filtered_right = _filtered_range_frames(df, right, conditions)
    if filtered_df is None or filtered_right is None:
        return False
    return _select_range_pair(filtered_df, filtered_right, conditions) is not None


def _get_extended_indices(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list[JoinCondition],
    keep: str,
    return_materialized_indices: bool,
) -> dict | None:
    """Build dual-range windows and filter later predicates in Rust.

    The first two compatible range predicates become five-element Rust anchor
    tuples. Later predicates retain their user order and are sent as residual
    tuples. For ``!=`` residuals, the shared helper attaches the authoritative
    null masks and extension-array flag.

    ``None`` means that no compatible pair of range predicates exists; the
    caller may then route to the single-anchor residual implementation. An
    empty dictionary means that the dual-range kernel was selected but no row
    survived its windows or residual predicates.

    Args:
        df: Left working dataframe with unique reset physical positions.
        right: Right working dataframe with the same physical layout expected
            by all predicates.
        conditions: Complete predicate list. Two compatible range triples are
            required at the front of the Rust contract; later triples are
            residual filters.
        keep: ``"first"``, ``"last"``, ``"any"``, or ``"all"``. The Rust
            kernel applies it after residual filtering.
        return_materialized_indices: Whether building-block callers require
            all surviving pairs. This is converted to the Rust wrapper's
            corresponding boolean flag.

    Returns:
        Materialized public index labels, an empty result when there are no
        matches, or ``None`` when the dual-range contract is not applicable.
    """
    filtered_df, filtered_right = _filtered_range_frames(df, right, conditions)
    if filtered_df is None or filtered_right is None:
        return {
            "left_index": np.array([], dtype=np.int64),
            "right_index": np.array([], dtype=np.int64),
        }
    selected = _select_range_pair(filtered_df, filtered_right, conditions)
    if selected is None:
        return None
    first_position, second_position, anchor = selected
    first = conditions[first_position]
    second = conditions[second_position]
    second_left = _convert_array_to_numpy(
        filtered_df.loc[anchor.left_index, second.left]._values
    )
    second_right_series = filtered_right.loc[anchor.right_index, second.right]
    second_right = _convert_array_to_numpy(array=second_right_series._values)
    predicates = [
        (
            anchor.left_array,
            anchor.left_index,
            anchor.right_array,
            anchor.right_index,
            first.op,
        ),
        (
            second_left,
            anchor.left_index,
            second_right,
            anchor.right_index,
            second.op,
        ),
    ]
    for position, condition in enumerate(conditions):
        if position in {first_position, second_position}:
            continue
        left_residual = filtered_df.loc[anchor.left_index, condition.left]
        right_residual = filtered_right.loc[anchor.right_index, condition.right]
        predicates.append(
            _build_residual_predicate(left_residual, right_residual, condition.op)
        )
    result = janitor_rs.range_join_extended_indices(predicates, keep)
    if result is None:
        return {
            "left_index": np.array([], dtype=np.int64),
            "right_index": np.array([], dtype=np.int64),
        }
    return result


def _aggregate_extended(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list[JoinCondition],
    aggfunc: list[tuple],
    reverse: bool,
    return_matched: bool,
) -> pd.DataFrame | None:
    """Aggregate a dual-range join, with an optional residual phase.

    The first two compatible range predicates always produce the two anchor
    windows. If there are no remaining predicates, the exact-two-range Rust
    API aggregates directly over their intersection. If residual predicates
    remain, the range-extended API evaluates them inside that intersection
    before updating aggregation state. Neither path materializes candidate
    pairs in Python.

    Args:
        df: Left dataframe with reset, unique physical positions.
        right: Right dataframe with reset, unique physical positions.
        conditions: Complete predicate list. Two compatible range predicates
            are selected as anchors; any remaining predicates are residuals
            evaluated in their original user order.
        aggfunc: Rust-supported ``(column, operation)`` requests.
        reverse: If false, aggregate right-side source values into left
            output slots. If true, aggregate left-side values into right
            output slots.
        return_matched: Whether Rust should return the per-output match mask.

    Returns:
        A materialized aggregation dataframe, an empty dataframe when the
        selected range layout has no matches, or ``None`` when no compatible
        dual-range pair exists and the caller should use the anchor fallback.

    Raises:
        TypeError: If the selected anchor dtype has no registered Rust kernel.
        ValueError: If Rust rejects predicate layout, aggregation requests, or
            output-position metadata.
    """
    filtered_df, filtered_right = _filtered_range_frames(df, right, conditions)
    if filtered_df is None or filtered_right is None:
        return _empty_aggregation_result(
            source=right if not reverse else df,
            aggfunc=aggfunc,
        )
    selected = _select_range_pair(filtered_df, filtered_right, conditions)
    if selected is None:
        return None
    first_position, second_position, anchor = selected
    first = conditions[first_position]
    second = conditions[second_position]
    second_left = _convert_array_to_numpy(
        filtered_df.loc[anchor.left_index, second.left]._values
    )
    second_right = _convert_array_to_numpy(
        filtered_right.loc[anchor.right_index, second.right]._values
    )
    source = (
        filtered_right.loc[anchor.right_index]
        if not reverse
        else filtered_df.loc[anchor.left_index]
    )
    output_index = (
        right.index.take(anchor.right_index)
        if reverse
        else df.index.take(anchor.left_index)
    )
    # The first tuple carries the physical-to-trimmed output maps. They are
    # needed by both APIs when null filtering or right-side sorting reordered
    # the compact layouts. The second anchor has the same aligned layouts but
    # does not need to repeat the ordering flag.
    predicates = [
        (
            anchor.left_array,
            anchor.left_index,
            anchor.right_array,
            anchor.right_index,
            anchor.left_index,
            anchor.right_index,
            anchor.right_index_is_ordered,
            first.op,
        ),
        (
            second_left,
            anchor.left_index,
            second_right,
            anchor.right_index,
            second.op,
        ),
    ]
    for position, condition in enumerate(conditions):
        if position in {first_position, second_position}:
            continue
        predicates.append(
            _build_residual_predicate(
                filtered_df.loc[anchor.left_index, condition.left],
                filtered_right.loc[anchor.right_index, condition.right],
                condition.op,
            )
        )
    # Exactly two anchors have no residual filtering and use the simpler
    # range_join_aggregate API. Once a third predicate is present, use the
    # extended API so Rust evaluates that predicate inside each intersection.
    if len(predicates) == 2:
        kernel = (
            janitor_rs.range_join_aggregate_reverse
            if reverse
            else janitor_rs.range_join_aggregate
        )
    else:
        kernel = (
            janitor_rs.range_join_extended_aggregate_reverse
            if reverse
            else janitor_rs.range_join_extended_aggregate
        )
    result = kernel(
        predicates,
        _aggregation_inputs(source=source, aggfunc=aggfunc),
        return_matched,
    )
    return _materialize_aggregation_result(
        result=result,
        output_index=output_index,
        source=source,
        aggfunc=aggfunc,
        return_matched=return_matched,
    )


def _get_indices(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list[tuple],
    keep: str,
    return_materialized_indices: bool,
) -> dict | None:
    """Compute indices for exactly two compatible range predicates.

    The first compatible predicate supplies the sorted right layout. The
    second right column is reordered through that layout and is passed to Rust
    as an ascending array. This means the Rust kernel only needs one simple
    partition-point implementation; PyJanitor owns sorting, null filtering,
    physical-position alignment, and the stable pairing of values with index
    labels.

    Args:
        df: Left working dataframe after conditional-join preparation.
        right: Right working dataframe after conditional-join preparation.
        conditions: Exactly two ``(left_column, right_column, operator)``
            triples, both using range operators.
        keep: Requested selection mode. ``"all"`` emits every pair in each
            intersected window; the other modes retain one right label per
            successful left row.
        return_materialized_indices: When true, return the range-building
            blocks as well as the materialized labels. Keep is ignored by the
            Rust wrapper in that mode.

    Returns:
        A dictionary of ``left_index`` and ``right_index`` arrays, with
        ``starts`` and ``ends`` when building blocks were requested, or
        ``None`` when the caller must use the extended fallback.

    Raises:
        ValueError: If the predicate count, dtype, layout, or Rust contract is
            invalid.
    """
    if len(conditions) != 2:
        raise ValueError("range join requires exactly two predicates")

    left_columns = {condition.left for condition in conditions}
    right_columns = {condition.right for condition in conditions}
    df = _maybe_remove_nulls_from_dataframe(df, left_columns)
    right = _maybe_remove_nulls_from_dataframe(right, right_columns)
    if df is None or right is None:
        empty = np.array([], dtype=np.int64)
        return {"left_index": empty, "right_index": empty}

    selected = _select_range_pair(df=df, right=right, conditions=conditions)
    if selected is None:
        # The pair is not suitable for the simple ascending kernel. The
        # caller must use the extended path, which can preserve correctness
        # without assuming a shared right layout.
        return None

    first_position, second_position, anchor = selected
    first = conditions[first_position]
    second = conditions[second_position]
    second_left = df.loc[anchor.left_index, second.left]
    second_right = right.loc[anchor.right_index, second.right]
    second_left_array = _convert_array_to_numpy(array=second_left._values)
    second_right_array = _convert_array_to_numpy(array=second_right._values)
    predicates = [
        (
            anchor.left_array,
            anchor.left_index,
            anchor.right_array,
            anchor.right_index,
            anchor.right_index_is_ordered,
            first.op,
        ),
        (
            second_left_array,
            anchor.left_index,
            second_right_array,
            anchor.right_index,
            True,
            second.op,
        ),
    ]
    result = janitor_rs.range_join_indices(
        predicates,
        keep,
        bool(return_materialized_indices),
    )
    if result is None:
        empty = np.array([], dtype=np.int64)
        return {"left_index": empty, "right_index": empty}
    return result
