import janitor_rs
import pandas as pd

from janitor.functions._conditional_join import _range_join, regions
from janitor.functions._conditional_join._aggregation_helpers import (
    _aggregation_inputs,
    _empty_aggregation_result,
    _materialize_aggregation_result,
)
from janitor.functions._conditional_join._helpers import (
    _build_residual_predicate,
    _convert_array_to_numpy,
)


def _get_indices(
    df: pd.DataFrame,
    right: pd.DataFrame,
    mapping: dict,
    return_matching_indices: bool,
    keep: str,
):
    """Build region-join indices through the Rust regions kernel.

    Parameters
    ----------
    df, right : pandas.DataFrame
        Null-filtered working join frames.
    mapping : dict
        Predicate groups produced by the conditional-join helper.
    return_matching_indices : bool
        Whether all passing pairs should be returned.
    keep : {"all", "first", "last", "any"}
        Selection mode when all matches are not requested.

    Returns
    -------
    dict
        Original left/right index arrays, or empty arrays when no pair
        matches.
    """
    primary_conditions = [mapping["ge_gt"], mapping["le_lt"]]
    residual_conditions = []
    residual_conditions.extend(mapping["le_or_ge"])
    residual_conditions.extend(mapping["equals"])
    residual_conditions.extend(mapping["not_equals"])
    residual_conditions = [condition for condition in residual_conditions if condition]
    return regions.get_indices(
        df=df,
        right=right,
        primary_conditions=primary_conditions,
        residual_conditions=residual_conditions,
        return_matching_indices=return_matching_indices,
        keep=keep,
    )


def _aggregate(
    df,
    right,
    conditions,
    aggfunc,
    reverse,
    return_matched,
):
    """Aggregate a dual-region join without materializing matching pairs.

    The first two compatible range predicates are converted into the two
    primary region anchors. Remaining predicates stay in their original order
    and are evaluated by Rust inside the region sweep.

    Args:
        df: Reset, null-filtered left working frame.
        right: Reset, null-filtered right working frame.
        conditions: Complete user-ordered predicate list.
        aggfunc: Rust-supported ``(column, operation)`` requests.
        reverse: Aggregate left values into right output slots when true.
        return_matched: Request Rust's per-output matched mask.

    Returns:
        A materialized aggregation dataframe, an empty dataframe when one
        side has no eligible rows, or ``None`` when no compatible pair of
        range anchors exists.
    """
    filtered_df, filtered_right = _range_join._filtered_range_frames(
        df, right, conditions
    )
    if filtered_df is None or filtered_right is None:
        return _empty_aggregation_result(
            source=right if not reverse else df,
            aggfunc=aggfunc,
        )

    selected = _range_join._select_range_pair(filtered_df, filtered_right, conditions)
    if selected is None:
        return None
    first_position, second_position, anchor = selected
    first = conditions[first_position]
    second = conditions[second_position]

    # The first anchor supplies the canonical physical left/right layouts.
    # The second anchor is read in those same layouts; the Rust region builder
    # uses the original IDs to align its independently built region paths.
    second_left = _convert_array_to_numpy(
        filtered_df.loc[anchor.left_index, second[0]]._values
    )
    second_right = _convert_array_to_numpy(
        filtered_right.loc[anchor.right_index, second[1]]._values
    )
    predicates = [
        (
            anchor.left_array,
            anchor.left_index,
            anchor.right_array,
            anchor.right_index,
            anchor.left_index,
            anchor.right_index,
            anchor.right_index_is_ordered,
            first[2],
        ),
        (
            second_left,
            anchor.left_index,
            second_right,
            anchor.right_index,
            second[2],
        ),
    ]
    for position, condition in enumerate(conditions):
        if position in {first_position, second_position}:
            continue
        predicates.append(
            _build_residual_predicate(
                filtered_df.loc[anchor.left_index, condition[0]],
                filtered_right.loc[anchor.right_index, condition[1]],
                condition[2],
            )
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
    aggregation_inputs = _aggregation_inputs(source, aggfunc)
    extended = len(predicates) > 2
    if extended:
        kernel = (
            janitor_rs.region_extended_aggregate_reverse
            if reverse
            else janitor_rs.region_extended_aggregate
        )
    else:
        kernel = (
            janitor_rs.region_aggregate_reverse
            if reverse
            else janitor_rs.region_aggregate
        )
    result = kernel(predicates, aggregation_inputs, return_matched)
    return _materialize_aggregation_result(
        result=result,
        output_index=output_index,
        source=source,
        aggfunc=aggfunc,
        return_matched=return_matched,
    )
