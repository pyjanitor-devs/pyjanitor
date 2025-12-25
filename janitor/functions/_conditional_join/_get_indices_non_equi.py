from __future__ import annotations

import numpy as np
import pandas as pd

from janitor.functions._conditional_join import (
    _dual_non_equi,
    _greater_than_indices,
    _helpers,
    _less_than_indices,
    _range_indices,
)


def _get_indices(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list,
    keep: str,
    return_matching_indices: bool,
) -> tuple:
    """
    Get indices, or aggregates, for multiple conditions,
    where `>/>=` or `</<=` is present
    """
    empty_array = np.array([], dtype=np.intp)
    mapping = _helpers._separate_conditions_based_on_op(conditions=conditions)
    _columns = (
        mapping["le_or_ge"],
        mapping["le_lt"],
        mapping["ge_gt"],
        mapping["equals"],
    )
    columns = []
    for entry in _columns:
        if not entry:
            continue
        if isinstance(entry, tuple):
            columns.append(entry)
        else:
            columns.extend(entry)
    left_columns = set()
    right_columns = set()
    for left_col, right_col, _ in columns:
        left_columns.add(left_col)
        right_columns.add(right_col)
    df = _helpers._maybe_remove_nulls_from_dataframe(df=df, columns=left_columns)
    if df is None:
        return {
            "left_index": empty_array,
            "right_index": empty_array,
        }
    right = _helpers._maybe_remove_nulls_from_dataframe(df=right, columns=right_columns)
    if right is None:
        return {
            "left_index": empty_array,
            "right_index": empty_array,
        }
    if (not mapping["is_range_join"]) and (len(mapping["le_or_ge"]) == 1):
        (left_on, right_on, op) = mapping["le_or_ge"][0]
        left_col = df[left_on]
        if not right[right_on].is_monotonic_increasing:
            right = right.sort_values(right_on, ignore_index=False, kind="stable")
        right_col = right[right_on]
        lt_or_le_check = op in _helpers.less_than_join_types
        if lt_or_le_check:
            outcome = _less_than_indices._le_lt_indices(
                left=left_col._values,
                left_index=left_col.index._values,
                right=right_col._values,
                strict=op == "<",
            )
        else:
            outcome = _greater_than_indices._ge_gt_indices(
                left=left_col._values,
                left_index=left_col.index._values,
                right=right_col._values,
                strict=op == ">",
            )
        if outcome is None:
            return {
                "left_index": empty_array,
                "right_index": empty_array,
            }
        if lt_or_le_check:
            left_index, starts = outcome
            ends = None
        else:
            left_index, ends = outcome
            starts = None
        _conditions = (
            mapping["equals"],
            mapping["not_equals"],
        )
        rest = []
        for entry in _conditions:
            if not entry:
                continue
            if isinstance(entry, tuple):
                rest.append(entry)
            else:
                rest.extend(entry)
        outcome = _helpers._get_positive_matches_conditions(
            df=df,
            right=right,
            conditions=rest,
            left_index=left_index,
            starts=starts,
            ends=ends,
        )
        if outcome is None:
            return {
                "left_index": empty_array,
                "right_index": empty_array,
            }
        return _helpers.build_indices_matches(
            left_index=left_index,
            right_index=right.index._values,
            counts_array=outcome["counts_array"],
            starts=starts,
            ends=ends,
            matches=outcome["matches"],
            total=outcome["total"],
            keep=keep,
        )

    if not mapping["is_range_join"]:
        first_condition, second_condition, *rest_ = mapping["le_or_ge"]
        others = (rest_, mapping["equals"], mapping["not_equals"])
        outcome = _dual_non_equi._get_indices(
            df=df,
            right=right,
            first_condition=first_condition,
            second_condition=second_condition,
        )

        rest = []
        for entry in others:
            if not entry:
                continue
            if isinstance(entry, tuple):
                rest.append(entry)
            else:
                rest.extend(entry)
        if outcome is None:
            return {
                "left_index": empty_array,
                "right_index": empty_array,
            }
        if not rest:
            return _dual_non_equi._build_indices(
                left_index=outcome["left_index"],
                right_index=outcome["right_index"],
                positions=outcome["positions"],
                counts_array=outcome["counts_array"],
                total=outcome["total"],
                keep=keep,
            )
        ends = outcome["counts_array"].cumsum()
        starts = np.empty(outcome["counts_array"].size, dtype=np.int64)
        starts[0] = 0
        starts[1:] = ends[:-1]
        out = _helpers._get_positive_matches_conditions_posns(
            df=df,
            right=right,
            conditions=rest,
            left_index=outcome["left_index"],
            right_index=outcome["right_index"],
            positions=outcome["positions"],
            starts=starts,
            ends=ends,
        )
        if out is None:
            return {
                "left_index": empty_array,
                "right_index": empty_array,
            }
        return _helpers._build_indices_positions(
            left_index=outcome["left_index"],
            right_index=outcome["right_index"],
            positions=out["positions"],
            starts=starts,
            ends=ends,
            counts_array=out["counts_array"],
            total=out["total"],
            keep=keep,
        )

    # is range join
    (_, r1_col, _) = mapping["le_lt"]
    (_, r2_col, _) = mapping["ge_gt"]
    others = (
        mapping["le_or_ge"],
        mapping["equals"],
        mapping["not_equals"],
    )
    rest = []
    for entry in others:
        if not entry:
            continue
        if isinstance(entry, tuple):
            rest.append(entry)
        else:
            rest.extend(entry)
    right_is_sorted = right[r1_col].is_monotonic_increasing
    if not right_is_sorted:
        # defensive approach,
        # in case of duplicates
        sorter = dict.fromkeys([r1_col, r2_col])
        sorter = [*sorter]
        right = right.sort_values(sorter, kind="stable", ignore_index=False)
    check = right[r2_col].is_monotonic_increasing
    if check:
        outcome = _range_indices._range_indices(
            df=df,
            right=right,
            ge_gt=mapping["ge_gt"],
            le_lt=mapping["le_lt"],
        )
        if outcome is None:
            return {
                "left_index": empty_array,
                "right_index": empty_array,
            }
        if not rest:
            return _range_indices._build_indices(
                left_index=outcome["left_index"],
                right_index=right.index._values,
                starts=outcome["starts"],
                ends=outcome["ends"],
                keep=keep,
                right_is_sorted=right_is_sorted,
                return_matching_indices=return_matching_indices,
            )
        out = _helpers._get_positive_matches_conditions(
            df=df,
            right=right,
            conditions=rest,
            left_index=outcome["left_index"],
            starts=outcome["starts"],
            ends=outcome["ends"],
        )
        if out is None:
            return {
                "left_index": empty_array,
                "right_index": empty_array,
            }
        return _helpers.build_indices_matches(
            left_index=outcome["left_index"],
            right_index=right.index._values,
            counts_array=out["counts_array"],
            starts=outcome["starts"],
            ends=outcome["ends"],
            matches=out["matches"],
            total=out["total"],
            keep=keep,
        )
    outcome = _dual_non_equi._get_indices(
        df=df,
        right=right,
        first_condition=mapping["ge_gt"],
        second_condition=mapping["le_lt"],
    )
    if outcome is None:
        return {
            "left_index": empty_array,
            "right_index": empty_array,
        }
    ends = outcome["counts_array"].cumsum()
    starts = np.empty(outcome["counts_array"].size, dtype=np.int64)
    starts[0] = 0
    starts[1:] = ends[:-1]
    if not rest:
        return _dual_non_equi._build_indices(
            left_index=outcome["left_index"],
            right_index=outcome["right_index"],
            starts=starts,
            ends=ends,
            positions=outcome["positions"],
            counts_array=outcome["counts_array"],
            total=outcome["total"],
            keep=keep,
        )
    out = _helpers._get_positive_matches_conditions_posns(
        df=df,
        right=right,
        conditions=rest,
        left_index=outcome["left_index"],
        right_index=outcome["right_index"],
        positions=outcome["positions"],
        starts=starts,
        ends=ends,
    )
    if out is None:
        return {
            "left_index": empty_array,
            "right_index": empty_array,
        }
    return _helpers._build_indices_positions(
        left_index=outcome["left_index"],
        right_index=outcome["right_index"],
        positions=out["positions"],
        starts=starts,
        ends=ends,
        counts_array=out["counts_array"],
        total=out["total"],
        keep=keep,
    )
