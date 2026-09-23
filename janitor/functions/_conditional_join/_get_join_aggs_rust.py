"""Fused Rust aggregation for single and extended conditional joins.

The Rust functions in this module's maps update aggregation state while they
discover matching candidates. This avoids building intermediate join-index
pairs solely to aggregate them.
"""

import janitor_rs
import numpy as np
import pandas as pd

from janitor.functions._conditional_join._get_join_aggs import _build_agg_label
from janitor.functions._conditional_join._helpers import (
    _convert_array_to_numpy,
    _get_boolean_args_for_ne,
    _maybe_remove_nulls_from_dataframe,
    _null_checks_cond_join,
    _sort_if_not_monotonic,
    greater_than_join_types,
    less_than_join_types,
)


def _null_positions(series: pd.Series) -> np.ndarray | None:
    """Return full-layout physical positions of null rows."""
    nulls = series.isna().to_numpy(dtype=bool)
    if not nulls.any():
        return None
    return _convert_array_to_numpy(array=series.index[nulls]._values)


def _aggregation_inputs(source: pd.DataFrame, aggfunc: list[tuple]) -> list[tuple]:
    """Prepare full-layout aggregation arrays and authoritative masks."""
    result = []
    for column_name, operation in aggfunc:
        series = source[column_name]
        result.append(
            (
                _convert_array_to_numpy(array=series._values),
                series.isna().to_numpy(dtype=bool),
                operation,
            )
        )
    return result


def _empty_result(source: pd.DataFrame, aggfunc: list[tuple]) -> pd.DataFrame:
    result = {}
    for column_name, operation in aggfunc:
        dtype = "int64" if operation == "size" else source[column_name].dtype
        result[_build_agg_label(column_name, operation)] = pd.array([], dtype=dtype)
    return pd.DataFrame(result, copy=False)


def _materialize_result(
    result,
    output_index: pd.Index,
    source: pd.DataFrame,
    aggfunc: list[tuple],
) -> pd.DataFrame:
    if result is None:
        return _empty_result(source, aggfunc)
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
            invalid = values == -1
            safe_values = values.copy()
            safe_values[invalid] = 0
            values = series.iloc[safe_values].copy()
            values.iloc[invalid] = pd.NA
            values = values.array
        elif operation in {"sum", "prod"} and pd.api.types.is_extension_array_dtype(
            series.dtype
        ):
            values = pd.array(values, dtype=series.dtype)
        output[_build_agg_label(column_name, operation)] = values[matched]
    return pd.DataFrame(output, copy=False, index=index)


def _kernel(name: str):
    try:
        return getattr(janitor_rs, name)
    except AttributeError as error:
        raise TypeError(f"Rust aggregation does not support dtype {name}") from error


def _single(
    df: pd.DataFrame,
    right: pd.DataFrame,
    condition: tuple,
    aggfunc: list[tuple],
    reverse: bool,
) -> pd.DataFrame:
    left_on, right_on, operation = condition
    left_series = df[left_on]
    right_series = right[right_on]
    left_positions = left_null_positions = right_positions = right_null_positions = None
    is_extension_array = False

    if operation in less_than_join_types.union(greater_than_join_types):
        left_outcome = _null_checks_cond_join(left_series)
        right_outcome = _null_checks_cond_join(right_series)
        if left_outcome is None or right_outcome is None:
            return _empty_result(df if reverse else right, aggfunc)
        left_values, _ = left_outcome
        right_values, _ = right_outcome
        right_values, _ = _sort_if_not_monotonic(right_values)
        left_work = df.loc[left_values.index]
        right_work = right.loc[right_values.index]
        left_array = _convert_array_to_numpy(left_values._values)
        right_array = _convert_array_to_numpy(right_values._values)
    elif operation == "!=":
        left_null = left_series.isna().to_numpy(dtype=bool)
        right_null = right_series.isna().to_numpy(dtype=bool)
        left_values = left_series.loc[~left_null]
        right_values = right_series.loc[~right_null]
        if left_values.empty or right_values.empty:
            right_sorted = right_values
        else:
            right_sorted, _ = _sort_if_not_monotonic(right_values)
        left_work = df
        right_work = right
        left_array = _convert_array_to_numpy(left_values._values)
        right_array = _convert_array_to_numpy(right_sorted._values)
        left_positions = _convert_array_to_numpy(left_values.index._values)
        right_positions = _convert_array_to_numpy(right_sorted.index._values)
        left_null_positions = _null_positions(left_series)
        right_null_positions = _null_positions(right_series)
        is_extension_array = bool(
            pd.api.types.is_extension_array_dtype(left_series.dtype)
        )
    else:
        raise ValueError("single Rust aggregation requires a non-equality predicate")

    dtype = left_array.dtype.name
    function_name = (
        "single_join_aggregate_reverse_" if reverse else "single_join_aggregate_"
    ) + dtype
    function = _kernel(function_name)
    result = function(
        left_array,
        right_array,
        operation,
        left_positions,
        left_null_positions,
        right_positions,
        right_null_positions,
        is_extension_array,
        _aggregation_inputs(right_work if not reverse else left_work, aggfunc),
    )
    return _materialize_result(
        result,
        right_work.index if reverse else left_work.index,
        right_work if not reverse else left_work,
        aggfunc,
    )


def _extended(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list[tuple],
    aggfunc: list[tuple],
    reverse: bool,
) -> pd.DataFrame:
    all_not_equal = all(operation == "!=" for _, _, operation in conditions)
    if all_not_equal:
        first_left_on, first_right_on, _ = conditions[0]
        left_series = df[first_left_on]
        right_series = right[first_right_on]
        left_null = left_series.isna()
        right_null = right_series.isna()
        left_values = left_series.loc[~left_null]
        right_values = right_series.loc[~right_null]
        if left_values.empty or right_values.empty:
            right_sorted = right_values
        else:
            right_sorted, _ = _sort_if_not_monotonic(right_values)
        predicates = [
            (
                _convert_array_to_numpy(left_values._values),
                _convert_array_to_numpy(left_series.index._values),
                _convert_array_to_numpy(left_values.index._values),
                _null_positions(left_series),
                _convert_array_to_numpy(right_sorted._values),
                _convert_array_to_numpy(right_series.index._values),
                _convert_array_to_numpy(right_sorted.index._values),
                _null_positions(right_series),
                True,
                bool(pd.api.types.is_extension_array_dtype(left_series.dtype)),
                "!=",
            )
        ]
        for left_on, right_on, operation in conditions[1:]:
            left_aligned = df[left_on]
            right_aligned = right[right_on]
            left_array = _convert_array_to_numpy(left_aligned._values)
            right_array = _convert_array_to_numpy(right_aligned._values)
            left_mask, right_mask, extension = _get_boolean_args_for_ne(
                operation, left_aligned, right_aligned
            )
            if left_mask is None and right_mask is None:
                predicates.append((left_array, right_array, operation))
            else:
                predicates.append(
                    (
                        left_array,
                        left_mask,
                        right_array,
                        right_mask,
                        bool(extension),
                        operation,
                    )
                )
        dtype = _convert_array_to_numpy(left_values._values).dtype.name
        function_name = (
            "single_join_extended_aggregate_reverse_"
            if reverse
            else "single_join_extended_aggregate_"
        ) + dtype
        result = _kernel(function_name)(
            predicates,
            _aggregation_inputs(right if not reverse else df, aggfunc),
        )
        return _materialize_result(
            result,
            right.index if reverse else df.index,
            right if not reverse else df,
            aggfunc,
        )

    first_position = next(
        position
        for position, (_, _, operation) in enumerate(conditions)
        if operation in less_than_join_types.union(greater_than_join_types)
    )
    non_ne_left = {left for left, _, operation in conditions if operation != "!="}
    non_ne_right = {
        right_name for _, right_name, operation in conditions if operation != "!="
    }
    filtered_df = _maybe_remove_nulls_from_dataframe(df, non_ne_left)
    filtered_right = _maybe_remove_nulls_from_dataframe(right, non_ne_right)
    if filtered_df is None or filtered_right is None:
        return _empty_result(df if reverse else right, aggfunc)
    first_left_on, first_right_on, first_operation = conditions[first_position]
    left_values = filtered_df[first_left_on]
    right_values = filtered_right[first_right_on]
    right_sorted, _ = _sort_if_not_monotonic(right_values)
    right_positions = _convert_array_to_numpy(right_sorted.index._values)
    sorted_right = filtered_right.loc[right_sorted.index]
    predicates = [
        (
            _convert_array_to_numpy(left_values._values),
            _convert_array_to_numpy(left_values.index._values),
            _convert_array_to_numpy(right_sorted._values),
            right_positions,
            True,
            first_operation,
        )
    ]
    for position, (left_on, right_on, operation) in enumerate(conditions):
        if position == first_position:
            continue
        left_aligned = filtered_df[left_on]
        right_aligned = filtered_right.loc[
            right_positions,
            right_on,
        ]
        left_array = _convert_array_to_numpy(left_aligned._values)
        right_array = _convert_array_to_numpy(right_aligned._values)
        if operation == "!=":
            left_mask, right_mask, extension = _get_boolean_args_for_ne(
                operation, left_aligned, right_aligned
            )
            if left_mask is None and right_mask is None:
                predicates.append((left_array, right_array, operation))
            else:
                predicates.append(
                    (
                        left_array,
                        left_mask,
                        right_array,
                        right_mask,
                        bool(extension),
                        operation,
                    )
                )
        else:
            predicates.append((left_array, right_array, operation))
    dtype = _convert_array_to_numpy(left_values._values).dtype.name
    function_name = (
        "single_join_extended_aggregate_reverse_"
        if reverse
        else "single_join_extended_aggregate_"
    ) + dtype
    result = _kernel(function_name)(
        predicates,
        _aggregation_inputs(sorted_right if not reverse else filtered_df, aggfunc),
    )
    return _materialize_result(
        result,
        sorted_right.index if reverse else filtered_df.index,
        sorted_right if not reverse else filtered_df,
        aggfunc,
    )


def _aggregate(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list[tuple],
    aggfunc: list[tuple],
    reverse: bool,
) -> pd.DataFrame:
    if len(conditions) == 1:
        return _single(df, right, conditions[0], aggfunc, reverse)
    return _extended(df, right, conditions, aggfunc, reverse)
