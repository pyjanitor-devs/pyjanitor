"""Shared preparation and materialization helpers for Rust aggregations."""

import numpy as np
import pandas as pd

from janitor.functions._conditional_join._get_join_aggs import _build_agg_label
from janitor.functions._conditional_join._helpers import _convert_array_to_numpy


def _aggregation_inputs(source: pd.DataFrame, aggfunc: list[tuple]) -> list[tuple]:
    """Prepare full-layout aggregation arrays and authoritative null masks."""
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


def _empty_aggregation_result(
    source: pd.DataFrame, aggfunc: list[tuple]
) -> pd.DataFrame:
    """Return an empty aggregation result with the requested output dtypes."""
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
    """Convert Rust aggregation slots into a pandas result frame."""
    if result is None:
        return _empty_aggregation_result(source, aggfunc)
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


def _aggregation_kernel(name: str):
    """Resolve a dtype-specific Rust aggregation function."""
    try:
        import janitor_rs

        return getattr(janitor_rs, name)
    except AttributeError as error:
        raise TypeError(f"Rust aggregation does not support dtype {name}") from error
