"""Fast path for exactly two ascending range predicates.

PyJanitor prepares the shared right layout here. Rust then computes the two
binary-search windows and applies ``keep`` without evaluating residual
predicates; calls with additional predicates use ``_single_non_equi_join_extended``.
"""

from __future__ import annotations

import janitor_rs
import numpy as np
import pandas as pd

from janitor.functions._conditional_join._helpers import (
    _convert_array_to_numpy,
    _maybe_remove_nulls_from_dataframe,
)

_RANGE_KERNELS = {
    "int64": janitor_rs.range_join_indices_int64,
    "int32": janitor_rs.range_join_indices_int32,
    "int16": janitor_rs.range_join_indices_int16,
    "int8": janitor_rs.range_join_indices_int8,
    "uint64": janitor_rs.range_join_indices_uint64,
    "uint32": janitor_rs.range_join_indices_uint32,
    "uint16": janitor_rs.range_join_indices_uint16,
    "uint8": janitor_rs.range_join_indices_uint8,
    "float64": janitor_rs.range_join_indices_f64,
    "float32": janitor_rs.range_join_indices_f32,
}


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
    partition-point implementation; PyJanitor owns sorting and alignment.
    """
    if len(conditions) != 2:
        raise ValueError("range join requires exactly two predicates")

    left_columns = {left for left, _, _ in conditions}
    right_columns = {right_name for _, right_name, _ in conditions}
    df = _maybe_remove_nulls_from_dataframe(df, left_columns)
    right = _maybe_remove_nulls_from_dataframe(right, right_columns)
    if df is None or right is None:
        empty = np.array([], dtype=np.int64)
        return {"left_index": empty, "right_index": empty}

    from janitor.functions._conditional_join._single_non_equi_join_extended import (
        _select_range_pair,
    )

    selected = _select_range_pair(df=df, right=right, conditions=conditions)
    if selected is None:
        # The pair is not suitable for the simple ascending kernel. The
        # caller must use the extended path, which can preserve correctness
        # without assuming a shared right layout.
        return None

    first_position, second_position, anchor = selected
    first = conditions[first_position]
    second = conditions[second_position]
    second_left = df.loc[anchor.left_index, second[0]]
    second_right = right.loc[anchor.right_index, second[1]]
    second_left_array = _convert_array_to_numpy(array=second_left._values)
    second_right_array = _convert_array_to_numpy(array=second_right._values)
    dtype_name = anchor.left_array.dtype.name
    try:
        kernel = _RANGE_KERNELS[dtype_name]
    except KeyError as error:
        raise TypeError(
            f"range join does not support dtype {anchor.left_array.dtype}"
        ) from error

    predicates = [
        (
            anchor.left_array,
            anchor.left_index,
            anchor.right_array,
            anchor.right_index,
            anchor.right_index_is_ordered,
            first[2],
        ),
        (
            second_left_array,
            anchor.left_index,
            second_right_array,
            anchor.right_index,
            True,
            second[2],
        ),
    ]
    result = kernel(
        predicates,
        keep,
        bool(return_materialized_indices),
    )
    if result is None:
        empty = np.array([], dtype=np.int64)
        return {"left_index": empty, "right_index": empty}
    return result
