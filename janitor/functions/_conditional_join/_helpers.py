"""Shared Python-side contracts for conditional joins.

This module is the boundary between the public pandas API and the specialised
join dispatchers. The dispatchers may use different Rust kernels, but they
all rely on the same conventions established here:

* dataframe rows are addressed by zero-based physical positions after the
  caller has reset the working indexes;
* ``slice(None)`` means that the complete working layout is valid and should
  not be eagerly materialised into an index array;
* non-null filtering removes rows that cannot participate in ordinary range
  predicates, while ``!=`` retains its separate null-mask semantics;
* sorted value arrays must carry their physical positions with them; and
* Rust results contain paired physical positions, never user-facing pandas
  labels.

The final materialisation helper converts those physical pairs into pandas
rows. Keeping that conversion here prevents each Rust-backed path from
implementing different outer-join, indicator, column-selection, or
``include_join_positions`` behavior.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Hashable, Sequence

import numpy as np
import pandas as pd
from pandas.core.dtypes.concat import concat_compat


class _JoinOperator(Enum):
    """
    List of operators used in conditional_join.
    """

    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_THAN_OR_EQUAL = ">="
    LESS_THAN_OR_EQUAL = "<="
    STRICTLY_EQUAL = "=="
    NOT_EQUAL = "!="


less_than_join_types = {
    _JoinOperator.LESS_THAN.value,
    _JoinOperator.LESS_THAN_OR_EQUAL.value,
}

greater_than_join_types = {
    _JoinOperator.GREATER_THAN.value,
    _JoinOperator.GREATER_THAN_OR_EQUAL.value,
}


def _empty_indices() -> dict[str, np.ndarray]:
    """Return the standard empty conditional-join result.

    Returns:
        A dictionary containing empty ``left_index`` and ``right_index``
        arrays, matching the other conditional-join paths.
    """
    empty = np.array([], dtype=np.intp)
    return {"left_index": empty, "right_index": empty}


def _materialize_or_return_indices(
    *,
    result: dict[str, np.ndarray],
    df: pd.DataFrame,
    right: pd.DataFrame,
    how: str,
    df_columns: Any,
    right_columns: Any,
    indicator: bool | str,
    include_join_positions: bool,
    return_matching_indices: bool,
    return_building_blocks: bool = False,
) -> dict[str, np.ndarray] | pd.DataFrame:
    """Return kernel output as indices or as a materialized join dataframe.

    The Rust-facing conditional-join kernels always produce physical
    ``left_index`` and ``right_index`` arrays. Public index helpers need that
    dictionary, while dataframe joins need those positions gathered into the
    requested join shape. Keeping this decision here ensures the single-range,
    all-``!=``, and dual-range dispatchers expose the same contract.

    Args:
        result: Kernel result containing physical row positions. Building
            blocks may add window arrays; those are never materialized.
        df: Prepared left dataframe addressed by physical positions.
        right: Prepared right dataframe addressed by physical positions.
        how: Join shape passed to the dataframe materializer.
        df_columns: Optional deprecated left-column selection.
        right_columns: Optional deprecated right-column selection.
        indicator: Whether to add the merge indicator column.
        include_join_positions: Whether to expose matched positions in the
            dataframe result index.
        return_matching_indices: Return the physical-position dictionary when
            true.
        return_building_blocks: Return the dictionary unchanged when true,
            even if dataframe output was otherwise requested.

    Returns:
        The physical index dictionary for index/building-block callers, or a
        materialized pandas dataframe for ordinary conditional joins.
    """
    if return_matching_indices or return_building_blocks:
        return result
    return _materialize_index_result(
        df=df,
        right=right,
        left_index=result["left_index"],
        right_index=result["right_index"],
        how=how,
        df_columns=df_columns,
        right_columns=right_columns,
        indicator=indicator,
        include_join_positions=include_join_positions,
    )


# copied from pandas/core/dtypes/missing.py
# seems function was introduced in 2.2.2
# we should support lesser versions - at least 2.0.0
def construct_1d_array_from_inferred_fill_value(
    value: object, length: int
) -> np.ndarray:
    """Create a repeated missing-value array using pandas' inferred dtype.

    This mirrors pandas' internal fill-value construction for extension and
    object dtypes. It is used when an outer join needs rows from one side to
    be padded with a dtype-compatible missing value rather than a hard-coded
    ``np.nan``.

    Args:
        value: Representative scalar or array-like value whose dtype should
            determine the fill array.
        length: Number of missing entries to create.

    Returns:
        A NumPy-compatible array containing ``length`` inferred missing
        values.
    """
    # Find our empty_value dtype by constructing an array
    #  from our value and doing a .take on it
    from pandas.core.algorithms import take_nd
    from pandas.core.construction import sanitize_array
    from pandas.core.indexes.base import Index

    arr = sanitize_array(value, Index(range(1)), copy=False)
    taker = -1 * np.ones(length, dtype=np.intp)
    return take_nd(arr, taker)


def _create_multiindex_column(df: pd.DataFrame, right: pd.DataFrame) -> tuple:
    """Namespace overlapping columns under ``left`` and ``right``.

    The helper mutates the two shallow working frames used during
    materialization. A leading level distinguishes columns originating from
    each side, while all original column levels remain unchanged beneath it.

    Args:
        df: Left working dataframe.
        right: Right working dataframe.

    Returns:
        The same two dataframes with MultiIndex columns containing a source
        namespace.
    """
    header = np.empty(df.columns.size, dtype="U4")
    header[:] = "left"
    header = [header]
    columns = [df.columns.get_level_values(n) for n in range(df.columns.nlevels)]
    header.extend(columns)
    df.columns = pd.MultiIndex.from_arrays(header)
    header = np.empty(right.columns.size, dtype="U5")
    header[:] = "right"
    header = [header]
    columns = [right.columns.get_level_values(n) for n in range(right.columns.nlevels)]
    header.extend(columns)
    right.columns = pd.MultiIndex.from_arrays(header)
    return df, right


def _preserve_object_dtype(array: np.ndarray, dtype) -> np.ndarray | pd.Series:
    """Guard a reindexed ``object``-dtype array against string inference.

    ``pandas``'s ``future.infer_string`` option (default-on since pandas 3.0)
    reinterprets a bare ``object`` ndarray of Python strings as the new
    ``str`` dtype the moment it is handed to the ``pd.DataFrame`` constructor
    — even though the source column was genuinely ``object`` (e.g. mixed
    types, or a string column the caller deliberately kept as ``object``).
    ``pd.merge`` never hits this because it reassigns dtypes at the block
    level instead of rebuilding columns from raw arrays. Wrapping the array
    in a dtype-tagged ``Series`` here is what makes a dict-of-arrays
    ``pd.DataFrame(...)`` call respect that dtype the same way.

    Non-object dtypes pass straight through unchanged.
    """
    if dtype == object:
        return pd.Series(array, dtype=object, copy=False)
    return array


def _materialize_index_result(
    df: pd.DataFrame,
    right: pd.DataFrame,
    left_index: np.ndarray,
    right_index: np.ndarray,
    how: str,
    df_columns: Any,
    right_columns: Any,
    indicator: bool | str,
    include_join_positions: bool,
) -> pd.DataFrame:
    """Materialize a conditional-join result from physical row positions.

    The join kernels return zero-based physical positions, not pandas index
    labels. This helper gathers matching rows, appends unmatched rows for
    outer-style joins, preserves extension-array dtypes while creating
    missing values, and optionally adds the merge indicator and matched
    physical positions as the result index.

    Args:
        df: Prepared left dataframe whose rows are addressed by physical
            positions.
        right: Prepared right dataframe whose rows are addressed by physical
            positions.
        left_index: Physical left positions for matched pairs.
        right_index: Physical right positions for matched pairs.
        how: Join shape: ``"inner"``, ``"left"``, ``"right"``, or
            ``"outer"``.
        df_columns: Deprecated left-column selection.
        right_columns: Deprecated right-column selection.
        indicator: ``False`` to omit the indicator, ``True`` for ``"_merge"``,
            or a custom indicator column name.
        include_join_positions: If true, use matched physical positions as a
            two-level result index. This is valid for inner joins only.

    Returns:
        The materialized joined dataframe.

    Raises:
        ValueError: If both deprecated column selectors are ``None`` or the
            requested indicator name collides with an output column.
    """
    # TODO: deprecate df_columns and right_columns
    # user can handle column renaming before the join
    if (df_columns is None) and (right_columns is None):
        raise ValueError("df_columns and right_columns cannot both be None.")
    if (df_columns is not None) and (df_columns != slice(None)):
        df = df.select_columns(df_columns)
    if (right_columns is not None) and (right_columns != slice(None)):
        right = right.select_columns(right_columns)
    if df_columns is None:
        df = pd.DataFrame([])
    elif right_columns is None:
        right = pd.DataFrame([])

    if not df.columns.intersection(right.columns).empty:
        df, right = _create_multiindex_column(df, right)

    def _add_indicator(
        indicator: bool | str,
        labels: tuple[str, ...],
        lengths: tuple[int, ...],
        columns: pd.Index,
    ) -> tuple[object, pd.Categorical]:
        """Build one categorical indicator array for all output segments."""
        name: object = "_merge" if isinstance(indicator, bool) else indicator
        if name in columns:
            raise ValueError(
                "Cannot use name of an existing column for indicator column"
            )
        if columns.nlevels > 1:
            name = tuple([name] + [""] * (columns.nlevels - 1))
        categories = ["left_only", "right_only", "both"]
        segments = []
        for label, length in zip(labels, lengths):
            if length:
                segment = pd.Categorical([label], categories=categories).repeat(length)
                segments.append(segment)
        if not segments:
            return name, pd.Categorical([], categories=categories)
        if len(segments) == 1:
            return name, segments[0]
        return name, concat_compat(segments)

    def _inner(
        left_positions: np.ndarray,
        right_positions: np.ndarray,
    ) -> pd.DataFrame:
        """Build matched rows without creating intermediate frames."""
        dictionary = {}
        for key, value in df.items():
            dictionary[key] = _preserve_object_dtype(
                value._values[left_positions], value.dtype
            )
        for key, value in right.items():
            dictionary[key] = _preserve_object_dtype(
                value._values[right_positions], value.dtype
            )
        if indicator:
            name, values = _add_indicator(
                indicator,
                ("both",),
                (left_positions.size,),
                df.columns.union(right.columns),
            )
            dictionary[name] = values
        if include_join_positions:
            index = pd.MultiIndex.from_arrays([left_positions, right_positions])
            return pd.DataFrame(dictionary, copy=False, index=index)
        return pd.DataFrame(dictionary, copy=False)

    if how == "inner":
        return _inner(left_index, right_index)

    left_unmatched = np.empty(0, dtype=np.intp)
    right_unmatched = np.empty(0, dtype=np.intp)
    if how in {"left", "outer"}:
        left_matched = np.zeros(len(df), dtype=bool)
        left_matched[left_index] = True
        left_unmatched = np.flatnonzero(~left_matched)
    if how in {"right", "outer"}:
        right_matched = np.zeros(len(right), dtype=bool)
        right_matched[right_index] = True
        right_unmatched = np.flatnonzero(~right_matched)

    dictionary = {}
    for key, value in df.items():
        array = value._values
        segments = [array[left_index]]
        if left_unmatched.size:
            segments.append(array[left_unmatched])
        if right_unmatched.size:
            segments.append(
                construct_1d_array_from_inferred_fill_value(
                    value=array[:1],
                    length=right_unmatched.size,
                )
            )
        if len(segments) == 1:
            dictionary[key] = _preserve_object_dtype(segments[0], value.dtype)
        else:
            dictionary[key] = _preserve_object_dtype(
                concat_compat(segments), value.dtype
            )

    for key, value in right.items():
        array = value._values
        segments = [array[right_index]]
        if left_unmatched.size:
            segments.append(
                construct_1d_array_from_inferred_fill_value(
                    value=array[:1],
                    length=left_unmatched.size,
                )
            )
        if right_unmatched.size:
            segments.append(array[right_unmatched])
        if len(segments) == 1:
            dictionary[key] = _preserve_object_dtype(segments[0], value.dtype)
        else:
            dictionary[key] = _preserve_object_dtype(
                concat_compat(segments), value.dtype
            )

    if indicator:
        labels = ["both"]
        lengths = [left_index.size]
        if left_unmatched.size:
            labels.append("left_only")
            lengths.append(left_unmatched.size)
        if right_unmatched.size:
            labels.append("right_only")
            lengths.append(right_unmatched.size)
        name, values = _add_indicator(
            indicator,
            tuple(labels),
            tuple(lengths),
            df.columns.union(right.columns),
        )
        dictionary[name] = values

    return pd.DataFrame(dictionary, copy=False)


@dataclass(frozen=True, slots=True)
class JoinCondition:
    """Normalized internal representation of one conditional-join predicate.

    The public API continues to accept three-element tuples. PyJanitor converts
    those tuples to this immutable object immediately after public validation,
    so routing and preparation code can use descriptive attributes instead of
    remembering whether field ``0``, ``1``, or ``2`` means the operator.

    Args:
        left: Left dataframe column label.
        right: Right dataframe column label.
        op: Comparison operator, such as ``"<"`` or ``"!="``.
    """

    left: Hashable
    right: Hashable
    op: str


def _normalize_conditions(conditions: Sequence[tuple]) -> list[JoinCondition]:
    """Convert validated public condition tuples to immutable objects.

    Args:
        conditions: Three-element public condition tuples, or conditions that
            have already been normalized by an internal caller.

    Returns:
        A new list containing one :class:`JoinCondition` per input predicate.
    """
    return [
        condition if isinstance(condition, JoinCondition) else JoinCondition(*condition)
        for condition in conditions
    ]


def _sort_if_not_monotonic(series: pd.Series) -> tuple[pd.Series, bool]:
    """Normalize a series to ascending order without losing row identity.

    An already increasing series is returned unchanged. A decreasing series
    is reversed, which preserves its values and index pairing without a full
    sort. Other non-monotonic series use a stable sort so duplicate values
    retain deterministic physical order. The returned pandas index is part of
    the contract: callers must use it to select and reorder every dependent
    right-side array, including residual predicates and aggregation inputs.

    Args:
        series: Non-null pandas series used as a Rust binary-search layout.

    Returns:
        ``(ordered_series, was_already_increasing)``. The boolean describes
        the input ordering, not whether sorting was required by the caller.
    """

    is_sorted = series.is_monotonic_increasing
    if is_sorted:
        return series, True
    if series.is_monotonic_decreasing:
        return series.iloc[::-1], False
    return series.sort_values(kind="stable"), False


def _convert_array_to_numpy(
    array: np.ndarray,
    na_value: int = 0,
) -> np.ndarray:
    """Convert pandas-backed values to the NumPy dtype expected by Rust.

    Nullable extension arrays need an explicit ``na_value`` before they can
    cross the PyO3 boundary. Datetime and timedelta values are viewed as their
    int64 nanosecond representation so the numeric Rust kernels can compare
    them without losing physical row alignment.

    Args:
        array: NumPy array, pandas extension array, or pandas-backed values.
        na_value: Scalar used for missing entries in non-mask value arrays.

    Returns:
        A NumPy array suitable for a dtype-specialized kernel.
    """
    if pd.api.types.is_extension_array_dtype(array):
        array_dtype = getattr(array.dtype, "numpy_dtype", None)
        if array_dtype is None:
            array = array.to_numpy(na_value=na_value, copy=False)
        else:
            array = array.to_numpy(dtype=array_dtype, na_value=na_value, copy=False)
    if pd.api.types.is_timedelta64_dtype(array):
        array = array.to_numpy(copy=False)
    if pd.api.types.is_datetime64_dtype(array) or pd.api.types.is_timedelta64_dtype(
        array
    ):
        array = array.view(np.int64)
    return array


def _build_residual_predicate(
    left: pd.Series,
    right: pd.Series,
    operation: str,
    left_index: pd.Index | slice = slice(None),
    right_index: pd.Index | slice = slice(None),
) -> tuple:
    """Build one residual predicate in the Rust tuple format.

    Residual arrays are already aligned to the anchor's physical layout. This
    helper only converts their values and, for ``!=``, attaches authoritative
    null masks; it does not sort, filter, or reset either series.

    Args:
        left: Left residual series in anchor-aligned physical order.
        right: Right residual series in the same aligned order.
        operation: String comparison operator.
        right_index: Right index positions to align the right series to.

    Returns:
        A three-element tuple containing ``left_array``, ``right_array``, and
        ``operation`` for ordinary predicates. For null-aware ``!=``
        predicates, returns a six-element tuple containing the two value
        arrays, their null masks, the extension-array flag, and the operator
        in the format expected by the Rust parser.

    Position invariant:
        ``left`` and ``right`` must already be in the anchor layout. This
        helper converts values and masks; it does not sort or independently
        align rows. The resulting arrays are indexed by the physical
        positions returned by the selected Rust kernel.
    """
    if left_index is None:
        left_index = slice(None)
    if right_index is None:
        right_index = slice(None)
    left = left.loc[left_index]
    left_array = _convert_array_to_numpy(array=left._values)
    right = right.loc[right_index]
    right_array = _convert_array_to_numpy(array=right._values)
    if operation != "!=":
        return (
            left_array,
            right_array,
            operation,
        )

    left_null_mask, right_null_mask, is_extension_array = _get_boolean_args_for_ne(
        op=operation,
        left=left,
        right=right,
    )
    if left_null_mask is None and right_null_mask is None:
        return (
            left_array,
            right_array,
            operation,
        )
    return (
        left_array,
        left_null_mask,
        right_array,
        right_null_mask,
        bool(is_extension_array),
        operation,
    )


def _get_boolean_args_for_ne(
    op: str, left: np.ndarray | None, right: np.ndarray | None
) -> tuple:
    """Build null masks and the extension-array flag for ``!=``.

    Ordinary range predicates remove null rows before dispatch. ``!=`` is
    different: NumPy-backed nulls match according to the dedicated all-`!=`
    contract, while pandas extension-array nulls do not match. The returned
    masks therefore travel with residual predicates so Rust can distinguish
    missing values from converted numeric sentinels.

    Args:
        op: Predicate operator; only ``!=`` requests masks.
        left: Left residual series or array-like values.
        right: Right residual series or array-like values.

    Returns:
        ``(left_null_mask, right_null_mask, is_extension_array)``. When no
        null is present, both masks are ``None`` and the flag is ``False``.
    """
    if op != "!=":
        return None, None, False
    left_booleans = left.isna()
    right_booleans = right.isna()
    if not any((left_booleans.any(), right_booleans.any())):
        return None, None, False
    is_extension_array = pd.api.types.is_extension_array_dtype(left)
    left_booleans = left_booleans.to_numpy(na_value=False, copy=False, dtype=np.bool_)
    right_booleans = right_booleans.to_numpy(na_value=False, copy=False, dtype=np.bool_)
    return left_booleans, right_booleans, is_extension_array


def _get_indexer_for_non_null_rows(df, columns_and_ops):
    """Return rows valid for ordinary range predicates.

    Null masks are combined with logical OR across the predicate columns: a
    row is excluded when any ordinary predicate column is null. ``!=`` is
    skipped because its null behavior is handled by its dedicated mask
    contract. ``slice(None)`` represents a fully valid layout and avoids
    allocating a full positional indexer; ``None`` means no usable row
    remains.

    Args:
        df: Working dataframe whose rows are addressed by physical position.
        columns_and_ops: ``(column_name, operator)`` pairs for one join side.

    Returns:
        ``None`` for an all-null usable side, ``slice(None)`` when no filtering
        is needed, or an indexer selecting the valid physical rows.
    """
    booleans = None
    for column, op in columns_and_ops:
        if op == _JoinOperator.NOT_EQUAL.value:
            continue
        column = df[column]
        new_booleans = column.isna()
        if booleans is None:
            booleans = new_booleans
        else:
            booleans = booleans | new_booleans
    if booleans.all():
        return None
    if booleans.any():
        return df.index[~booleans]
    return slice(None)
