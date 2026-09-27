"""Isolated regions dispatch for dual and multi-predicate range joins.

The Rust boundary uses these tuple shapes:

* primary anchors: ``(left, left_index, right, right_index, op)``;
* residual predicates: ``(left, right, op)`` after PyJanitor has removed
  nulls. Nullable ``!=`` residuals may use the existing six-element mask
  form.

The first two predicates are the primary inequality anchors. They are sorted
independently and Rust aligns them by their original index labels. Later
predicates remain aligned to that physical layout and are evaluated as
residual filters.
"""

from __future__ import annotations

import janitor_rs
import numpy as np
import pandas as pd

from janitor.functions._conditional_join import _dual_non_equi, _helpers


def _prepare_pair(df: pd.DataFrame, right: pd.DataFrame, condition):
    """Prepare one sorted, null-free primary anchor.

    Parameters
    ----------
    df, right : pandas.DataFrame
        Working left and right frames after null removal.
    condition : tuple
        ``(left_column, right_column, operator)`` inequality triple.

    Returns
    -------
    tuple
        ``(left_values, left_index, right_values, right_index, operator)``.

    Returns the five-element Rust anchor tuple
    ``(left, left_index, right, right_index, op)``.
    """
    left_on, right_on, op = condition
    left = df[left_on]
    right_values = right[right_on]
    left, _ = _helpers._sort_if_not_monotonic(left)
    right_values, _ = _helpers._sort_if_not_monotonic(right_values)
    return (
        _helpers._convert_array_to_numpy(left._values),
        _helpers._convert_array_to_numpy(left.index._values),
        _helpers._convert_array_to_numpy(right_values._values),
        _helpers._convert_array_to_numpy(right_values.index._values),
        op,
    )


def _align_pair(pair, left_index, right_index):
    """Align the second anchor to the first anchor's original layouts.

    Parameters
    ----------
    pair : tuple
        A five-element prepared anchor.
    left_index, right_index : array-like
        The first anchor's original identifiers and canonical layouts.

    Returns
    -------
    tuple
        The second anchor reordered to the first anchor's layouts.

    Notes
    -----
    ``left_index`` and ``right_index`` are unique original identifiers, so
    reindexing preserves the first anchor's physical layouts without a
    cross-pair dtype conversion.
    """
    left = pd.Series(pair[0], index=pair[1]).reindex(left_index)
    right = pd.Series(pair[2], index=pair[3]).reindex(right_index)
    return left.to_numpy(), left_index, right.to_numpy(), right_index, pair[4]


def _prepare_residual(df, right, condition, left_index, right_index):
    """Build an aligned residual predicate tuple.

    Parameters
    ----------
    df, right : pandas.DataFrame
        Working join frames.
    condition : tuple
        ``(left_column, right_column, operator)`` residual triple.
    left_index, right_index : array-like
        Canonical physical layouts established by the first anchor.

    Returns
    -------
    tuple
        ``(left_values, right_values, operator)`` aligned to the anchors.
    """
    left_on, right_on, op = condition
    return (
        _helpers._convert_array_to_numpy(df.loc[left_index, left_on]._values),
        _helpers._convert_array_to_numpy(right.loc[right_index, right_on]._values),
        op,
    )


def _build_primary_regions(df, right, first_condition, second_condition):
    """Prepare and align the two minimum inequality anchors.

    Parameters
    ----------
    df, right : pandas.DataFrame
        Null-filtered working frames.
    first_condition, second_condition : tuple
        ``(left_column, right_column, operator)`` inequality triples.

    Returns
    -------
    tuple
        Two five-element anchors, with the second aligned to the first.
    """
    first = _prepare_pair(df, right, first_condition)
    second = _prepare_pair(df, right, second_condition)
    return first, _align_pair(second, first[1], first[3])


def _as_predicates(df, right, primary_conditions, residual_conditions):
    """Return aligned primary anchors followed by aligned residuals.

    Parameters
    ----------
    df, right : pandas.DataFrame
        Working join frames.
    primary_conditions : sequence of tuple
        Exactly two inequality triples used to build regions.
    residual_conditions : sequence of tuple
        Additional predicates evaluated after both primary regions pass.

    Returns
    -------
    list of tuple
        Rust-compatible primary and residual predicate tuples.
    """
    first, second = _build_primary_regions(df, right, *primary_conditions)
    predicates = [first, second]
    predicates.extend(
        _prepare_residual(df, right, condition, first[1], first[3])
        for condition in residual_conditions
    )
    return predicates


def _result_dict(result):
    """Normalize a Rust result to PyJanitor's index mapping.

    Parameters
    ----------
    result : dict or None
        Optional dictionary returned by Rust.

    Returns
    -------
    dict
        ``left_index`` and ``right_index`` NumPy arrays.
    """
    empty = np.array([], dtype=np.intp)
    if result is None:
        return {"left_index": empty, "right_index": empty}
    return {
        "left_index": np.asarray(result["left_index"]),
        "right_index": np.asarray(result["right_index"]),
    }


def get_indices(
    df,
    right,
    primary_conditions,
    residual_conditions=(),
    return_matching_indices=True,
    keep="all",
):
    """Build dual regions, then apply residual predicates and ``keep``.

    Parameters
    ----------
    df, right : pandas.DataFrame
        The left and right join inputs after the caller's null policy.
    primary_conditions : sequence of tuple
        Exactly two ``(left_column, right_column, operator)`` inequalities.
    residual_conditions : sequence of tuple, optional
        Additional predicates evaluated after both primary regions pass.
    return_matching_indices : bool, default True
        Request all matches rather than a single ``keep`` selection.
    keep : {"all", "first", "last", "any"}
        Selection semantics when all matches are not requested.

    Returns
    -------
    dict
        Matching original left and right index arrays.

    Raises
    ------
    ValueError
        If exactly two primary anchors are not supplied.
    """
    if len(primary_conditions) != 2:
        raise ValueError("regions require exactly two primary predicates")
    predicates = _as_predicates(df, right, primary_conditions, residual_conditions)
    selection = "all" if return_matching_indices else keep
    if len(predicates) == 2:
        result = janitor_rs.region_indices(predicates, selection)
    else:
        result = janitor_rs.region_indices_extended(predicates, selection)
    return _result_dict(result)


def aggregate(
    df,
    right,
    primary_conditions,
    residual_conditions,
    aggregation,
):
    """Prepare predicates for dual or multi-predicate aggregation.

    Parameters
    ----------
    df, right : pandas.DataFrame
        Working join frames.
    primary_conditions : sequence of tuple
        Exactly two inequality triples used as region anchors.
    residual_conditions : sequence of tuple
        Additional predicates evaluated by the aggregation implementation.
    aggregation : callable
        Existing aggregation dispatcher receiving the prepared predicates.

    Returns
    -------
    object
        The value returned by ``aggregation``.

    ``aggregation`` receives the prepared predicate list and is responsible
    for selecting the existing forward or reverse range-aggregation kernel.
    Residual predicates remain after the two primary anchors and are evaluated
    before aggregation state is updated.
    """
    if len(primary_conditions) != 2:
        raise ValueError("regions require exactly two primary predicates")
    predicates = _as_predicates(df, right, primary_conditions, residual_conditions)
    return aggregation(predicates)


def python_fallback(df, right, first_condition, second_condition):
    """Run the legacy implementation for compatibility comparisons.

    Parameters
    ----------
    df, right : pandas.DataFrame
        Working join frames.
    first_condition, second_condition : tuple
        The two primary inequality triples.

    Returns
    -------
    dict or None
        The legacy implementation's result.
    """
    return _dual_non_equi._get_indices(
        df=df,
        right=right,
        first_condition=first_condition,
        second_condition=second_condition,
    )
