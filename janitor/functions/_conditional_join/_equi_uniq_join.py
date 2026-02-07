from __future__ import annotations

import janitor_rs
import numpy as np
import pandas as pd

from janitor.functions._conditional_join import (
    _helpers,
)


def _get_indices(
    df: pd.DataFrame,
    right: pd.DataFrame,
    mapping: dict,
):
    empty_array = np.array([], dtype=np.intp)
    l_cols = []
    r_cols = []
    for left_col, right_col, _ in mapping["equals"]:
        l_cols.append(df[left_col]._values)
        r_cols.append(right[right_col]._values)
    if len(l_cols) > 1:
        l_cols = pd.MultiIndex.from_arrays(l_cols)
        r_cols = pd.MultiIndex.from_arrays(r_cols)
    else:
        l_cols = pd.Index(l_cols[0])
        r_cols = pd.Index(r_cols[0])
    left_index = df.index._values
    right_index = right.index._values
    indexers = r_cols.get_indexer(l_cols)
    booleans = indexers != -1
    if not booleans.any():
        return {
            "left_index": empty_array,
            "right_index": empty_array,
        }
    if not booleans.all() and not any(
        (
            mapping["le_or_ge"],
            mapping["le_lt"],
            mapping["ge_gt"],
            mapping["not_equals"],
        )
    ):
        indexers = indexers[booleans]
        left_index = left_index[booleans]
        right_index = right_index[indexers]
        return {
            "left_index": left_index,
            "right_index": right_index,
        }
    if not any(
        (
            mapping["le_or_ge"],
            mapping["le_lt"],
            mapping["ge_gt"],
            mapping["not_equals"],
        )
    ):
        right_index = right_index[indexers]
        return {
            "left_index": left_index,
            "right_index": right_index,
        }
    rest = mapping["le_or_ge"]
    rest.append(mapping["le_lt"])
    rest.append(mapping["ge_gt"])
    rest.extend(mapping["not_equals"])
    rest = filter(None, rest)
    rest = dict.fromkeys(rest)
    outcome = _helpers._update_positions_no_range_(
        df=df, right=right, conditions=rest, positions=indexers
    )
    if outcome is None:
        return {
            "left_index": empty_array,
            "right_index": empty_array,
        }
    left_index = janitor_rs.index_trim_positions(
        index=left_index,
        positions=outcome["positions"],
        length=outcome["total"],
    )
    right_index = janitor_rs.build_positional_index(
        index=right_index,
        positions=outcome["positions"],
        length=outcome["total"],
    )
    return {
        "left_index": left_index,
        "right_index": right_index,
    }
