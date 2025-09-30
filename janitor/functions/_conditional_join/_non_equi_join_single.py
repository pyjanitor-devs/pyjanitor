from __future__ import annotations

import numpy as np
import pandas as pd

from . import _aggs, _helpers


def _single_le_ge_join(
    df: pd.DataFrame,
    right: pd.DataFrame,
    condition: tuple,
    return_ranges: bool,
    aggfunc: list[tuple] | None,
    keep: str,
) -> tuple | dict | None:
    """
    Compute single join on '</<=/>/>='
    """
    left_on, right_on, op = condition
    booleans = _helpers._maybe_remove_nulls_from_dataframe(
        df=df, columns=[left_on], return_bools=True
    )
    if booleans is None:
        return None
    right = _helpers._maybe_remove_nulls_from_dataframe(
        df=right, columns=[right_on]
    )
    if right is None:
        return None
    right_is_sorted = right[right_on].is_monotonic_increasing
    if not right_is_sorted:
        right = right.sort_values(right_on, ignore_index=False, kind="stable")
    len_df = len(df)
    indices = {
        "starts": np.empty(len_df, dtype=np.intp),
        "ends": np.empty(len_df, dtype=np.intp),
        "booleans": booleans.to_numpy(np.int8, copy=False),
        "sizes": np.empty(len_df, dtype=np.intp),
        "right_is_sorted": right_is_sorted,
    }
    indices = _helpers._update_search_indices(
        left=df[left_on]._values,
        right=right[right_on]._values,
        indices=indices,
        op=op,
        first_time=True,
    )
    if indices is None:
        return None
    if aggfunc is not None:
        booleans = indices["booleans"]
        df_index = df.index._values
        indices["counts_array"] = indices["sizes"]
        if not booleans.all():
            booleans = booleans.astype(np.bool_, copy=False)
            df_index = df.index._values[booleans]
            indices["counts_array"] = indices["counts_array"][booleans]
        results = _aggs.compute_aggfunc_result(
            aggfunc=aggfunc,
            agg_frame=right,
            indices=indices,
            total=indices["matches"],
        )
        return {"aggregates": results, "df_index": df_index}
    if keep == "all":
        total = indices["total"]
    else:
        total = indices["matches"]
    return _helpers._build_indices_single_equi_or_true_range_join(
        left_index=df.index._values,
        right_index=right.index._values,
        starts=indices["starts"],
        ends=indices["ends"],
        right_is_sorted=right_is_sorted,
        return_ranges=return_ranges,
        total=total,
        keep=keep,
        booleans=indices["booleans"],
    )
