from __future__ import annotations

import numpy as np
import pandas as pd

from . import aggs, helpers


def _single_le_ge_join(
    df: pd.DataFrame,
    right: pd.DataFrame,
    condition: tuple,
    return_ranges: bool,
    aggfunc: list[tuple],
    keep: str,
) -> tuple:
    """
    Compute aggregate on '</<=/>/>='
    """
    left_on, right_on, op = condition
    booleans = helpers._maybe_remove_nulls_from_dataframe(
        df=df, columns=[left_on], return_bools=True
    )
    if booleans is None:
        return None
    right = helpers._maybe_remove_nulls_from_dataframe(
        df=right, columns=[right_on]
    )
    if right is None:
        return None
    right_is_sorted = True
    if not right[right_on].is_monotonic_increasing:
        right = right.sort_values(right_on, ignore_index=False, kind="stable")
        right_is_sorted = False
    len_df = len(df)
    len_right = len(right)
    starts = np.zeros(len_df, dtype=np.intp)
    ends = np.empty(len_df, dtype=np.intp)
    ends[:] = len_right
    sizes = np.zeros(len_df, dtype=np.intp)
    booleans = booleans.astype(np.int8, copy=False)
    indices = {
        "starts": starts,
        "ends": ends,
        "booleans": booleans,
        "sizes": sizes,
        "right_is_sorted": right_is_sorted,
    }
    indices = helpers._update_search_indices(
        left=df[left_on]._values,
        right=right[right_on]._values,
        indices=indices,
        op=op,
    )
    if indices is None:
        return None
    if aggfunc is not None:
        booleans = indices["booleans"]
        if not booleans.all():
            booleans = booleans.astype(np.bool_, copy=False)
            df_index = df.index._values[booleans]
            indices["counts_array"] = indices["sizes"][booleans]
        else:
            indices["counts_array"] = indices["sizes"]
            df_index = df.index._values
        results = aggs.compute_aggfunc_result(
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
    return helpers._build_indices_single_equi_or_true_range_join(
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
