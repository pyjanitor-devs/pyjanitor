"""
Numba utility functions for conditional join.
"""

import numpy as np
from janitor.utils import import_message

try:
    import numba as nb
except ImportError:
    import_message(
        submodule="conditional_join",
        package="numba",
        conda_channel="conda-forge",
        pip_install=False,
    )


@nb.njit(cache=True, parallel=True)
def _numba_keep_first(
    search_indices: np.ndarray, right_index: np.ndarray, len_right: int
) -> np.ndarray:
    """Parallelized output for first match."""
    length_of_indices = search_indices.size
    right_c = np.empty(shape=length_of_indices, dtype=np.intp)
    if len_right:
        for ind in nb.prange(length_of_indices):
            right_c[ind] = right_index[
                search_indices[ind] : len_right  # noqa:E203
            ].min()
    else:
        for ind in nb.prange(length_of_indices):
            right_c[ind] = right_index[
                : search_indices[ind]
            ].min()  # noqa:E203
    return right_c


@nb.njit(cache=True, parallel=True)
def _numba_keep_last(
    search_indices: np.ndarray, right_index: np.ndarray, len_right: int
) -> np.ndarray:
    """Parallelized output for last match."""
    length_of_indices = search_indices.size
    right_c = np.empty(shape=length_of_indices, dtype=np.intp)
    if len_right:
        for ind in nb.prange(length_of_indices):
            right_c[ind] = right_index[
                search_indices[ind] : len_right  # noqa:E203
            ].max()
    else:
        for ind in nb.prange(length_of_indices):
            right_c[ind] = right_index[
                : search_indices[ind]
            ].max()  # noqa:E203
    return right_c
