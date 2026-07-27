"""Tests for to_scipy_sparse."""

import importlib

import numpy as np
import pytest
import scipy.sparse
import xarray as xr
from helpers import running_on_ci

import janitor  # noqa: F401

# Skip all tests if the `sparse` package is not installed
pytestmark = pytest.mark.skipif(
    (importlib.util.find_spec("sparse") is None) & ~running_on_ci(),
    reason="Sparse-conversion tests relying on the `sparse` package "
    "only required for CI",
)


@pytest.mark.xarray
def test_to_scipy_sparse_roundtrips_dense_values(da):
    """Converting to scipy.sparse and back to a dense array is lossless."""

    mat = da.to_scipy_sparse()

    assert scipy.sparse.issparse(mat)
    np.testing.assert_array_equal(mat.toarray(), da.data)


@pytest.mark.xarray
def test_to_scipy_sparse_keeps_only_nonzero_entries():
    """Only the non-zero entries of the DataArray are stored."""

    values = np.array([[0, 0, 3], [4, 0, 0]])
    da = xr.DataArray(values, dims=["row", "col"])

    mat = da.to_scipy_sparse()

    assert mat.nnz == 2
    np.testing.assert_array_equal(mat.toarray(), values)


@pytest.mark.xarray
def test_to_scipy_sparse_raises_on_wrong_ndim():
    """to_scipy_sparse only supports 2-dimensional DataArrays."""

    da_1d = xr.DataArray(np.array([0, 1, 2]), dims=["row"])

    with pytest.raises(ValueError):
        da_1d.to_scipy_sparse()
