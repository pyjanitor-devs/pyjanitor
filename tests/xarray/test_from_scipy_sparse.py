"""Tests for from_scipy_sparse."""

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
def test_from_scipy_sparse_restores_values():
    """The values of the scipy.sparse matrix are restored correctly."""

    values = np.array([[0, 0, 3], [4, 0, 0]])
    template = xr.DataArray(np.zeros((2, 3)), dims=["row", "col"])
    mat = scipy.sparse.csr_matrix(values)

    new_da = template.from_scipy_sparse(mat)

    np.testing.assert_array_equal(new_da.data.todense(), values)


@pytest.mark.xarray
def test_from_scipy_sparse_keeps_dim_names():
    """Dimension names are taken from the template DataArray."""

    template = xr.DataArray(np.zeros((2, 3)), dims=["row", "col"])
    mat = scipy.sparse.csr_matrix(np.array([[0, 0, 3], [4, 0, 0]]))

    new_da = template.from_scipy_sparse(mat)

    assert new_da.dims == ("row", "col")


@pytest.mark.xarray
def test_from_scipy_sparse_does_not_require_matching_shape_by_default():
    """Shapes of template and scipy_sparse_mat need not match, since
    use_coords defaults to False."""

    template = xr.DataArray(
        np.zeros((5, 5)),
        dims=["row", "col"],
        coords=dict(row=range(5), col=range(5)),
    )
    mat = scipy.sparse.csr_matrix(np.array([[0, 0, 3], [4, 0, 0]]))

    new_da = template.from_scipy_sparse(mat)

    assert new_da.shape == (2, 3)


@pytest.mark.xarray
def test_from_scipy_sparse_roundtrips_with_to_scipy_sparse(da):
    """to_scipy_sparse followed by from_scipy_sparse restores the
    original data."""

    mat = da.to_scipy_sparse()
    new_da = da.from_scipy_sparse(mat, use_coords=True)

    np.testing.assert_array_equal(new_da.data.todense(), da.data)
