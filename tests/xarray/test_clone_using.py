"""Tests for clone_using."""
import numpy as np
import pytest
import xarray as xr

import janitor  # noqa: F401


@pytest.mark.xarray
def test_successful_cloning_coords(da):
    """Test that clone_using coordinates works correctly."""

    # with copying coords
    new_da: xr.DataArray = da.clone_using(np.random.randn(*da.data.shape))

    with pytest.raises(AssertionError):
        np.testing.assert_equal(new_da.data, da.data)

    assert all(
        (
            new_coord == old_coord
            for new_coord, old_coord in zip(new_da.coords, da.coords)
        )
    )
    assert new_da.dims == da.dims


@pytest.mark.xarray
def test_successful_cloning_no_coords(da):
    """Test that cloning works without coordinates."""

    new_da: xr.DataArray = da.clone_using(
        np.random.randn(*da.data.shape), use_coords=False
    )

    with pytest.raises(AssertionError):
        np.testing.assert_equal(new_da.data, da.data)

    assert new_da.dims == da.dims


@pytest.mark.xarray
def test_metadata_cloning(da):
    """Test that metadata gets cloned over."""
    new_da: xr.DataArray = da.clone_using(
        np.random.randn(*da.data.shape), use_attrs=True, new_name="new_name"
    )

    assert new_da.name != da.name
    assert new_da.attrs == da.attrs


@pytest.mark.xarray
def test_no_coords_errors(da: xr.DataArray):
    """Test that errors are raised when dims do not match."""
    # number of dims should match
    with pytest.raises(ValueError):
        da.clone_using(np.random.randn(10, 10, 10), use_coords=False)

    # shape of each axis does not need to match
    da.clone_using(np.random.randn(10, 10), use_coords=False)


@pytest.mark.xarray
def test_coords_errors(da: xr.DataArray):
    # number of dims should match
    with pytest.raises(ValueError):
        da.clone_using(np.random.randn(10, 10, 10), use_coords=False)

    # shape of each axis must match when using coords
    with pytest.raises(ValueError):
        da.clone_using(np.random.randn(10, 10), use_coords=True)
