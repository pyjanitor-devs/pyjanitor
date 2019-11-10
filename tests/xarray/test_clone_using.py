import pandas as pd
import xarray as xr
import numpy as np
import pytest

import janitor


@pytest.fixture
def da():
    da = xr.DataArray(
        np.random.randint(0, 100, size=(512, 1024)),
        dims=["random_ax_1", "random_ax_2"],
        coords=dict(
            random_ax_1=np.linspace(0, 1, 512),
            random_ax_2=np.logspace(-2, 2, 1024),
        ),
        name="blarg",
        attrs=dict(a=3, b=["asdf", "fdsa"]),
    )
    return da


@pytest.mark.xarray
def test_successful_cloning_coords(da):

    # with copying coords
    new_da: xr.DataArray = da.clone_using(np.random.randn(*da.data.shape))

    with pytest.raises(AssertionError):
        np.testing.assert_equal(new_da.data, da.data)

    assert [
        new_coord == old_coord
        for new_coord, old_coord in zip(new_da.coords, da.coords)
    ]
    assert new_da.dims == da.dims


@pytest.mark.xarray
def test_successful_cloning_no_coords(da):

    new_da: xr.DataArray = da.clone_using(
        np.random.randn(*da.data.shape), use_coords=False
    )

    with pytest.raises(AssertionError):
        np.testing.assert_equal(new_da.data, da.data)

    assert new_da.dims == da.dims


@pytest.mark.xarray
def test_metadata_cloning(da):
    new_da: xr.DataArray = da.clone_using(
        np.random.randn(*da.data.shape), use_attrs=True, new_name="new_name"
    )

    assert new_da.name != da.name
    assert new_da.attrs == da.attrs


@pytest.mark.xarray
def test_no_coords_errors(da: xr.DataArray):
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
