"""Fixtures for xarray tests."""
import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def da():
    """
    Input testing DataArray for clone_using and convert_datetime_to_number.

    It creates a two-dimensional array of random integers adds axis coordinates
    that are either linearly or log-spaced increments.

    Included is a simple metadata dictionary passed as `attrs`.

    .. # noqa: DAR201
    """
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
