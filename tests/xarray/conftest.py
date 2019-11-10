import pytest
import xarray as xr
import numpy as np


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
