"""Tests for datetime_conversion."""
import numpy as np
import pytest
import xarray as xr


@pytest.mark.xarray
def test_datetime_conversion(da):
    """Test that datetime conversion works on DataArrays."""
    seconds_arr = np.arange(512)

    # dataarrays
    new_da = da.assign_coords(
        random_ax_1=1e9 * seconds_arr * np.timedelta64(1, "ns")
    ).convert_datetime_to_number("m", dim="random_ax_1")

    # account for rounding errors
    np.testing.assert_array_almost_equal(
        new_da.coords["random_ax_1"].data, 1 / 60 * seconds_arr
    )

    # datasets
    new_ds = xr.Dataset(
        dict(
            array=da.assign_coords(
                random_ax_1=1e9 * seconds_arr * np.timedelta64(1, "ns")
            )
        )
    ).convert_datetime_to_number("m", dim="random_ax_1")

    np.testing.assert_array_almost_equal(
        new_ds.coords["random_ax_1"].data, 1 / 60 * seconds_arr
    )
