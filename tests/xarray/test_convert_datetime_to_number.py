import numpy as np
import pytest


@pytest.mark.xarray
def test_datetime_conversion(da):
    seconds_arr = np.arange(512)

    new_da = da.assign_coords(
        random_ax_1=1e9 * seconds_arr * np.timedelta64(1, "ns")
    ).convert_datetime_to_number("m", dim="random_ax_1")

    # account for rounding errors
    np.testing.assert_array_almost_equal(
        new_da.coords["random_ax_1"].data, 1 / 60 * seconds_arr
    )
