"""
Functions to augment XArray DataArrays and Datasets with additional
functionality.
"""


from typing import Union

import numpy as np
from pandas_flavor import (
    register_xarray_dataarray_method,
    register_xarray_dataset_method,
)

import xarray as xr


@register_xarray_dataarray_method
def clone_using(
    da: xr.DataArray,
    np_arr: np.array,
    use_coords: bool = True,
    use_attrs: bool = False,
    new_name: str = None,
) -> xr.DataArray:
    """
    Given a NumPy array, return an XArray `DataArray` which contains the same
    dimension names and (optionally) coordinates and other properties as the
    supplied `DataArray`.

    This is similar to `xr.DataArray.copy()` with more specificity for
    the type of cloning you would like to perform - the different properties
    that you desire to mirror in the new `DataArray`.

    If the coordinates from the source `DataArray` are not desired, the shape
    of the source and new NumPy arrays don't need to match.
    The number of dimensions do, however.

    Usage example - making a new `DataArray` from a previous one, keeping the
    dimension names but dropping the coordinates (the input NumPy array is of a
    different size):

        >>> import xarray as xr
        >>> import janitor.xarray
        >>> da = xr.DataArray(
        ...     np.zeros((512, 1024)), dims=["ax_1", "ax_2"],
        ...     coords=dict(ax_1=np.linspace(0, 1, 512),
        ...                 ax_2=np.logspace(-2, 2, 1024)),
        ...     name="original"
        ... )
        >>> new_da = da.clone_using(
        ...     np.ones((4, 6)), new_name='new_and_improved', use_coords=False
        ... )
        >>> new_da
        <xarray.DataArray 'new_and_improved' (ax_1: 4, ax_2: 6)>
        array([[1., 1., 1., 1., 1., 1.],
               [1., 1., 1., 1., 1., 1.],
               [1., 1., 1., 1., 1., 1.],
               [1., 1., 1., 1., 1., 1.]])
        Dimensions without coordinates: ax_1, ax_2

    :param da: The `DataArray` supplied by the method itself.
    :param np_arr: The NumPy array which will be wrapped in a new `DataArray`
        given the properties copied over from the source `DataArray`.
    :param use_coords: If `True`, use the coordinates of the source
        `DataArray` for the coordinates of the newly-generated array. Shapes
        must match in this case. If `False`, only the number of dimensions
        must match.
    :param use_attrs: If `True`, copy over the `attrs` from the source
        `DataArray`.
        The data inside `attrs` itself is not copied, only the mapping.
        Otherwise, use the supplied attrs.
    :param new_name: If set, use as the new name of the returned `DataArray`.
        Otherwise, use the name of `da``.
    :return: A `DataArray` styled like the input `DataArray` containing the
        NumPy array data.
    :raises ValueError: if number of dimensions in `NumPy` array and
        `DataArray` do not match.
    :raises ValueError: if shape of `NumPy` array and `DataArray`
        do not match.
    """

    if np_arr.ndim != da.ndim:
        raise ValueError(
            "Number of dims in the NumPy array and the DataArray "
            "must match."
        )

    if use_coords and not all(
        np_ax_len == da_ax_len
        for np_ax_len, da_ax_len in zip(np_arr.shape, da.shape)
    ):
        raise ValueError(
            "Input NumPy array and DataArray must have the same "
            "shape if copying over coordinates."
        )

    return xr.DataArray(
        np_arr,
        dims=da.dims,
        coords=da.coords if use_coords else None,
        attrs=da.attrs.copy() if use_attrs else None,
        name=new_name if new_name is not None else da.name,
    )


@register_xarray_dataset_method
@register_xarray_dataarray_method
def convert_datetime_to_number(
    da_or_ds: Union[xr.DataArray, xr.Dataset],
    time_units: str,
    dim: str = "time",
):
    """
    Convert the coordinates of a datetime axis to a human-readable float
    representation.

    Usage example to convert a `DataArray`'s time dimension coordinates from
    a minutes to seconds:

        >>> import numpy as np
        >>> import xarray as xr
        >>> import janitor.xarray
        >>> timepoints = 5
        >>> da = xr.DataArray(
        ...     np.array([2, 8, 0, 1, 7, 7]),
        ...     dims='time',
        ...     coords=dict(time=np.arange(6) * np.timedelta64(1, 'm'))
        ... )
        >>> da_minutes = da.convert_datetime_to_number('s', dim='time')
        >>> da_minutes
        <xarray.DataArray (time: 6)>
        array([2, 8, 0, 1, 7, 7])
        Coordinates:
          * time     (time) float64 0.0 60.0 120.0 180.0 240.0 300.0

    :param da_or_ds: XArray object.
    :param time_units: Numpy timedelta string specification for the unit you
        would like to convert the coordinates to.
    :param dim: the time dimension whose coordinates are datetime objects.
    :return: The original XArray object with the time dimension reassigned.
    """

    times = da_or_ds.coords[dim].data / np.timedelta64(1, time_units)

    return da_or_ds.assign_coords({dim: times})
