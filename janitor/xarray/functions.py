"""
Functions to augment XArray DataArrays and Datasets with additional
functionality.
"""

from typing import Union

import numpy as np
import pandas_flavor as pf
import xarray as xr

from ..utils import import_message

try:
    import sparse
except ImportError:
    import_message(
        submodule="xarray",
        package="sparse",
        conda_channel="conda-forge",
        pip_install=True,
    )


@pf.register_xarray_dataarray_method
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

    Examples:
        Making a new `DataArray` from a previous one, keeping the
        dimension names but dropping the coordinates (the input NumPy array
        is of a different size):

        >>> import xarray as xr
        >>> import janitor.xarray
        >>> da = xr.DataArray(
        ...     np.zeros((512, 1024)),
        ...     dims=["ax_1", "ax_2"],
        ...     coords=dict(ax_1=np.linspace(0, 1, 512), ax_2=np.logspace(-2, 2, 1024)),
        ...     name="original",
        ... )
        >>> new_da = da.clone_using(
        ...     np.ones((4, 6)),
        ...     new_name="new_and_improved",
        ...     use_coords=False,
        ... )
        >>> new_da
        <xarray.DataArray 'new_and_improved' (ax_1: 4, ax_2: 6)> Size: 192B
        array([[1., 1., 1., 1., 1., 1.],
               [1., 1., 1., 1., 1., 1.],
               [1., 1., 1., 1., 1., 1.],
               [1., 1., 1., 1., 1., 1.]])
        Dimensions without coordinates: ax_1, ax_2

    Args:
        da: The `DataArray` supplied by the method itself.
        np_arr: The NumPy array which will be wrapped in a new `DataArray`
            given the properties copied over from the source `DataArray`.
        use_coords: If `True`, use the coordinates of the source
            `DataArray` for the coordinates of the newly-generated array.
            Shapes must match in this case. If `False`, only the number of
            dimensions must match.
        use_attrs: If `True`, copy over the `attrs` from the source
            `DataArray`.
            The data inside `attrs` itself is not copied, only the mapping.
            Otherwise, use the supplied attrs.
        new_name: If set, use as the new name of the returned `DataArray`.
            Otherwise, use the name of `da`.

    Raises:
        ValueError: If number of dimensions in `NumPy` array and
            `DataArray` do not match.
        ValueError: If shape of `NumPy` array and `DataArray`
            do not match.

    Returns:
        A `DataArray` styled like the input `DataArray` containing the
            NumPy array data.
    """

    if np_arr.ndim != da.ndim:
        raise ValueError(
            "Number of dims in the NumPy array and the DataArray must match."
        )

    if use_coords and not all(
        np_ax_len == da_ax_len for np_ax_len, da_ax_len in zip(np_arr.shape, da.shape)
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


@pf.register_xarray_dataset_method
@pf.register_xarray_dataarray_method
def convert_datetime_to_number(
    da_or_ds: Union[xr.DataArray, xr.Dataset],
    time_units: str,
    dim: str = "time",
) -> Union[xr.DataArray, xr.Dataset]:
    """Convert the coordinates of a datetime axis to a human-readable float
    representation.

    Examples:
        Convert a `DataArray`'s time dimension coordinates from
        minutes to seconds:

        >>> import numpy as np
        >>> import xarray as xr
        >>> import janitor.xarray
        >>> timepoints = 5
        >>> da = xr.DataArray(
        ...     np.array([2, 8, 0, 1, 7, 7]),
        ...     dims="time",
        ...     coords=dict(time=np.arange(6) * np.timedelta64(1, "m")),
        ... )
        >>> da_minutes = da.convert_datetime_to_number("s", dim="time")
        >>> da_minutes
        <xarray.DataArray (time: 6)> Size: 48B
        array([2, 8, 0, 1, 7, 7])
        Coordinates:
          * time     (time) float64 48B 0.0 60.0 120.0 180.0 240.0 300.0

    Args:
        da_or_ds: XArray object.
        time_units: Numpy timedelta string specification for the unit you
            would like to convert the coordinates to.
        dim: The time dimension whose coordinates are datetime objects.

    Returns:
        The original XArray object with the time dimension reassigned.
    """

    times = da_or_ds.coords[dim].data / np.timedelta64(1, time_units)

    return da_or_ds.assign_coords({dim: times})


@pf.register_xarray_dataarray_method
def to_scipy_sparse(da: xr.DataArray):
    """
    Convert a 2-dimensional `DataArray` to a `scipy.sparse` matrix.

    Uses the [sparse](https://github.com/pydata/sparse) package as an
    intermediate step, so only non-zero entries are kept. `scipy.sparse`
    only supports 2-dimensional matrices, so `da` must be 2-dimensional.

    Examples:
        Converting a mostly-zero `DataArray` to a `scipy.sparse` matrix:

        >>> import numpy as np
        >>> import xarray as xr
        >>> import janitor.xarray
        >>> da = xr.DataArray(
        ...     np.array([[0, 0, 3], [4, 0, 0]]),
        ...     dims=["row", "col"],
        ... )
        >>> mat = da.to_scipy_sparse()  # doctest: +SKIP
        >>> mat.toarray()  # doctest: +SKIP
        array([[0, 0, 3],
               [4, 0, 0]])

    Args:
        da: The `DataArray` supplied by the method itself. Must have
            exactly 2 dimensions.

    Raises:
        ValueError: If `da` does not have exactly 2 dimensions.

    Returns:
        A `scipy.sparse` CSR matrix containing the non-zero values of `da`.
    """
    if da.ndim != 2:
        raise ValueError(
            "`to_scipy_sparse` only supports 2-dimensional DataArrays, "
            f"but the supplied DataArray has {da.ndim} dimension(s)."
        )

    if isinstance(da.data, sparse.COO):
        return da.data.tocsr()

    return sparse.COO.from_numpy(np.asarray(da.data)).tocsr()


@pf.register_xarray_dataarray_method
def from_scipy_sparse(
    da: xr.DataArray,
    scipy_sparse_mat,
    use_coords: bool = False,
    use_attrs: bool = False,
    new_name: str = None,
) -> xr.DataArray:
    """
    Given a 2-dimensional `scipy.sparse` matrix, return a `DataArray`
    that mirrors the dimension names (and, optionally, coordinates and
    attrs) of the supplied `DataArray`.

    This is the inverse of `to_scipy_sparse`, and is implemented as a
    thin wrapper around `clone_using`, converting `scipy_sparse_mat` to
    a `sparse.COO`-backed array first. Since `scipy_sparse_mat` will
    generally not have the same shape as `da`, `use_coords` defaults to
    `False` here (unlike in `clone_using`), so shapes are not required
    to match unless you explicitly ask for coordinates to be copied over.

    Examples:
        Rebuilding a `DataArray` from a `scipy.sparse` matrix, keeping
        only the dimension names of the original:

        >>> import numpy as np
        >>> import scipy.sparse
        >>> import xarray as xr
        >>> import janitor.xarray
        >>> da = xr.DataArray(
        ...     np.zeros((2, 3)),
        ...     dims=["row", "col"],
        ...     name="original",
        ... )
        >>> mat = scipy.sparse.csr_matrix([[0, 0, 3], [4, 0, 0]])
        >>> new_da = da.from_scipy_sparse(mat)  # doctest: +SKIP
        >>> new_da.dims  # doctest: +SKIP
        ('row', 'col')
        >>> new_da.data.todense()  # doctest: +SKIP
        array([[0, 0, 3],
               [4, 0, 0]])

    Args:
        da: The `DataArray` supplied by the method itself, used as the
            template for dimension names and (optionally) coordinates
            and attrs.
        scipy_sparse_mat: A 2-dimensional `scipy.sparse` matrix.
        use_coords: If `True`, use the coordinates of `da` for the
            coordinates of the newly-generated array. `da` and
            `scipy_sparse_mat` must have the same shape in that case.
            Defaults to `False`.
        use_attrs: If `True`, copy over the `attrs` from `da`.
        new_name: If set, use as the name of the returned `DataArray`.
            Otherwise, use the name of `da`.

    Returns:
        A new `DataArray`, with `sparse.COO`-backed data restored from
            `scipy_sparse_mat`.
    """
    coo = sparse.COO.from_scipy_sparse(scipy_sparse_mat)

    return da.clone_using(
        coo,
        use_coords=use_coords,
        use_attrs=use_attrs,
        new_name=new_name,
    )
