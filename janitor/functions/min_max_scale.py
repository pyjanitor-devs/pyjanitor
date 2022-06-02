from __future__ import annotations

import pandas_flavor as pf
import pandas as pd

from janitor.utils import deprecated_alias
from janitor.utils import deprecated_kwargs


@pf.register_dataframe_method
@deprecated_kwargs(
    "old_min",
    "old_max",
    "new_min",
    "new_max",
    message=(
        "The keyword argument {argument!r} of {func_name!r} is deprecated. "
        "Please use 'feature_range' instead."
    ),
)
@deprecated_alias(col_name="column_name")
def min_max_scale(
    df: pd.DataFrame,
    feature_range: tuple[int | float, int | float] = (0, 1),
    column_name: str | int | list[str | int] | pd.Index = None,
    entire_data: bool = False,
) -> pd.DataFrame:
    """
    Scales data to between a minimum and maximum value.

    If `minimum` and `maximum` are provided, the true min/max of the
    `DataFrame` or column is ignored in the scaling process and replaced with
    these values, instead.

    One can optionally set a new target minimum and maximum value using the
    `feature_range[0]` and `feature_range[1]` keyword arguments.
    This will result in the transformed data being bounded between
    `feature_range[0]` and `feature_range[1]`.

    If a particular column name is specified, then only that column of data
    are scaled. Otherwise, the entire dataframe is scaled.

    Example: Basic usage.

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({'a':[1, 2], 'b':[0, 1]})
        >>> df.min_max_scale()
             a    b
        0  0.5  0.0
        1  1.0  0.5

    Example: Setting custom minimum and maximum.

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({'a':[1, 2], 'b':[0, 1]})
        >>> df.min_max_scale(feature_range=(0, 100))
               a     b
        0   50.0   0.0
        1  100.0  50.0

    Example: Apply min-max to the selected columns.

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({'a':[1, 2], 'b':[0, 1]})
        >>> df.min_max_scale(feature_range=(0, 100), column_name=['a', 'b'])
               a      b
        0    0.0    0.0
        1  100.0  100.0
        >>> df.min_max_scale(feature_range=(0, 100), column_name='a')
               a  b
        0    0.0  0
        1  100.0  1

    The aforementioned example might be applied to something like scaling the
    isoelectric points of amino acids. While technically they range from
    approx 3-10, we can also think of them on the pH scale which ranges from
    1 to 14. Hence, 3 gets scaled not to 0 but approx. 0.15 instead, while 10
    gets scaled to approx. 0.69 instead.

    :param df: A pandas DataFrame.
    :param feature_range: (optional) Desired range of transformed data.
    :param column_name: (optional) The column on which to perform scaling.
    :returns: A pandas DataFrame with scaled data.
    :raises ValueError: if `feature_range` isn't tuple type.
    :raises ValueError: if the length of `feature_range` isn't equal to two.
    :raises ValueError: if the element of `feature_range` isn't number type.
    :raises ValueError: if `feature_range[1]` <= `feature_range[0]`.
    """

    if not (
        isinstance(feature_range, (tuple, list))
        and len(feature_range) == 2
        and all((isinstance(i, (int, float))) for i in feature_range)
        and feature_range[1] > feature_range[0]
    ):
        raise ValueError(
            "`feature_range` should be a range type contains number element, "
            "the first element must be greater than the second one"
        )

    if column_name is not None:
        df = df.copy()  # Avoid to change the original DataFrame.

        old_feature_range = df[column_name].pipe(min_max_value, entire_data)
        df[column_name] = df[column_name].pipe(
            apply_min_max,
            *old_feature_range,
            *feature_range,
        )
    else:
        old_feature_range = df.pipe(min_max_value, entire_data)
        df = df.pipe(
            apply_min_max,
            *old_feature_range,
            *feature_range,
        )

    return df


def min_max_value(df: pd.DataFrame, entire_data: bool) -> tuple:
    """
    Return the minimum and maximum of DataFrame.

    Use the `entire_data` flag to control returning entire data or each column.
    """

    if entire_data:
        mmin = df.min().min()
        mmax = df.max().max()
    else:
        mmin = df.min()
        mmax = df.max()

    return mmin, mmax


def apply_min_max(
    df: pd.DataFrame,
    old_min: int | float | pd.Series,
    old_max: int | float | pd.Series,
    new_min: int | float | pd.Series,
    new_max: int | float | pd.Series,
) -> pd.DataFrame:
    """
    Apply minimax scaler to DataFrame.

    Notes
    -----
    - Inputting minimum and maximum type
        - int or float : It will apply minimax to the entire DataFrame.
        - Series : It will apply minimax to each column.
    """

    old_range = old_max - old_min
    new_range = new_max - new_min

    return (df - old_min) * new_range / old_range + new_min
