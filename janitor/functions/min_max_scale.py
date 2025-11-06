from __future__ import annotations

import pandas as pd
import pandas_flavor as pf

from janitor.utils import deprecated_alias, deprecated_kwargs


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
    jointly: bool = False,
) -> pd.DataFrame:
    """Scales DataFrame to between a minimum and maximum value.

    One can optionally set a new target **minimum** and **maximum** value
    using the `feature_range` keyword argument.

    If `column_name` is specified, then only that column(s) of data is scaled.
    Otherwise, the entire dataframe is scaled.
    If `jointly` is `True`, the `column_names` provided entire dataframe will
    be regnozied as the one to jointly scale. Otherwise, each column of data
    will be scaled separately.

    Examples:
        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({"a": [1, 2], "b": [0, 1]})
        >>> df.min_max_scale()
             a    b
        0  0.0  0.0
        1  1.0  1.0
        >>> df.min_max_scale(jointly=True)
             a    b
        0  0.5  0.0
        1  1.0  0.5

        Setting custom minimum and maximum.

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({"a": [1, 2], "b": [0, 1]})
        >>> df.min_max_scale(feature_range=(0, 100))
               a      b
        0    0.0    0.0
        1  100.0  100.0
        >>> df.min_max_scale(feature_range=(0, 100), jointly=True)
               a     b
        0   50.0   0.0
        1  100.0  50.0

        Apply min-max to the selected columns.

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({"a": [1, 2], "b": [0, 1], "c": [1, 0]})
        >>> df.min_max_scale(
        ...     feature_range=(0, 100),
        ...     column_name=["a", "c"],
        ... )
               a  b      c
        0    0.0  0  100.0
        1  100.0  1    0.0
        >>> df.min_max_scale(
        ...     feature_range=(0, 100),
        ...     column_name=["a", "c"],
        ...     jointly=True,
        ... )
               a  b     c
        0   50.0  0  50.0
        1  100.0  1   0.0
        >>> df.min_max_scale(feature_range=(0, 100), column_name="a")
               a  b  c
        0    0.0  0  1
        1  100.0  1  0

        The aforementioned example might be applied to something like scaling the
        isoelectric points of amino acids. While technically they range from
        approx 3-10, we can also think of them on the pH scale which ranges from
        1 to 14. Hence, 3 gets scaled not to 0 but approx. 0.15 instead, while 10
        gets scaled to approx. 0.69 instead.

    !!! summary "Version Changed"

        - 0.24.0
            - Deleted `old_min`, `old_max`, `new_min`, and `new_max` options.
            - Added `feature_range`, and `jointly` options.

    Args:
        df: A pandas DataFrame.
        feature_range: Desired range of transformed data.
        column_name: The column on which to perform scaling.
        jointly: Scale the entire data if True.

    Raises:
        ValueError: If `feature_range` isn't tuple type.
        ValueError: If the length of `feature_range` isn't equal to two.
        ValueError: If the element of `feature_range` isn't number type.
        ValueError: If `feature_range[1]` <= `feature_range[0]`.

    Returns:
        A pandas DataFrame with scaled data.
    """  # noqa: E501

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

        old_feature_range = df[column_name].pipe(_min_max_value, jointly)
        df[column_name] = df[column_name].pipe(
            _apply_min_max,
            *old_feature_range,
            *feature_range,
        )
    else:
        old_feature_range = df.pipe(_min_max_value, jointly)
        df = df.pipe(
            _apply_min_max,
            *old_feature_range,
            *feature_range,
        )

    return df


def _min_max_value(df: pd.DataFrame, jointly: bool) -> tuple:
    """
    Return the minimum and maximum of DataFrame.

    Use the `jointly` flag to control returning entire data or each column.

    .. # noqa: DAR101
    .. # noqa: DAR201
    """
    mmin = df.min()
    mmax = df.max()
    if jointly:
        if not isinstance(mmin, int):
            mmin = mmin.min()
        if not isinstance(mmax, int):
            mmax = mmax.max()
    return mmin, mmax


def _apply_min_max(
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

    .. # noqa: DAR101
    .. # noqa: DAR201
    """

    old_range = old_max - old_min
    new_range = new_max - new_min

    return (df - old_min) * new_range / old_range + new_min
