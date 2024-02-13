"""Implementation of the `jitter` function."""

from typing import Hashable, Iterable, Optional

import numpy as np
import pandas as pd
import pandas_flavor as pf

from janitor.utils import check


@pf.register_dataframe_method
def jitter(
    df: pd.DataFrame,
    column_name: Hashable,
    dest_column_name: str,
    scale: np.number,
    clip: Optional[Iterable[np.number]] = None,
    random_state: Optional[np.number] = None,
) -> pd.DataFrame:
    """Adds Gaussian noise (jitter) to the values of a column.

    A new column will be created containing the values of the original column
    with Gaussian noise added.
    For each value in the column, a Gaussian distribution is created
    having a location (mean) equal to the value
    and a scale (standard deviation) equal to `scale`.
    A random value is then sampled from this distribution,
    which is the jittered value.
    If a tuple is supplied for `clip`,
    then any values of the new column less than `clip[0]`
    will be set to `clip[0]`,
    and any values greater than `clip[1]` will be set to `clip[1]`.
    Additionally, if a numeric value is supplied for `random_state`,
    this value will be used to set the random seed used for sampling.
    NaN values are ignored in this method.

    This method mutates the original DataFrame.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({"a": [3, 4, 5, np.nan]})
        >>> df
             a
        0  3.0
        1  4.0
        2  5.0
        3  NaN
        >>> df.jitter("a", dest_column_name="a_jit", scale=1, random_state=42)
             a     a_jit
        0  3.0  3.496714
        1  4.0  3.861736
        2  5.0  5.647689
        3  NaN       NaN

    Args:
        df: A pandas DataFrame.
        column_name: Name of the column containing
            values to add Gaussian jitter to.
        dest_column_name: The name of the new column containing the
            jittered values that will be created.
        scale: A positive value multiplied by the original
            column value to determine the scale (standard deviation) of the
            Gaussian distribution to sample from. (A value of zero results in
            no jittering.)
        clip: An iterable of two values (minimum and maximum) to clip
            the jittered values to, default to None.
        random_state: An integer or 1-d array value used to set the random
            seed, default to None.

    Raises:
        TypeError: If `column_name` is not numeric.
        ValueError: If `scale` is not a numerical value
            greater than `0`.
        ValueError: If `clip` is not an iterable of length `2`.
        ValueError: If `clip[0]` is greater than `clip[1]`.

    Returns:
        A pandas DataFrame with a new column containing
            Gaussian-jittered values from another column.
    """

    # Check types
    check("scale", scale, [int, float])

    # Check that `column_name` is a numeric column
    if not np.issubdtype(df[column_name].dtype, np.number):
        raise TypeError(f"{column_name} must be a numeric column.")

    if scale <= 0:
        raise ValueError("`scale` must be a numeric value greater than 0.")
    values = df[column_name]
    if random_state is not None:
        np.random.seed(random_state)
    result = np.random.normal(loc=values, scale=scale)
    if clip:
        # Ensure `clip` has length 2
        if len(clip) != 2:
            raise ValueError("`clip` must be an iterable of length 2.")
        # Ensure the values in `clip` are ordered as min, max
        if clip[1] < clip[0]:
            raise ValueError(
                "`clip[0]` must be less than or equal to `clip[1]`."
            )
        result = np.clip(result, *clip)
    df[dest_column_name] = result

    return df
