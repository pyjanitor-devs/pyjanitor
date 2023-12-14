"""Implementation of the `change_index_dtype` function."""
import pandas as pd
import pandas_flavor as pf

from janitor.utils import check


@pf.register_dataframe_method
def change_index_dtype(
    df: pd.DataFrame, dtype, axis: str = "index"
) -> pd.DataFrame:
    """Cast an index to a specified dtype ``dtype``.

    This method does not mutate the original DataFrame.

    Examples:
        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({
        ...     "class": ["bird", "bird", "bird", "mammal", "mammal"],
        ...     "max_speed": [389, 389, 24, 80, 21],
        ...     "type": ["falcon", "falcon", "parrot", "Lion", "Monkey"],
        ... })
        >>> df
            class  max_speed    type
        0    bird        389  falcon
        1    bird        389  falcon
        2    bird         24  parrot
        3  mammal         80    Lion
        4  mammal         21  Monkey
        >>> grouped_df = df.groupby("class").agg(["mean", "median"])
        >>> grouped_df  # doctest: +NORMALIZE_WHITESPACE
                 max_speed
                      mean median
        class
        bird    267.333333  389.0
        mammal   50.500000   50.5
        >>> grouped_df.collapse_levels(sep="_")  # doctest: +NORMALIZE_WHITESPACE
                max_speed_mean  max_speed_median
        class
        bird        267.333333             389.0
        mammal       50.500000              50.5

    Before applying `.collapse_levels`, the `.agg` operation returns a
    multi-level column DataFrame whose columns are `(level 1, level 2)`:

    ```python
    [("max_speed", "mean"), ("max_speed", "median")]
    ```

    `.collapse_levels` then flattens the column MultiIndex into a single
    level index with names:

    ```python
    ["max_speed_mean", "max_speed_median"]
    ```

    Args:
        df: A pandas DataFrame.
        dtype : str, data type or Mapping
            of index name/position -> data type.
            Use a str, numpy.dtype, pandas.ExtensionDtype,
            Python type to cast the entire Index
            to the same type.
            Alternatively, use a mapping, e.g. {index_name: dtype, ...},
            where index_name is an index name/position and dtype is a numpy.dtype
            or Python type to cast one or more of the DataFrame's
            Index to specific types.
        axis: 'index/columns'. Determines which axis to change the dtype(s).

    Returns:
        A pandas DataFrame with new Index.
    """  # noqa: E501

    check("axis", axis, [str])
    if axis not in {"index", "columns"}:
        raise ValueError("axis should be either index or columns.")

    df = df[:]
    current_index = getattr(df, axis)
    if not isinstance(current_index, pd.MultiIndex):
        current_index = current_index.astype(dtype)
        setattr(df, axis, current_index)
        return df

    if not isinstance(dtype, dict):
        dtype = {
            level_number: dtype
            for level_number in range(current_index.nlevels)
        }

    all_str = all(isinstance(level, str) for level in dtype)
    all_int = all(isinstance(level, int) for level in dtype)
    if not all_str | all_int:
        raise TypeError(
            "The levels in the dictionary "
            "should be either all strings or all integers."
        )

    dtype = {
        current_index._get_level_number(label): _dtype
        for label, _dtype in dtype.items()
    }

    new_levels = []
    codes = current_index.codes
    levels = current_index.levels

    for level_number in range(current_index.nlevels):
        _index = levels[level_number]
        if level_number in dtype:
            _dtype = dtype[level_number]
            _index = _index.astype(_dtype)
        new_levels.append(_index)

    current_index = pd.MultiIndex(
        levels=new_levels,
        codes=codes,
        names=current_index.names,
        copy=False,
        verify_integrity=False,
    )
    setattr(df, axis, current_index)
    return df
