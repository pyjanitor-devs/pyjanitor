"""Implementation of the `change_index_dtype` function."""

from __future__ import annotations

from typing import Union

import pandas as pd
import pandas_flavor as pf

from janitor.utils import check


@pf.register_dataframe_method
def change_index_dtype(
    df: pd.DataFrame, dtype: Union[str, dict], axis: str = "index"
) -> pd.DataFrame:
    """Cast an index to a specified dtype ``dtype``.

    This method does not mutate the original DataFrame.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> import janitor
        >>> rng = np.random.default_rng(seed=0)
        >>> np.random.seed(0)
        >>> tuples = list(zip(*[['bar', 'bar', 'baz', 'baz',
        ...             'foo', 'foo', 'qux', 'qux'],
        ...              [1.0, 2.0, 1.0, 2.0,
        ...               1.0, 2.0, 1.0, 2.0]]))
        >>> idx = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
        >>> df = pd.DataFrame(np.random.randn(8, 2), index=idx, columns=['A', 'B'])
        >>> df
                             A         B
        first second
        bar   1.0     1.764052  0.400157
              2.0     0.978738  2.240893
        baz   1.0     1.867558 -0.977278
              2.0     0.950088 -0.151357
        foo   1.0    -0.103219  0.410599
              2.0     0.144044  1.454274
        qux   1.0     0.761038  0.121675
              2.0     0.443863  0.333674
        >>> outcome=df.change_index_dtype(dtype=str)
        >>> outcome
                             A         B
        first second
        bar   1.0     1.764052  0.400157
              2.0     0.978738  2.240893
        baz   1.0     1.867558 -0.977278
              2.0     0.950088 -0.151357
        foo   1.0    -0.103219  0.410599
              2.0     0.144044  1.454274
        qux   1.0     0.761038  0.121675
              2.0     0.443863  0.333674
        >>> outcome.index.dtypes
        first     object
        second    object
        dtype: object
        >>> outcome=df.change_index_dtype(dtype={'second':int})
        >>> outcome
                             A         B
        first second
        bar   1       1.764052  0.400157
              2       0.978738  2.240893
        baz   1       1.867558 -0.977278
              2       0.950088 -0.151357
        foo   1      -0.103219  0.410599
              2       0.144044  1.454274
        qux   1       0.761038  0.121675
              2       0.443863  0.333674
        >>> outcome.index.dtypes
        first     object
        second     int64
        dtype: object
        >>> outcome=df.change_index_dtype(dtype={0:'category',1:int})
        >>> outcome
                             A         B
        first second
        bar   1       1.764052  0.400157
              2       0.978738  2.240893
        baz   1       1.867558 -0.977278
              2       0.950088 -0.151357
        foo   1      -0.103219  0.410599
              2       0.144044  1.454274
        qux   1       0.761038  0.121675
              2       0.443863  0.333674
        >>> outcome.index.dtypes
        first     category
        second       int64
        dtype: object

    Args:
        df: A pandas DataFrame.
        dtype : Use a str or dtype to cast the entire Index
            to the same type.
            Alternatively, use a dictionary to change the MultiIndex
            to new dtypes.
        axis: Determines which axis to change the dtype(s).
            Should be either 'index' or 'columns'.

    Returns:
        A pandas DataFrame with new Index.
    """  # noqa: E501

    check("axis", axis, [str])
    if axis not in {"index", "columns"}:
        raise ValueError("axis should be either index or columns.")

    df = df[:]
    current_index = getattr(df, axis)
    if not isinstance(current_index, pd.MultiIndex):
        if isinstance(dtype, dict):
            raise TypeError(
                "Changing the dtype via a dictionary "
                "is not supported for a single index."
            )
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
