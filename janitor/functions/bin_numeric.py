from typing import Hashable, Optional, Union, Sequence
import pandas_flavor as pf
import pandas as pd

from janitor.utils import deprecated_alias


ScalarSequence = Sequence[float]


@pf.register_dataframe_method
@deprecated_alias(
    from_column="from_column_name",
    to_column="to_column_name",
    num_bins="bins",
)
def bin_numeric(
    df: pd.DataFrame,
    from_column_name: Hashable,
    to_column_name: Hashable,
    bins: Optional[Union[int, ScalarSequence, pd.IntervalIndex]] = 5,
    **kwargs,
) -> pd.DataFrame:
    """
    Generate a new column that labels bins for a specified numeric column.

    This method mutates the original DataFrame.

    A wrapper around the pandas [`cut()`][pd_cut_docs] function to bin data of
    one column, generating a new column with the results.

    [pd_cut_docs]: https://pandas.pydata.org/docs/reference/api/pandas.cut.html

    ```python
    import pandas as pd
    import janitor
    df = (
        pd.DataFrame(...)
        .bin_numeric(
            from_column_name='col1',
            to_column_name='col1_binned',
            bins=3,
            labels=['1-2', '3-4', '5-6'],
        )
    )
    ```

    :param df: A pandas DataFrame.
    :param from_column_name: The column whose data you want binned.
    :param to_column_name: The new column to be created with the binned data.
    :param bins: The binning strategy to be utilized. Read the `pd.cut`
        documentation for more details.
    :param **kwargs: Additional kwargs to pass to `pd.cut`, except `retbins`.
    :return: A pandas DataFrame.
    :raises ValueError: If `retbins` is passed in as a kwarg.
    """
    if "retbins" in kwargs:
        raise ValueError("`retbins` is not an acceptable keyword argument.")

    df[str(to_column_name)] = pd.cut(
        df[str(from_column_name)],
        bins=bins,
        **kwargs,
    )

    return df
