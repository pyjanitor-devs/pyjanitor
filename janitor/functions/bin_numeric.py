from typing import Optional, Union, Sequence
import pandas_flavor as pf
import pandas as pd

from janitor.utils import check, check_column, deprecated_alias


ScalarSequence = Sequence[float]


@pf.register_dataframe_method
@deprecated_alias(
    from_column="from_column_name",
    to_column="to_column_name",
    num_bins="bins",
)
def bin_numeric(
    df: pd.DataFrame,
    from_column_name: str,
    to_column_name: str,
    bins: Optional[Union[int, ScalarSequence, pd.IntervalIndex]] = 5,
    **kwargs,
) -> pd.DataFrame:
    """
    Generate a new column that labels bins for a specified numeric column.

    This method does not mutate the original DataFrame.

    A wrapper around the pandas [`cut()`][pd_cut_docs] function to bin data of
    one column, generating a new column with the results.

    [pd_cut_docs]: https://pandas.pydata.org/docs/reference/api/pandas.cut.html

    Example: Binning a numeric column with specific bin edges.

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({"a": [3, 6, 9, 12, 15]})
        >>> df.bin_numeric(
        ...     from_column_name="a", to_column_name="a_binned",
        ...     bins=[0, 5, 11, 15],
        ... )
            a  a_binned
        0   3    (0, 5]
        1   6   (5, 11]
        2   9   (5, 11]
        3  12  (11, 15]
        4  15  (11, 15]

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

    check("from_column_name", from_column_name, [str])
    check("to_column_name", to_column_name, [str])
    check_column(df, from_column_name)

    df = df.assign(
        **{
            to_column_name: pd.cut(df[from_column_name], bins=bins, **kwargs),
        }
    )

    return df
