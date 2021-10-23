from typing import Hashable, Optional
import pandas_flavor as pf
import pandas as pd

from janitor.utils import deprecated_alias


@pf.register_dataframe_method
@deprecated_alias(from_column="from_column_name", to_column="to_column_name")
def bin_numeric(
    df: pd.DataFrame,
    from_column_name: Hashable,
    to_column_name: Hashable,
    num_bins: int = 5,
    labels: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate a new column that labels bins for a specified numeric column.

    This method mutates the original DataFrame.

    Makes use of pandas `cut()` function to bin data of one column,
    generating a new column with the results.

    ```python
        import pandas as pd
        import janitor
        df = (
            pd.DataFrame(...)
            .bin_numeric(
                from_column_name='col1',
                to_column_name='col1_binned',
                num_bins=3,
                labels=['1-2', '3-4', '5-6']
                )
        )
    ```

    :param df: A pandas DataFrame.
    :param from_column_name: The column whose data you want binned.
    :param to_column_name: The new column to be created with the binned data.
    :param num_bins: The number of bins to be utilized.
    :param labels: Optionally rename numeric bin ranges with labels. Number of
        label names must match number of bins specified.
    :return: A pandas DataFrame.
    :raises ValueError: if number of labels do not match number of bins.
    """
    if not labels:
        df[str(to_column_name)] = pd.cut(
            df[str(from_column_name)], bins=num_bins
        )
    else:
        if not len(labels) == num_bins:
            raise ValueError("Number of labels must match number of bins.")

        df[str(to_column_name)] = pd.cut(
            df[str(from_column_name)], bins=num_bins, labels=labels
        )

    return df
