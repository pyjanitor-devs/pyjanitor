from typing import Hashable, Iterable, Union
import pandas_flavor as pf
import pandas as pd


@pf.register_dataframe_method
def take_first(
    df: pd.DataFrame,
    subset: Union[Hashable, Iterable[Hashable]],
    by: Hashable,
    ascending: bool = True,
) -> pd.DataFrame:
    """
    Take the first row within each group specified by `subset`.

    This method does not mutate the original DataFrame.

    ```python
        import pandas as pd
        import janitor

        data = {
            "a": ["x", "x", "y", "y"],
            "b": [0, 1, 2, 3]
        }
        df = pd.DataFrame(data)

        df.take_first(subset="a", by="b")
    ```

    :param df: A pandas DataFrame.
    :param subset: Column(s) defining the group.
    :param by: Column to sort by.
    :param ascending: Whether or not to sort in ascending order, `bool`.
    :returns: A pandas DataFrame.
    """
    result = df.sort_values(by=by, ascending=ascending).drop_duplicates(
        subset=subset, keep="first"
    )

    return result
