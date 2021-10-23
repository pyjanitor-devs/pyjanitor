import pandas_flavor as pf
import pandas as pd
from janitor.utils import FILLTYPE, check, check_column
from operator import methodcaller


@pf.register_dataframe_method
def fill_direction(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Provide a method-chainable function for filling missing values
    in selected columns.

    It is a wrapper for `pd.Series.ffill` and `pd.Series.bfill`,
    and pairs the column name with one of `up`, `down`, `updown`,
    and `downup`.

    ```python
        import pandas as pd
        import janitor as jn

        df

                 text  code
        0      ragnar   NaN
        1         NaN   2.0
        2  sammywemmy   3.0
        3         NaN   NaN
        4      ginger   5.0
    ```


    Fill on a single column:

    ```python
        df.fill_direction(code = 'up')

                 text  code
        0      ragnar   2.0
        1         NaN   2.0
        2  sammywemmy   3.0
        3         NaN   5.0
        4      ginger   5.0
    ```

    Fill on multiple columns:

    ```python
        df.fill_direction(text = 'down', code = 'down')

                 text  code
        0      ragnar   NaN
        1      ragnar   2.0
        2  sammywemmy   3.0
        3  sammywemmy   3.0
        4      ginger   5.0
    ```

    Fill multiple columns in different directions:

    ```python
        df.fill_direction(text = 'up', code = 'down')

                 text  code
        0      ragnar   NaN
        1  sammywemmy   2.0
        2  sammywemmy   3.0
        3      ginger   3.0
        4      ginger   5.0
    ```

    Functional usage syntax:

    ```python
        import pandas as pd
        import janitor as jn

        df = pd.DataFrame(...)
        df = jn.fill_direction(
                    df = df,
                    column_1 = direction_1,
                    column_2 = direction_2,
                )
    ```

    Method-chaining usage syntax:

    ```python
        import pandas as pd
        import janitor as jn

        df = pd.DataFrame(...)
               .fill_direction(
                    column_1 = direction_1,
                    column_2 = direction_2,
                )
    ```

    :param df: A pandas DataFrame.
    :param kwargs: Key - value pairs of columns and directions.
        Directions can be either `down`, `up`, `updown`
        (fill up then down) and `downup` (fill down then up).
    :returns: A pandas DataFrame with modified column(s).
    :raises ValueError: if column supplied is not in the DataFrame.
    :raises ValueError: if direction supplied is not one of `down`, `up`,
        `updown`, or `downup`.
    """

    if not kwargs:
        return df

    fill_types = {fill.name for fill in FILLTYPE}
    for column_name, fill_type in kwargs.items():
        check("column_name", column_name, [str])
        check("fill_type", fill_type, [str])
        if fill_type.upper() not in fill_types:
            raise ValueError(
                """
                fill_type should be one of
                up, down, updown, or downup.
                """
            )

    check_column(df, kwargs)

    new_values = {}
    for column_name, fill_type in kwargs.items():
        direction = FILLTYPE[f"{fill_type.upper()}"].value
        if len(direction) == 1:
            direction = methodcaller(direction[0])
            output = direction(df[column_name])
        else:
            direction = [methodcaller(entry) for entry in direction]
            output = _chain_func(df[column_name], *direction)
        new_values[column_name] = output

    return df.assign(**new_values)


def _chain_func(column: pd.Series, *funcs):
    """
    Apply series of functions consecutively
    to a Series.
    https://blog.finxter.com/how-to-chain-multiple-function-calls-in-python/
    """
    new_value = column.copy()
    for func in funcs:
        new_value = func(new_value)
    return new_value
