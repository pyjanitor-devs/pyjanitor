import inspect
import pandas_flavor as pf
import pandas as pd

from janitor.utils import check, check_column, deprecated_alias


@pf.register_dataframe_method
@deprecated_alias(column="column_name")
def process_text(
    df: pd.DataFrame,
    column_name: str,
    string_function: str,
    **kwargs: str,
) -> pd.DataFrame:
    """
    Apply a Pandas string method to an existing column and return a dataframe.

    This function aims to make string cleaning easy, while chaining,
    by simply passing the string method name to the ``process_text`` function.
    This modifies an existing column; it does not create a new column.
    New columns can be created via pyjanitor's `transform_columns`.


    A list of all the string methods in Pandas can be accessed `here
    <https://pandas.pydata.org/docs/user_guide/text.html#method-summary>`__.

    Example:



        import pandas as pd
        import janitor as jn

                 text  code
        0      Ragnar     1
        1  sammywemmy     2
        2      ginger     3

        df.process_text(column_name = "text",
                        string_function = "lower")

          text          code
        0 ragnar         1
        1 sammywemmy     2
        2 ginger         3

    For string methods with parameters, simply pass the keyword arguments::

        df.process_text(
            column_name = "text",
            string_function = "extract",
            pat = r"(ag)",
            expand = False,
            flags = re.IGNORECASE
            )

          text     code
        0 ag        1
        1 NaN       2
        2 NaN       3

    Functional usage syntax:



        import pandas as pd
        import janitor as jn

        df = pd.DataFrame(...)
        df = jn.process_text(
            df = df,
            column_name,
            string_function = "string_func_name_here",
            kwargs
            )

    Method-chaining usage syntax:



        import pandas as pd
        import janitor as jn

        df = (
            pd.DataFrame(...)
            .process_text(
                column_name,
                string_function = "string_func_name_here",
                kwargs
                )
        )


    :param df: A pandas dataframe.
    :param column_name: String column to be operated on.
    :param string_function: Pandas string method to be applied.
    :param kwargs: Keyword arguments for parameters of the `string_function`.
    :returns: A pandas dataframe with modified column(s).
    :raises KeyError: if ``string_function`` is not a Pandas string method.
    :raises TypeError: if the wrong ``kwarg`` is supplied.
    :raises ValueError: if `column_name` not found in dataframe.

    .. # noqa: DAR402
    """
    check("column_name", column_name, [str])
    check("string_function", string_function, [str])
    check_column(df, [column_name])

    pandas_string_methods = [
        func.__name__
        for _, func in inspect.getmembers(pd.Series.str, inspect.isfunction)
        if not func.__name__.startswith("_")
    ]

    if string_function not in pandas_string_methods:
        raise KeyError(f"{string_function} is not a Pandas string method.")

    result = getattr(df[column_name].str, string_function)(**kwargs)

    if isinstance(result, pd.DataFrame):
        raise ValueError(
            """
            The outcome of the processed text is a DataFrame,
            which is not supported in `process_text`.
            """
        )

    return df.assign(**{column_name: result})
