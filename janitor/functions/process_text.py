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
    **kwargs,
) -> pd.DataFrame:
    """
    Apply a Pandas string method to an existing column.

    This function aims to make string cleaning easy, while chaining,
    by simply passing the string method name,
    along with keyword arguments, if any, to the function.

    This modifies an existing column; it does not create a new column;
    new columns can be created via pyjanitor's
    [`transform_columns`][janitor.functions.transform_columns.transform_columns].


    A list of all the string methods in Pandas can be accessed [here](https://pandas.pydata.org/docs/user_guide/text.html#method-summary).


    Example:

        >>> import pandas as pd
        >>> import janitor
        >>> import re
        >>> df = pd.DataFrame({"text": ["Ragnar", "sammywemmy", "ginger"],
        ... "code": [1, 2, 3]})
        >>> df
                 text  code
        0      Ragnar     1
        1  sammywemmy     2
        2      ginger     3
        >>> df.process_text(column_name="text", string_function="lower")
                 text  code
        0      ragnar     1
        1  sammywemmy     2
        2      ginger     3

    For string methods with parameters, simply pass the keyword arguments:

        >>> df.process_text(
        ...     column_name="text",
        ...     string_function="extract",
        ...     pat=r"(ag)",
        ...     expand=False,
        ...     flags=re.IGNORECASE,
        ... )
          text  code
        0   ag     1
        1  NaN     2
        2  NaN     3

    :param df: A pandas DataFrame.
    :param column_name: String column to be operated on.
    :param string_function: pandas string method to be applied.
    :param kwargs: Keyword arguments for parameters of the `string_function`.
    :returns: A pandas DataFrame with modified column.
    :raises KeyError: If `string_function` is not a Pandas string method.
    :raises ValueError: If the text function returns a DataFrame, instead of a Series.
    """  # noqa: E501

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
            "The outcome of the processed text is a DataFrame, "
            "which is not supported in `process_text`."
        )

    return df.assign(**{column_name: result})
