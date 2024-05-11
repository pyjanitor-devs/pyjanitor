from janitor.utils import import_message

from .pivot_longer import _pivot_longer_dot_value

try:
    import polars as pl
except ImportError:
    import_message(
        submodule="polars",
        package="polars",
        conda_channel="conda-forge",
        pip_install=True,
    )


def pivot_longer_spec(
    df: pl.DataFrame,
    spec: pl.DataFrame,
) -> pl.DataFrame:
    """A declarative interface to pivot a DataFrame from wide to long form,
    where you describe how the data will be unpivoted,
    using a DataFrame. This gives you, the user,
    more control over unpivoting, where you create a “spec”
    DataFrame that describes exactly how data stored in the column names
    becomes variables.

    !!! info "New in version 0.28.0"

    Examples:
        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame(
        ...     {
        ...         "Sepal.Length": [5.1, 5.9],
        ...         "Sepal.Width": [3.5, 3.0],
        ...         "Petal.Length": [1.4, 5.1],
        ...         "Petal.Width": [0.2, 1.8],
        ...         "Species": ["setosa", "virginica"],
        ...     }
        ... )
        >>> df
           Sepal.Length  Sepal.Width  Petal.Length  Petal.Width    Species
        0           5.1          3.5           1.4          0.2     setosa
        1           5.9          3.0           5.1          1.8  virginica
        >>> spec = {'.name':['Sepal.Length','Petal.Length',
        ...                  'Sepal.Width','Petal.Width'],
        ...         '.value':['Length','Length','Width','Width'],
        ...         'part':['Sepal','Petal','Sepal','Petal']}
        >>> spec = pd.DataFrame(spec)
        >>> spec
                  .name  .value   part
        0  Sepal.Length  Length  Sepal
        1  Petal.Length  Length  Petal
        2   Sepal.Width   Width  Sepal
        3   Petal.Width   Width  Petal
        >>> pivot_longer_spec(df=df,spec=spec)
             Species   part  Length  Width
        0     setosa  Sepal     5.1    3.5
        1  virginica  Sepal     5.9    3.0
        2     setosa  Petal     1.4    0.2
        3  virginica  Petal     5.1    1.8

    Args:
        df: The source DataFrame to unpivot.
        spec: A specification DataFrame.
            This is useful for more complex pivots
            because it gives you greater control
            on how the metadata stored in the column names
            turns into columns in the result.
            Must be a DataFrame containing character .name and .value columns.
            Additional columns in spec should be named to match columns
            in the long format of the dataset and contain values
            corresponding to columns pivoted from the wide format.
            Note that these additional columns should not already exist in the
            source DataFrame.
    Raises:
        KeyError: If '.name' or '.value' is missing from the spec's columns.
        ValueError: If the labels in spec['.name'] is not unique.

    Returns:
        A polars DataFrame.
    """
    if ".name" not in spec.columns:
        raise KeyError(
            "Kindly ensure the spec DataFrame has a `.name` column."
        )
    if ".value" not in spec.columns:
        raise KeyError(
            "Kindly ensure the spec DataFrame has a `.value` column."
        )
    if spec.select(pl.col(".name").is_duplicated().any()).item():
        raise ValueError("The labels in the `.name` column should be unique.")

    exclude = set(df.columns).intersection(spec.columns)
    if exclude:
        raise ValueError(
            f"Labels {*exclude, } in the spec dataframe already exist "
            "as column labels in the source dataframe. "
            "Kindly ensure the spec DataFrame's columns "
            "are not present in the source DataFrame."
        )

    if spec.columns[:2] != [".name", ".value"]:
        raise ValueError(
            "The first two columns of the spec DataFrame "
            "should be '.name' and '.value', "
            "with '.name' coming before '.value'."
        )

    return _pivot_longer_dot_value(
        df=df,
        spec=spec,
    )
