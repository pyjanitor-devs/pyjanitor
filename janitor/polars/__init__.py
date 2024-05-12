from typing import Mapping, Union

from polars.type_aliases import ColumnNameOrSelector, PolarsDataType

from janitor.utils import import_message

from .pivot_longer import _pivot_longer, _pivot_longer_dot_value

try:
    import polars as pl
except ImportError:
    import_message(
        submodule="polars",
        package="polars",
        conda_channel="conda-forge",
        pip_install=True,
    )


@pl.api.register_dataframe_namespace("janitor")
class PolarsFrame:
    def __init__(self, df: pl.DataFrame) -> pl.DataFrame:
        self._df = df

    def pivot_longer(
        self,
        index: ColumnNameOrSelector = None,
        column_names: ColumnNameOrSelector = None,
        names_to: Union[list, tuple, str] = "variable",
        values_to: str = "value",
        names_sep: str = None,
        names_pattern: str = None,
        names_transform: Union[
            Mapping[
                Union[ColumnNameOrSelector, PolarsDataType], PolarsDataType
            ],
            PolarsDataType,
        ] = None,
    ) -> pl.DataFrame:
        """
        Unpivots a DataFrame from *wide* to *long* format.

        It is modeled after the `pivot_longer` function in R's tidyr package,
        and also takes inspiration from the `melt` function in R's data.table package.

        This function is useful to massage a DataFrame into a format where
        one or more columns are considered measured variables, and all other
        columns are considered as identifier variables.

        All measured variables are *unpivoted* (and typically duplicated) along the
        row axis.

        Examples:
            >>> import polars as pl
            >>> import polars.selectors as cs
            >>> import janitor.polars
            >>> df = pl.DataFrame(
            ...     {
            ...         "Sepal.Length": [5.1, 5.9],
            ...         "Sepal.Width": [3.5, 3.0],
            ...         "Petal.Length": [1.4, 5.1],
            ...         "Petal.Width": [0.2, 1.8],
            ...         "Species": ["setosa", "virginica"],
            ...     }
            ... )
            >>> df
            shape: (2, 5)
            ┌──────────────┬─────────────┬──────────────┬─────────────┬───────────┐
            │ Sepal.Length ┆ Sepal.Width ┆ Petal.Length ┆ Petal.Width ┆ Species   │
            │ ---          ┆ ---         ┆ ---          ┆ ---         ┆ ---       │
            │ f64          ┆ f64         ┆ f64          ┆ f64         ┆ str       │
            ╞══════════════╪═════════════╪══════════════╪═════════════╪═══════════╡
            │ 5.1          ┆ 3.5         ┆ 1.4          ┆ 0.2         ┆ setosa    │
            │ 5.9          ┆ 3.0         ┆ 5.1          ┆ 1.8         ┆ virginica │
            └──────────────┴─────────────┴──────────────┴─────────────┴───────────┘

            Replicate polars' [melt](https://docs.pola.rs/py-polars/html/reference/dataframe/api/polars.DataFrame.melt.html#polars-dataframe-melt):
            >>> df.janitor.pivot_longer(index = 'Species')
            shape: (8, 3)
            ┌───────────┬──────────────┬───────┐
            │ Species   ┆ variable     ┆ value │
            │ ---       ┆ ---          ┆ ---   │
            │ str       ┆ str          ┆ f64   │
            ╞═══════════╪══════════════╪═══════╡
            │ setosa    ┆ Sepal.Length ┆ 5.1   │
            │ virginica ┆ Sepal.Length ┆ 5.9   │
            │ setosa    ┆ Sepal.Width  ┆ 3.5   │
            │ virginica ┆ Sepal.Width  ┆ 3.0   │
            │ setosa    ┆ Petal.Length ┆ 1.4   │
            │ virginica ┆ Petal.Length ┆ 5.1   │
            │ setosa    ┆ Petal.Width  ┆ 0.2   │
            │ virginica ┆ Petal.Width  ┆ 1.8   │
            └───────────┴──────────────┴───────┘

            Split the column labels into individual columns:
            >>> df.janitor.pivot_longer(
            ...     index = 'Species',
            ...     names_to = ('part', 'dimension'),
            ...     names_sep = '.',
            ... ).select('Species','part','dimension','value')
            shape: (8, 4)
            ┌───────────┬───────┬───────────┬───────┐
            │ Species   ┆ part  ┆ dimension ┆ value │
            │ ---       ┆ ---   ┆ ---       ┆ ---   │
            │ str       ┆ str   ┆ str       ┆ f64   │
            ╞═══════════╪═══════╪═══════════╪═══════╡
            │ setosa    ┆ Sepal ┆ Length    ┆ 5.1   │
            │ virginica ┆ Sepal ┆ Length    ┆ 5.9   │
            │ setosa    ┆ Sepal ┆ Width     ┆ 3.5   │
            │ virginica ┆ Sepal ┆ Width     ┆ 3.0   │
            │ setosa    ┆ Petal ┆ Length    ┆ 1.4   │
            │ virginica ┆ Petal ┆ Length    ┆ 5.1   │
            │ setosa    ┆ Petal ┆ Width     ┆ 0.2   │
            │ virginica ┆ Petal ┆ Width     ┆ 1.8   │
            └───────────┴───────┴───────────┴───────┘

            Retain parts of the column names as headers:
            >>> df.janitor.pivot_longer(
            ...     index = 'Species',
            ...     names_to = ('part', '.value'),
            ...     names_sep = '.',
            ... ).select('Species','part','Length','Width')
            shape: (4, 4)
            ┌───────────┬───────┬────────┬───────┐
            │ Species   ┆ part  ┆ Length ┆ Width │
            │ ---       ┆ ---   ┆ ---    ┆ ---   │
            │ str       ┆ str   ┆ f64    ┆ f64   │
            ╞═══════════╪═══════╪════════╪═══════╡
            │ setosa    ┆ Sepal ┆ 5.1    ┆ 3.5   │
            │ virginica ┆ Sepal ┆ 5.9    ┆ 3.0   │
            │ setosa    ┆ Petal ┆ 1.4    ┆ 0.2   │
            │ virginica ┆ Petal ┆ 5.1    ┆ 1.8   │
            └───────────┴───────┴────────┴───────┘

            Split the column labels based on regex:
            >>> df = pl.DataFrame({"id": [1], "new_sp_m5564": [2], "newrel_f65": [3]})
            >>> df
            shape: (1, 3)
            ┌─────┬──────────────┬────────────┐
            │ id  ┆ new_sp_m5564 ┆ newrel_f65 │
            │ --- ┆ ---          ┆ ---        │
            │ i64 ┆ i64          ┆ i64        │
            ╞═════╪══════════════╪════════════╡
            │ 1   ┆ 2            ┆ 3          │
            └─────┴──────────────┴────────────┘
            >>> df.janitor.pivot_longer(
            ...     index = 'id',
            ...     names_to = ('diagnosis', 'gender', 'age'),
            ...     names_pattern = r"new_?(.+)_(.)(\\d+)",
            ... ).select('id','diagnosis','gender','age','value')
            shape: (2, 5)
            ┌─────┬───────────┬────────┬──────┬───────┐
            │ id  ┆ diagnosis ┆ gender ┆ age  ┆ value │
            │ --- ┆ ---       ┆ ---    ┆ ---  ┆ ---   │
            │ i64 ┆ str       ┆ str    ┆ str  ┆ i64   │
            ╞═════╪═══════════╪════════╪══════╪═══════╡
            │ 1   ┆ sp        ┆ m      ┆ 5564 ┆ 2     │
            │ 1   ┆ rel       ┆ f      ┆ 65   ┆ 3     │
            └─────┴───────────┴────────┴──────┴───────┘

            Convert the dtypes of specific columns with `names_transform`:
            >>> (
            ...     df.janitor.pivot_longer(
            ...         index="id",
            ...         names_to=("diagnosis", "gender", "age"),
            ...         names_pattern=r"new_?(.+)_(.)(\\d+)",
            ...         names_transform={"age": pl.Int32},
            ...     ).select('id','diagnosis','gender','age','value')
            ... )
            shape: (2, 5)
            ┌─────┬───────────┬────────┬──────┬───────┐
            │ id  ┆ diagnosis ┆ gender ┆ age  ┆ value │
            │ --- ┆ ---       ┆ ---    ┆ ---  ┆ ---   │
            │ i64 ┆ str       ┆ str    ┆ i32  ┆ i64   │
            ╞═════╪═══════════╪════════╪══════╪═══════╡
            │ 1   ┆ sp        ┆ m      ┆ 5564 ┆ 2     │
            │ 1   ┆ rel       ┆ f      ┆ 65   ┆ 3     │
            └─────┴───────────┴────────┴──────┴───────┘

            Use multiple `.value` to reshape the dataframe:
            >>> df = pl.DataFrame(
            ...     [
            ...         {
            ...             "x_1_mean": 10,
            ...             "x_2_mean": 20,
            ...             "y_1_mean": 30,
            ...             "y_2_mean": 40,
            ...             "unit": 50,
            ...         }
            ...     ]
            ... )
            >>> df
            shape: (1, 5)
            ┌──────────┬──────────┬──────────┬──────────┬──────┐
            │ x_1_mean ┆ x_2_mean ┆ y_1_mean ┆ y_2_mean ┆ unit │
            │ ---      ┆ ---      ┆ ---      ┆ ---      ┆ ---  │
            │ i64      ┆ i64      ┆ i64      ┆ i64      ┆ i64  │
            ╞══════════╪══════════╪══════════╪══════════╪══════╡
            │ 10       ┆ 20       ┆ 30       ┆ 40       ┆ 50   │
            └──────────┴──────────┴──────────┴──────────┴──────┘
            >>> df.janitor.pivot_longer(
            ...     index="unit",
            ...     names_to=(".value", "time", ".value"),
            ...     names_pattern=r"(x|y)_([0-9])(_mean)",
            ... ).select('unit','time','x_mean','y_mean')
            shape: (2, 4)
            ┌──────┬──────┬────────┬────────┐
            │ unit ┆ time ┆ x_mean ┆ y_mean │
            │ ---  ┆ ---  ┆ ---    ┆ ---    │
            │ i64  ┆ str  ┆ i64    ┆ i64    │
            ╞══════╪══════╪════════╪════════╡
            │ 50   ┆ 1    ┆ 10     ┆ 30     │
            │ 50   ┆ 2    ┆ 20     ┆ 40     │
            └──────┴──────┴────────┴────────┘

        !!! info "New in version 0.28.0"

        Args:
            index: Column(s) or selector(s) to use as identifier variables.
            column_names: Column(s) or selector(s) to unpivot.
            names_to: Name of new column as a string that will contain
                what were previously the column names in `column_names`.
                The default is `variable` if no value is provided. It can
                also be a list/tuple of strings that will serve as new column
                names, if `name_sep` or `names_pattern` is provided.
                If `.value` is in `names_to`, new column names will be extracted
                from part of the existing column names and overrides `values_to`.
            values_to: Name of new column as a string that will contain what
                were previously the values of the columns in `column_names`.
            names_sep: Determines how the column name is broken up, if
                `names_to` contains multiple values. It takes the same
                specification as polars' `str.split` method.
            names_pattern: Determines how the column name is broken up.
                It can be a regular expression containing matching groups.
                It takes the same
                specification as polars' `str.extract_groups` method.
            names_transform: Use this option to change the types of columns that
                have been transformed to rows.
                This does not applies to the values' columns.
                It takes the same specification as
                [polar's cast](https://docs.pola.rs/py-polars/html/reference/dataframe/api/polars.DataFrame.cast.html)
                function.
                Applicable only if one of names_sep
                or names_pattern is provided.

        Returns:
            A polars DataFrame that has been unpivoted from wide to long
                format.
        """  # noqa: E501
        return _pivot_longer(
            df=self._df,
            index=index,
            column_names=column_names,
            names_pattern=names_pattern,
            names_sep=names_sep,
            names_to=names_to,
            values_to=values_to,
            names_transform=names_transform,
        )


@pl.api.register_lazyframe_namespace("janitor")
class PolarsLazyFrame:
    def __init__(self, df: pl.LazyFrame) -> pl.LazyFrame:
        self._df = df

    def pivot_longer(
        self,
        index: ColumnNameOrSelector = None,
        column_names: ColumnNameOrSelector = None,
        names_to: Union[list, tuple, str] = "variable",
        values_to: str = "value",
        names_sep: str = None,
        names_pattern: str = None,
        names_transform: Union[
            Mapping[
                Union[ColumnNameOrSelector, PolarsDataType], PolarsDataType
            ],
            PolarsDataType,
        ] = None,
    ) -> pl.LazyFrame:
        """
        Unpivots a LazyFrame from *wide* to *long* format.

        It is modeled after the `pivot_longer` function in R's tidyr package,
        and also takes inspiration from the `melt` function in R's data.table package.

        This function is useful to massage a LazyFrame into a format where
        one or more columns are considered measured variables, and all other
        columns are considered as identifier variables.

        All measured variables are *unpivoted* (and typically duplicated) along the
        row axis.

        Examples:
            >>> import polars as pl
            >>> import polars.selectors as cs
            >>> import janitor.polars
            >>> df = pl.LazyFrame(
            ...     {
            ...         "Sepal.Length": [5.1, 5.9],
            ...         "Sepal.Width": [3.5, 3.0],
            ...         "Petal.Length": [1.4, 5.1],
            ...         "Petal.Width": [0.2, 1.8],
            ...         "Species": ["setosa", "virginica"],
            ...     }
            ... )
            >>> df.collect()
            shape: (2, 5)
            ┌──────────────┬─────────────┬──────────────┬─────────────┬───────────┐
            │ Sepal.Length ┆ Sepal.Width ┆ Petal.Length ┆ Petal.Width ┆ Species   │
            │ ---          ┆ ---         ┆ ---          ┆ ---         ┆ ---       │
            │ f64          ┆ f64         ┆ f64          ┆ f64         ┆ str       │
            ╞══════════════╪═════════════╪══════════════╪═════════════╪═══════════╡
            │ 5.1          ┆ 3.5         ┆ 1.4          ┆ 0.2         ┆ setosa    │
            │ 5.9          ┆ 3.0         ┆ 5.1          ┆ 1.8         ┆ virginica │
            └──────────────┴─────────────┴──────────────┴─────────────┴───────────┘

            >>> df.janitor.pivot_longer(index = 'Species').collect()
            shape: (8, 3)
            ┌───────────┬──────────────┬───────┐
            │ Species   ┆ variable     ┆ value │
            │ ---       ┆ ---          ┆ ---   │
            │ str       ┆ str          ┆ f64   │
            ╞═══════════╪══════════════╪═══════╡
            │ setosa    ┆ Sepal.Length ┆ 5.1   │
            │ virginica ┆ Sepal.Length ┆ 5.9   │
            │ setosa    ┆ Sepal.Width  ┆ 3.5   │
            │ virginica ┆ Sepal.Width  ┆ 3.0   │
            │ setosa    ┆ Petal.Length ┆ 1.4   │
            │ virginica ┆ Petal.Length ┆ 5.1   │
            │ setosa    ┆ Petal.Width  ┆ 0.2   │
            │ virginica ┆ Petal.Width  ┆ 1.8   │
            └───────────┴──────────────┴───────┘

        !!! info "New in version 0.28.0"

        Args:
            index: Column(s) or selector(s) to use as identifier variables.
            column_names: Column(s) or selector(s) to unpivot.
            names_to: Name of new column as a string that will contain
                what were previously the column names in `column_names`.
                The default is `variable` if no value is provided. It can
                also be a list/tuple of strings that will serve as new column
                names, if `name_sep` or `names_pattern` is provided.
                If `.value` is in `names_to`, new column names will be extracted
                from part of the existing column names and overrides `values_to`.
            values_to: Name of new column as a string that will contain what
                were previously the values of the columns in `column_names`.
            names_sep: Determines how the column name is broken up, if
                `names_to` contains multiple values. It takes the same
                specification as polars' `str.split` method.
            names_pattern: Determines how the column name is broken up.
                It can be a regular expression containing matching groups.
                It takes the same
                specification as polars' `str.extract_groups` method.
            names_transform: Use this option to change the types of columns that
                have been transformed to rows.
                This does not applies to the values' columns.
                It takes the same specification as
                [polar's cast](https://docs.pola.rs/py-polars/html/reference/dataframe/api/polars.DataFrame.cast.html)
                function.
                Applicable only if one of names_sep
                or names_pattern is provided.

        Returns:
            A polars LazyFrame that has been unpivoted from wide to long
                format.
        """  # noqa: E501
        return _pivot_longer(
            df=self._df,
            index=index,
            column_names=column_names,
            names_pattern=names_pattern,
            names_sep=names_sep,
            names_to=names_to,
            values_to=values_to,
            names_transform=names_transform,
        )


def pivot_longer_spec(
    df: Union[pl.DataFrame, pl.LazyFrame],
    spec: pl.DataFrame,
) -> Union[pl.DataFrame, pl.LazyFrame]:
    """A declarative interface to pivot a DataFrame from wide to long form,
    where you describe how the data will be unpivoted,
    using a DataFrame. This gives you, the user,
    more control over unpivoting, where you create a “spec”
    DataFrame that describes exactly how data stored in the column names
    becomes variables.

    !!! info "New in version 0.28.0"

    Examples:
        >>> import pandas as pd
        >>> import janitor.polars
        >>> df = pl.DataFrame(
        ...     {
        ...         "Sepal.Length": [5.1, 5.9],
        ...         "Sepal.Width": [3.5, 3.0],
        ...         "Petal.Length": [1.4, 5.1],
        ...         "Petal.Width": [0.2, 1.8],
        ...         "Species": ["setosa", "virginica"],
        ...     }
        ... )
        >>> df
        shape: (2, 5)
        ┌──────────────┬─────────────┬──────────────┬─────────────┬───────────┐
        │ Sepal.Length ┆ Sepal.Width ┆ Petal.Length ┆ Petal.Width ┆ Species   │
        │ ---          ┆ ---         ┆ ---          ┆ ---         ┆ ---       │
        │ f64          ┆ f64         ┆ f64          ┆ f64         ┆ str       │
        ╞══════════════╪═════════════╪══════════════╪═════════════╪═══════════╡
        │ 5.1          ┆ 3.5         ┆ 1.4          ┆ 0.2         ┆ setosa    │
        │ 5.9          ┆ 3.0         ┆ 5.1          ┆ 1.8         ┆ virginica │
        └──────────────┴─────────────┴──────────────┴─────────────┴───────────┘
        >>> spec = {'.name':['Sepal.Length','Petal.Length',
        ...                  'Sepal.Width','Petal.Width'],
        ...         '.value':['Length','Length','Width','Width'],
        ...         'part':['Sepal','Petal','Sepal','Petal']}
        >>> spec = pl.DataFrame(spec)
        >>> spec
        shape: (4, 3)
        ┌──────────────┬────────┬───────┐
        │ .name        ┆ .value ┆ part  │
        │ ---          ┆ ---    ┆ ---   │
        │ str          ┆ str    ┆ str   │
        ╞══════════════╪════════╪═══════╡
        │ Sepal.Length ┆ Length ┆ Sepal │
        │ Petal.Length ┆ Length ┆ Petal │
        │ Sepal.Width  ┆ Width  ┆ Sepal │
        │ Petal.Width  ┆ Width  ┆ Petal │
        └──────────────┴────────┴───────┘
        >>> df.pipe(pivot_longer_spec,spec=spec)
        shape: (4, 4)
        ┌───────────┬────────┬───────┬───────┐
        │ Species   ┆ Length ┆ Width ┆ part  │
        │ ---       ┆ ---    ┆ ---   ┆ ---   │
        │ str       ┆ f64    ┆ f64   ┆ str   │
        ╞═══════════╪════════╪═══════╪═══════╡
        │ setosa    ┆ 5.1    ┆ 3.5   ┆ Sepal │
        │ virginica ┆ 5.9    ┆ 3.0   ┆ Sepal │
        │ setosa    ┆ 1.4    ┆ 0.2   ┆ Petal │
        │ virginica ┆ 5.1    ┆ 1.8   ┆ Petal │
        └───────────┴────────┴───────┴───────┘

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


__all__ = ["PolarsFrame", "PolarsLazyFrame", "pivot_longer_spec"]
