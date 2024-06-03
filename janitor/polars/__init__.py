from __future__ import annotations

from polars.type_aliases import ColumnNameOrSelector

from janitor.utils import check, import_message

from .clean_names import _clean_column_names, _clean_expr_names
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

    def clean_names(
        self,
        strip_underscores: str | bool = None,
        case_type: str = "lower",
        remove_special: bool = False,
        strip_accents: bool = False,
        truncate_limit: int = None,
    ) -> pl.DataFrame:
        """
        Clean the column names in a polars DataFrame.

        Examples:
            >>> import polars as pl
            >>> import janitor.polars
            >>> df = pl.DataFrame(
            ...     {
            ...         "Aloha": range(3),
            ...         "Bell Chart": range(3),
            ...         "Animals@#$%^": range(3)
            ...     }
            ... )
            >>> df
            shape: (3, 3)
            ┌───────┬────────────┬──────────────┐
            │ Aloha ┆ Bell Chart ┆ Animals@#$%^ │
            │ ---   ┆ ---        ┆ ---          │
            │ i64   ┆ i64        ┆ i64          │
            ╞═══════╪════════════╪══════════════╡
            │ 0     ┆ 0          ┆ 0            │
            │ 1     ┆ 1          ┆ 1            │
            │ 2     ┆ 2          ┆ 2            │
            └───────┴────────────┴──────────────┘
            >>> df.janitor.clean_names(remove_special=True)
            shape: (3, 3)
            ┌───────┬────────────┬─────────┐
            │ aloha ┆ bell_chart ┆ animals │
            │ ---   ┆ ---        ┆ ---     │
            │ i64   ┆ i64        ┆ i64     │
            ╞═══════╪════════════╪═════════╡
            │ 0     ┆ 0          ┆ 0       │
            │ 1     ┆ 1          ┆ 1       │
            │ 2     ┆ 2          ┆ 2       │
            └───────┴────────────┴─────────┘

        !!! info "New in version 0.28.0"

        Args:
            strip_underscores: Removes the outer underscores from all
                column names. Default None keeps outer underscores. Values can be
                either 'left', 'right' or 'both' or the respective shorthand 'l',
                'r' and True.
            case_type: Whether to make the column names lower or uppercase.
                Current case may be preserved with 'preserve',
                while snake case conversion (from CamelCase or camelCase only)
                can be turned on using "snake".
                Default 'lower' makes all characters lowercase.
            remove_special: Remove special characters from the column names.
                Only letters, numbers and underscores are preserved.
            strip_accents: Whether or not to remove accents from
                the labels.
            truncate_limit: Truncates formatted column names to
                the specified length. Default None does not truncate.

        Returns:
            A polars DataFrame.
        """  # noqa: E501
        return self._df.rename(
            lambda col: _clean_column_names(
                obj=col,
                strip_accents=strip_accents,
                strip_underscores=strip_underscores,
                case_type=case_type,
                remove_special=remove_special,
                truncate_limit=truncate_limit,
            )
        )

    def pivot_longer(
        self,
        index: ColumnNameOrSelector = None,
        column_names: ColumnNameOrSelector = None,
        names_to: list | tuple | str = "variable",
        values_to: str = "value",
        names_sep: str = None,
        names_pattern: str = None,
        names_transform: pl.Expr = None,
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

        For more granular control on the unpivoting, have a look at
        `pivot_longer_spec`.

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
            ...     names_pattern = r"new_?(.+)_(.)([0-9]+)",
            ... ).select('id','diagnosis','gender','age','value').sort(by=pl.all())
            shape: (2, 5)
            ┌─────┬───────────┬────────┬──────┬───────┐
            │ id  ┆ diagnosis ┆ gender ┆ age  ┆ value │
            │ --- ┆ ---       ┆ ---    ┆ ---  ┆ ---   │
            │ i64 ┆ str       ┆ str    ┆ str  ┆ i64   │
            ╞═════╪═══════════╪════════╪══════╪═══════╡
            │ 1   ┆ rel       ┆ f      ┆ 65   ┆ 3     │
            │ 1   ┆ sp        ┆ m      ┆ 5564 ┆ 2     │
            └─────┴───────────┴────────┴──────┴───────┘

            Convert the dtypes of specific columns with `names_transform`:
            >>> df.janitor.pivot_longer(
            ...     index = "id",
            ...     names_pattern=r"new_?(.+)_(.)([0-9]+)",
            ...     names_to=("diagnosis", "gender", "age"),
            ...     names_transform=pl.col('age').cast(pl.Int32),
            ... ).select("id", "diagnosis", "gender", "age", "value").sort(by=pl.all())
            shape: (2, 5)
            ┌─────┬───────────┬────────┬──────┬───────┐
            │ id  ┆ diagnosis ┆ gender ┆ age  ┆ value │
            │ --- ┆ ---       ┆ ---    ┆ ---  ┆ ---   │
            │ i64 ┆ str       ┆ str    ┆ i32  ┆ i64   │
            ╞═════╪═══════════╪════════╪══════╪═══════╡
            │ 1   ┆ rel       ┆ f      ┆ 65   ┆ 3     │
            │ 1   ┆ sp        ┆ m      ┆ 5564 ┆ 2     │
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
            ... ).select('unit','time','x_mean','y_mean').sort(by=pl.all())
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
                Accepts a polars expression or a list of polars expressions.
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

    def pivot_longer(
        self,
        index: ColumnNameOrSelector = None,
        column_names: ColumnNameOrSelector = None,
        names_to: list | tuple | str = "variable",
        values_to: str = "value",
        names_sep: str = None,
        names_pattern: str = None,
        names_transform: pl.Expr = None,
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

        For more granular control on the unpivoting, have a look at
        `pivot_longer_spec`.

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
                Accepts a polars expression or a list of polars expressions.
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


@pl.api.register_expr_namespace("janitor")
class PolarsExpr:
    def __init__(self, expr: pl.Expr) -> pl.Expr:
        self._expr = expr

    def clean_names(
        self,
        strip_underscores: str | bool = None,
        case_type: str = "lower",
        remove_special: bool = False,
        strip_accents: bool = False,
        enforce_string: bool = False,
        truncate_limit: int = None,
    ) -> pl.Expr:
        """
        Clean the labels in a polars Expression.

        Examples:
            >>> import polars as pl
            >>> import janitor.polars
            >>> df = pl.DataFrame({"raw": ["Abçdê fgí j"]})
            >>> df
            shape: (1, 1)
            ┌─────────────┐
            │ raw         │
            │ ---         │
            │ str         │
            ╞═════════════╡
            │ Abçdê fgí j │
            └─────────────┘

            Clean the column values:
            >>> df.with_columns(pl.col("raw").janitor.clean_names(strip_accents=True))
            shape: (1, 1)
            ┌─────────────┐
            │ raw         │
            │ ---         │
            │ str         │
            ╞═════════════╡
            │ abcde_fgi_j │
            └─────────────┘

        !!! info "New in version 0.28.0"

        Args:
            strip_underscores: Removes the outer underscores
                from all labels in the expression.
                Default None keeps outer underscores.
                Values can be either 'left', 'right'
                or 'both' or the respective shorthand 'l',
                'r' and True.
            case_type: Whether to make the labels in the expression lower or uppercase.
                Current case may be preserved with 'preserve',
                while snake case conversion (from CamelCase or camelCase only)
                can be turned on using "snake".
                Default 'lower' makes all characters lowercase.
            remove_special: Remove special characters from the values in the expression.
                Only letters, numbers and underscores are preserved.
            strip_accents: Whether or not to remove accents from
                the expression.
            enforce_string: Whether or not to cast the expression to a string type.
            truncate_limit: Truncates formatted labels in the expression to
                the specified length. Default None does not truncate.

        Returns:
            A polars Expression.
        """
        return _clean_expr_names(
            obj=self._expr,
            strip_accents=strip_accents,
            strip_underscores=strip_underscores,
            case_type=case_type,
            remove_special=remove_special,
            enforce_string=enforce_string,
            truncate_limit=truncate_limit,
        )


def pivot_longer_spec(
    df: pl.DataFrame | pl.LazyFrame,
    spec: pl.DataFrame,
) -> pl.DataFrame | pl.LazyFrame:
    """
    A declarative interface to pivot a DataFrame
    from wide to long form,
    where you describe how the data will be unpivoted,
    using a DataFrame. This gives you, the user,
    more control over the transformation to long form,
    using a *spec* DataFrame that describes exactly
    how data stored in the column names
    becomes variables.

    It can come in handy for situations where
    `janitor.polars.pivot_longer`
    seems inadequate for the transformation.

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
            At a minimum, the spec DataFrame
            must have a `.name` column
            and a `.value` column.
            The `.name` column  should contain the
            columns in the source DataFrame that will be
            transformed to long form.
            The `.value` column gives the name of the column
            that the values in the source DataFrame will go into.
            Additional columns in the spec DataFrame
            should be named to match columns
            in the long format of the dataset and contain values
            corresponding to columns pivoted from the wide format.
            Note that these additional columns should not already exist
            in the source DataFrame.

    Raises:
        KeyError: If `.name` or `.value` is missing from the spec's columns.
        ValueError: If the labels in `spec['.name']` is not unique.

    Returns:
        A polars DataFrame/LazyFrame.
    """
    check("spec", spec, [pl.DataFrame])
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
