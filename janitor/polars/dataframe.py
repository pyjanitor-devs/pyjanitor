from __future__ import annotations

from typing import Any

from polars.type_aliases import ColumnNameOrSelector

from janitor.utils import import_message

from .clean_names import _clean_column_names
from .complete import _complete
from .pivot_longer import _pivot_longer
from .row_to_names import _row_to_names

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
class PolarsDataFrame:
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
            >>> df.janitor.pivot_longer(index = 'Species').sort(by=pl.all())
            shape: (8, 3)
            ┌───────────┬──────────────┬───────┐
            │ Species   ┆ variable     ┆ value │
            │ ---       ┆ ---          ┆ ---   │
            │ str       ┆ str          ┆ f64   │
            ╞═══════════╪══════════════╪═══════╡
            │ setosa    ┆ Petal.Length ┆ 1.4   │
            │ setosa    ┆ Petal.Width  ┆ 0.2   │
            │ setosa    ┆ Sepal.Length ┆ 5.1   │
            │ setosa    ┆ Sepal.Width  ┆ 3.5   │
            │ virginica ┆ Petal.Length ┆ 5.1   │
            │ virginica ┆ Petal.Width  ┆ 1.8   │
            │ virginica ┆ Sepal.Length ┆ 5.9   │
            │ virginica ┆ Sepal.Width  ┆ 3.0   │
            └───────────┴──────────────┴───────┘

            Split the column labels into individual columns:
            >>> df.janitor.pivot_longer(
            ...     index = 'Species',
            ...     names_to = ('part', 'dimension'),
            ...     names_sep = '.',
            ... ).select('Species','part','dimension','value').sort(by=pl.all())
            shape: (8, 4)
            ┌───────────┬───────┬───────────┬───────┐
            │ Species   ┆ part  ┆ dimension ┆ value │
            │ ---       ┆ ---   ┆ ---       ┆ ---   │
            │ str       ┆ str   ┆ str       ┆ f64   │
            ╞═══════════╪═══════╪═══════════╪═══════╡
            │ setosa    ┆ Petal ┆ Length    ┆ 1.4   │
            │ setosa    ┆ Petal ┆ Width     ┆ 0.2   │
            │ setosa    ┆ Sepal ┆ Length    ┆ 5.1   │
            │ setosa    ┆ Sepal ┆ Width     ┆ 3.5   │
            │ virginica ┆ Petal ┆ Length    ┆ 5.1   │
            │ virginica ┆ Petal ┆ Width     ┆ 1.8   │
            │ virginica ┆ Sepal ┆ Length    ┆ 5.9   │
            │ virginica ┆ Sepal ┆ Width     ┆ 3.0   │
            └───────────┴───────┴───────────┴───────┘

            Retain parts of the column names as headers:
            >>> df.janitor.pivot_longer(
            ...     index = 'Species',
            ...     names_to = ('part', '.value'),
            ...     names_sep = '.',
            ... ).select('Species','part','Length','Width').sort(by=pl.all())
            shape: (4, 4)
            ┌───────────┬───────┬────────┬───────┐
            │ Species   ┆ part  ┆ Length ┆ Width │
            │ ---       ┆ ---   ┆ ---    ┆ ---   │
            │ str       ┆ str   ┆ f64    ┆ f64   │
            ╞═══════════╪═══════╪════════╪═══════╡
            │ setosa    ┆ Petal ┆ 1.4    ┆ 0.2   │
            │ setosa    ┆ Sepal ┆ 5.1    ┆ 3.5   │
            │ virginica ┆ Petal ┆ 5.1    ┆ 1.8   │
            │ virginica ┆ Sepal ┆ 5.9    ┆ 3.0   │
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

    def row_to_names(
        self,
        row_numbers: int | list = 0,
        remove_rows: bool = False,
        remove_rows_above: bool = False,
        separator: str = "_",
    ) -> pl.DataFrame:
        """
        Elevates a row, or rows, to be the column names of a DataFrame.

        Examples:
            Replace column names with the first row.

            >>> import polars as pl
            >>> import janitor.polars
            >>> df = pl.DataFrame({
            ...     "a": ["nums", '6', '9'],
            ...     "b": ["chars", "x", "y"],
            ... })
            >>> df
            shape: (3, 2)
            ┌──────┬───────┐
            │ a    ┆ b     │
            │ ---  ┆ ---   │
            │ str  ┆ str   │
            ╞══════╪═══════╡
            │ nums ┆ chars │
            │ 6    ┆ x     │
            │ 9    ┆ y     │
            └──────┴───────┘
            >>> df.janitor.row_to_names(0, remove_rows=True)
            shape: (2, 2)
            ┌──────┬───────┐
            │ nums ┆ chars │
            │ ---  ┆ ---   │
            │ str  ┆ str   │
            ╞══════╪═══════╡
            │ 6    ┆ x     │
            │ 9    ┆ y     │
            └──────┴───────┘
            >>> df.janitor.row_to_names(row_numbers=[0,1], remove_rows=True)
            shape: (1, 2)
            ┌────────┬─────────┐
            │ nums_6 ┆ chars_x │
            │ ---    ┆ ---     │
            │ str    ┆ str     │
            ╞════════╪═════════╡
            │ 9      ┆ y       │
            └────────┴─────────┘

            Remove rows above the elevated row and the elevated row itself.

            >>> df = pl.DataFrame({
            ...     "a": ["bla1", "nums", '6', '9'],
            ...     "b": ["bla2", "chars", "x", "y"],
            ... })
            >>> df
            shape: (4, 2)
            ┌──────┬───────┐
            │ a    ┆ b     │
            │ ---  ┆ ---   │
            │ str  ┆ str   │
            ╞══════╪═══════╡
            │ bla1 ┆ bla2  │
            │ nums ┆ chars │
            │ 6    ┆ x     │
            │ 9    ┆ y     │
            └──────┴───────┘
            >>> df.janitor.row_to_names(1, remove_rows=True, remove_rows_above=True)
            shape: (2, 2)
            ┌──────┬───────┐
            │ nums ┆ chars │
            │ ---  ┆ ---   │
            │ str  ┆ str   │
            ╞══════╪═══════╡
            │ 6    ┆ x     │
            │ 9    ┆ y     │
            └──────┴───────┘

        !!! info "New in version 0.28.0"

        Args:
            row_numbers: Position of the row(s) containing the variable names.
                Note that indexing starts from 0. It can also be a list.
                Defaults to 0 (first row).
            remove_rows: Whether the row(s) should be removed from the DataFrame.
            remove_rows_above: Whether the row(s) above the selected row should
                be removed from the DataFrame.
            separator: Combines the labels into a single string,
                if row_numbers is a list of integers. Default is '_'.

        Returns:
            A polars DataFrame.
        """  # noqa: E501
        return _row_to_names(
            self._df,
            row_numbers=row_numbers,
            remove_rows=remove_rows,
            remove_rows_above=remove_rows_above,
            separator=separator,
        )

    def complete(
        self,
        *columns: ColumnNameOrSelector,
        fill_value: dict | Any | pl.Expr = None,
        explicit: bool = True,
        sort: bool = False,
        by: ColumnNameOrSelector = None,
    ) -> pl.DataFrame:
        """
        Turns implicit missing values into explicit missing values

        It is modeled after tidyr's `complete` function.
        In a way, it is the inverse of `pl.drop_nulls`,
        as it exposes implicitly missing rows.

        If the combination involves multiple columns, pass it as a struct,
        with an alias - the name of the struct should not exist in the DataFrame.

        If new values need to be introduced, a polars Expression
        with the new values can be passed, as long as the polars Expression
        has a name that already exists in the DataFrame.

        It is up to the user to ensure that the polars expression returns
        unique values and/or sorted values.

        Note that if the polars expression evaluates to a struct,
        then the fields, not the name, should already exist in the DataFrame.

        Examples:
            >>> import polars as pl
            >>> import janitor.polars
            >>> df = pl.DataFrame(
            ...     dict(
            ...         group=(1, 2, 1, 2),
            ...         item_id=(1, 2, 2, 3),
            ...         item_name=("a", "a", "b", "b"),
            ...         value1=(1, None, 3, 4),
            ...         value2=range(4, 8),
            ...     )
            ... )
            >>> df
            shape: (4, 5)
            ┌───────┬─────────┬───────────┬────────┬────────┐
            │ group ┆ item_id ┆ item_name ┆ value1 ┆ value2 │
            │ ---   ┆ ---     ┆ ---       ┆ ---    ┆ ---    │
            │ i64   ┆ i64     ┆ str       ┆ i64    ┆ i64    │
            ╞═══════╪═════════╪═══════════╪════════╪════════╡
            │ 1     ┆ 1       ┆ a         ┆ 1      ┆ 4      │
            │ 2     ┆ 2       ┆ a         ┆ null   ┆ 5      │
            │ 1     ┆ 2       ┆ b         ┆ 3      ┆ 6      │
            │ 2     ┆ 3       ┆ b         ┆ 4      ┆ 7      │
            └───────┴─────────┴───────────┴────────┴────────┘

            Generate all possible combinations of
            `group`, `item_id`, and `item_name`
            (whether or not they appear in the data)
            >>> with pl.Config(tbl_rows=-1):
            ...     df.janitor.complete("group", "item_id", "item_name", sort=True)
            shape: (12, 5)
            ┌───────┬─────────┬───────────┬────────┬────────┐
            │ group ┆ item_id ┆ item_name ┆ value1 ┆ value2 │
            │ ---   ┆ ---     ┆ ---       ┆ ---    ┆ ---    │
            │ i64   ┆ i64     ┆ str       ┆ i64    ┆ i64    │
            ╞═══════╪═════════╪═══════════╪════════╪════════╡
            │ 1     ┆ 1       ┆ a         ┆ 1      ┆ 4      │
            │ 1     ┆ 1       ┆ b         ┆ null   ┆ null   │
            │ 1     ┆ 2       ┆ a         ┆ null   ┆ null   │
            │ 1     ┆ 2       ┆ b         ┆ 3      ┆ 6      │
            │ 1     ┆ 3       ┆ a         ┆ null   ┆ null   │
            │ 1     ┆ 3       ┆ b         ┆ null   ┆ null   │
            │ 2     ┆ 1       ┆ a         ┆ null   ┆ null   │
            │ 2     ┆ 1       ┆ b         ┆ null   ┆ null   │
            │ 2     ┆ 2       ┆ a         ┆ null   ┆ 5      │
            │ 2     ┆ 2       ┆ b         ┆ null   ┆ null   │
            │ 2     ┆ 3       ┆ a         ┆ null   ┆ null   │
            │ 2     ┆ 3       ┆ b         ┆ 4      ┆ 7      │
            └───────┴─────────┴───────────┴────────┴────────┘

            Cross all possible `group` values with the unique pairs of
            `(item_id, item_name)` that already exist in the data.
            For such situations, where there is a group of columns,
            pass it in as a struct:
            >>> with pl.Config(tbl_rows=-1):
            ...     df.janitor.complete(
            ...         "group",
            ...         pl.struct("item_id", "item_name").unique().sort().alias("rar"),
            ...         sort=True
            ...     )
            shape: (8, 5)
            ┌───────┬─────────┬───────────┬────────┬────────┐
            │ group ┆ item_id ┆ item_name ┆ value1 ┆ value2 │
            │ ---   ┆ ---     ┆ ---       ┆ ---    ┆ ---    │
            │ i64   ┆ i64     ┆ str       ┆ i64    ┆ i64    │
            ╞═══════╪═════════╪═══════════╪════════╪════════╡
            │ 1     ┆ 1       ┆ a         ┆ 1      ┆ 4      │
            │ 1     ┆ 2       ┆ a         ┆ null   ┆ null   │
            │ 1     ┆ 2       ┆ b         ┆ 3      ┆ 6      │
            │ 1     ┆ 3       ┆ b         ┆ null   ┆ null   │
            │ 2     ┆ 1       ┆ a         ┆ null   ┆ null   │
            │ 2     ┆ 2       ┆ a         ┆ null   ┆ 5      │
            │ 2     ┆ 2       ┆ b         ┆ null   ┆ null   │
            │ 2     ┆ 3       ┆ b         ┆ 4      ┆ 7      │
            └───────┴─────────┴───────────┴────────┴────────┘

            Fill in nulls:
            >>> with pl.Config(tbl_rows=-1):
            ...     df.janitor.complete(
            ...         "group",
            ...         pl.struct("item_id", "item_name").unique().sort().alias('rar'),
            ...         fill_value={"value1": 0, "value2": 99},
            ...         explicit=True,
            ...         sort=True,
            ...     )
            shape: (8, 5)
            ┌───────┬─────────┬───────────┬────────┬────────┐
            │ group ┆ item_id ┆ item_name ┆ value1 ┆ value2 │
            │ ---   ┆ ---     ┆ ---       ┆ ---    ┆ ---    │
            │ i64   ┆ i64     ┆ str       ┆ i64    ┆ i64    │
            ╞═══════╪═════════╪═══════════╪════════╪════════╡
            │ 1     ┆ 1       ┆ a         ┆ 1      ┆ 4      │
            │ 1     ┆ 2       ┆ a         ┆ 0      ┆ 99     │
            │ 1     ┆ 2       ┆ b         ┆ 3      ┆ 6      │
            │ 1     ┆ 3       ┆ b         ┆ 0      ┆ 99     │
            │ 2     ┆ 1       ┆ a         ┆ 0      ┆ 99     │
            │ 2     ┆ 2       ┆ a         ┆ 0      ┆ 5      │
            │ 2     ┆ 2       ┆ b         ┆ 0      ┆ 99     │
            │ 2     ┆ 3       ┆ b         ┆ 4      ┆ 7      │
            └───────┴─────────┴───────────┴────────┴────────┘

            Limit the fill to only the newly created
            missing values with `explicit = FALSE`
            >>> with pl.Config(tbl_rows=-1):
            ...     df.janitor.complete(
            ...         "group",
            ...         pl.struct("item_id", "item_name").unique().sort().alias('rar'),
            ...         fill_value={"value1": 0, "value2": 99},
            ...         explicit=False,
            ...         sort=True,
            ...     )
            shape: (8, 5)
            ┌───────┬─────────┬───────────┬────────┬────────┐
            │ group ┆ item_id ┆ item_name ┆ value1 ┆ value2 │
            │ ---   ┆ ---     ┆ ---       ┆ ---    ┆ ---    │
            │ i64   ┆ i64     ┆ str       ┆ i64    ┆ i64    │
            ╞═══════╪═════════╪═══════════╪════════╪════════╡
            │ 1     ┆ 1       ┆ a         ┆ 1      ┆ 4      │
            │ 1     ┆ 2       ┆ a         ┆ 0      ┆ 99     │
            │ 1     ┆ 2       ┆ b         ┆ 3      ┆ 6      │
            │ 1     ┆ 3       ┆ b         ┆ 0      ┆ 99     │
            │ 2     ┆ 1       ┆ a         ┆ 0      ┆ 99     │
            │ 2     ┆ 2       ┆ a         ┆ null   ┆ 5      │
            │ 2     ┆ 2       ┆ b         ┆ 0      ┆ 99     │
            │ 2     ┆ 3       ┆ b         ┆ 4      ┆ 7      │
            └───────┴─────────┴───────────┴────────┴────────┘

            >>> df = pl.DataFrame(
            ...     {
            ...         "Year": [1999, 2000, 2004, 1999, 2004],
            ...         "Taxon": [
            ...             "Saccharina",
            ...             "Saccharina",
            ...             "Saccharina",
            ...             "Agarum",
            ...             "Agarum",
            ...         ],
            ...         "Abundance": [4, 5, 2, 1, 8],
            ...     }
            ... )
            >>> df
            shape: (5, 3)
            ┌──────┬────────────┬───────────┐
            │ Year ┆ Taxon      ┆ Abundance │
            │ ---  ┆ ---        ┆ ---       │
            │ i64  ┆ str        ┆ i64       │
            ╞══════╪════════════╪═══════════╡
            │ 1999 ┆ Saccharina ┆ 4         │
            │ 2000 ┆ Saccharina ┆ 5         │
            │ 2004 ┆ Saccharina ┆ 2         │
            │ 1999 ┆ Agarum     ┆ 1         │
            │ 2004 ┆ Agarum     ┆ 8         │
            └──────┴────────────┴───────────┘

            Expose missing years from 1999 to 2004 -
            pass a polars expression with the new dates,
            and ensure the expression's name already exists
            in the DataFrame:
            >>> expression = pl.int_range(1999,2005).alias('Year')
            >>> with pl.Config(tbl_rows=-1):
            ...     df.janitor.complete(expression,'Taxon',sort=True)
            shape: (12, 3)
            ┌──────┬────────────┬───────────┐
            │ Year ┆ Taxon      ┆ Abundance │
            │ ---  ┆ ---        ┆ ---       │
            │ i64  ┆ str        ┆ i64       │
            ╞══════╪════════════╪═══════════╡
            │ 1999 ┆ Agarum     ┆ 1         │
            │ 1999 ┆ Saccharina ┆ 4         │
            │ 2000 ┆ Agarum     ┆ null      │
            │ 2000 ┆ Saccharina ┆ 5         │
            │ 2001 ┆ Agarum     ┆ null      │
            │ 2001 ┆ Saccharina ┆ null      │
            │ 2002 ┆ Agarum     ┆ null      │
            │ 2002 ┆ Saccharina ┆ null      │
            │ 2003 ┆ Agarum     ┆ null      │
            │ 2003 ┆ Saccharina ┆ null      │
            │ 2004 ┆ Agarum     ┆ 8         │
            │ 2004 ┆ Saccharina ┆ 2         │
            └──────┴────────────┴───────────┘

            Expose missing rows per group:
            >>> df = pl.DataFrame(
            ...     {
            ...         "state": ["CA", "CA", "HI", "HI", "HI", "NY", "NY"],
            ...         "year": [2010, 2013, 2010, 2012, 2016, 2009, 2013],
            ...         "value": [1, 3, 1, 2, 3, 2, 5],
            ...     }
            ... )
            >>> df
            shape: (7, 3)
            ┌───────┬──────┬───────┐
            │ state ┆ year ┆ value │
            │ ---   ┆ ---  ┆ ---   │
            │ str   ┆ i64  ┆ i64   │
            ╞═══════╪══════╪═══════╡
            │ CA    ┆ 2010 ┆ 1     │
            │ CA    ┆ 2013 ┆ 3     │
            │ HI    ┆ 2010 ┆ 1     │
            │ HI    ┆ 2012 ┆ 2     │
            │ HI    ┆ 2016 ┆ 3     │
            │ NY    ┆ 2009 ┆ 2     │
            │ NY    ┆ 2013 ┆ 5     │
            └───────┴──────┴───────┘
            >>> low = pl.col('year').min()
            >>> high = pl.col('year').max().add(1)
            >>> new_year_values=pl.int_range(low,high).alias('year')
            >>> with pl.Config(tbl_rows=-1):
            ...     df.janitor.complete(new_year_values,by='state',sort=True)
            shape: (16, 3)
            ┌───────┬──────┬───────┐
            │ state ┆ year ┆ value │
            │ ---   ┆ ---  ┆ ---   │
            │ str   ┆ i64  ┆ i64   │
            ╞═══════╪══════╪═══════╡
            │ CA    ┆ 2010 ┆ 1     │
            │ CA    ┆ 2011 ┆ null  │
            │ CA    ┆ 2012 ┆ null  │
            │ CA    ┆ 2013 ┆ 3     │
            │ HI    ┆ 2010 ┆ 1     │
            │ HI    ┆ 2011 ┆ null  │
            │ HI    ┆ 2012 ┆ 2     │
            │ HI    ┆ 2013 ┆ null  │
            │ HI    ┆ 2014 ┆ null  │
            │ HI    ┆ 2015 ┆ null  │
            │ HI    ┆ 2016 ┆ 3     │
            │ NY    ┆ 2009 ┆ 2     │
            │ NY    ┆ 2010 ┆ null  │
            │ NY    ┆ 2011 ┆ null  │
            │ NY    ┆ 2012 ┆ null  │
            │ NY    ┆ 2013 ┆ 5     │
            └───────┴──────┴───────┘


        !!! info "New in version 0.28.0"

        Args:
            *columns: This refers to the columns to be completed.
                It can be a string or a column selector or a polars expression.
                A polars expression can be used to introduced new values,
                as long as the polars expression has a name that already exists
                in the DataFrame.
                It is up to the user to ensure that the polars expression returns
                unique values.
            fill_value: Scalar value or polars expression to use instead of nulls
                for missing combinations. A dictionary, mapping columns names
                to a scalar value is also accepted.
            explicit: Determines if only implicitly missing values
                should be filled (`False`), or all nulls existing in the LazyFrame
                (`True`). `explicit` is applicable only
                if `fill_value` is not `None`.
            sort: Sort the DataFrame based on *columns.
            by: Column(s) to group by.
                The explicit missing rows are returned per group.

        Returns:
            A polars DataFrame.
        """  # noqa: E501
        return _complete(
            df=self._df,
            columns=columns,
            fill_value=fill_value,
            explicit=explicit,
            sort=sort,
            by=by,
        )
