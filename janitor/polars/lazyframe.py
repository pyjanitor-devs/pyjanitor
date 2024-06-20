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


@pl.api.register_lazyframe_namespace("janitor")
class PolarsLazyFrame:
    def __init__(self, df: pl.LazyFrame) -> pl.LazyFrame:
        self._df = df

    def clean_names(
        self,
        strip_underscores: str | bool = None,
        case_type: str = "lower",
        remove_special: bool = False,
        strip_accents: bool = False,
        truncate_limit: int = None,
    ) -> pl.LazyFrame:
        """
        Clean the column names in a polars LazyFrame.

        Examples:
            >>> import polars as pl
            >>> import janitor.polars
            >>> df = pl.LazyFrame(
            ...     {
            ...         "Aloha": range(3),
            ...         "Bell Chart": range(3),
            ...         "Animals@#$%^": range(3)
            ...     }
            ... )
            >>> df.collect()
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
            >>> df.janitor.clean_names(remove_special=True).collect()
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
            A polars LazyFrame.
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

            >>> df.janitor.pivot_longer(index = 'Species').sort(by=pl.all()).collect()
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

    def row_to_names(
        self,
        row_numbers: int | list = 0,
        remove_rows: bool = False,
        remove_rows_above: bool = False,
        separator: str = "_",
    ) -> pl.LazyFrame:
        """
        Elevates a row, or rows, to be the column names of a DataFrame.

        Examples:
            Replace column names with the first row.

            >>> import polars as pl
            >>> import janitor.polars
            >>> df = pl.LazyFrame({
            ...     "a": ["nums", '6', '9'],
            ...     "b": ["chars", "x", "y"],
            ... })
            >>> df.collect()
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
            >>> df.janitor.row_to_names(0, remove_rows=True).collect()
            shape: (2, 2)
            ┌──────┬───────┐
            │ nums ┆ chars │
            │ ---  ┆ ---   │
            │ str  ┆ str   │
            ╞══════╪═══════╡
            │ 6    ┆ x     │
            │ 9    ┆ y     │
            └──────┴───────┘
            >>> df.janitor.row_to_names(row_numbers=[0,1], remove_rows=True).collect()
            shape: (1, 2)
            ┌────────┬─────────┐
            │ nums_6 ┆ chars_x │
            │ ---    ┆ ---     │
            │ str    ┆ str     │
            ╞════════╪═════════╡
            │ 9      ┆ y       │
            └────────┴─────────┘

            Remove rows above the elevated row and the elevated row itself.

            >>> df = pl.LazyFrame({
            ...     "a": ["bla1", "nums", '6', '9'],
            ...     "b": ["bla2", "chars", "x", "y"],
            ... })
            >>> df.collect()
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
            >>> df.janitor.row_to_names(1, remove_rows=True, remove_rows_above=True).collect()
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
            separator: If `row_numbers` is a list of numbers, this parameter
                determines how the labels will be combined into a single string.

        Returns:
            A polars LazyFrame.
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
    ) -> pl.LazyFrame:
        """
        Turns implicit missing values into explicit missing values.

        It is modeled after tidyr's `complete` function.
        In a way, it is the inverse of `pl.drop_nulls`,
        as it exposes implicitly missing rows.

        If the combination involves multiple columns, pass it as a struct,
        with an alias - the name of the struct should not exist in the LazyFrame.

        If new values need to be introduced, a polars Expression
        with the new values can be passed, as long as the polars Expression
        has a name that already exists in the LazyFrame.

        It is up to the user to ensure that the polars expression returns
        unique values.

        Note that if the polars expression evaluates to a struct,
        then the fields, not the name, should already exist in the LazyFrame.

        Examples:
            >>> import polars as pl
            >>> import janitor.polars
            >>> df = pl.LazyFrame(
            ...     dict(
            ...         group=(1, 2, 1, 2),
            ...         item_id=(1, 2, 2, 3),
            ...         item_name=("a", "a", "b", "b"),
            ...         value1=(1, None, 3, 4),
            ...         value2=range(4, 8),
            ...     )
            ... )
            >>> df.collect()
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
            ...     df.janitor.complete("group", "item_id", "item_name", sort=True).collect()
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

        !!! info "New in version 0.28.0"

        Args:
            *columns: This refers to the columns to be completed.
                It can be a string or a column selector or a polars expression.
                A polars expression can be used to introduced new values,
                as long as the polars expression has a name that already exists
                in the LazyFrame.
                It is up to the user to ensure that the polars expression returns
                unique values.
            fill_value: Scalar value or polars expression to use instead of nulls
                for missing combinations. A dictionary, mapping columns names
                to a scalar value is also accepted.
            explicit: Determines if only implicitly missing values
                should be filled (`False`), or all nulls existing in the LazyFrame
                (`True`). `explicit` is applicable only
                if `fill_value` is not `None`.
            sort: Sort the LazyFrame based on *columns.
            by: Column(s) to group by.
                The explicit missing rows are returned per group.

        Returns:
            A polars LazyFrame.
        """  # noqa: E501
        return _complete(
            df=self._df,
            columns=columns,
            fill_value=fill_value,
            explicit=explicit,
            sort=sort,
            by=by,
        )
