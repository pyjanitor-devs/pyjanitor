from typing import Any, Iterable, Optional, Union

from polars.type_aliases import IntoExpr

from janitor.utils import import_message

from .pivot_longer import _pivot_longer

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
        index: Union[IntoExpr, Iterable[IntoExpr], None] = None,
        column_names: Union[IntoExpr, Iterable[IntoExpr], None] = None,
        names_to: Optional[Union[list, tuple, str]] = "variable",
        values_to: Optional[Union[list, tuple, str]] = "value",
        names_sep: Optional[Union[str, None]] = None,
        names_pattern: Optional[Union[list, tuple, str, None]] = None,
        names_transform: Optional[Any] = pl.Utf8,
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
            ... )
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
            ... )
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
            ... )
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
            ...     )
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
            ... )
            shape: (2, 4)
            ┌──────┬──────┬────────┬────────┐
            │ unit ┆ time ┆ x_mean ┆ y_mean │
            │ ---  ┆ ---  ┆ ---    ┆ ---    │
            │ i64  ┆ str  ┆ i64    ┆ i64    │
            ╞══════╪══════╪════════╪════════╡
            │ 50   ┆ 1    ┆ 10     ┆ 30     │
            │ 50   ┆ 2    ┆ 20     ┆ 40     │
            └──────┴──────┴────────┴────────┘

            Reshape the dataframe by passing a sequence to `names_pattern`:
            >>> df = pl.DataFrame({'hr1': [514, 573],
            ...                    'hr2': [545, 526],
            ...                    'team': ['Red Sox', 'Yankees'],
            ...                    'year1': [2007, 2007],
            ...                    'year2': [2008, 2008]})
            >>> df
            shape: (2, 5)
            ┌─────┬─────┬─────────┬───────┬───────┐
            │ hr1 ┆ hr2 ┆ team    ┆ year1 ┆ year2 │
            │ --- ┆ --- ┆ ---     ┆ ---   ┆ ---   │
            │ i64 ┆ i64 ┆ str     ┆ i64   ┆ i64   │
            ╞═════╪═════╪═════════╪═══════╪═══════╡
            │ 514 ┆ 545 ┆ Red Sox ┆ 2007  ┆ 2008  │
            │ 573 ┆ 526 ┆ Yankees ┆ 2007  ┆ 2008  │
            └─────┴─────┴─────────┴───────┴───────┘
            >>> df.janitor.pivot_longer(
            ...     index = 'team',
            ...     names_to = ['year', 'hr'],
            ...     names_pattern = ['year', 'hr']
            ... )
            shape: (4, 3)
            ┌─────────┬─────┬──────┐
            │ team    ┆ hr  ┆ year │
            │ ---     ┆ --- ┆ ---  │
            │ str     ┆ i64 ┆ i64  │
            ╞═════════╪═════╪══════╡
            │ Red Sox ┆ 514 ┆ 2007 │
            │ Yankees ┆ 573 ┆ 2007 │
            │ Red Sox ┆ 545 ┆ 2008 │
            │ Yankees ┆ 526 ┆ 2008 │
            └─────────┴─────┴──────┘

            Multiple `values_to`:
            >>> df = pl.DataFrame(
            ...         {
            ...             "City": ["Houston", "Austin", "Hoover"],
            ...             "State": ["Texas", "Texas", "Alabama"],
            ...             "Name": ["Aria", "Penelope", "Niko"],
            ...             "Mango": [4, 10, 90],
            ...             "Orange": [10, 8, 14],
            ...             "Watermelon": [40, 99, 43],
            ...             "Gin": [16, 200, 34],
            ...             "Vodka": [20, 33, 18],
            ...         },
            ...     )
            >>> df
            shape: (3, 8)
            ┌─────────┬─────────┬──────────┬───────┬────────┬────────────┬─────┬───────┐
            │ City    ┆ State   ┆ Name     ┆ Mango ┆ Orange ┆ Watermelon ┆ Gin ┆ Vodka │
            │ ---     ┆ ---     ┆ ---      ┆ ---   ┆ ---    ┆ ---        ┆ --- ┆ ---   │
            │ str     ┆ str     ┆ str      ┆ i64   ┆ i64    ┆ i64        ┆ i64 ┆ i64   │
            ╞═════════╪═════════╪══════════╪═══════╪════════╪════════════╪═════╪═══════╡
            │ Houston ┆ Texas   ┆ Aria     ┆ 4     ┆ 10     ┆ 40         ┆ 16  ┆ 20    │
            │ Austin  ┆ Texas   ┆ Penelope ┆ 10    ┆ 8      ┆ 99         ┆ 200 ┆ 33    │
            │ Hoover  ┆ Alabama ┆ Niko     ┆ 90    ┆ 14     ┆ 43         ┆ 34  ┆ 18    │
            └─────────┴─────────┴──────────┴───────┴────────┴────────────┴─────┴───────┘

            >>> df.janitor.pivot_longer(
            ...     index=["City", "State"],
            ...     column_names=cs.numeric(),
            ...     names_to=("Fruit", "Drink"),
            ...     values_to=("Pounds", "Ounces"),
            ...     names_pattern=["M|O|W", "G|V"],
            ...     )
            shape: (9, 6)
            ┌─────────┬─────────┬────────────┬────────┬───────┬────────┐
            │ City    ┆ State   ┆ Fruit      ┆ Pounds ┆ Drink ┆ Ounces │
            │ ---     ┆ ---     ┆ ---        ┆ ---    ┆ ---   ┆ ---    │
            │ str     ┆ str     ┆ str        ┆ i64    ┆ str   ┆ i64    │
            ╞═════════╪═════════╪════════════╪════════╪═══════╪════════╡
            │ Houston ┆ Texas   ┆ Mango      ┆ 4      ┆ Gin   ┆ 16     │
            │ Austin  ┆ Texas   ┆ Mango      ┆ 10     ┆ Gin   ┆ 200    │
            │ Hoover  ┆ Alabama ┆ Mango      ┆ 90     ┆ Gin   ┆ 34     │
            │ Houston ┆ Texas   ┆ Orange     ┆ 10     ┆ Vodka ┆ 20     │
            │ Austin  ┆ Texas   ┆ Orange     ┆ 8      ┆ Vodka ┆ 33     │
            │ Hoover  ┆ Alabama ┆ Orange     ┆ 14     ┆ Vodka ┆ 18     │
            │ Houston ┆ Texas   ┆ Watermelon ┆ 40     ┆ null  ┆ null   │
            │ Austin  ┆ Texas   ┆ Watermelon ┆ 99     ┆ null  ┆ null   │
            │ Hoover  ┆ Alabama ┆ Watermelon ┆ 43     ┆ null  ┆ null   │
            └─────────┴─────────┴────────────┴────────┴───────┴────────┘

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
                `values_to` can also be a list/tuple
                and requires that `names_pattern` is also a list/tuple.
            names_sep: Determines how the column name is broken up, if
                `names_to` contains multiple values. It takes the same
                specification as polars' `str.split` method.
            names_pattern: Determines how the column name is broken up.
                It can be a regular expression containing matching groups.
                It takes the same
                specification as polars' `str.extract_groups` method.
                `names_pattern` can also be a list/tuple of regular expressions.
                Under the hood it is processed with polars' `str.contains` function.
                For a list/tuple of regular expressions,
                `names_to` must also be a list/tuple and the lengths of both
                arguments must match.
            names_transform: Use this option to change the types of columns that
                have been transformed to rows.
                This does not applies to the values' columns.
                It can be a single valid polars dtype,
                or a dictionary pairing the new column names
                with a valid polars dtype.
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
