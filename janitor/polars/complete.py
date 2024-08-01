"""complete implementation for polars."""

from __future__ import annotations

from typing import Any

from janitor.utils import check, import_message

from .polars_flavor import register_dataframe_method, register_lazyframe_method

try:
    import polars as pl
    import polars.selectors as cs
    from polars._typing import ColumnNameOrSelector
except ImportError:
    import_message(
        submodule="polars",
        package="polars",
        conda_channel="conda-forge",
        pip_install=True,
    )


@register_lazyframe_method
@register_dataframe_method
def expand(
    df: pl.DataFrame | pl.LazyFrame,
    *columns: ColumnNameOrSelector,
    sort: bool = False,
    by: ColumnNameOrSelector = None,
) -> pl.DataFrame | pl.LazyFrame:
    """
    Creates a DataFrame from a cartesian combination of all inputs.

    Inspiration is from tidyr's expand() function.

    expand() is often useful with
    [pl.DataFrame.join](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.join.html)
    to convert implicit
    missing values to explicit missing values - similar to
    [`complete`][janitor.polars.complete.complete].

    It can also be used to figure out which combinations are missing
    (e.g identify gaps in your DataFrame).

    The variable `columns` parameter can be a string,
    a ColumnSelector, a polars expression, or a polars Series.

    `expand` can also be applied to a LazyFrame.

    Examples:
        >>> import polars as pl
        >>> import janitor.polars
        >>> data = [{'type': 'apple', 'year': 2010, 'size': 'XS'},
        ...         {'type': 'orange', 'year': 2010, 'size': 'S'},
        ...         {'type': 'apple', 'year': 2012, 'size': 'M'},
        ...         {'type': 'orange', 'year': 2010, 'size': 'S'},
        ...         {'type': 'orange', 'year': 2011, 'size': 'S'},
        ...         {'type': 'orange', 'year': 2012, 'size': 'M'}]
        >>> df = pl.DataFrame(data)
        >>> df
        shape: (6, 3)
        ┌────────┬──────┬──────┐
        │ type   ┆ year ┆ size │
        │ ---    ┆ ---  ┆ ---  │
        │ str    ┆ i64  ┆ str  │
        ╞════════╪══════╪══════╡
        │ apple  ┆ 2010 ┆ XS   │
        │ orange ┆ 2010 ┆ S    │
        │ apple  ┆ 2012 ┆ M    │
        │ orange ┆ 2010 ┆ S    │
        │ orange ┆ 2011 ┆ S    │
        │ orange ┆ 2012 ┆ M    │
        └────────┴──────┴──────┘

        Get unique observations:
        >>> df.expand('type',sort=True)
        shape: (2, 1)
        ┌────────┐
        │ type   │
        │ ---    │
        │ str    │
        ╞════════╡
        │ apple  │
        │ orange │
        └────────┘
        >>> df.expand('size',sort=True)
        shape: (3, 1)
        ┌──────┐
        │ size │
        │ ---  │
        │ str  │
        ╞══════╡
        │ M    │
        │ S    │
        │ XS   │
        └──────┘
        >>> df.expand('type', 'size',sort=True)
        shape: (6, 2)
        ┌────────┬──────┐
        │ type   ┆ size │
        │ ---    ┆ ---  │
        │ str    ┆ str  │
        ╞════════╪══════╡
        │ apple  ┆ M    │
        │ apple  ┆ S    │
        │ apple  ┆ XS   │
        │ orange ┆ M    │
        │ orange ┆ S    │
        │ orange ┆ XS   │
        └────────┴──────┘
        >>> with pl.Config(tbl_rows=-1):
        ...     df.expand('type','size','year',sort=True)
        shape: (18, 3)
        ┌────────┬──────┬──────┐
        │ type   ┆ size ┆ year │
        │ ---    ┆ ---  ┆ ---  │
        │ str    ┆ str  ┆ i64  │
        ╞════════╪══════╪══════╡
        │ apple  ┆ M    ┆ 2010 │
        │ apple  ┆ M    ┆ 2011 │
        │ apple  ┆ M    ┆ 2012 │
        │ apple  ┆ S    ┆ 2010 │
        │ apple  ┆ S    ┆ 2011 │
        │ apple  ┆ S    ┆ 2012 │
        │ apple  ┆ XS   ┆ 2010 │
        │ apple  ┆ XS   ┆ 2011 │
        │ apple  ┆ XS   ┆ 2012 │
        │ orange ┆ M    ┆ 2010 │
        │ orange ┆ M    ┆ 2011 │
        │ orange ┆ M    ┆ 2012 │
        │ orange ┆ S    ┆ 2010 │
        │ orange ┆ S    ┆ 2011 │
        │ orange ┆ S    ┆ 2012 │
        │ orange ┆ XS   ┆ 2010 │
        │ orange ┆ XS   ┆ 2011 │
        │ orange ┆ XS   ┆ 2012 │
        └────────┴──────┴──────┘

        Get observations that only occur in the data:
        >>> df.expand(pl.struct('type','size'),sort=True).unnest('type')
        shape: (4, 2)
        ┌────────┬──────┐
        │ type   ┆ size │
        │ ---    ┆ ---  │
        │ str    ┆ str  │
        ╞════════╪══════╡
        │ apple  ┆ M    │
        │ apple  ┆ XS   │
        │ orange ┆ M    │
        │ orange ┆ S    │
        └────────┴──────┘
        >>> df.expand(pl.struct('type','size','year'),sort=True).unnest('type')
        shape: (5, 3)
        ┌────────┬──────┬──────┐
        │ type   ┆ size ┆ year │
        │ ---    ┆ ---  ┆ ---  │
        │ str    ┆ str  ┆ i64  │
        ╞════════╪══════╪══════╡
        │ apple  ┆ M    ┆ 2012 │
        │ apple  ┆ XS   ┆ 2010 │
        │ orange ┆ M    ┆ 2012 │
        │ orange ┆ S    ┆ 2010 │
        │ orange ┆ S    ┆ 2011 │
        └────────┴──────┴──────┘

        Expand the DataFrame to include new observations:
        >>> with pl.Config(tbl_rows=-1):
        ...     df.expand('type','size',pl.int_range(2010,2014).alias('new_year'),sort=True)
        shape: (24, 3)
        ┌────────┬──────┬──────────┐
        │ type   ┆ size ┆ new_year │
        │ ---    ┆ ---  ┆ ---      │
        │ str    ┆ str  ┆ i64      │
        ╞════════╪══════╪══════════╡
        │ apple  ┆ M    ┆ 2010     │
        │ apple  ┆ M    ┆ 2011     │
        │ apple  ┆ M    ┆ 2012     │
        │ apple  ┆ M    ┆ 2013     │
        │ apple  ┆ S    ┆ 2010     │
        │ apple  ┆ S    ┆ 2011     │
        │ apple  ┆ S    ┆ 2012     │
        │ apple  ┆ S    ┆ 2013     │
        │ apple  ┆ XS   ┆ 2010     │
        │ apple  ┆ XS   ┆ 2011     │
        │ apple  ┆ XS   ┆ 2012     │
        │ apple  ┆ XS   ┆ 2013     │
        │ orange ┆ M    ┆ 2010     │
        │ orange ┆ M    ┆ 2011     │
        │ orange ┆ M    ┆ 2012     │
        │ orange ┆ M    ┆ 2013     │
        │ orange ┆ S    ┆ 2010     │
        │ orange ┆ S    ┆ 2011     │
        │ orange ┆ S    ┆ 2012     │
        │ orange ┆ S    ┆ 2013     │
        │ orange ┆ XS   ┆ 2010     │
        │ orange ┆ XS   ┆ 2011     │
        │ orange ┆ XS   ┆ 2012     │
        │ orange ┆ XS   ┆ 2013     │
        └────────┴──────┴──────────┘

        Filter for missing observations:
        >>> columns = ('type','size','year')
        >>> with pl.Config(tbl_rows=-1):
        ...     df.expand(*columns).join(df, how='anti', on=columns).sort(by=pl.all())
        shape: (13, 3)
        ┌────────┬──────┬──────┐
        │ type   ┆ size ┆ year │
        │ ---    ┆ ---  ┆ ---  │
        │ str    ┆ str  ┆ i64  │
        ╞════════╪══════╪══════╡
        │ apple  ┆ M    ┆ 2010 │
        │ apple  ┆ M    ┆ 2011 │
        │ apple  ┆ S    ┆ 2010 │
        │ apple  ┆ S    ┆ 2011 │
        │ apple  ┆ S    ┆ 2012 │
        │ apple  ┆ XS   ┆ 2011 │
        │ apple  ┆ XS   ┆ 2012 │
        │ orange ┆ M    ┆ 2010 │
        │ orange ┆ M    ┆ 2011 │
        │ orange ┆ S    ┆ 2012 │
        │ orange ┆ XS   ┆ 2010 │
        │ orange ┆ XS   ┆ 2011 │
        │ orange ┆ XS   ┆ 2012 │
        └────────┴──────┴──────┘

        Expand within each group, using `by`:
        >>> with pl.Config(tbl_rows=-1):
        ...     df.expand('year','size',by='type',sort=True)
        shape: (10, 3)
        ┌────────┬──────┬──────┐
        │ type   ┆ year ┆ size │
        │ ---    ┆ ---  ┆ ---  │
        │ str    ┆ i64  ┆ str  │
        ╞════════╪══════╪══════╡
        │ apple  ┆ 2010 ┆ M    │
        │ apple  ┆ 2010 ┆ XS   │
        │ apple  ┆ 2012 ┆ M    │
        │ apple  ┆ 2012 ┆ XS   │
        │ orange ┆ 2010 ┆ M    │
        │ orange ┆ 2010 ┆ S    │
        │ orange ┆ 2011 ┆ M    │
        │ orange ┆ 2011 ┆ S    │
        │ orange ┆ 2012 ┆ M    │
        │ orange ┆ 2012 ┆ S    │
        └────────┴──────┴──────┘

    !!! info "New in version 0.28.0"

    Args:
        *columns: This refers to the columns to be completed.
            It can be a string or a column selector or a polars expression.
            A polars expression can be used to introduced new values,
            as long as the polars expression has a name that already exists
            in the DataFrame.
        sort: Sort the DataFrame based on *columns.
        by: Column(s) to group by.

    Returns:
        A polars DataFrame/LazyFrame.
    """  # noqa: E501
    if not columns:
        return df
    uniques, _ = _expand(df=df, columns=columns, by=by, sort=sort)
    return uniques


@register_lazyframe_method
@register_dataframe_method
def complete(
    df: pl.DataFrame | pl.LazyFrame,
    *columns: ColumnNameOrSelector,
    fill_value: dict | Any | pl.Expr = None,
    explicit: bool = True,
    sort: bool = False,
    by: ColumnNameOrSelector = None,
) -> pl.DataFrame | pl.LazyFrame:
    """
    Turns implicit missing values into explicit missing values

    It is modeled after tidyr's `complete` function.
    In a way, it is the inverse of `pl.drop_nulls`,
    as it exposes implicitly missing rows.

    If new values need to be introduced, a polars Expression
    or a polars Series with the new values can be passed,
    as long as the polars Expression/Series
    has a name that already exists in the DataFrame.

    `complete` can also be applied to a LazyFrame.

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
        ...     df.complete("group", "item_id", "item_name", sort=True)
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
        >>> with pl.Config(tbl_rows=-1):
        ...     df.select(
        ...         "group", pl.struct("item_id", "item_name"), "value1", "value2"
        ...     ).complete("group", "item_id", sort=True).unnest("item_id")
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
        ...     df.select(
        ...         "group", pl.struct("item_id", "item_name"), "value1", "value2"
        ...     ).complete(
        ...         "group",
        ...         "item_id",
        ...         fill_value={"value1": 0, "value2": 99},
        ...         explicit=True,
        ...         sort=True,
        ...     ).unnest("item_id")
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
        missing values with `explicit = FALSE`:
        >>> with pl.Config(tbl_rows=-1):
        ...     df.select(
        ...         "group", pl.struct("item_id", "item_name"), "value1", "value2"
        ...     ).complete(
        ...         "group",
        ...         "item_id",
        ...         fill_value={"value1": 0, "value2": 99},
        ...         explicit=False,
        ...         sort=True,
        ...     ).unnest("item_id").sort(pl.all())
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
        ...     df.complete(expression,'Taxon',sort=True)
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
        ...     df.complete(new_year_values,by='state',sort=True)
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
        A polars DataFrame/LazyFrame.
    """  # noqa: E501
    if not columns:
        return df
    return _complete(
        df=df,
        columns=columns,
        fill_value=fill_value,
        explicit=explicit,
        sort=sort,
        by=by,
    )


def _expand(
    df: pl.DataFrame | pl.LazyFrame,
    columns: tuple[ColumnNameOrSelector],
    sort: bool,
    by: ColumnNameOrSelector,
) -> pl.DataFrame | pl.LazyFrame:
    """
    This function computes the final output for the `complete` function.

    A DataFrame, with rows of missing values, if any, is returned.
    """

    check("sort", sort, [bool])
    _columns = []
    for column in columns:
        if isinstance(column, str):
            col = pl.col(column).unique()
        elif cs.is_selector(column):
            col = column.as_expr().unique()
        elif isinstance(column, pl.Expr):
            col = column.unique()
        elif isinstance(column, pl.Series):
            col = column.unique()
        else:
            raise TypeError(
                f"The argument passed to the columns parameter "
                "should either be a string, a column selector, "
                "a polars expression, or a polars Series; "
                f"instead got - {type(column)}."
            )
        if sort:
            col = col.sort()
        _columns.append(col)
    by_does_not_exist = by is None
    if by_does_not_exist:
        _columns = [column.implode() for column in _columns]
        uniques = df.select(_columns)
        uniques_schema = uniques.collect_schema()
        _columns = uniques_schema.names()
    else:
        uniques = df.group_by(by, maintain_order=sort).agg(_columns)
        uniques_schema = uniques.collect_schema()
        _columns = cs.expand_selector(
            uniques_schema, cs.exclude(by), strict=False
        )
    for column in _columns:
        uniques = uniques.explode(column)

    return uniques, uniques_schema


def _complete(
    df: pl.DataFrame | pl.LazyFrame,
    columns: tuple[ColumnNameOrSelector],
    fill_value: dict | Any | pl.Expr,
    explicit: bool,
    sort: bool,
    by: ColumnNameOrSelector,
) -> pl.DataFrame | pl.LazyFrame:
    """
    This function computes the final output for the `complete` function.

    A DataFrame, with rows of missing values, if any, is returned.
    """
    check("explicit", explicit, [bool])
    uniques, uniques_schema = _expand(df=df, columns=columns, sort=sort, by=by)

    df_columns = df.collect_schema()
    columns_to_fill = df_columns.keys() ^ uniques_schema.keys()
    if (fill_value is None) or not columns_to_fill:
        return uniques.join(
            df, on=uniques_schema.names(), how="left", coalesce=True
        )
    idx = None
    columns_to_select = df_columns.names()
    if not explicit:
        idx = "".join(columns_to_select)
        idx = f"{idx}_"
        df = df.with_row_index(name=idx)
    df = uniques.join(df, on=uniques_schema.names(), how="left", coalesce=True)
    # exclude columns that were not used
    # to generate the combinations
    exclude_columns = uniques_schema.names()
    if idx:
        exclude_columns.append(idx)
    _columns = set(columns_to_select).difference(exclude_columns)

    if isinstance(fill_value, dict):
        _columns = _columns.intersection(fill_value)
        fill_value = [
            pl.col(column_name).fill_null(value=fill_value[column_name])
            for column_name in _columns
        ]
    else:
        fill_value = [
            pl.col(column).fill_null(value=fill_value) for column in _columns
        ]
    if not explicit:
        condition = pl.col(idx).is_null()
        fill_value = [
            pl.when(condition).then(_fill_value).otherwise(pl.col(column_name))
            for column_name, _fill_value in zip(_columns, fill_value)
        ]
    df = df.with_columns(fill_value)

    return df.select(columns_to_select)
