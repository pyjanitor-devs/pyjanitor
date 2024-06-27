"""expland implementation for polars."""

from __future__ import annotations

from janitor.utils import check, import_message

from .polars_flavor import register_dataframe_method, register_lazyframe_method

try:
    import polars as pl
    import polars.selectors as cs
    from polars.type_aliases import ColumnNameOrSelector
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
    *columns: tuple[ColumnNameOrSelector],
    by: ColumnNameOrSelector = None,
    sort: bool = False,
) -> pl.DataFrame | pl.LazyFrame:
    """
    Creates a DataFrame from a cartesian combination of all inputs.

    Inspiration is from tidyr's expand() function.

    If `by` is present, the DataFrame is *expanded* per group.

    `expand` can also be applied to a LazyFrame.

    !!! info "New in version 0.28.0"

    Examples:
        >>> import pandas as pd
        >>> import janitor
        >>> data = [{'type': 'apple', 'year': 2010, 'size': 'XS'},
        ...         {'type': 'orange', 'year': 2010, 'size': 'S'},
        ...         {'type': 'apple', 'year': 2012, 'size': 'M'},
        ...         {'type': 'orange', 'year': 2010, 'size': 'S'},
        ...         {'type': 'orange', 'year': 2011, 'size': 'S'},
        ...         {'type': 'orange', 'year': 2012, 'size': 'M'}]
        >>> df = pd.DataFrame(data)
        >>> df
             type  year size
        0   apple  2010   XS
        1  orange  2010    S
        2   apple  2012    M
        3  orange  2010    S
        4  orange  2011    S
        5  orange  2012    M

        Get unique observations:
        >>> df.expand('type')
             type
        0   apple
        1  orange
        >>> df.expand('size')
          size
        0   XS
        1    S
        2    M
        >>> df.expand('type', 'size')
             type size
        0   apple   XS
        1   apple    S
        2   apple    M
        3  orange   XS
        4  orange    S
        5  orange    M
        >>> df.expand('type','size','year')
              type size  year
        0    apple   XS  2010
        1    apple   XS  2012
        2    apple   XS  2011
        3    apple    S  2010
        4    apple    S  2012
        5    apple    S  2011
        6    apple    M  2010
        7    apple    M  2012
        8    apple    M  2011
        9   orange   XS  2010
        10  orange   XS  2012
        11  orange   XS  2011
        12  orange    S  2010
        13  orange    S  2012
        14  orange    S  2011
        15  orange    M  2010
        16  orange    M  2012
        17  orange    M  2011

        Get observations that only occur in the data:
        >>> df.expand(['type','size'])
             type size
        0   apple   XS
        1  orange    S
        2   apple    M
        3  orange    M
        >>> df.expand(['type','size','year'])
             type size  year
        0   apple   XS  2010
        1  orange    S  2010
        2   apple    M  2012
        3  orange    S  2011
        4  orange    M  2012

        Expand the DataFrame to include new observations:
        >>> df.expand('type','size',{'new_year':range(2010,2014)})
              type size  new_year
        0    apple   XS      2010
        1    apple   XS      2011
        2    apple   XS      2012
        3    apple   XS      2013
        4    apple    S      2010
        5    apple    S      2011
        6    apple    S      2012
        7    apple    S      2013
        8    apple    M      2010
        9    apple    M      2011
        10   apple    M      2012
        11   apple    M      2013
        12  orange   XS      2010
        13  orange   XS      2011
        14  orange   XS      2012
        15  orange   XS      2013
        16  orange    S      2010
        17  orange    S      2011
        18  orange    S      2012
        19  orange    S      2013
        20  orange    M      2010
        21  orange    M      2011
        22  orange    M      2012
        23  orange    M      2013

        Filter for missing observations:
        >>> combo = df.expand('type','size','year')
        >>> anti_join = df.merge(combo, how='right', indicator=True)
        >>> anti_join.query("_merge=='right_only").drop(columns="_merge")
              type  year size
        1    apple  2012   XS
        2    apple  2011   XS
        3    apple  2010    S
        4    apple  2012    S
        5    apple  2011    S
        6    apple  2010    M
        8    apple  2011    M
        9   orange  2010   XS
        10  orange  2012   XS
        11  orange  2011   XS
        14  orange  2012    S
        16  orange  2010    M
        18  orange  2011    M

        Expand within each group, using `by`:
        >>> df.expand('year','size',by='type')
                year size
        type
        apple   2010   XS
        apple   2010    M
        apple   2012   XS
        apple   2012    M
        orange  2010    S
        orange  2010    M
        orange  2011    S
        orange  2011    M
        orange  2012    S
        orange  2012    M

    Args:
        df: A pandas DataFrame/LazyFrame.
        columns: Specification of columns to expand.
        by: If present, the DataFrame is expanded per group.

    Returns:
        A polars DataFrame/LazyFrame.
    """
    if not columns:
        return df
    check("sort", sort, [bool])
    _columns = []
    for column in columns:
        if isinstance(column, str):
            col = pl.col(column)
            if sort:
                col = col.sort()
            _columns.append(col.implode())
        elif cs.is_selector(column):
            col = column.as_expr()
            if sort:
                col = col.sort()
            _columns.append(col.implode())
        elif isinstance(column, (pl.Expr, pl.Series)):
            _columns.append(column)
        else:
            raise TypeError(
                f"The argument passed to the columns parameter "
                "should either be a string, a column selector, "
                "or a polars expression, instead got - "
                f"{type(column)}."
            )
    by_does_not_exist = by is None
    if by_does_not_exist:
        df = df.select(_columns)
    else:
        df = df.group_by(by, maintain_order=sort).agg(_columns)
    for column in df.columns:
        df = df.explode(column)
    return df
