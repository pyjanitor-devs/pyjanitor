"""
# General Functions

pyjanitor's general-purpose data cleaning functions.
"""

# NOTE: Instructions for future contributors:

# 1. Place the source code of the functions in a file named after the function.
# 2. Place utility functions in the same file.
# 3. If you use a utility function from another source file,
# please refactor it out to `janitor.functions.utils`.
# 4. Import the function into this file so that it shows up in the top-level API.
# 5. Sort the imports in alphabetical order.
# 6. Try to group related functions together (e.g. see `convert_date.py`)
# 7. Never import utils.

from .add_columns import add_columns
from .also import also
from .bin_numeric import bin_numeric
from .case_when import case_when
from .change_index_dtype import change_index_dtype
from .change_type import change_type
from .clean_names import clean_names
from .coalesce import coalesce
from .collapse_levels import collapse_levels
from .complete import complete
from .concatenate_columns import concatenate_columns
from .conditional_join import conditional_join, get_join_indices
from .convert_date import (
    convert_excel_date,
    convert_matlab_date,
    convert_unix_date,
)
from .count_cumulative_unique import count_cumulative_unique
from .currency_column_to_numeric import currency_column_to_numeric
from .deconcatenate_column import deconcatenate_column
from .drop_constant_columns import drop_constant_columns
from .drop_duplicate_columns import drop_duplicate_columns
from .dropnotnull import dropnotnull
from .encode_categorical import encode_categorical
from .expand_column import expand_column
from .expand_grid import cartesian_product, expand, expand_grid
from .explode_index import explode_index
from .factorize_columns import factorize_columns
from .fill import fill_direction, fill_empty
from .filter import filter_column_isin, filter_date, filter_on, filter_string
from .find_replace import find_replace
from .flag_nulls import flag_nulls
from .get_dupes import get_dupes
from .groupby_agg import groupby_agg
from .groupby_topk import groupby_topk
from .impute import impute
from .jitter import jitter
from .join_apply import join_apply
from .label_encode import label_encode
from .limit_column_characters import limit_column_characters
from .min_max_scale import min_max_scale
from .move import move
from .pivot import pivot_longer, pivot_longer_spec, pivot_wider
from .process_text import process_text
from .remove_columns import remove_columns
from .remove_empty import remove_empty
from .rename_columns import rename_column, rename_columns
from .reorder_columns import reorder_columns
from .round_to_fraction import round_to_fraction
from .row_to_names import row_to_names
from .select import (
    DropLabel,
    get_columns,
    get_index_labels,
    select,
    select_columns,
    select_rows,
)
from .shuffle import shuffle
from .sort_column_value_order import sort_column_value_order
from .sort_naturally import sort_naturally
from .take_first import take_first
from .then import then
from .to_datetime import to_datetime
from .toset import toset
from .transform_columns import transform_column, transform_columns
from .truncate_datetime import truncate_datetime_dataframe
from .update_where import update_where
from .utils import (
    unionize_dataframe_categories,
)

__all__ = [
    "add_columns",
    "also",
    "bin_numeric",
    "cartesian_product",
    "case_when",
    "change_type",
    "change_index_dtype",
    "clean_names",
    "coalesce",
    "collapse_levels",
    "complete",
    "concatenate_columns",
    "conditional_join",
    "convert_excel_date",
    "convert_matlab_date",
    "convert_unix_date",
    "count_cumulative_unique",
    "currency_column_to_numeric",
    "deconcatenate_column",
    "drop_constant_columns",
    "drop_duplicate_columns",
    "dropnotnull",
    "encode_categorical",
    "expand",
    "expand_column",
    "expand_grid",
    "explode_index",
    "factorize_columns",
    "fill_direction",
    "fill_empty",
    "filter_date",
    "filter_column_isin",
    "filter_on",
    "filter_string",
    "find_replace",
    "flag_nulls",
    "get_dupes",
    "get_join_indices",
    "groupby_agg",
    "groupby_topk",
    "impute",
    "jitter",
    "join_apply",
    "label_encode",
    "limit_column_characters",
    "min_max_scale",
    "move",
    "pivot_longer",
    "pivot_longer_spec",
    "pivot_wider",
    "process_text",
    "remove_columns",
    "remove_empty",
    "rename_column",
    "rename_columns",
    "reorder_columns",
    "round_to_fraction",
    "row_to_names",
    "select_columns",
    "select_rows",
    "select",
    "shuffle",
    "sort_column_value_order",
    "sort_naturally",
    "take_first",
    "then",
    "to_datetime",
    "toset",
    "transform_column",
    "transform_columns",
    "truncate_datetime_dataframe",
    "update_where",
    "unionize_dataframe_categories",
    "DropLabel",
    "get_index_labels",
    "get_columns",
]
