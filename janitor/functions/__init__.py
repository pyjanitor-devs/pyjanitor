"""
# General Functions

pyjanitor's general-purpose data cleaning functions.

NOTE: Instructions for future contributors:

1. Place the source code of the functions in a file named after the function.
2. Place utility functions in the same file.
3. If you use a utility function from another source file,
please refactor it out to `janitor.functions.utils`.
4. Import the function into this file so that it shows up in the top-level API.
5. Sort the imports in alphabetical order.
6. Try to group related functions together (e.g. see `convert_date.py`)
7. Never import utils.
"""


from .add_columns import add_columns
from .also import also
from .bin_numeric import bin_numeric
from .case_when import case_when
from .change_type import change_type
from .clean_names import clean_names
from .coalesce import coalesce
from .collapse_levels import collapse_levels
from .complete import complete
from .concatenate_columns import concatenate_columns
from .conditional_join import conditional_join
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
from .expand_grid import expand_grid
from .factorize_columns import factorize_columns
from .fill import fill_direction, fill_empty
from .filter import filter_date, filter_column_isin, filter_on, filter_string
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
from .pivot import pivot_longer, pivot_wider
from .process_text import process_text
from .remove_columns import remove_columns
from .remove_empty import remove_empty
from .rename_columns import rename_column, rename_columns
from .reorder_columns import reorder_columns
from .round_to_fraction import round_to_fraction
from .row_to_names import row_to_names
from .select_columns import select_columns
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
from .utils import patterns, unionize_dataframe_categories
