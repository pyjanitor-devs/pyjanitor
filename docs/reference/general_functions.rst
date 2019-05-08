=================
General functions
=================
.. currentmodule:: janitor

Modify columns
~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: janitor.functions/

    expand_column
    concatenate_columns
    deconcatenate_column
    remove_columns
    add_column
    add_columns
    transform_column
    transform_columns
    rename_column
    reorder_columns
    reset_index_inplace
    collapse_levels
    change_type
    limit_column_characters
    row_to_names
    clean_names

Modify values
~~~~~~~~~~~~~
.. autosummary::
   :toctree: janitor.functions/

    fill_empty
    convert_excel_date
    convert_matlab_date
    convert_unix_date
    remove_empty
    coalesce
    find_replace
    dropnotnull
    update_where

Preprocessing
~~~~~~~~~~~~~
.. autosummary::
   :toctree: janitor.functions/

    min_max_scale
    impute
    label_encode
    encode_categorical
