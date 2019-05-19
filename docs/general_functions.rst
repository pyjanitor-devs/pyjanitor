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
    collapse_levels
    change_type
    limit_column_characters
    row_to_names
    clean_names
    currency_column_to_numeric
    groupby_agg
    drop_duplicate_columns

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
    round_to_fraction
    update_where
    to_datetime

Filtering
~~~~~~~~~
.. autosummary::
   :toctree: janitor.functions/

    take_first
    filter_string
    filter_on
    filter_date
    filter_column_isin
    select_columns
    dropnotnull

Preprocessing
~~~~~~~~~~~~~
.. autosummary::
   :toctree: janitor.functions/

    bin_numeric
    encode_categorical
    impute
    label_encode
    min_max_scale

Other
~~~~~
.. autosummary::
   :toctree: janitor.functions/

   then
