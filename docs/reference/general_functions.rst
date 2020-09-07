=================
General Functions
=================
.. currentmodule:: janitor

Modify columns
~~~~~~~~~~~~~~
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
    rename_columns
    reorder_columns
    collapse_levels
    change_type
    limit_column_characters
    row_to_names
    clean_names
    currency_column_to_numeric
    groupby_agg
    join_apply
    drop_duplicate_columns
    process_text

Modify values
~~~~~~~~~~~~~
.. autosummary::
    :toctree: janitor.functions/

    fill_empty
    fill_direction
    convert_excel_date
    convert_matlab_date
    convert_unix_date
    remove_empty
    coalesce
    find_replace
    round_to_fraction
    update_where
    to_datetime
    jitter

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
    get_dupes

Preprocessing
~~~~~~~~~~~~~
.. autosummary::
    :toctree: janitor.functions/

    bin_numeric
    encode_categorical
    impute
    label_encode
    min_max_scale
    get_features_targets

Other
~~~~~
.. autosummary::
    :toctree: janitor.functions/

    then
    shuffle
    count_cumulative_unique
    sort_naturally
    expand_grid
    flag_nulls
    move
    toset
    unionize_dataframe_categories
    groupby_topk
    complete
    
