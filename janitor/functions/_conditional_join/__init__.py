"""Internal building blocks for :func:`janitor.conditional_join`.

The join kernels communicate with result and aggregation code using compact
positional representations. ``starts`` and ``ends`` are per-row half-open
boundaries into right-hand candidates; ``matches`` is a flat survivor mask;
and ``positions`` is an integer tape indexing ``right_index``. ``left_index``
and ``right_index`` carry the original dataframe index values (labels), while
``positions`` entries are positional offsets into the right-side array. Not
every representation is present for every join shape.
"""
