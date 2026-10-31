"""Internal building blocks for :func:`janitor.conditional_join`.

The join kernels communicate with result and aggregation code using compact
positional representations. ``starts`` and ``ends`` are per-row half-open
boundaries into right-hand candidates; ``matches`` is a flat survivor mask;
and ``positions`` is an integer tape indexing ``right_index``. Both frames are
normalized to ``range(len(frame))`` before join discovery, so ``left_index``
and ``right_index`` hold row positions rather than the caller's index labels.
Not every representation is present for every join shape.
"""
