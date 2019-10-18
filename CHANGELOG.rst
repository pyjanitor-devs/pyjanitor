new version (on deck)
=====================
- [ENH] Series toset() functionality #570
- [ENH] Added option to coalesce function to not delete coalesced columns.
- [ENH] Added functionality to deconcatenate tuple/list/collections in a column to deconcatenate_column
- [ENH] Fix error message when length of new_column_names is wrong
- [DOC] Fixed several examples of functional syntax in ``functions.py``.
- [DOC] Fix #noqa comments showing up in docs.
- [ENH] Add unionizing a group of dataframes' categoricals.

v0.18.2
=======
- [INF] Make requirements.txt smaller
- [ENH] Add a reset_index parameter to shuffle
- [DOC] Added contribution page link to readme
- [DOC] fix example for ``update_where``, provide a bit more detail, and expand the bad_values example notebook to demonstrate its use by @anzelpwj.
- [INF] Fix pytest marks by @ericmjl (issue #520)
- [ENH] add example notebook with use of finance submodule methods by @rahosbach
- [DOC] added a couple of admonitions for Windows users. h/t @anzelpwj for debugging
   help when a few tests failed for `win32`
- [ENH] Pyjanitor for PySpark @zjpoh
- [ENH] Add pyspark clean_names @zjpoh
- [ENH] Convert asserts to raise exceptions by @hectormz
- [ENH] Add decorator functions for missing and error handling @jiafengkevinchen
- [DOC] Update README with functional ``pandas`` API example.
- [INF] Move ``get_features_targets()`` to new ``ml.py`` module by @hectormz
- [ENH] Add chirality to morgan fingerprints in janitor.chemistry submodule by @Clayton-Springer
- [INF] ``import_message`` suggests python dist. appropriate installs by @hectormz
- [ENH] Add count_cumulative_unique() method to janitor.functions submodule by @rahosbach
- [ENH] Add ``update_where()`` method to ``janitor.spark.functions`` submodule by @zjpoh

v0.18.1
=======
- [ENH] extend find_replace functionality to allow both exact match and
  regular-expression-based fuzzy match by @shandou
- [ENH] add preserve_position kwarg to deconcatenate_column with tests
  by @shandou and @ericmjl
- [DOC] add contributions that did not leave ``git`` traces by @ericmjl
- [ENH] add inflation adjustment in finance submodule by @rahosbach
- [DOC] clarified how new functions should be implemented by @shandou
- [ENH] add optional removal of accents on functions.clean_names, enabled by
  default by @mralbu
- [ENH] add camelCase conversion to snake_case on ``clean_names`` by @ericmjl,
  h/t @jtaylor for sharing original
- [ENH] Added ``null_flag`` function which can mark null values in rows.
  Implemented by @anzelpwj
- [ENH] add engineering submodule with unit conversion method by @rahosbach
- [DOC] add PyPI project description
- [ENH] add example notebook with use of finance submodule methods
  by @rahosbach

For changes that happened prior to v0.18.1,
please consult the closed PRs,
which can be found here_.

.. _here: https://github.com/ericmjl/pyjanitor/pulls?q=is%3Apr+is%3Aclosed

We thank all contributors
who have helped make ``pyjanitor``
the package that it is today.
