release_number (on deck)
========================
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
