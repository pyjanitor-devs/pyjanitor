# Changelog

## [Unreleased]
-   [BUG] Fix conditional join issue for multiple conditions, where pd.eval fails to evaluate if numexpr is installed. #898 @samukweku
-   [ENH] Added `case_when` to handle multiple conditionals and replacement values. Issue #736. @robertmitchellv
-   [ENH] Deprecate `new_column_names` and `merge_frame` from `process_text`. Only existing columns are supported. @samukweku
-   [ENH] `complete` uses `pd.merge` internally, providing a simpler logic, with some speed improvements in certain cases over `pd.reindex`. @samukweku
-   [ENH] `expand_grid` returns a MultiIndex DataFrame, allowing the user to decide how to manipulate the columns. @samukweku
-   [INF] Simplify a bit linting, use pre-commit as the CI linting checker. @Zeroto521
-   [ENH] Fix bug in `pivot_longer` for wrong output when `names_pattern` is a sequence with a single value. Issue #885 @samukweku
-   [ENH] Deprecate `aggfunc` from `pivot_wider`; aggregation can be chained with pandas' `groupby`.
-   [ENH] `As_Categorical` deprecated from `encode_categorical`; a tuple of `(categories, order)` suffices for **kwargs. @samukweku
-   [ENH] Deprecate `names_sort` from `pivot_wider`.@samukweku
-   [ENH] Add `softmax` to `math` module. Issue #902. @loganthomas

## [v0.21.2] - 2021-09-01

-   [ENH] Fix warning message in `coalesce`, from bfill/fill;`coalesce` now uses variable arguments. Issue #882 @samukweku
-   [INF] Add SciPy as explicit dependency in `base.in`. Issue #895 @ericmjl

## [v0.21.1] - 2021-08-29

-   [DOC] Fix references and broken links in AUTHORS.rst. @loganthomas
-   [DOC] Updated Broken links in the README and contributing docs. @nvamsikrishna05
-   [INF] Update pre-commit hooks and remove mutable references. Issue #844. @loganthomas
-   [INF] Add GitHub Release pointer to auto-release script. Issue #818. @loganthomas
-   [INF] Updated black version in github actions code-checks to match pre-commit hooks. @nvamsikrishna05
-   [ENH] Add reset_index flag to row_to_names function. @fireddd
-   [ENH] Updated `label_encode` to use pandas factorize instead of scikit-learn LabelEncoder. @nvamsikrishna05
-   [INF] Removed the scikit-learn package from the dependencies from environment-dev.yml and base.in files. @nvamsikrishna05
-   [ENH] Add function to remove constant columns. @fireddd
-   [ENH] Added `factorize_columns` method which will deprecate the `label_encode` method in future release. @nvamsikrishna05
-   [DOC] Delete Read the Docs project and remove all readthedocs.io references from the repo. Issue #863. @loganthomas
-   [DOC] Updated various documentation sources to reflect pyjanitor-dev ownership. @loganthomas
-   [INF] Fix `isort` automatic checks. Issue #845. @loganthomas
-   [ENH] `complete` function now uses variable args (\*args) - @samukweku
-   [EHN] Set `expand_column`'s `sep` default is `"|"`, same to `pandas.Series.str.get_dummies`. Issue #876. @Zeroto521
-   [ENH] Deprecate `limit` from fill_direction. fill_direction now uses kwargs. @samukweku
-   [ENH] Added `conditional_join` function that supports joins on non-equi operators. @samukweku
-   [INF] Speed up pytest via `-n` (pytest-xdist) option. Issue #881. @Zeroto521
-   [DOC] Add list mark to keep `select_columns`'s example same style. @Zeroto521
-   [ENH] Updated `rename_columns` to take optional function argument for mapping. @nvamsikrishna05

## [v0.21.0] - 2021-07-16

-   [ENH] Drop `fill_value` parameter from `complete`. Users can use `fillna` instead. @samukweku
-   [BUG] Fix bug in `pivot_longer` with single level columns. @samukweku
-   [BUG] Disable exchange rates API until we can find another one to hit. @ericmjl
-   [ENH] Change `coalesce` to return columns; also use `bfill`, `ffill`,
    which is faster than `combine_first` @samukweku
-   [ENH] Use `eval` for string conditions in `update_where`. @samukweku
-   [ENH] Add clearer error messages for `pivot_longer`. h/t to @tdhock
    for the observation. Issue #836 @samukweku
-   [ENH] `select_columns` now uses variable arguments (\*args),
    to provide a simpler selection without the need for lists. - @samukweku
-   [ENH] `encode_categoricals` refactored to use generic functions
    via `functools.dispatch`. - @samukweku
-   [ENH] Updated convert_excel_date to throw meaningful error when values contain non-numeric. @nvamsikrishna05

## [v0.20.14] - 2021-03-25

-   [ENH] Add `dropna` parameter to groupby_agg. @samukweku
-   [ENH] `complete` adds a `by` parameter to expose explicit missing values per group, via groupby. @samukweku
-   [ENH] Fix check_column to support single inputs - fixes `label_encode`. @zbarry

## [v0.20.13] - 2021-02-25

-   [ENH] Performance improvements to `expand_grid`. @samukweku
-   [HOTFIX] Add `multipledispatch` to pip requirements. @ericmjl

## [v0.20.12] - 2021-02-25

-   [INF] Auto-release GitHub action maintenance. @loganthomas

## [v0.20.11] - 2021-02-24

-   [INF] Setup auto-release GitHub action. @loganthomas
-   [INF] Deploy `darglint` package for docstring linting. Issue #745. @loganthomas
-   [ENH] Added optional truncation to `clean_names` function. Issue #753. @richardqiu
-   [ENH] Added `timeseries.flag_jumps()` function. Issue #711. @loganthomas
-   [ENH] `pivot_longer` can handle multiple values in paired columns, and can reshape
    using a list/tuple of regular expressions in `names_pattern`. @samukweku
-   [ENH] Replaced default numeric conversion of dataframe with a `dtypes` parameter,
    allowing the user to control the data types. - @samukweku
-   [INF] Loosen dependency specifications. Switch to pip-tools for managing
    dependencies. Issue #760. @MinchinWeb
-   [DOC] added pipenv installation instructions @evan-anderson
-   [ENH] Add `pivot_wider` function, which is the inverse of the `pivot_longer`
    function. @samukweku
-   [INF] Add `openpyxl` to `environment-dev.yml`. @samukweku
-   [ENH] Reduce code by reusing existing functions for fill_direction. @samukweku
-   [ENH] Improvements to `pivot_longer` function, with improved speed and cleaner code.
    `dtypes` parameter dropped; user can change dtypes with pandas' `astype` method, or
    pyjanitor's `change_type` method. @samukweku
-   [ENH] Add kwargs to `encode_categorical` function, to create ordered categorical columns,
    or categorical columns with explicit categories. @samukweku
-   [ENH] Improvements to `complete` method. Use `pd.merge` to handle duplicates and
    null values. @samukweku
-   [ENH] Add `new_column_names` parameter to `process_text`, allowing a user to
    create a new column name after processing a text column. Also added a `merge_frame`
    parameter, allowing dataframe merging, if the result of the text processing is a
    dataframe.@samukweku
-   [ENH] Add `aggfunc` parameter to pivot_wider. @samukweku
-   [ENH] Modified the `check` function in utils to verify if a value is a callable. @samukweku
-   [ENH] Add a base `_select_column` function, using `functools.singledispatch`,
    to allow for flexible columns selection. @samukweku
-   [ENH] pivot_longer and pivot_wider now support janitor.select_columns syntax,
    allowing for more flexible and dynamic column selection. @samukweku

## [v0.20.10]

-   [ENH] Added function `sort_timestamps_monotonically` to timeseries functions @UGuntupalli
-   [ENH] Added the complete function for converting implicit missing values
    to explicit ones. @samukweku
-   [ENH] Further simplification of expand_grid. @samukweku
-   [BUGFIX] Added copy() method to original dataframe, to avoid mutation. Issue #729. @samukweku
-   [ENH] Added `also` method for running functions in chain with no return values.
-   [DOC] Added a `timeseries` module section to website docs. Issue #742. @loganthomas
-   [ENH] Added a `pivot_longer` function, a wrapper around `pd.melt` and similar to
    tidyr's `pivot_longer` function. Also added an example notebook. @samukweku
-   [ENH] Fixed code to returns error if `fill_value` is not a dictionary. @samukweku
-   [INF] Welcome bot (.github/config.yml) for new users added. Issue #739. @samukweku

## [v0.20.9]

-   [ENH] Updated groupby_agg function to account for null entries in the `by` argument. @samukweku
-   [ENH] Added function `groupby_topk` to janitor functions @mphirke

## [v0.20.8]

-   [ENH] Upgraded `update_where` function to use either the pandas query style,
    or boolean indexing via the `loc` method. Also updated `find_replace` function to use the `loc`
    method directly, instead of routing it through the `update_where` function. @samukweku
-   [INF] Update `pandas` minimum version to 1.0.0. @hectormz
-   [DOC] Updated the general functions API page to show all available functions. @samukweku
-   [DOC] Fix the few lacking type annotations of functions. @VPerrollaz
-   [DOC] Changed the signature from str to Optional[str] when initialized by None. @VPerrollaz
-   [DOC] Add the Optional type for all signatures of the API. @VPerrollaz
-   [TST] Updated test_expand_grid to account for int dtype difference in Windows OS @samukweku
-   [TST] Make importing `pandas` testing functions follow uniform pattern. @hectormz
-   [ENH] Added `process_text` wrapper function for all Pandas string methods. @samukweku
-   [TST] Only skip tests for non-installed libraries on local machine. @hectormz
-   [DOC] Fix minor issues in documentation. @hectormz
-   [ENH] Added `fill_direction` function for forward/backward fills on missing values
    for selected columns in a dataframe. @samukweku
-   [ENH] Simpler logic and less lines of code for expand_grid function @samukweku

## [v0.20.7]

-   [TST] Add a test for transform_column to check for nonmutation. @VPerrollaz
-   [ENH] Contributed `expand_grid` function by @samukweku

## [v0.20.6]

-   [DOC] Pep8 all examples. @VPerrollaz
-   [TST] Add docstrings to tests @hectormz
-   [INF] Add `debug-statements`, `requirements-txt-fixer`, and `interrogate` to `pre-commit`. @hectormz
-   [ENH] Upgraded transform_column to use df.assign underneath the hood,
    and also added option to transform column elementwise (via apply)
    or columnwise (thus operating on a series). @ericmjl

## [v0.20.5]

-   [INF] Replace `pycodestyle` with `flake8` in order to add `pandas-vet` linter @hectormz
-   [ENH] `select_columns()` now raises `NameError` if column label in
    `search_columns_labels` is missing from `DataFrame` columns. @smu095

## [v0.20.1]

-   [DOC] Added an example for groupby_agg in general functions @samukweku
-   [ENH] Contributed `sort_naturally()` function. @ericmjl

## [v0.20.0]

-   [DOC] Edited transform_column dest_column_name kwarg description to be clearer on defaults by @evan-anderson.
-   [ENH] Replace `apply()` in favor of `pandas` functions in several functions. @hectormz
-   [ENH] Add `ecdf()` Series function by @ericmjl.
-   [DOC] Update API policy for clarity. @ericmjl
-   [ENH] Enforce string conversion when cleaning names. @ericmjl
-   [ENH] Change `find_replace` implementation to use keyword arguments to specify columns to perform find and replace on. @ericmjl
-   [ENH] Add `jitter()` dataframe function by @rahosbach

## [v0.19.0]

-   [ENH] Add xarray support and clone_using / convert_datetime_to_number funcs by @zbarry.

## [v0.18.3]

-   [ENH] Series toset() functionality #570 @eyaltrabelsi
-   [ENH] Added option to coalesce function to not delete coalesced columns. @gddcunh
-   [ENH] Added functionality to deconcatenate tuple/list/collections in a column to deconcatenate_column @zbarry
-   [ENH] Fix error message when length of new_column_names is wrong @DollofCutty
-   [DOC] Fixed several examples of functional syntax in `functions.py`. @bdice
-   [DOC] Fix #noqa comments showing up in docs by @hectormz
-   [ENH] Add unionizing a group of dataframes' categoricals. @zbarry
-   [DOC] Fix contributions hyperlinks in `AUTHORS.rst` and contributions by @hectormz
-   [INF] Add `pre-commit` hooks to repository by @ericmjl
-   [DOC] Fix formatting code in `CONTRIBUTING.rst` by @hectormz
-   [DOC] Changed the typing for most "column_name(s)" to Hashable rather than enforcing strings, to more closely match Pandas API by @dendrondal
-   [INF] Edited pycodestyle and Black parameters to avoid venvs by @dendrondal

## [v0.18.2]

-   [INF] Make requirements.txt smaller @eyaltrabelsi
-   [ENH] Add a reset_index parameter to shuffle @eyaltrabelsi
-   [DOC] Added contribution page link to readme @eyaltrabelsi
-   [DOC] fix example for `update_where`, provide a bit more detail, and expand the bad_values example notebook to demonstrate its use by @anzelpwj.
-   [INF] Fix pytest marks by @ericmjl (issue #520)
-   [ENH] add example notebook with use of finance submodule methods by @rahosbach
-   [DOC] added a couple of admonitions for Windows users. h/t @anzelpwj for debugging
    help when a few tests failed for `win32` @Ram-N
-   [ENH] Pyjanitor for PySpark @zjpoh
-   [ENH] Add pyspark clean_names @zjpoh
-   [ENH] Convert asserts to raise exceptions by @hectormz
-   [ENH] Add decorator functions for missing and error handling @jiafengkevinchen
-   [DOC] Update README with functional `pandas` API example. @ericmjl
-   [INF] Move `get_features_targets()` to new `ml.py` module by @hectormz
-   [ENH] Add chirality to morgan fingerprints in janitor.chemistry submodule by @Clayton-Springer
-   [INF] `import_message` suggests python dist. appropriate installs by @hectormz
-   [ENH] Add count_cumulative_unique() method to janitor.functions submodule by @rahosbach
-   [ENH] Add `update_where()` method to `janitor.spark.functions` submodule by @zjpoh

## [v0.18.1]

-   [ENH] extend find_replace functionality to allow both exact match and
    regular-expression-based fuzzy match by @shandou
-   [ENH] add preserve_position kwarg to deconcatenate_column with tests
    by @shandou and @ericmjl
-   [DOC] add contributions that did not leave `git` traces by @ericmjl
-   [ENH] add inflation adjustment in finance submodule by @rahosbach
-   [DOC] clarified how new functions should be implemented by @shandou
-   [ENH] add optional removal of accents on functions.clean_names, enabled by
    default by @mralbu
-   [ENH] add camelCase conversion to snake_case on `clean_names` by @ericmjl,
    h/t @jtaylor for sharing original
-   [ENH] Added `null_flag` function which can mark null values in rows.
    Implemented by @anzelpwj
-   [ENH] add engineering submodule with unit conversion method by @rahosbach
-   [DOC] add PyPI project description
-   [ENH] add example notebook with use of finance submodule methods
    by @rahosbach

For changes that happened prior to v0.18.1,
please consult the closed PRs,
which can be found [here](https://github.com/pyjanitor-devs/pyjanitor/pulls?q=is%3Apr+is%3Aclosed).

We thank all contributors
who have helped make `pyjanitor`
the package that it is today.

[Unreleased]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.21.2...HEAD

[v0.21.2]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.21.1...v0.21.2

[v0.21.1]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.21.0...v0.21.1

[v0.21.0]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.20.14...v0.21.0

[v0.20.14]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.20.13...v0.20.14

[v0.20.13]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.20.12...v0.20.13

[v0.20.12]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.20.11...v0.20.12

[v0.20.11]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.20.10...v0.20.11

[v0.20.10]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.20.9...v0.20.10

[v0.20.9]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.20.8...v0.20.9

[v0.20.8]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.20.7...v0.20.8

[v0.20.7]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.20.5...v0.20.7

[v0.20.6]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.20.5...v0.20.7

[v0.20.5]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.20.1...v0.20.5

[v0.20.1]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.20.0...v0.20.1

[v0.20.0]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.19.0...v0.20.0

[v0.19.0]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.18.3...v0.19.0

[v0.18.3]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.18.2...v0.18.3

[v0.18.2]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.18.1...v0.18.2

[v0.18.1]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.18.0...v0.18.1
