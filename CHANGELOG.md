# Changelog

## [Unreleased]

-   [DOC] Rewrite the introductory example to chain `clean_names`, `remove_empty`, and `encode_categorical`, so copying it does not emit `FutureWarning`. - Issue #1442

## [v0.32.66] - 2026-10-30

-   [BUG] Leave a numeric first column unchanged in `adorn_pct_formatting`, `adorn_ns`, and `adorn_rounding`. Column position 0 is the row identifier, matching `adorn_totals` and `adorn_percentages`. Based on work by @dylanpulver (#1677) and @SumanGouda (#1696). - Issue #1676

## [v0.32.65] - 2026-10-30

-   [ENH] Change the `join_agg` default for `return_matched` from `True` to `False`; callers that need the boolean `matched` index level must now pass `return_matched=True`. - PR #1829 @samukweku

## [v0.32.64] - 2026-10-28

-   [ENH] Vectorise groupby `rle_id` (one pass over all groups instead of a Python function per group) and reuse the groupby's grouping, so `level=`, `dropna=False`, duplicate index labels and multi-key equally sized groups (e.g. a single row) give correct IDs. The ID column is now always `int64` (`float64` when grouped rows with missing keys are dropped), including for `string[pyarrow]` columns, which previously raised, and `Int64`/`string[python]` columns, which gave `Int64` IDs. - Issue #1827 @samukweku

## [v0.32.63] - 2026-10-28

-   [BUG] Keep new columns visible on the groupby returned by `assign` and `rle_id`; reused groupbys and chained calls previously read the original frame's cached columns and pandas_flavor accessors.

## [v0.32.62] - 2026-10-28

-   [ENH] Speed up homogeneous NumPy `.value` groups in `pivot_longer` without extra copies for fragmented frames. - Issue #1821 @senanurcetin

## [v0.32.61] - 2026-10-27

-   [ENH] Support equality-only joins in `conditional_join`; null keys never
    match, unlike `pd.merge`.

## [v0.32.60] - 2026-10-26

-   No user-facing changes; see the generated release notes.

## [v0.32.59] - 2026-10-26

-   [BUG] Treat mixed missing sentinels as one category in `drop_constant_columns`, preserving partially missing columns and removing all-missing columns. - Issue #1781 @senanurcetin

## [v0.32.58] - 2026-10-25

-   No user-facing changes; see the generated release notes.

## [v0.32.57] - 2026-10-25

-   [BUG] Keep the sorted row order of polars `complete(sort=True)` on LazyFrames with polars 2.0, where left joins no longer preserve the left frame's order by default. - Issue #1797 @raams95

## [v0.32.56] - 2026-10-24

-   No user-facing changes; see the generated release notes.

## [v0.32.55] - 2026-10-24

-   [DOC] Clean up historical changelog entries to improve documentation clarity. - PR #1807 @samukweku

## [v0.32.54] - 2026-10-23

-   [TST] Improve read_commandline test reliability by using pytest's tmp_path fixture instead of a shared temporary file. - PR #1767 @samukweku

## [v0.32.53] - 2026-10-23

-   [BUG] Fix datetime resolution assertions for pandas 3.1, including Excel and MATLAB date conversions. - @samukweku

## [v0.32.52] - 2026-10-23

-   [BUG] Update MultiIndex selector expectations for pandas 3.1 compatibility. - @samukweku
-   [INF] Update the black dependency from 26.5.1 to 26.10.1. - @dependabot[bot]

## [v0.32.51] - 2026-10-22

-   [ENH] Preserve pandas DataFrame.select for compatibility. - @samukweku
-   [BUG] Stop registering deprecated `select` as a pandas DataFrame, Series, or GroupBy method to avoid collision with pandas 3.1 `DataFrame.select`; method-style calls on Series and GroupBy objects now raise `AttributeError`, while on pandas >=3.1 old `df.select(columns=...)`, `df.select(index=...)`, and `df.select(axis=...)` calls can silently create columns named `columns`, `index`, and `axis`. Use `jn.select_columns`/`jn.select_rows` or `df.select_columns`/`df.select_rows` instead. - Issue #1798 @samukweku

## [v0.32.50] - 2026-10-22

-   [BUG] Keep backslashes in polars `make_clean_names`, as in the pandas `clean_names`. - Issue #1771 @Rodrigo-Palma
-   [INF] Update certifi and virtualenv dependencies. - @dependabot[bot]

## [v0.32.49] - 2026-10-22

-   [BUG] Preserve user warning filters on import, use visible `FutureWarning` for deprecated APIs, and report warnings at the caller's location. - Issue #1775 @BALOGUN-DAVID

## [v0.32.48] - 2026-10-20

-   [BUG] Raise clear, API-specific errors when empty selectors, including zero-column frames, are passed to `move`, `pivot_longer`, `pivot_wider`, or tuple-based `summarise`. - Issue #1784

## [v0.32.47] - 2026-10-20

-   [BUG] Use empty lists as defaults for conditional-join projections. - @samukweku

## [v0.32.46] - 2026-10-19

-   [INF] Upgrade the janitor-rs dependency to 0.6.5. - @samukweku

## [v0.32.45] - 2026-10-19

-   [BUG] Empty list-like selectors in `select`, `select_columns`, and `select_rows` now select nothing; `DropLabel([])` drops nothing, and `invert=True` selects the complement. Empty dictionaries now select nothing on MultiIndex axes as well. - Issue #1782 @samukweku

## [v0.32.44] - 2026-10-19

-   [ENH] Add `output_columns` to `conditional_join` as the preferred side-aware output projection while retaining `df_columns` and `right_columns` for compatibility. - Issue #1751
-   [BUG] Preserve duplicate columns selected by `conditional_join` output projections.
-   [BUG] Preserve unmatched rows when `conditional_join` drops one output side in non-inner joins.
-   [BUG] Avoid deprecation warnings from `get_join_indices` and `join_agg` internal projections.
-   [INF] Update prefix-dev/setup-pixi from 0.11.0 to 0.11.2. - @dependabot[bot]

## [v0.32.43] - 2026-10-19

-   [DOC] Document `slice(None)` selection for all labels on any index, with end-to-end test coverage. - PR #1778 @samukweku

## [v0.32.42] - 2026-10-18

-   [DOC] Add missing version annotations throughout the documentation. - Issue #1188 @BALOGUN-DAVID

## [v0.32.41] - 2026-10-16

-   [BUG] Include the README in package metadata so PyPI renders the project description. - @samukweku

## [v0.32.40] - 2026-10-16

-   [BUG] Fix Polars make_clean_names handling of @ in column names. - Issue #1758 @sumangouda

## [v0.32.39] - 2026-10-15

-   [PERF] Speed up equality-led `conditional_join`, `get_join_indices`, and `join_agg` calls with range predicates. With multiple range predicates, the first range predicate in the order supplied sets the search layout and is paired with the first later range predicate of the opposing direction; put the most selective range predicate first. - Issue #1752, PR #1765 @samukweku
-   [BUG] Fix equality joins with mixed-type object keys and duplicate right keys raising `TypeError`. - PR #1765 @samukweku
-   [ENH] `join_agg` with an equality predicate now omits output rows whose equality key has no match, and returns an empty result when no key matches, consistent with `get_join_indices`. Rows whose key matches but fail the other predicates are still returned with `matched=False`. - Issue #1752, PR #1765 @samukweku

## [v0.32.38] - 2026-10-15

-   [DOC] Add usage examples to the read_csvs docstring. - Issue #1755 @iMalcolmS

## [v0.32.37] - 2026-10-13

-   [INF] Skip preview deployments for pull requests from forks and update prefix-dev/setup-pixi to 0.11.0. - @dependabot[bot]

## [v0.32.36] - 2026-10-07

-   [BUG] Fix millisecond timestamp detection in convert_unix_date for pandas 3. - Issue #1756 @raashish1601

## [v0.32.35] - 2026-10-05

-   [DOC] Clarify index normalization in join aggregation helpers and related documentation. - @tunglambk

## [v0.32.34] - 2026-10-04

-   [ENH] Add anti joins to conditional_join. - @samukweku

## [v0.32.33] - 2026-10-04

-   [ENH] Add cumulative envelopes and documentation for dual-range joins. - @samukweku
-   [BUG] Fix cumulative datetime-bound normalization and all-unmatched join_agg output. - @samukweku

## [v0.32.32] - 2026-10-04

-   [DOC] Improve get_join_indices and join_agg reference documentation. - @samukweku

## [v0.32.31] - 2026-10-04

-   [ENH] Route conditional joins and join aggregations through Rust kernels and update janitor-rs to 0.6.2. - @samukweku
-   [BUG] Preserve object-column dtypes in conditional-join results. - @samukweku
-   [DEP] Deprecate the Numba-based conditional-join dispatch. - @samukweku

## [v0.32.30] - 2026-10-04

-   [DOC] Correct internal docstring parameter names and descriptions. - PR #1742 @VenishPaneliya

## [v0.32.29] - 2026-10-02

-   [BUG] Collapse every run of underscores in Polars make_clean_names. - Issue #1740 @Rodrigo-Palma

## [v0.32.28] - 2026-10-01

-   [BUG] Fix row_to_names for slices without a start or stop. - Issue #1738 @Rodrigo-Palma
-   [DEP] Deprecate df_columns and right_columns in conditional_join. - Issue #1712 @sumangouda

## [v0.32.27] - 2026-09-03

-   [PERF] Avoid materializing not-equal pairs for conditional_join with keep=first or keep=last. - Issue #1651, PR #1681 @tunglambk

## [v0.32.26] - 2026-09-02

-   [BUG] Scope auto-release validation to generated files and improve release-note validation and formatting retries. - @samukweku
-   [DOC] Reorganize and clarify conditional-join and join-aggregation documentation. - @samukweku

## [v0.32.25] - 2026-08-31

-   [PERF] Reduce peak memory in `_build_indexer_reorder_contents` for wide frames (reps >= 8) using single NumPy allocation; tall frames with few repetitions retain the original reshape path. - Issue #1655 @Anupam2400
-   [ENH] `conditional_join` now picks the most selective `<`/`<=`/`>`/`>=` predicate as its binary-search anchor (instead of the first one supplied) when `keep` is `'first'` or `'last'`, fixing pathological slowdowns from unfavorable predicate ordering; the choice is estimated from a fixed-size sample so the selection cost no longer scales with input size; `keep='all'` output is unaffected. - Issue #1641 @samukweku
-   [BUG] Fix `conditional_join` crash for `keep='last'` joins with a single `<`/`<=` window and multiple non-equi conditions - a missing `counts` argument to the Rust `index_starts_only_keep_last` call. - Issue #1641 @samukweku
-   [ENH] Avoid copying column data during `conditional_join` input validation. - Issue #1645, PR #1642 @samukweku
-   [ENH] Limit `conditional_join` matching work to columns referenced by join conditions. - Issue #1646, PR #1647 @samukweku
-   [ENH] Speed up `conditional_join` with an unsorted right join key and `keep="first"` or `keep="last"`. - Issue #1125, PR #1644 @samukweku
-   [ENH] Replace `axis=1` `.apply()` with vectorized operations in `compare_df_cols` for 4-7x performance improvement. - Issue #1630 @Anupam2400
-   [DOC] Add documentation explaining when cumulative-event aggregation is preferable to range join aggregations. - Issue #1702 @sumangouda
-   [INF] Automate contributor recognition: weekly workflow discovers new commit authors and regenerates the all-contributors table; backfilled 30 missing contributors. - Issue #1623

## [v0.32.24] - 2026-08-17

-   [ENH] Replace `.apply()` with `.map()` in `find_replace` for ~6x performance improvement. - Issue #1422 @Anupam2400
-   [ENH] Add `drop_first` parameter to `expand_column`. - Issue #368 @manav252
-   [ENH] Add `strip_whitespace` to `clean_names` function. - Issue #1385
-   [ENH] Improve `polars.complete` and `polars.expand`, avoiding the potentially expensive schema computation on lazy frames. @samukweku
-   [TST] Fix 'HealthCheck' failure in 'test_ecdf_string' by adding missing '@settings' decorator. @mjsr84
-   [INF] Update CI and development dependencies.

## [v0.32.23] - 2026-04-07

-   [INF] Update prefix-dev/setup-pixi and pygments dependencies.

## [v0.32.22] - 2026-03-30

-   [INF] Update codecov/codecov-action to version 6.

## [v0.32.21] - 2026-03-28

-   [ENH] Improve pandas pivot_longer behavior. - @samukweku

## [v0.32.20] - 2026-02-18

-   [ENH] Add by methods to groupby objects. - @samukweku

## [v0.32.19] - 2026-02-11

-   [ENH] Add `pd.col` support - Issue #1590 @samukweku

## [v0.32.18] - 2026-02-08

-   [ENH] Add `assign` method to groupby object. - Issue #1467 @samukweku

## [v0.32.17] - 2026-02-08

-   [DEP] Deprecate mutate and improve its warning message. - @samukweku

## [v0.32.16] - 2026-02-07

-   [ENH] Add `scale_mad` for robust median/MAD scaling. - PR #1530 @ShachiMistry

## [v0.32.15] - 2026-02-07

-   [ENH] Add DataFrame column comparison utilities and improve compare_df_cols performance. - @samukweku

## [v0.32.14] - 2026-02-07

-   [INF] Improve release versioning and clean up obsolete release marker files.

## [v0.32.13] - 2026-02-07

-   [INF] Correct version entries during the automated release transition.

## [v0.32.12] - 2026-02-07

-   [DEP] `summarise` deprecated. - Issue #1045 @samukweku

## [v0.32.11] - 2026-02-07

-   [ENH] Add R janitor utility functions, tabyl statistical tests, and mixed date/time conversion utilities.
-   [BUG] Improve pandas 3.0 compatibility across date parsing, regex handling, and copy-on-write behavior.

## [v0.32.10] - 2026-02-07

-   [BUG] Fix the auto-release versioning workflow and prepare Python 3.8+ compatibility.
-   [ENH] Improve equi joins and document the `rle_id` function.
-   [ENH] Add `include_join_positions` parameter to `conditional_join`; added limited support for join aggregations via the `join_agg` function. - Issue #1497 @samukweku

## [v0.32.9] - 2026-01-22

-   [ENH] Add the rle_id function for run-length encoding. - Issue #1435 @emmanuel-ferdman
-   [INF] Update CI and development dependencies.

## [v0.32.8] - 2026-01-02

-   [ENH] Add tabyl frequency tables. - Issue #1556 @samukweku

## [v0.32.7] - 2025-12-30

-   [ENH] Implement adorn functions for tabyl formatting.
-   [BUG] Fix adorn_ns index alignment.

## [v0.32.6] - 2025-12-25

-   [ENH] Refactor conditional_join internals. - @samukweku

## [v0.32.5] - 2025-12-24

-   [ENH] Add currency symbol support to accounting cleaning style.
-   [TST] Add coverage for additional currency symbols.

## [v0.32.4] - 2025-12-23

-   [TST] Isolate read_csvs tests with tmp_path for parallel execution.
-   [INF] Enable automated patch releases on merges to dev and remove obsolete automation.

## [v0.32.3] - 2025-12-11

-   [ENH] Undeprecate `select_columns` and `select_rows` - Issue #1514 @samukweku
-   [ENH] `clean_names()` now replaces `@` with `_` when between word characters - Issue #1257 @Kevin-Smith77

## [v0.32.2] - 2025-12-06

-   [ENH] switch to register_dataframe_groupby_method - Issue #1534 @bryanpmayfield

## [v0.32.1] - 2025-11-06

-   [ENH] update `pypa/gh-action-pypi-publish` from master to release/v1 - Issue #1524 @samukweku

## [v0.32.0] - 2025-11-06

-   [ENH] `return_ragged_arrays` deprecated; get_join_indices function now returns a dictionary - Issue #520 @samukweku
-   [ENH] Reverse deprecation of `pivot_wider()` -- Issue #1464
-   [ENH] Add accessor and method for pandas DataFrameGroupBy objects. - Issue #587 @samukweku
-   [ENH] Call mutate/summarise directly on groupby objects instead. Also add `ungroup` method to expose underlying dataframe of a grouped object. - Issue #1511 @samukweku
-   [BUG] Fix incorrect output of `pivot_wider`, where `index` and `names_from` is provided, and values_from is None. - Issue #1509 @samukweku

## [v0.31.0] - 2025-03-07

-   [ENH] Added support for pd.Series.select - Issue #1394 @samukweku
-   [ENH] Added support for janitor.mutate - Issue #1226 @samukweku
-   [ENH] Added support for janitor.summarise - Issue #1225 @samukweku
-   [ENH] Added support for janitor.alias - Issue #1449 @samukweku

## [v0.30.0] - 2024-12-04

## [v0.29.2] - 2024-09-28

## [v0.29.1] - 2024-09-23

## [v0.29.0] - 2024-09-15

-   [DOC] Un-deprecate `join_apply` as no alternative currently exists - Issue #1399 @lbeltrame

## [v0.28.1] - 2024-08-09

## [v0.28.0] - 2024-08-03

-   [ENH] Added a `cartesian_product` function, as well as an `expand` method for pandas. - Issue #1293 @samukweku
-   [ENH] Improve `pivot_longer` when `sort_by_appearance` is True. Added `pivot_longer_spec` for more control on how the dataframe should be unpivoted. -@samukweku #1361
-   [ENH] Added `convert_excel_date` and `convert_matlab_date` methods for polars - Issue #1352
-   [ENH] Added a `complete` method for polars. - Issue #1352 @samukweku
-   [ENH] Added a `pivot_longer` method, and a `pivot_longer_spec` function for polars - Issue #1352 @samukweku
-   [ENH] Added a `row_to_names` method for polars. Issue #1352 @samukweku
-   [ENH] `read_commandline` function now supports polars - Issue #1352 @samukweku
-   [ENH] `xlsx_cells` function now supports polars - Issue #1352 @samukweku
-   [ENH] `xlsx_table` function now supports polars - Issue #1352 @samukweku
-   [ENH] Added a `clean_names` method for polars - it can be used to clean the column names, or clean column values . Issue #1343 @samukweku
-   [ENH] Improved performance for non-equi joins when using numba - @samukweku PR #1341
-   [ENH] pandas Index,Series, DataFrame now supported in the `complete` method. - PR #1369 @samukweku
-   [ENH] Improve performance for `first/last` in conditional_join, when the join columns in the right dataframe are sorted. - PR #1382 @samukweku

## [v0.27.0] - 2024-03-21

-   [BUG] Fix logic for groupby in complete. Index support deprecated. Fix deprecation warning for fillna in `complete` PR #1289 @samukweku
-   [ENH] `select` function now supports variable arguments - PR #1288 @samukweku
-   [ENH] `conditional_join` now supports timedelta dtype. - PR #1297 @samukweku
-   [ENH] `get_join_indices` function added - returns only join indices between two dataframes. Issue #1310 @samukweku
-   [ENH] `explode_index` function added. - Issue #1283
-   [ENH] `conditional_join` now supports timedelta dtype. - PR #1297
-   [ENH] `change_index_dtype` added. - @samukweku Issue #1314
-   [ENH] Add `glue` and `axis` parameters to `collapse_levels`. - Issue #211 @samukweku
-   [ENH] `row_to_names` now supports multiple rows conversion to columns. - @samukweku Issue #1333
-   [ENH] Fix warnings from Pandas. `truncate_datetime` now uses a vectorized option. -@samukweku #1337

## [v0.26.0] - 2023-09-18

-   [ENH] `clean_names` can now be applied to column values. Issue #995 @samukweku
-   [BUG] Fix ImportError - Issue #1285 @samukweku

## [v0.25.0] - 2023-07-27

-   [INF] Replace `pytest.ini` file with `pyproject.toml` file. PR #1204 @Zeroto521
-   [INF] Extract docstrings tests from all tests. PR #1205 @Zeroto521
-   [BUG] Address the `TypeError` when importing v0.24.0 (issue #1201 @xujiboy and @joranbeasley)
-   [INF] Fixed issue with missing PyPI README. PR #1216 @thatlittleboy
-   [INF] Update some `mkdocs` compatibility code. PR #1231 @thatlittleboy
-   [INF] Migrated docstring style from Sphinx to Google for better compatibility with `mkdocstrings`. PR #1235 @thatlittleboy
-   [INF] Prevent selection of chevrons (`>>>`) and outputs in Example code blocks. PR #1237 @thatlittleboy
-   [DEPR] Add deprecation warnings for `process_text`, `rename_column`, `rename_columns`, `filter_on`, `remove_columns`, `fill_direction`. Issue #1045 @samukweku
-   [ENH] `pivot_longer` now supports named groups where `names_pattern` is a regular expression. A dictionary can now be passed to `names_pattern`, and is internally evaluated as a list/tuple of regular expressions. Issue #1209 @samukweku
-   [ENH] Improve selection in `conditional_join`. Issue #1223 @samukweku
-   [ENH] Add `col` class for selecting columns within an expression. Currently limited to use within `conditional_join`. PR #1260 @samukweku.
-   [ENH] Performance improvement for range joins in `conditional_join`, when `use_numba = False`. Performance improvement for equi-join and a range join, when `use_numba = True`, for many to many join with wide ranges. PR #1256, #1267 @samukweku
-   [DEPR] Add deprecation warning for `pivot_wider`. Issue #1045 @samukweku
-   [BUG] Fix string column selection on a MultiIndex. Issue #1265. @samukweku

## [v0.24.0] - 2022-11-12

-   [ENH] Add lazy imports to speed up the time taken to load pyjanitor (part 2)
-   [DOC] Updated developer guide docs.
-   [ENH] Allow column selection/renaming within conditional_join. Issue #1102. Also allow first or last match. Issue #1020 @samukweku.
-   [ENH] New decorator `deprecated_kwargs` for breaking API. #1103 @Zeroto521
-   [ENH] Extend select_columns to support non-string columns. Issue #1105 @samukweku
-   [ENH] Performance improvement for groupby_topk. Issue #1093 @samukweku
-   [ENH] `min_max_scale` drop `old_min` and `old_max` to fit sklearn's method API. Issue #1068 @Zeroto521
-   [ENH] Add `jointly` option for `min_max_scale` support to transform each column values or entire values. Default transform each column, similar behavior to `sklearn.preprocessing.MinMaxScaler`. (Issue #1067, PR #1112, PR #1123) @Zeroto521
-   [INF] Require pyspark minimal version is v3.2.0 to cut duplicates codes. Issue #1110 @Zeroto521
-   [ENH] Add support for extension arrays in `expand_grid`. Issue #1121 @samukweku
-   [ENH] Add `names_expand` and `index_expand` parameters to `pivot_wider` for exposing missing categoricals. Issue #1108 @samukweku
-   [ENH] Add fix for slicing error when selecting columns in `pivot_wider`. Issue #1134 @samukweku
-   [ENH] `dropna` parameter added to `pivot_longer`. Issue #1132 @samukweku
-   [INF] Update `mkdocstrings` version and to fit its new coming features. PR #1138 @Zeroto521
-   [BUG] Force `math.softmax` returning `Series`. PR #1139 @Zeroto521
-   [INF] Set independent environment for building documentation. PR #1141 @Zeroto521
-   [DOC] Add local documentation preview via github action artifact. PR #1149 @Zeroto521
-   [ENH] Enable `encode_categorical` handle 2 (or more ) dimensions array. PR #1153 @Zeroto521
-   [TST] Fix testcases failing on Window. Issue #1160 @Zeroto521, and @samukweku
-   [INF] Cancel old workflow runs via Github Action `concurrency`. PR #1161 @Zeroto521
-   [ENH] Faster computation for non-equi join, with a numba engine. Speed improvement for left/right joins when `sort_by_appearance` is False. Issue #1102 @samukweku
-   [BUG] Avoid `change_type` mutating original `DataFrame`. PR #1162 @Zeroto521
-   [ENH] The parameter `column_name` of `change_type` totally supports inputing multi-column now. #1163 @Zeroto521
-   [ENH] Fix error when `sort_by_appearance=True` is combined with `dropna=True`. Issue #1168 @samukweku
-   [ENH] Add explicit default parameter to `case_when` function. Issue #1159 @samukweku
-   [BUG] pandas 1.5.x `_MergeOperation` doesn't have `copy` keyword anymore. Issue #1174 @Zeroto521
-   [ENH] `select_rows` function added for flexible row selection. Generic `select` function added as well. Add support for MultiIndex selection via dictionary. Issue #1124 @samukweku
-   [TST] Compat with macos and window, to fix `FailedHealthCheck` Issue #1181 @Zeroto521
-   [INF] Merge two docs CIs (`docs-preview.yml` and `docs.yml`) to one. And add `documentation` pytest mark. PR #1183 @Zeroto521
-   [INF] Merge `codecov.yml` (only works for the dev branch pushing event) into `tests.yml` (only works for PR event). PR #1185 @Zeroto521
-   [TST] Fix failure for test/timeseries/test_fill_missing_timestamp. Issue #1184 @samukweku
-   [BUG] Import `DataDescription` to fix: `AttributeError: 'DataFrame' object has no attribute 'data_description'`. PR #1191 @Zeroto521

## [v0.23.1] - 2022-05-03

-   [DOC] Updated `fill.py` and `update_where.py` documentation with working examples.
-   [ENH] Deprecate `num_bins` from `bin_numeric` in favour of `bins`, and allow generic `**kwargs` to be passed into `pd.cut`. Issue #969. @thatlittleboy
-   [ENH] Fix `concatenate_columns` not working on category inputs @zbarry
-   [INF] Simplify CI system @ericmjl
-   [ENH] Added "read_commandline" function to janitor.io @BaritoneBeard
-   [BUG] Fix bug with the complement parameter of `filter_on`. Issue #988. @thatlittleboy
-   [ENH] Add `xlsx_table`, for reading tables from an Excel sheet. @samukweku
-   [ENH] minor improvements for conditional_join; equality only joins are no longer supported; there has to be at least one non-equi join present. @samukweku
-   [BUG] `sort_column_value_order` no longer mutates original dataframe.
-   [BUG] Extend `fill_empty`'s `column_names` type range. Issue #998. @Zeroto521
-   [BUG] Removed/updated error-inducing default arguments in `row_to_names` (#1004) and `round_to_fraction` (#1005). @thatlittleboy
-   [ENH] `patterns` deprecated in favour of importing `re.compile`. #1007 @samukweku
-   [ENH] Changes to kwargs in `encode_categorical`, where the values can either be a string or a 1D array. #1021 @samukweku
-   [ENH] Add `fill_value` and `explicit` parameters to the `complete` function. #1019 @samukweku
-   [ENH] Performance improvement for `expand_grid`. @samukweku
-   [BUG] Make `factorize_columns` (PR #1028) and `truncate_datetime_dataframe` (PR #1040) functions non-mutating. @thatlittleboy
-   [BUG] Fix SettingWithCopyWarning and other minor bugs when using `truncate_datetime_dataframe`, along with further performance improvements (PR #1040). @thatlittleboy
-   [ENH] Performance improvement for `conditional_join`. @samukweku
-   [ENH] Multiple `.value` is now supported in `pivot_longer`. Multiple values_to is also supported, when names_pattern is a list or tuple. `names_transform` parameter added, for efficient dtype transformation of unpivoted columns. #1034, #1048, #1051 @samukweku
-   [ENH] Add `xlsx_cells` for reading a spreadsheet as a table of individual cells. #929 @samukweku.
-   [ENH] Let `filter_string` suit parameters of `Series.str.contains` Issue #1003 and #1047. @Zeroto521
-   [ENH] `names_glue` in `pivot_wider` now takes a string form, using str.format_map under the hood. `levels_order` is also deprecated. @samukweku
-   [BUG] Fixed bug in `transform_columns` which ignored the `column_names` specification when `new_column_names` dictionary was provided as an argument, issue #1063. @thatlittleboy
-   [BUG] `count_cumulative_unique` no longer modifies the column being counted in the output when `case_sensitive` argument is set to False, issue #1065. @thatlittleboy
-   [BUG] Fix for gcc missing error in dev container
-   [DOC] Added a step in the dev guide to install `Remote Container` in VS Code. @ashenafiyb
-   [DOC] Convert `expand_column` and `find_replace` code examples to doctests, issue #972. @gahjelle
-   [DOC] Convert `expand_column` code examples to doctests, issue #972. @gahjelle
-   [DOC] Convert `get_dupes` code examples to doctests, issue #972. @ethompsy
-   [DOC] Convert `engineering` code examples to doctests, issue #972 @ashenafiyb
-   [DOC] Convert `groupby_topk` code examples to doctests, issue #972. @ethompsy
-   [DOC] Add doctests to `math`, issue #972. @gahjelle
-   [DOC] Add doctests to `math` and `ml`, issue #972. @gahjelle
-   [DOC] Add doctests to `math`, `ml`, and `xarray`, issue #972. @gahjelle

## [v0.22.0] - 2021-11-21

-   [BUG] Fix conditional join issue for multiple conditions, where pd.eval fails to evaluate if numexpr is installed. #898 @samukweku
-   [ENH] Added `case_when` to handle multiple conditionals and replacement values. Issue #736. @robertmitchellv
-   [ENH] Deprecate `new_column_names` and `merge_frame` from `process_text`. Only existing columns are supported. @samukweku
-   [ENH] `complete` uses `pd.merge` internally, providing a simpler logic, with some speed improvements in certain cases over `pd.reindex`. @samukweku
-   [ENH] `expand_grid` returns a MultiIndex DataFrame, allowing the user to decide how to manipulate the columns. @samukweku
-   [INF] Simplify a bit linting, use pre-commit as the CI linting checker. @Zeroto521
-   [ENH] Fix bug in `pivot_longer` for wrong output when `names_pattern` is a sequence with a single value. Issue #885 @samukweku
-   [ENH] Deprecate `aggfunc` from `pivot_wider`; aggregation can be chained with pandas' `groupby`.
-   [ENH] `As_Categorical` deprecated from `encode_categorical`; a tuple of `(categories, order)` suffices for \*\*kwargs. @samukweku
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
-   [ENH] Set `expand_column`'s `sep` default is `"|"`, same to `pandas.Series.str.get_dummies`. Issue #876. @Zeroto521
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

[Unreleased]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.66...HEAD

[v0.32.66]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.65...v0.32.66

[v0.32.65]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.64...v0.32.65

[v0.32.64]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.63...v0.32.64

[v0.32.63]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.62...v0.32.63

[v0.32.62]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.61...v0.32.62

[v0.32.61]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.60...v0.32.61

[v0.32.60]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.59...v0.32.60

[v0.32.59]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.58...v0.32.59

[v0.32.58]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.57...v0.32.58

[v0.32.57]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.56...v0.32.57

[v0.32.56]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.55...v0.32.56

[v0.32.55]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.54...v0.32.55

[v0.32.54]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.53...v0.32.54

[v0.32.53]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.52...v0.32.53

[v0.32.52]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.51...v0.32.52

[v0.32.51]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.50...v0.32.51

[v0.32.50]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.49...v0.32.50

[v0.32.49]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.48...v0.32.49

[v0.32.48]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.47...v0.32.48

[v0.32.47]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.46...v0.32.47

[v0.32.46]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.45...v0.32.46

[v0.32.45]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.44...v0.32.45

[v0.32.44]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.43...v0.32.44

[v0.32.43]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.42...v0.32.43

[v0.32.42]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.41...v0.32.42

[v0.32.41]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.40...v0.32.41

[v0.32.40]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.39...v0.32.40

[v0.32.39]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.38...v0.32.39

[v0.32.38]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.37...v0.32.38

[v0.32.37]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.36...v0.32.37

[v0.32.36]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.35...v0.32.36

[v0.32.35]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.34...v0.32.35

[v0.32.34]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.33...v0.32.34

[v0.32.33]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.32...v0.32.33

[v0.32.32]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.31...v0.32.32

[v0.32.31]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.30...v0.32.31

[v0.32.30]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.29...v0.32.30

[v0.32.29]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.28...v0.32.29

[v0.32.28]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.27...v0.32.28

[v0.32.27]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.26...v0.32.27

[v0.32.26]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.25...v0.32.26

[v0.32.25]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.24...v0.32.25

[v0.32.24]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.23...v0.32.24

[v0.32.23]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.22...v0.32.23

[v0.32.22]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.21...v0.32.22

[v0.32.21]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.20...v0.32.21

[v0.32.20]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.19...v0.32.20

[v0.32.19]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.18...v0.32.19

[v0.32.18]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.17...v0.32.18

[v0.32.17]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.16...v0.32.17

[v0.32.16]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.15...v0.32.16

[v0.32.15]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.14...v0.32.15

[v0.32.14]: https://github.com/pyjanitor-devs/pyjanitor/compare/vpyjanitor0.32.12%3D...v0.32.14

[v0.32.13]: https://github.com/pyjanitor-devs/pyjanitor/compare/vpyjanitor0.32.11%3D...vpyjanitor0.32.12%3D

[v0.32.12]: https://github.com/pyjanitor-devs/pyjanitor/compare/vpyjanitor0.32.10%3D...vpyjanitor0.32.11%3D

[v0.32.11]: https://github.com/pyjanitor-devs/pyjanitor/compare/vpyjanitor0.32.9%3D...vpyjanitor0.32.10%3D

[v0.32.10]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.9...vpyjanitor0.32.9%3D

[v0.32.9]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.8...v0.32.9

[v0.32.8]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.7...v0.32.8

[v0.32.7]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.6...v0.32.7

[v0.32.6]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.5...v0.32.6

[v0.32.5]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.4...v0.32.5

[v0.32.4]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.3...v0.32.4

[v0.32.3]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.2...v0.32.3

[v0.32.2]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.1...v0.32.2

[v0.32.1]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.32.0...v0.32.1

[v0.32.0]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.31.0...v0.32.0

[v0.31.0]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.30.0...v0.31.0

[v0.30.0]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.29.2...v0.30.0

[v0.29.2]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.29.1...v0.29.2

[v0.29.1]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.29.0...v0.29.1

[v0.29.0]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.28.1...v0.29.0

[v0.28.1]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.28.0...v0.28.1

[v0.28.0]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.27.0...v0.28.0

[v0.27.0]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.26.0...v0.27.0

[v0.26.0]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.25.0...v0.26.0

[v0.25.0]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.24.0...v0.25.0

[v0.24.0]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.23.1...v0.24.0

[v0.23.1]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.22.0...v0.23.1

[v0.22.0]: https://github.com/pyjanitor-devs/pyjanitor/compare/v0.21.2...v0.22.0

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
