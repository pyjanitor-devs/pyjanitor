# AGENTS.md

This file provides guidance to LLM agents working with code in this repository.
It serves as the agent's "constitution" for pyjanitor development.

---

## Agent Constitution

### Self-Improvement Protocol

**CRITICAL RULE**: This file is a living document. Agents MUST update it when:

1. **User Corrections**: If the user corrects you on anything, immediately record
   the correction in this file (AGENTS.md) in an appropriate section, then
   continue with what you were doing, applying the correction.

2. **Discovered Patterns**: If you discover a pattern, convention, or best
   practice not documented here while working on the codebase, add it to the
   appropriate section.

3. **Command Updates**: If you find that a command has changed, been deprecated,
   or a better alternative exists, update the Commands section.

4. **Anti-Patterns**: If you make a mistake and learn from it, document the
   anti-pattern in the appropriate section to prevent future occurrences.

**How to Update**: Add new learnings to the `## Learned Patterns` section at
the bottom of this file. The maintainer will periodically review and integrate
these into the main sections.

### Core Principles

- **Read Before Edit**: Always read and understand relevant files before
  proposing changes.
- **Minimal Changes**: Make the smallest change necessary to accomplish the
  task.
- **Test-Driven**: Always run tests after making code changes.
- **Document**: Keep docstrings up-to-date using Google-style format.
- **Lint Markdown**: Always run `markdownlint` on markdown files after editing.

### Cross-repository compatibility gate (janitor-rs #225)

Every janitor-rs change needs an explicit pyjanitor impact assessment before
merge:

- Rust-only internals, benchmarks, or refactors: document why no pyjanitor
  change is needed and run the Rust test suite.
- Python-visible behavior, signatures, dtypes, errors, or performance
  contracts: use coordinated PRs or add explicit compatibility tests in both
  repositories, linking janitor-rs #225.
- Release-affecting changes: validate pyjanitor against the published
  janitor-rs wheel before the pyjanitor change merges.

The PR description must state the classification, the pyjanitor impact, and
the validation evidence. A paired pyjanitor PR is required when the
Python-visible contract changes; Rust-only work does not require a no-op
pyjanitor PR.

---

## Project Overview

pyjanitor is a Python implementation of the R package janitor. It provides a
clean, chainable API for extending pandas with powerful and readable
data-cleaning functions.

**Key Design Philosophy**:

- Methods are chainable (fluent interface)
- Methods are registered via `pandas_flavor` as DataFrame methods
- All methods return a DataFrame (immutability pattern - no mutation)
- Functions follow a consistent signature pattern: `df` first, then parameters

---

## Development Environment

### Package Manager

**This project uses `pixi` for dependency management and environment setup.**

**⚠️ CRITICAL FOR LLM AGENTS**: All Python commands MUST be run within a pixi
context. Never run Python commands directly without the `pixi run` prefix.

```bash
# ✅ CORRECT
pixi run python -c "import janitor; print(janitor.__version__)"
pixi run pytest tests/functions/test_clean_names.py -v
pixi run python scripts/my_script.py

# ❌ INCORRECT - will fail or use wrong environment
python -c "import janitor; ..."
pytest tests/
python scripts/my_script.py
```

### Environment Setup

```bash
# Enter development shell
pixi shell

# Or prefix individual commands
pixi run <command>
```

### Available Pixi Environments

| Environment | Purpose | Features |
|-------------|---------|----------|
| `default` | Standard development | tests, setup |
| `docs` | Documentation building | mkdocs, mkdocstrings |
| `tests` | Running test suite | pytest, hypothesis |
| `biology` | Biology module development | biopython |
| `chemistry` | Chemistry module development | rdkit, tqdm |
| `engineering` | Engineering module development | unyt |
| `spark` | PySpark development | pyspark |
| `py311`/`py312`/`py313` | Python version testing | Specific Python versions |

To run commands in a specific environment:

```bash
pixi run -e <environment> <command>
```

---

## Commands Reference

### Essential Commands

| Task | Command |
|------|---------|
| Run all tests | `pixi run test` |
| Run specific test | `pixi run pytest tests/functions/test_clean_names.py` |
| Run tests matching pattern | `pixi run pytest -k "test_clean_names" -v` |
| Run tests with coverage | `pixi run pytest --cov=janitor` |
| Build documentation | `pixi run docs` |
| Serve docs locally | `pixi run serve-docs` |
| Run linting | `pixi run lint` |
| Format code | `pixi run format` |
| Run all checks | `pixi run check` |
| Install pre-commit hooks | `pixi run start` |

### Testing Commands

```bash
# Run full test suite with parallel execution
pixi run pytest -v -n auto --color=yes

# Run tests for a specific module
pixi run pytest tests/functions/ -v
pixi run pytest tests/polars/ -v
pixi run pytest tests/chemistry/ -v

# Run doctests in source code
pixi run pytest --doctest-modules janitor/

# Run tests with specific marker
pixi run pytest -m "functions" -v
pixi run pytest -m "biology" -v
pixi run pytest -m "chemistry" -v

# Run a single test function
pixi run pytest tests/functions/test_clean_names.py::test_clean_names_method_chain
```

### Documentation Commands

```bash
# Build docs
pixi run docs

# Serve docs with live reload
pixi run serve-docs

# Build docs in specific environment
pixi run -e docs build-docs
```

### Code Quality Commands

```bash
# Run all pre-commit hooks
pixi run lint

# Format code with ruff
pixi run format

# Check import sorting
pixi run isort

# Run full style check
pixi run style
```

### Markdown Linting

**Always run `markdownlint` on markdown files after editing them.**

```bash
# Lint a markdown file
markdownlint AGENTS.md

# Lint all markdown files
markdownlint "**/*.md"

# If markdownlint is not on PATH, install it globally:
pixi global install markdownlint-cli
```

### Notebook Commands

```bash
# Convert Jupyter notebook to Marimo format
uvx marimo convert <notebook.ipynb> -o <output.py>

# Edit Marimo notebook with live reload
uvx marimo edit --watch <notebook.py>

# Run Marimo notebook
uvx marimo run <notebook.py>
```

**⚠️ CRITICAL**: Always use `uvx marimo convert` to convert Jupyter notebooks.
Do NOT manually convert or create conversion scripts.

---

## Project Structure

```text
pyjanitor/
├── janitor/                    # Source code
│   ├── __init__.py            # Package entry point
│   ├── functions/             # Core DataFrame methods
│   │   ├── __init__.py
│   │   ├── clean_names.py     # Example: clean_names function
│   │   └── ...
│   ├── polars/                # Polars-specific implementations
│   ├── spark/                 # PySpark implementations
│   ├── xarray/                # xarray implementations
│   ├── biology.py             # Biology-specific functions
│   ├── chemistry.py           # Chemistry-specific functions
│   ├── engineering.py         # Engineering-specific functions
│   ├── finance.py             # Finance-specific functions
│   ├── io.py                  # I/O functions
│   ├── math.py                # Math functions
│   ├── ml.py                  # Machine learning functions
│   ├── timeseries.py          # Time series functions
│   └── utils.py               # Utility functions
├── tests/                      # Test files (mirrors source structure)
│   ├── conftest.py            # Shared pytest fixtures
│   ├── functions/             # Tests for functions/
│   ├── polars/                # Tests for polars/
│   ├── chemistry/             # Tests for chemistry
│   └── ...
├── examples/
│   └── notebooks/             # Marimo notebooks (.py files)
├── mkdocs/                    # Documentation source
└── pyproject.toml             # Project configuration
```

---

## Development Patterns

### Adding a New Function

1. **Create the function** in the appropriate module
   (e.g., `janitor/functions/my_function.py`)
2. **Register as DataFrame method** using `@pf.register_dataframe_method`
3. **Export in `__init__.py`** of the parent package
4. **Write tests** in `tests/functions/test_my_function.py`
5. **Add docstring** with Google-style format including Examples section
6. **Update documentation** if needed

### Function Template

```python
"""Description of the module."""

from __future__ import annotations

import pandas as pd
import pandas_flavor as pf


@pf.register_dataframe_method
def my_function(
    df: pd.DataFrame,
    param1: str,
    param2: int = 10,
) -> pd.DataFrame:
    """Short description of what the function does.

    Longer description with more details about behavior.

    Examples:
        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({"col": [1, 2, 3]})
        >>> df.my_function("value")
           col
        0    1
        1    2
        2    3

    Args:
        df: The pandas DataFrame object.
        param1: Description of param1.
        param2: Description of param2. Defaults to 10.

    Returns:
        A pandas DataFrame with the transformation applied.

    Raises:
        ValueError: If param1 is invalid.
    """
    # Implementation - always work on a copy
    df = df.copy()
    # ... transformation logic ...
    return df
```

### Test Template

```python
import pandas as pd
import pytest


@pytest.mark.functions
def test_my_function_basic(dataframe):
    """Test my_function with default args."""
    result = dataframe.my_function("value")
    expected = ...
    assert result.equals(expected)


@pytest.mark.functions
def test_my_function_with_param(dataframe):
    """Test my_function with custom param2."""
    result = dataframe.my_function("value", param2=20)
    # assertions...


@pytest.mark.functions
def test_my_function_error():
    """Test my_function raises ValueError for invalid input."""
    df = pd.DataFrame({"col": [1, 2, 3]})
    with pytest.raises(ValueError, match="expected error message"):
        df.my_function("invalid")
```

### Code Style Rules

- **Line length**: 88 characters (ruff default)
- **Docstrings**: Google-style format
- **Type hints**: Required for function signatures
- **Imports**: Sorted by ruff/isort (stdlib, third-party, local)
- **Formatting**: Handled by ruff-format (double quotes, 4-space indent)

### Pre-commit Hooks

The project uses these pre-commit hooks (auto-run on commit):

| Hook | Purpose |
|------|---------|
| `check-yaml` | Validate YAML files |
| `end-of-file-fixer` | Ensure files end with newline |
| `trailing-whitespace` | Remove trailing whitespace |
| `check-added-large-files` | Prevent large files |
| `nbstripout` | Strip notebook output |
| `interrogate` | Check docstring coverage (>55%) |
| `pydoclint` | Validate docstring format |
| `ruff-check` | Lint Python code |
| `ruff-format` | Format Python code |

---

## Testing Patterns

### Available Fixtures (from conftest.py)

| Fixture | Description |
|---------|-------------|
| `dataframe` | Basic DataFrame with mixed column types |
| `multilevel_dataframe` | DataFrame with MultiIndex columns |
| `multiindex_dataframe` | DataFrame with tuple column names |
| `date_dataframe` | DataFrame with date column |
| `null_df` | DataFrame with null values |
| `missingdata_df` | DataFrame with missing data |
| `biodf` | Biology-related test data |
| `chemdf` | Chemistry-related test data (SMILES) |
| `df_duplicated_columns` | DataFrame with duplicate column names |
| `df_constant_columns` | DataFrame with constant value columns |

### Test Markers

Use pytest markers to categorize tests:

```python
@pytest.mark.functions      # General function tests
@pytest.mark.biology        # Biology module tests
@pytest.mark.chemistry      # Chemistry module tests
@pytest.mark.finance        # Finance module tests
@pytest.mark.engineering    # Engineering module tests
@pytest.mark.polars         # Polars method tests
@pytest.mark.spark_functions # PySpark function tests
@pytest.mark.xarray         # xarray function tests
@pytest.mark.timeseries     # Time series tests
@pytest.mark.turtle         # Slow tests (>5 seconds)
```

### Running Specific Test Categories

```bash
# Run only function tests
pixi run pytest -m "functions" -v

# Run only biology tests (requires biology environment)
pixi run -e biology pytest -m "biology" -v

# Exclude slow tests
pixi run pytest -m "not turtle" -v
```

---

## Common Anti-Patterns to Avoid

### ❌ DON'T

1. **Don't run Python/pytest without pixi**

   ```bash
   # Wrong
   python script.py
   pytest tests/
   ```

2. **Don't mutate input DataFrames**

   ```python
   # Wrong
   def my_func(df):
       df["new_col"] = 1  # Mutates input!
       return df
   ```

3. **Don't manually convert notebooks**

   ```bash
   # Wrong - don't write custom conversion scripts
   python convert_notebook.py
   ```

4. **Don't forget to add tests**
   - Every new function needs corresponding tests

5. **Don't skip docstrings**
   - Interrogate enforces >55% docstring coverage

6. **Don't forget to lint markdown**
   - Always run `markdownlint` on markdown files after editing

### ✅ DO

1. **Always use pixi run**

   ```bash
   pixi run pytest tests/
   pixi run python script.py
   ```

2. **Work on copies**

   ```python
   def my_func(df):
       df = df.copy()
       df["new_col"] = 1
       return df
   ```

3. **Use uvx marimo for notebooks**

   ```bash
   uvx marimo convert notebook.ipynb -o notebook.py
   ```

4. **Write tests alongside code**

5. **Write Google-style docstrings with examples**

6. **Run markdownlint on markdown files**

   ```bash
   markdownlint AGENTS.md
   # Install if not on PATH: pixi global install markdownlint-cli
   ```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: janitor` | Use `pixi run` or `pixi shell` |
| Tests failing with import errors | Use correct pixi environment |
| Pre-commit hooks failing | Run `pixi run lint` for details |
| Docstring coverage failing | Add docstrings to functions |
| rdkit import error | Use `pixi run -e chemistry` |
| markdownlint not found | `pixi global install markdownlint-cli` |

### Environment Issues

```bash
# Reinstall environment
pixi install

# Update lock file
pixi lock

# Clean and reinstall
rm -rf .pixi && pixi install
```

---

## Learned Patterns

<!--
This section is for agents to record new learnings.
Add entries in the format:

### [Date] Learning Title

**Context**: What you were doing
**Learning**: What you discovered
**Recommendation**: How to apply this learning
-->

### [2026-10-19] Empty selectors and DropLabel

**Context**: Issue #1782 requires empty list-like selectors to select nothing.
**Learning**: `DropLabel([])` has inverse semantics and must drop nothing,
thereby selecting everything; it must be tested separately from a direct empty
selector.

### [2025-12-19] Always Run markdownlint

**Context**: Editing AGENTS.md file
**Learning**: Markdown files should be linted with `markdownlint` to ensure
consistent formatting and catch issues like long lines.
**Recommendation**: After editing any markdown file, run `markdownlint <file>`.
If not installed, use `pixi global install markdownlint-cli`.

### [2026-02-07] Open PRs with GitHub CLI

**Context**: User requested opening a PR after pushing changes.
**Learning**: Use `gh pr create` to open PRs when requested.
**Recommendation**: After pushing to the branch, create the PR using the GitHub
CLI.

### [2026-08-06] Build Documentation in the Docs Environment

**Context**: Validating a documentation-only change.
**Learning**: `pixi run docs` selects the default environment, which does not
include MkDocs. The documentation task is available in the `docs` environment.
**Recommendation**: Run `pixi run -e docs build-docs` to build documentation.

### [2026-09-03] Distinguish Public Docs from Maintainer Comments

**Context**: Reorganizing the `conditional_join` and `join_agg` documentation.
**Learning**: Implementation details are not needed in user-facing generated
documentation, but they can still be useful to maintainers working on the code.
**Recommendation**: Keep public docstrings focused on user-visible behavior,
arguments, limitations, and examples. Retain implementation-only explanations
as source comments when they clarify non-obvious code or algorithms for
maintainers.

### [2026-09-18] conditional_join Indexes Are Row Positions, Not Labels

**Context**: Correcting index-contract prose in `_helpers.py` and the
`_conditional_join_compute` comment.
**Learning**: `_conditional_join_compute` resets both frames with
`df.index = pd.RangeIndex(len(df))` before join discovery, so `left_index` and
`right_index` are 0-based row positions rather than original index labels.
**Recommendation**: Before editing index prose under
`janitor/functions/_conditional_join/`, check the index reset in
`_conditional_join_compute`. Describe `left_index`/`right_index` as row
positions, not dataframe labels.

---

### [2026-09-23] Conditional inequality joins are delegated to Rust

**Context**: Routing conditional-join inequality predicates through the
janitor-rs single and extended kernels.
**Learning**: Both single-condition and multiple-condition `!=` joins are
now delegated to Rust. Single predicates use `single_non_equi_join.rs`. Multiple
predicates use `single_non_equi_join_extended.rs`: mixed joins are seeded by a range
predicate, while all-`!=` joins build flat candidate pairs before applying
residual predicates.

PyJanitor remains responsible for resetting both frames to unique
`RangeIndex` values, filtering null rows from non-`!=` predicates,
stably sorting right-hand values, preserving physical position maps, and
providing authoritative null masks. Rust trusts those alignments and does not
sort, infer nullness, or reconstruct dataframe positions.

`return_building_blocks` is a PyJanitor-level request. Internally it asks
the relevant Rust path to retain all surviving candidates; range paths may
return compact windows, while `!=` paths return materialized pairs.

**Recommendation**: Keep the Python/Rust boundary explicit. Use
``single_non_equi_join.rs`` for one predicate, ``single_non_equi_join_extended.rs`` for multiple
predicates, and apply residual predicates before final ``keep`` selection.

### [2026-09-23] Conditional-join aggregation is fused at the Rust boundary

**Context**: Adding aggregation support for single and extended non-equality
conditional joins.
**Learning**: `join_agg` sends supported single range, single `!=`, mixed
range-led, and all-`!=` predicates directly to dedicated Rust aggregation
kernels. Those kernels update aggregation state while comparing candidates and
do not materialize left/right join-index pairs. Forward aggregation produces
one result slot per matched left row; reverse aggregation produces one result
slot per matched right row. The aggregation API has no `keep` parameter.

PyJanitor still owns dataframe preparation: it resets both frames to physical
`RangeIndex` positions, removes null rows from non-`!=` predicates, stably
sorts the right-side range values, preserves the filtered-to-physical position
maps, and supplies authoritative null masks for `!=`. Aggregation input arrays
remain full-layout arrays for the side being aggregated, even when predicate
values are filtered or sorted.

**Recommendation**: Keep aggregation adapters in their own module and preserve
the distinction between predicate layout and aggregation-source layout. When
mapping Rust `min`/`max` results back to pandas, treat the returned positions as
physical source positions and materialize the source values only after the
aggregation result has been filtered by its matched mask.

Numeric aggregation follows the pandas reduction contract. Signed integer
`sum` and `prod` inputs are promoted to `int64`; unsigned integer inputs are
promoted to `uint64`; `float32` and `float64` retain their respective dtypes;
and `min`/`max` retain the source dtype. Boolean aggregation is out of scope.
The wildcard `(*, "size")` and `(*, null_mask, "count")` contracts remain
dtype-independent. The right-index ordering flag is relevant only to
index-building selection; aggregation consumes all surviving candidates and
does not use it.

### [2026-09-25] Keep single-anchor and dual-range extended joins separate

**Context**: Clarifying the multi-predicate range-join routing contract.
**Learning**: The single-extended path uses the first range predicate as its
only binary-search anchor and evaluates every later predicate as a residual
filter. The dedicated range-join path is responsible for two confirmed range
anchors and their intersected windows.

**Recommendation**: Do not pass a second-range optimization flag to the
single-extended Rust kernel. Route confirmed dual-range calls through the
range-join implementation; use the single-extended path for one anchor plus
residual predicates.

### [2026-10-02] Conditional-join index uniqueness is a Python-side invariant

**Context**: Designing the single range-predicate Rust boundary.
**Learning**: PyJanitor guarantees that the left and right physical index
arrays contain unique positions. They are not necessarily ordered. Ordering
must therefore be represented separately by the right-layout ordering flag and
must not be inferred from uniqueness.

**Recommendation**: Rust may rely on uniqueness for physical-position
identity, but must use `right_index_is_ordered` when deciding whether sorted
range boundaries can directly implement `first` or `last`. Otherwise, compute
the extrema over the physical right positions in the candidate window.

### [2026-10-04] Anchor selection must retain predicate occurrence positions

**Context**: Documenting the `enumerate` calls used when selecting range
anchors in the conditional-join dispatchers.
**Learning**: Anchor selection tracks the original condition position, not only
the condition value. Duplicate predicate tuples can be equal by value, so a
condition object or value-based comparison cannot identify which occurrence
was selected. The position is also used to remove exactly the selected anchor
occurrences from the residual predicate list while preserving input order.

**Recommendation**: Keep `enumerate` in anchor-selection and residual-filtering
loops, and explain both the duplicate-occurrence and original-order invariants
when changing those loops.

### [2026-10-04] Coordinated Rust/Python development sources

When Pyjanitor and `janitor-rs` are developed together on stacked branches,
use a temporary `[tool.uv.sources]` override in `pyproject.toml` to exercise
the in-flight Rust package locally and in CI. Keep the `[project]` dependency
on the published PyPI range unchanged.

Prefer a pinned Rust commit SHA for reproducible installs. Update the SHA when
the Rust branch moves, and remove the override before proposing the stack for
merge into `dev`; the override is development infrastructure, not the
published-package contract.

### [2026-10-14] Pixi CI must consume the paired janitor-rs PR

When a pyjanitor change depends on unreleased Python-visible exports from a
janitor-rs PR, add a matching Git source under
`[tool.pixi.pypi-dependencies]`, pinned through the Rust PR branch and the
generated `pixi.lock` commit. Keep the `[project]` dependency as a compatible
published range (for example `>=0.6.2,<0.7`), rather than an exact old release,
so Pixi can solve the Git package and CI can build the paired wheel.

This override is required for every stacked pyjanitor/janitor-rs change that
uses unreleased Rust exports; otherwise CI installs the published janitor-rs
wheel and fails with missing-attribute errors. Remove the override and restore
the release pin before merging the stack into `dev`.

### [2026-10-14] join_agg Does Not Guarantee a Complete Output Row Set

**Context**: Reviewing which rows `join_agg` returns across equality, range,
and `!=` paths.
**Learning**: The result is indexed by physical row positions and, like
`pd.merge`, is not meant to be aligned back to the input. Rows with null
join-condition values are excluded except in joins that use only `!=`.
When an equality predicate is present and no key matches, the result is
empty, matching `get_join_indices`. Otherwise, whether an unmatched row
appears depends on the path: reverse aggregation with duplicate right keys
can keep right rows whose key is absent on the left, while other equality
paths drop them.
**Recommendation**: Do not treat these differences as defects or add
coverage-mask work to make every path return the same rows. Rows that do
appear without a partner carry identity values and `matched=False`.

### [2026-10-15] Condition-order guidance covers every multi-range join

**Context**: User correction while documenting range-anchor selection in
`conditional_join`.
**Learning**: The advice to put the most selective range predicate first is
not specific to `join_algorithm="regions"`. The first range predicate sets the
search layout for every join with multiple range predicates: range-only joins
on the default path, equality joins combined with range predicates, and the
regions algorithm.
**Recommendation**: When documenting condition order, describe all three
paths together. Do not phrase the guidance as applying "especially" to one
algorithm.

### [2026-10-23] Datetime tests must assert converted values

**Context**: A datetime-resolution compatibility fix changed an exact dtype
assertion to a broad datetime-type check, which allowed an intentionally shifted
MATLAB date conversion to pass.

**Learning**: A dtype assertion only verifies the result type. Date-conversion
tests must also assert the converted values so incorrect offsets remain visible.

**Recommendation**: When loosening datetime-resolution assertions, retain or
add value assertions, normalizing resolution only where the conversion contract
does not require a specific precision.

### [2026-10-25] Changelog reconstruction must preserve authored entries

**Context**: Reconstructing accumulated changelog entries from generated
release notes and Git history for issue #1806.
**Learning**: Generated release notes can summarize intermediate commits that
were later reverted, omit hand-written issue/PR attribution, and collapse
semantic changes into performance summaries. Historical changelog work must
start from the original authored entries and verify generated summaries
against the final squashed diff before assigning them to a release.
**Recommendation**: Preserve original wording and contributor credits where
available, keep issue and PR numbers accurate, separate behavior changes from
performance notes, and check for reverted intermediate changes.

Verify each entry against the target tag tree and final squashed diff: do not
duplicate changes across releases, and do not retain generated summaries for
test-only regressions or intermediate commits that were later removed.

### [2026-10-25] Empty releases still need changelog sections

**Context**: Reviewing automated changelog rotation for releases triggered by
merges that add no `[Unreleased]` entries.
**Learning**: Returning early for an empty `[Unreleased]` body leaves the
version absent from `CHANGELOG.md` and stale comparison links, even though the
release and generated `docs/releases/v*.md` file still exist.
**Recommendation**: Always create the version heading and comparison links;
use a clear placeholder when there are no user-facing changes, and test that
path explicitly.

### [2026-10-25] Release changelog entries need attribution and stable formatting

**Context**: Repairing the `v0.32.55` changelog after release automation ran
before changelog rotation was deployed.
**Learning**: A release section must not retain the empty-release placeholder
when it contains an attributed change. Generated bullets and comparison links
must preserve the existing changelog spacing conventions.
**Recommendation**: Treat the placeholder as mutually exclusive with real
entries, use the repository's three-space list marker style, and keep a blank
line between comparison-link definitions.

### [2026-10-25] Changelog release guards cover the complete tag range

**Context**: Issue #1808 showed that PR checks can remain green after release
rotation, and bounded or malformed tag selection can miss long-lived PRs or
fail before validation runs.

**Learning**: Validate `previous_tag..HEAD` before rotation or publishing.
Use the merge ref's first parent for PR checks, parse uncolored diffs
defensively, and allow paired historical edits without allowing new entries
through text matching. Intentional backfills use the explicit manual override.

**Recommendation**: Avoid shell tag filtering in the default guard. If tag
filtering is ever needed, use `git tag -l 'v[0-9]*'` and test it against stray
non-version tags.

### [2026-10-28] Never `copy.copy` a groupby; use `_copy_groupby`

**Context**: PR #1826. Groupby `assign`, `rle_id` and the deprecated `mutate`
shallow-copied a `DataFrameGroupBy` and swapped in a new `obj`. The copy
shared pandas' `_cache` (stale `_obj_with_exclusions`) and pandas_flavor
accessors cached on the instance (bound to the original groupby), so new
columns were silently missing from whole-frame aggregations and
`get_columns`, and chained calls raised `KeyError`/`ValueError`.

**Learning**: `janitor.functions.utils._copy_groupby` keeps the grouper reuse
(the reason `grp.assign(...)` + aggregation is 3-9x faster than regrouping)
and drops `_cache` plus every instance entry that shadows a class-level
pandas_flavor `CachedAccessor`, whichever object it was bound to. Do not
detect accessors by an `_obj` attribute: `getattr(frame, "_obj")` returns a
column named `_obj`. Walk `type(grp).__mro__` checking `base.__dict__`
(first hit wins); `inspect.getattr_static` gives the same answer but is ~5x
slower per copy (25 vs 5 us). Mutating `obj` in place after the copy (as
`assign` does per column) needs `_cache` cleared again. Deprecated functions
that produce groupbys still go through the helper, because their output feeds
live methods. Re-assigning a grouping column keeps the old groups (documented
in `assign`).

**Recommendation**: Any new code that returns a modified groupby must call
`_copy_groupby` and test reuse (original groupby already aggregated or
accessed) and chaining, not only a fresh groupby.

### [2026-10-30] `read_archive` keeps `engine` for the dataframe library

**Context**: Adding `read_archive` to `janitor.io` after review on #1440.
**Learning**: IO readers take `engine="pandas"` or `engine="polars"` and
forward `**kwargs` to the underlying reader. `engine` is not forwarded:
pandas `read_csv` / `read_excel` and polars `read_excel` use the same
parameter name for a different choice. Missing polars is reported with
`import_message`, then re-raised as `ImportError`. Polars Excel reads need
the fastexcel package because polars defaults to its calamine engine.
**Recommendation**: Match `read_commandline` when adding another engine
switch. Do not pass `engine` through in `**kwargs`.

### [2026-10-30] Public default changes require a full call-site and documentation audit

**Context**: Changing the public `join_agg` default for `return_matched`.
**Learning**: A default change must update every test and example that relies
on the old output shape. Tests that validate optional metadata must pass the
flag explicitly, and property-test helpers must derive expected shape from
that requested flag rather than from the result under test.

**Recommendation**: Run all affected test modules, audit changelog and PR
metadata, and strengthen the default-behavior regression test with an exact
expected frame.

### [2026-10-30] adorn row identifier is column position 0

**Context**: Issue #1676. `adorn_pct_formatting`, `adorn_ns`, and
`adorn_rounding` formatted a numeric first column. `adorn_totals` and
`adorn_percentages` already skipped it by comparing labels to `columns[0]`.
**Learning**: The identifier is positional on the frame being adorned.
A label comparison also skips later columns that reuse that name, and
assigning with `df[label]` writes every column that shares the label.
`adorn_ns` must read the identifier from `df`: column 0 of a caller-supplied
`ns` can be a real count. Counts stored by `adorn_percentages` share `df`'s
columns and have to be matched by position. `select_dtypes(include=[np.number])`
includes nullable `Int64` and `Float64` columns and excludes booleans.
**Recommendation**: Choose adorn targets with `adorn_data_positions` and
assign with `DataFrame.isetitem`. Pair `adorn_ns` counts with
`adorn_count_pairs`.

## Version History

- **2025-12-19**: Initial comprehensive AGENTS.md with self-improvement protocol
- **2025-12-19**: Added markdownlint requirement and fixed line length issues
