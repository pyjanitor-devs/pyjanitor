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
- **Adversarial Review**: Before treating a PR as done, get a fresh-context
  review of the diff (e.g. a subagent with no memory of how the change was
  built) to actively try to refute correctness claims, not just check style.
  Implementer bias won't catch what an independent read will - this repo's
  join/merge code in particular has subtle correctness invariants (dtype,
  null, and extension-array semantics; output ordering guarantees) that are
  easy to assume rather than verify.

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

`pixi run docs` uses the `default` environment, which doesn't include
MkDocs - use the `docs` environment for anything that actually builds
documentation.

```bash
# Serve docs with live reload
pixi run serve-docs

# Build docs (in the docs environment - MkDocs isn't in default)
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

### Writing PRs and Issues

For changes that involve non-obvious algorithmic or performance reasoning
(not every PR/issue - a typo fix or a one-line dependency bump doesn't need
this), include a plain-language **ELI5** section explaining the change
without jargon: what was wrong, why, and what changed - in terms someone
unfamiliar with the internals could follow. Place it right after the
technical summary/background, before any benchmark numbers or detailed
analysis.

PR body shape:

```markdown
## Summary
<technical summary of the change and why>

## ELI5
<plain-language explanation - only if the change needs one>

## Benchmark
<before/after numbers, if performance-related>

## Test plan
<what was run to verify it>
```

Issue body shape:

```markdown
## Background
<what's wrong and why, technically>

## ELI5
<plain-language explanation - only if the issue needs one>

## Measured impact
<numbers, if applicable>

## Ask
<what the fix should do>

## Related
<links to parent/sibling issues>
```

See PR #1644 and issues #1641/#1660 for real examples of this shape.

For any `Benchmark`/`Measured impact` section, extend the row-count range
far enough to show whether the effect shrinks away or plateaus at scale
(e.g. out to 10M-50M rows for join/row-wise operations), not just a few
thousand rows. A regression or speedup that looks small at 10k rows can
look very different at 50M - #1660's table is a real example: a ratio
that looked like it might fade out below 30k rows instead held flat at
~1.8-1.9x all the way to 50M, which is the number that actually matters
for deciding whether it's worth fixing.

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

Pixi usage, notebook conversion, and markdown linting have their own
canonical rules already (Core Principles, Development Environment,
Notebook Commands, Markdown Linting) - not repeated here.

### ❌ DON'T

1. **Don't mutate input DataFrames**

   ```python
   # Wrong
   def my_func(df):
       df["new_col"] = 1  # Mutates input!
       return df
   ```

2. **Don't forget to add tests**
   - Every new function needs corresponding tests

3. **Don't skip docstrings**
   - Interrogate enforces >55% docstring coverage

### ✅ DO

1. **Work on copies**

   ```python
   def my_func(df):
       df = df.copy()
       df["new_col"] = 1
       return df
   ```

2. **Write tests alongside code**

3. **Write Google-style docstrings with examples**

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

### [2026-02-07] Open PRs with GitHub CLI

**Context**: User requested opening a PR after pushing changes.
**Learning**: Use `gh pr create` to open PRs when requested.
**Recommendation**: After pushing to the branch, create the PR using the GitHub
CLI.

<!--
[2025-12-19] Always Run markdownlint, [2026-08-06] Build Documentation in
the Docs Environment, and [2026-08-21] Adversarially Review Every PR
Before It's Done were integrated into Core Principles, Documentation
Commands, and Core Principles respectively during a 2026-08-21 cleanup
pass, and removed from here to avoid restating the same rule twice.
-->

---

## Version History

- **2025-12-19**: Initial comprehensive AGENTS.md with self-improvement protocol
- **2025-12-19**: Added markdownlint requirement and fixed line length issues
- **2026-08-21**: Added mandatory adversarial review before every PR
- **2026-08-21**: Added PR/issue writing convention (ELI5 for non-obvious changes)
- **2026-08-21**: Added convention to extend benchmarks to scale (10M-50M rows)
- **2026-08-21**: Cleanup pass - removed 5x-duplicated markdownlint rule, 3x
  -duplicated pixi/notebook rules, and Learned Patterns entries already
  integrated into main sections
