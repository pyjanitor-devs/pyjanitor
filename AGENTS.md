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
- **ELI5 Comments**: Use ELI5-style explanations generously in new or
  modified code -- comments and docstrings should explain the mechanism
  (why it works, what trade-off it makes), not just restate what the code
  does.

The `pixi run`, `markdownlint`, and `uvx marimo convert` rules are each
stated once, in their own section below (Package Manager, Markdown
Linting, Notebook Commands) -- not repeated throughout this file.

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

The `pixi run`, notebook-conversion, and markdownlint rules live in their
own sections above (Package Manager, Notebook Commands, Markdown Linting)
-- not repeated here. This list covers anti-patterns with no other home.

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

### [2026-08-06] Build Documentation in the Docs Environment

**Context**: Validating a documentation-only change.
**Learning**: `pixi run docs` selects the default environment, which does not
include MkDocs. The documentation task is available in the `docs` environment.
**Recommendation**: Run `pixi run -e docs build-docs` to build documentation.

### [2026-08-22] Always Use a New Worktree for a New Issue/Branch

**Context**: Working on issue #1653 in the shared `~/github/pyjanitor` clone
while another concurrent session had issue #1648 checked out in a sibling
worktree (`~/github/pyjanitor-1648`); the shared clone's HEAD ended up
detached mid-session from that other work, which was only caught by
inspecting `git status`/`git reflog` before committing.
**Learning**: The shared main clone directory gets reused across sessions
and branches, so starting a new issue there risks colliding with another
in-progress checkout, or committing on top of the wrong branch/detached
HEAD.
**Recommendation**: For every new issue or branch, create a dedicated
`git worktree add ~/github/pyjanitor-<issue> -b <branch> origin/dev`
(or the equivalent for the current base branch) instead of checking out a
new branch in the shared clone. Never leave uncommitted work or run
destructive git operations in the shared clone without first checking
`git status`/`git branch --show-current` for signs of concurrent use.

### [2026-08-22] Use ELI5 Comments Generously

**Context**: Requested while adding new NumPy kernels to
`_agg_functions.py` (Issue #1653) with a non-obvious tie/null/NaN
contract to preserve.
**Learning**: Terse docstrings (e.g. the file's existing "Compute min")
are fine for simple pass-throughs, but new or non-obvious logic --
especially anything with a subtle invariant, a workaround, or a
trade-off a reader wouldn't guess -- benefits from an explicit ELI5
explanation of the mechanism, not just a restatement of what the code
does.
**Recommendation**: When writing or substantially modifying a function,
default to adding a short ELI5 paragraph explaining *why* it works that
way, particularly for dispatch/threshold logic, non-obvious algorithms,
and anything replicating an existing implementation's exact quirky
behavior.

---

## Version History

- **2025-12-19**: Initial comprehensive AGENTS.md with self-improvement protocol
- **2025-12-19**: Added markdownlint requirement and fixed line length issues
