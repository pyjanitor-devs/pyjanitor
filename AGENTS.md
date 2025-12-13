# AGENTS.md

This file provides guidance to LLM agents when working with code in this repository.

## Agent Constitution

**CRITICAL RULE**: If the user corrects you on anything, you must:

1. Immediately record the correction in this file (AGENTS.md) in an appropriate section
2. Continue with what you were doing, applying the correction

This ensures that corrections become part of the permanent knowledge base for all future agent interactions.

## Development Environment

**Package Manager**: This project uses `pixi` for dependency management and environment setup.

**CRITICAL FOR LLM AGENTS**: All Python commands must be run within a pixi run context. Never run Python commands directly without the `pixi run` prefix. This is essential for proper dependency management and environment isolation.

**CRITICAL**: This includes Python scripts - always use `pixi run python <script>` instead of `python <script>`.

**Examples**:

- ✅ `pixi run python -c "import json; ..."` (correct)
- ❌ `python -c "import json; ..."` (incorrect - will fail)
- ✅ `pixi run python scripts/my_script.py` (correct)
- ❌ `python scripts/my_script.py` (incorrect - will fail)
- ✅ `pixi run pytest tests/` (correct)
- ❌ `pytest tests/` (incorrect - will fail)

**Key Commands**:

- `pixi run test` - Run the test suite with pytest
- `pixi run docs` - Build documentation with mkdocs
- `pixi run serve-docs` - Serve documentation locally
- `pixi run lint` - Run pre-commit hooks
- `pixi run format` - Format code with ruff
- `pixi run check` - Run all checks (tests, docs, linting, formatting)

**Environment Setup**: Use `pixi shell` to enter the development environment, or prefix commands with `pixi run`.

**Testing Environment**: Tests must be run within the pixi environment using `pixi run test` or `pixi run pytest`.

**Pre-commit Hooks**: The project uses pre-commit hooks with Ruff, interrogate (docstring coverage), pydoclint, and other tools. Hooks run automatically on commit.

## Project Overview

pyjanitor is a Python implementation of the R package janitor. It provides a clean user-friendly API for extending pandas with powerful and readable data-cleaning functions.

## Development Patterns

### Code Style

- **Docstrings**: Use Google-style docstrings
- **Testing**: Always add tests when making code changes (tests mirror source structure in `tests/` directory)
- **Linting**: Automatic linting tools handle formatting (Ruff replaces Black and Flake8)

### Notebooks

- All notebooks in `examples/notebooks/` should be converted to Marimo format (`.py` files)
- **CRITICAL**: Always use `uvx marimo convert <notebook.ipynb> -o <output.py>` to convert Jupyter notebooks to Marimo format. Do not manually convert or create conversion scripts.
- Marimo notebooks are Python files with special cell markers using the `marimo` library
- When creating or editing notebooks, use the marimo format
- To edit marimo notebooks, use `uvx marimo edit --watch <notebook.py>`

