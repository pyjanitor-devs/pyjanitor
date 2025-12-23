# Multi-Python Environment Testing

This document explains how `pyjanitor` configures and uses multiple Python versions for testing across different environments. Understanding this system helps developers maintain CI/CD pipelines and ensures compatibility across Python versions.

## Why Multi-Python Testing?

Python evolves with each release, introducing new features, deprecations, and behavioral changes. Testing against multiple Python versions ensures that `pyjanitor`:

- **Maintains compatibility** across supported Python versions
- **Catches version-specific bugs** early in development
- **Validates dependencies** work correctly with different Python versions
- **Provides confidence** that code changes don't break existing functionality

Currently, `pyjanitor` tests against Python 3.11, 3.12, and 3.13 to cover the range of versions in active use.

## How Pixi Features Work

Pixi uses a **feature-based** system to manage different configurations. Features are reusable sets of dependencies that can be combined to create environments. This is similar to Rust's feature flags or conda's feature system.

In `pyproject.toml`, we define features for each Python version:

```toml
[tool.pixi.feature.py311.dependencies]
python = "3.11.*"

[tool.pixi.feature.py312.dependencies]
python = "3.12.*"

[tool.pixi.feature.py313.dependencies]
python = "3.13.*"
```

Each feature pins a specific Python version using semantic versioning. The `.*` wildcard allows pixi to select the latest patch version (e.g., 3.11.5, 3.12.2) while maintaining the major.minor version constraint.

## How Pixi Environments Combine Features

Environments in pixi are named configurations that combine multiple features. This allows us to create isolated testing environments with specific Python versions while sharing common dependencies.

In `pyproject.toml`, we define environments that combine Python version features with testing features:

```toml
[tool.pixi.environments]
py311 = { features = ["tests", "setup", "py311"] }
py312 = { features = ["tests", "setup", "py312"] }
py313 = { features = ["tests", "setup", "py313"] }
```

Each environment includes:

- **`tests`**: Testing dependencies (pytest, pytest-cov, hypothesis, etc.)
- **`setup`**: Development setup tools (pre-commit hooks)
- **`py311/py312/py313`**: The specific Python version

When pixi creates an environment, it:

1. Resolves all dependencies from the combined features
2. Ensures Python version constraints are satisfied
3. Creates an isolated environment with those exact versions
4. Caches the environment for faster subsequent builds

## How GitHub Actions Uses These Environments

GitHub Actions runs tests in parallel across multiple Python versions using a **matrix strategy**. The workflow configuration orchestrates this:

```yaml
strategy:
  fail-fast: false
  matrix:
    environment: [py311, py312, py313]
```

This creates three parallel jobs, one for each environment. Each job:

1. Checks out the repository
2. Sets up pixi with the specified environment
3. Runs tests in that environment's isolated Python version

The `setup-pixi` action installs the environment:

```yaml
- name: Setup Pixi Environment
  uses: prefix-dev/setup-pixi@v0.9.3
  with:
    pixi-version: latest
    cache: true
    cache-write: true
    environments: ${{ matrix.environment }}
```

The `environments` parameter tells pixi which environment to install. Pixi will:

- Check if the environment is already cached
- If not cached, resolve and install all dependencies
- Create the isolated environment with the correct Python version

## Running Commands in Specific Environments

When running tests, we specify the environment using the `-e` flag:

```yaml
- name: Run unit tests
  run: pixi run -e ${{ matrix.environment }} pytest ...
```

The `-e` flag ensures commands run in the correct environment's Python version. Without it, pixi would use the default environment, which might have a different Python version.

## The Complete Flow

Here's how everything works together:

1. **Developer pushes code** → GitHub Actions triggers
2. **Matrix strategy creates jobs** → One job per Python version (3.11, 3.12, 3.13)
3. **Each job sets up pixi** → Installs the corresponding environment (py311, py312, py313)
4. **Pixi resolves dependencies** → Combines features to get exact Python version + test dependencies
5. **Tests run in isolation** → Each environment has its own Python interpreter
6. **Results are reported** → All jobs must pass for the workflow to succeed

## Benefits of This Approach

- **Isolation**: Each Python version is tested in a completely isolated environment
- **Reproducibility**: Same environment configuration locally and in CI
- **Caching**: Pixi caches environments, making CI runs faster
- **Maintainability**: Adding a new Python version only requires adding a feature and environment
- **Flexibility**: Easy to test different combinations (e.g., Python 3.13 with different pandas versions)

## Adding a New Python Version

To add support for a new Python version (e.g., Python 3.14):

1. **Add a feature** in `pyproject.toml`:

   ```toml
   [tool.pixi.feature.py314.dependencies]
   python = "3.14.*"
   ```

2. **Add an environment**:

   ```toml
   py314 = { features = ["tests", "setup", "py314"] }
   ```

3. **Update the GitHub Actions matrix**:

   ```yaml
   matrix:
     environment: [py311, py312, py313, py314]
   ```

That's it! Pixi and GitHub Actions will handle the rest automatically.

## Troubleshooting

If tests fail in a specific Python version:

- **Check the environment**: Verify the Python version with `pixi run -e py311 python --version`
- **Check dependencies**: Some packages may not support all Python versions
- **Check caching**: Clear pixi cache if dependencies seem stale: `pixi clean`
- **Check logs**: GitHub Actions logs show which environment and Python version was used

## Summary

Multi-Python testing in `pyjanitor` uses pixi's feature and environment system to create isolated test environments for each Python version. GitHub Actions orchestrates parallel test runs across these environments. This approach provides comprehensive version coverage while maintaining simplicity and reproducibility.
