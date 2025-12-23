# How to Get Started Developing pyjanitor

This guide walks you through setting up a development environment
for `pyjanitor` and making your first contribution.

## Prerequisites

Before you begin, ensure you have:

- Git installed and configured
- A GitHub account
- [Pixi](https://pixi.sh) installed for dependency management

## Step 1: Fork and Clone the Repository

Fork the [`pyjanitor` repository][repo] on GitHub, then clone your fork locally:

[repo]: https://github.com/pyjanitor-devs/pyjanitor

```bash
git clone git@github.com:<your_github_username>/pyjanitor.git
cd pyjanitor
```

## Step 2: Set Up the Development Environment

Install the pixi environment and set up the project:

```bash
# Install pixi environment (this installs all dependencies)
pixi install

# Install pyjanitor in development mode and set up pre-commit hooks
pixi run start
```

The `start` command installs `pyjanitor` in editable mode
and configures pre-commit hooks
that will automatically check your code before commits.

## Step 3: Verify Your Setup

Run the test suite to ensure everything is working:

```bash
# Run all tests
pixi run test

# Or run only fast tests (excludes slow "turtle" tests)
pixi run pytest -m "not turtle"
```

If all tests pass, your environment is ready for development.

!!! note "Optional Dependencies"

    When you run tests locally,
    tests in `chemistry.py`, `biology.py`, and `spark.py`
    are automatically skipped if you don't have
    the optional dependencies (e.g., `rdkit`, `pyspark`) installed.
    These will still run in CI where all dependencies are available.

## Step 4: Plan Your Contribution

Before writing code, discuss your planned changes on the [GitHub issue tracker][issuetracker].
This helps:

- Ensure your idea aligns with the project's direction
- Avoid duplicate work
- Get feedback early
- Track who is working on what

[issuetracker]: https://github.com/pyjanitor-devs/pyjanitor

## Step 5: Create a Feature Branch

Create a new branch from the `dev` branch for your work:

```bash
git fetch origin dev
git checkout -b <name-of-your-bugfix-or-feature> origin/dev
```

Use a descriptive branch name that indicates what you're working on
(e.g., `fix-clean-names-underscore`, `add-new-function`).

## Step 6: Make Your Changes

As you write code, keep these guidelines in mind:

### Code Standards

- Follow existing code patterns and style
- Write clear, readable code
- Add docstrings following the
  [Google style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
- Use the docstring sections:
  **Examples**, **Args**, **Raises**, **Returns**, **Yields**
  (in that order when applicable)

### Commit Practices

- **Commit early and often**: Make frequent, smaller commits with clear messages
- **Stay updated**: Regularly sync with the `dev` branch:

  ```bash
  git fetch origin dev
  git rebase origin/dev
  ```

### Write Tests

Every feature or bugfix needs tests. Tests should:

- Run quickly (ideally under 2 seconds)
- Use `@settings(max_examples=10, timeout=None)` when using Hypothesis
- Be placed in the `tests/` directory mirroring the source structure

### Working with Marimo Notebooks

Notebooks are in Marimo format (`.py` files). To edit them:

```bash
# Navigate to the notebooks directory
cd examples/notebooks

# Edit a notebook (replace notebook_name.py with the actual filename)
uvx marimo edit --watch notebook_name.py
```

You must be in the `examples/notebooks` directory
when running the `marimo edit` command.

## Step 7: Check Your Code

Before committing, verify your code passes all checks:

```bash
# Run tests
pixi run test

# Or run fast tests only
pixi run pytest -m "not turtle"

# Check code style and formatting
pixi run lint
```

Pre-commit hooks will automatically run when you commit,
but you can also run them manually:

```bash
pixi run lint
```

!!! info "Pre-commit Hooks"

    Pre-commit hooks check code style automatically before commits.
    They do **not** run your tests locally.
    All tests are run in CI on GitHub Actions before your pull request is accepted.

## Step 8: Commit and Push Your Changes

Once your code is ready, commit and push:

```bash
git add .
git commit -m "Your detailed description of your changes"
git push origin <name-of-your-bugfix-or-feature>
```

If pre-commit hooks fail, fix the issues they report before committing.

## Step 9: Submit a Pull Request

1. Go to the [pyjanitor repository][repo] on GitHub
2. Click "New Pull Request"
3. Select your branch and set the target branch to **`dev`** (not `main` or `master`)
4. Fill out the pull request template with details about your changes
5. Submit the pull request

## Step 10: Address Review Feedback

GitHub Actions will automatically run:

- Code style checks
- Docstring coverage
- Test coverage
- Documentation discovery
- All tests across Python 3.11, 3.12, and 3.13

If any checks fail, review the logs and fix the issues.
Maintainers may also request changes during code review.
Update your branch and push new commits to address feedback.

## Common Development Tasks

All development tasks are available as pixi commands:

- `pixi run start` - Set up development environment
  (install package and pre-commit hooks)
- `pixi run test` - Run the test suite
- `pixi run lint` - Run linting checks
- `pixi run format` - Format code
- `pixi run docs` - Build documentation
- `pixi run serve-docs` - Serve documentation locally
- `pixi run check` - Run all checks (tests, docs, linting, formatting)

## Tips

### Running Specific Tests

Run a subset of tests:

```bash
# Run tests for a specific module
pixi run pytest tests/functions/test_clean_names.py

# Run a specific test
pixi run pytest tests/functions/test_clean_names.py::test_clean_names_basic
```

### Viewing Documentation Locally

Preview documentation while developing:

```bash
pixi run serve-docs
```

This starts a local server (usually at `http://127.0.0.1:8000`)
where you can view the documentation.

## Code Compatibility

`pyjanitor` supports Python 3.11, 3.12, and 3.13.
All contributed code must maintain compatibility with these versions.
Tests run automatically across all supported Python versions in CI.

## Troubleshooting

### TLS Certificate Errors Behind a Firewall

If you're behind a corporate firewall or proxy that performs TLS inspection,
you may encounter certificate errors like:

```text
Error:   × failed to solve requirements of environment 'tests' for platform 'linux-64'
  ├─▶ Request failed after 3 retries
  ├─▶ error sending request for url (https://prefix.dev/...)
  ├─▶ client error (Connect)
  ╰─▶ invalid peer certificate: UnknownIssuer
```

To work around this, use the `--tls-no-verify` flag when adding packages:

```bash
pixi add --pypi <package-name> --tls-no-verify
```

Alternatively, you can configure pixi globally to skip TLS verification
by setting the environment variable:

```bash
export PIXI_TLS_NO_VERIFY=true
pixi install
```

**Note:** Disabling TLS verification reduces security.
Only use this workaround in trusted network environments
where you understand the implications.

## Getting Help

If you encounter issues or have questions:

- Check existing [GitHub issues](https://github.com/pyjanitor-devs/pyjanitor/issues)
- Ask questions in your pull request
- Reach out to maintainers on GitHub

We're here to help make your contribution experience smooth and educational!
