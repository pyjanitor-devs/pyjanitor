# Development Guide

For those of you who are interested in contributing code to the project,
many previous contributors have wrestled with
a variety of ways of getting set up.
While we can't cover every single last configuration,
we could cover some of the more common cases.
Here they are for your benefit!

## Development Containers with VSCode

As of 29 May 2020, development containers are supported!
This is the preferred way to get you started up and running,
as it creates a uniform setup environment
that is much easier for the maintainers to debug,
because you are provided with a pre-built and clean development environment
free of any assumptions of your own system.
You don't have to wrestle with conda wait times if you don't want to!

To get started:

1. Fork the repository.
2. Ensure you have Docker running on your local machine.
3. Ensure you have VSCode running on your local machine.
4. In VS Code, Install an extension called `Remote - Containers`.
5. In Visual Studio Code,
    click on the quick actions Status Bar item in the lower left corner.
6. Then select "Remote Containers: Clone Repository In Container Volume".
7. Enter in the URL of your fork of `pyjanitor`.

VSCode will pull down the prebuilt Docker container,
git clone the repository for you inside an isolated Docker volume,
and mount the repository directory inside your Docker container.

Follow best practices to submit a pull request by making a feature branch.
Now, hack away, and submit in your pull request!

You shouldn't be able to access the cloned repo
on your local hard drive.
If you do want local access, then clone the repo locally first
before selecting "Remote Containers: Open Folder In Container".

If you find something is broken because a utility is missing in the container,
submit a PR with the appropriate build command inserted in the Dockerfile.
Care has been taken to document what each step does,
so please read the in-line documentation in the Dockerfile carefully.

## Manual Setup

### Fork the repository

Firstly, begin by forking the [`pyjanitor` repo][repo] on GitHub.
Then, clone your fork locally:

[repo]: https://github.com/pyjanitor-devs/pyjanitor

```bash
git clone git@github.com:<your_github_username>/pyjanitor.git
```

### Setup the conda environment

Now, install your cloned repo into a conda environment.
Assuming you have conda installed,
this is how you set up your fork for local development

```bash
cd pyjanitor/
# Activate the pyjanitor conda environment
source activate pyjanitor-dev

# Create your conda environment
conda env create -f environment-dev.yml

# Install PyJanitor in development mode
python setup.py develop

# Register current virtual environment as a Jupyter Python kernel
python -m ipykernel install --user --name pyjanitor-dev --display-name "PyJanitor development"
```
If you plan to write any notebooks,
make sure they run correctly inside the environment by
selecting the correct kernel from the top right corner of JupyterLab!

!!! note "PyCharm Users"

    For PyCharm users,
    here are some `instructions <PYCHARM_USERS.html>`__  to get your Conda environment set up.

### Install the pre-commit hooks

`pre-commit` hooks are available
to run code formatting checks automagically before git commits happen.
If you did not have these installed before,
run the following commands:

```bash
# Update your environment to install pre-commit
conda env update -f environment-dev.yml
# Install pre-commit hooks
pre-commit install
```

### Build docs locally

You should also be able to preview the docs locally.
To do this, from the main `pyjanitor` directory:

```bash
python -m mkdocs serve
```
The command above allows you to view the documentation locally in your browser.

If you get any errors about importing modules when running `mkdocs serve`,
first activate the development environment:

```bash
source activate pyjanitor-dev || conda activate pyjanitor-dev
```

### Plan out the change you'd like to contribute

The old adage rings true:

> failing to plan means planning to fail.

We'd encourage you to flesh out the idea you'd like to contribute
on the GitHub issue tracker before embarking on a contribution.
Submitting new code, in particular,
is one where the maintainers will need more consideration,
after all, any new code submitted introduces a new maintenance burden,
unless you the contributor would like to join the maintainers team!

To kickstart the discussion,
submit an issue to the [`pyjanitor` GitHub issue tracker][issuetracker]
describing your planned changes.
The issue tracker also helps us keep track of who is working on what.

[issuetracker]: https://github.com/pyjanitor-devs/pyjanitor

### Create a branch for local development

New contributions to `pyjanitor`
should be done in a new branch that you have
based off the latest version of the `dev` branch.

To create a new branch:

```bash
git checkout -b <name-of-your-bugfix-or-feature> dev
```

### Write the Code

As you work, remember to adhere to the coding standards and practices that pyjanitor follows. If in doubt, refer to existing code or bring up your questions in the GitHub issue you created. Some tips for writing code:

**Commit Early, Commit Often:** Make frequent, smaller commits. This helps to track progress and makes it easier for maintainers to follow your work. Include useful commit messages that describe the changes you're making:

**Stay Updated with dev branch:** Regularly pull the latest changes from the dev branch to ensure your feature branch is up-to-date, reducing the likelihood of merge conflicts:

```bash
git fetch origin dev
git rebase origin/dev
```

**Write Tests:** For every feature or bugfix, accompanying tests are essential.
They ensure the feature works as expected or the bug is truly fixed.
Tests should ideally run in less than 2 seconds.
If using Hypothesis for testing,
apply the `@settings(max_examples=10, timeout=None)` decorator.

### Check your environment

To ensure that your environment is properly set up, run the following command:

```bash
python -m pytest -m "not turtle"
```

If all tests pass then your environment is setup for
development and you are ready to contribute 🥳.

### Check your code

When you're done making changes,
commit your staged files with a meaningful message.
If installed correctly, you will automatically run pre-commit hooks
that check code for code style adherence.
These same checks will be run on GitHub Actions,
so no worries if you don't have the running locally.
If the pre-commit hooks fail,
be sure to fix the issues (as raised by them) before committing.
If you feel lost on how to fix the code,
please feel free to ping the maintainers on GitHub -
we can take things slowly to get it right,
and make this an educational opportunity for all who come by!

!!! tip
    You can run `python -m pytest -m "not turtle"` to run the fast tests.

!!! note "Running test locally"
    When you run tests locally,
    the tests in `chemistry.py`, `biology.py`, `spark.py`
    are automatically skipped if you don't have
    the optional dependencies (e.g. `rdkit`) installed.

!!! info
    * pre-commit **does not run** your tests locally rather all tests are run in continuous integration (CI).
    * All tests must pass in CI before the pull request is accepted,
    and the continuous integration system up on GitHub Actions
    will help run all of the tests before they are committed to the repository.

### Commit your changes

Now you can commit your changes and push your branch to GitHub:

```bash
git add .
git commit -m "Your detailed description of your changes."
git push origin <name-of-your-bugfix-or-feature>
```

### Submit a pull request through the GitHub website

Congratulations 🎉🎉🎉, you've made it to the penultimate step;
your code is ready to be checked and reviewed by the maintainers!
Head over to the GitHub website and create a pull request.
When you are picking out which branch to merge into,
a.k.a. the target branch, be sure to select `dev` (not `master`).

### Fix any remaining issues

It's rare, but you might at this point still encounter issues,
as the continuous integration (CI) system on GitHub Actions checks your code.
Some of these might not be your fault;
rather, it might well be the case that your code fell a little bit out of date
as others' pull requests are merged into the repository.

In any case, if there are any issues, the pipeline will fail out.
We check for code style, docstring coverage, test coverage, and doc discovery.
If you're comfortable looking at the pipeline logs, feel free to do so;
they are open to all to view.
Otherwise, one of the dev team members
can help you with reviewing the code checks.

## Code Compatibility

pyjanitor supports Python 3.6+,
so all contributed code must maintain this compatibility.

## Docstring Style

We follow the Google docstring style, please read [Napoleon's documentation](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) for a detailed introduction.

We are using the following docstring section identifiers -- please stick to them if you are contributing a docstring change:

- **Examples:** for sample code blocks demonstrating the use of pyjanitor. keep example blocks in the `pycon` (python-console) style, i.e., input code prefixed by `>>> ` and `... `, and output code with no prefix.
- **Args:** for function parameters
- **Raises:** for exceptions
- **Returns:** for function return value(s)
- **Yields:** for generator yield value(s)

If possible, it is preferable to stick to this section ordering within each docstring.

## Tips

To run a subset of tests:

```bash
pytest tests.test_functions
```
