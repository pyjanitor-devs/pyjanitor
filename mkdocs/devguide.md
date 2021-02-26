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
4. In Visual Studio Code,
    click on the quick actions Status Bar item in the lower left corner.
5. Then select "Remote Containers: Clone Repository In Container Volume".
6. Enter in the URL of your fork of `pyjanitor`.

VSCode will pull down the prebuilt Docker container,
git clone the repository for you inside an isolated Docker volume,
and mount the repository directory inside your Docker container.

Follow best practices to submit a pull request by making a feature branch.
Now, hack away, and submit in your pull request!

You shouln't be able to access the cloned repo
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

[repo]: https://github.com/ericmjl/pyjanitor

```bash
git clone git@github.com:your_name_here/pyjanitor.git
```

### Setup the conda environment

Now, install your local copy into a conda environment.
Assuming you have conda installed,
this is how you set up your fork for local development

```bash
cd pyjanitor/
make install
```

This also installs your new conda environment as a Jupyter-accessible kernel.
If you plan to write any notebooks,
to run correctly inside the environment,
make sure you select the correct kernel from the top right corner of JupyterLab!

!!! note "Windows Users"

    If you are on Windows,
    you may need to install `make` before you can run the install.
    You can get it from `conda-forge`::

    ```bash
    conda install -c defaults -c conda-forge make
    ```

    You should be able to run `make` now.
    The command above installs `make` to the `~/Anaconda3/Library/bin` directory.

!!! note "PyCharm Users"

    For PyCharm users,
    here are some `instructions <PYCHARM_USERS.html>`__  to get your Conda environment set up.

### Install the pre-commit hooks.

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

You should also be able to build the docs locally.
To do this, from the main `pyjanitor` directory:

```bash
make docs
```

The command above allows you to view the documentation locally in your browser.
`Sphinx (a python documentation generator) <http://www.sphinx-doc.org/en/stable/usage/quickstart.html>`_ builds and renders the html for you,
and you can find the html files by navigating to `pyjanitor/docs/_build`,
and then you can find the correct html file.
To see the main pyjanitor page,
open the `index.html` file.

!!! note "Errors with documentation builds"

    If you get any errors about importing modules when running `make docs`,
    first activate the development environment:

    ```bash
    source activate pyjanitor-dev || conda activate pyjanitor-dev
    ```

Sphinx uses `rst files (restructured text) <http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_ as its markdown language.
To edit documentation,
go to the rst file that corresponds to the html file you would like to edit.
Make the changes directly in the rst file with the correct markup.
Save the file and rebuild the html pages using the same commands as above to see what your changes look like in html.

### Plan out the change you'd like to contribute

The old adage rings true: failing to plan means planning to fail.
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

[issuetracker]: https://github.com/ericmjl/pyjanitor/issues

### Create a branch for local development

New contributions to `pyjanitor`
should be done in a new branch that you have
based off the latest version of the `dev` branch.

To create a new branch:

```bash
git checkout -b name-of-your-bugfix-or-feature dev
```

Now you can make your changes locally.

### Check your code

When you're done making changes,
check that your changes are properly formatted and that all tests still pass::

```bash
make check
```

If any of the checks fail, you can apply the checks individually (to save time):

* Automated code formatting: `make style`
* Code styling problems check: `make lint`
* Code unit testing: `make test`

Styling problems must be resolved before the pull request can be accepted.

`make test` runs all `pyjanitor`'s unit tests
to probe whether changes to the source code have potentially introduced bugs.
These tests must also pass before the pull request is accepted,
and the continuous integration system up on GitHub Actions
will help run all of the tests before they are committed to the repository.

When you run the test locally,
the tests in `chemistry.py`, `biology.py`, `spark.py`
are automatically skipped if you don't have
the optional dependencies (e.g. `rdkit`) installed.

### Commit your changes

Now you can commit your changes and push your branch to GitHub:

```bash
git add .
git commit -m "Your detailed description of your changes."
git push origin name-of-your-bugfix-or-feature
```

### Submit a pull request through the GitHub website

Congratulations, you've made it to the penultimate step;
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

## Tips

To run a subset of tests:

```bash
pytest tests.test_functions
```
