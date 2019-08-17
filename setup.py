import re
from pathlib import Path

from setuptools import setup


def requirements():
    with open("requirements.txt", "r+") as f:
        return f.read()


def generate_long_description() -> str:
    """
    Extra chunks from README for PyPI description.

    Target chunks must be contained within `.. pypi-doc` pair comments,
    so there must be an even number of comments in README.

    :returns: Extracted description from README

    """
    # Read the contents of README file
    this_directory = Path(__file__).parent
    with open(this_directory / "README.rst", encoding="utf-8") as f:
        readme = f.read()

    # Find pypi-doc comments in README
    indices = [m.start() for m in re.finditer(".. pypi-doc", readme)]
    assert (
        len(indices) % 2 == 0
    ), "Odd number of `.. pypi-doc` comments in README"

    # Loop through pairs of comments and save text between pairs
    long_description = ""
    for i in range(0, len(indices), 2):
        start_index = indices[i] + 11
        end_index = indices[i + 1]
        long_description += readme[start_index:end_index]
    return long_description


setup(
    name="pyjanitor",
    version="0.18.1",
    description="Tools for cleaning pandas DataFrames",
    author="Eric J. Ma",
    author_email="ericmajinglong@gmail.com",
    url="https://github.com/ericmjl/pyjanitor",
    packages=["janitor"],
    install_requires=requirements(),
    python_requires=">=3.6",
    long_description=generate_long_description(),
    long_description_content_type="text/x-rst",
)
