import re
from pathlib import Path

from setuptools import find_packages, setup


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
    if len(indices) % 2 != 0:
        raise Exception("Odd number of `.. pypi-doc` comments in README")

    # Loop through pairs of comments and save text between pairs
    long_description = ""
    for i in range(0, len(indices), 2):
        start_index = indices[i] + 11
        end_index = indices[i + 1]
        long_description += readme[start_index:end_index]
    return long_description


extra_spark = ["pyspark"]
extra_biology = ["biopython"]
extra_chemistry = ["rdkit"]
extra_engineering = ["unyt"]
extra_all = extra_biology + extra_engineering + extra_spark

setup(
    name="pyjanitor",
    version="0.20.5",
    description="Tools for cleaning pandas DataFrames",
    author="Eric J. Ma",
    author_email="ericmajinglong@gmail.com",
    url="https://github.com/ericmjl/pyjanitor",
    license="MIT",
    # packages=["janitor", "janitor.xarray", "janitor.spark"],
    packages=find_packages(),
    install_requires=requirements(),
    extras_require={
        "all": extra_all,
        "biology": extra_biology,
        # "chemistry": extra_chemistry, should be inserted once rdkit
        # fixes https://github.com/rdkit/rdkit/issues/1812
        "engineering": extra_engineering,
        "spark": extra_spark,
    },
    python_requires=">=3.6",
    long_description=generate_long_description(),
    long_description_content_type="text/x-rst",
)
