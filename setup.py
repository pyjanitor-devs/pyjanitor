"""Setup script."""
import codecs
import os
import re
from pathlib import Path
from pprint import pprint

from setuptools import find_packages, setup

HERE = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    # intentionally *not* adding an encoding option to open
    return codecs.open(os.path.join(HERE, *parts), "r").read()


def read_requirements(*parts):
    """
    Return requirements from parts.

    Given a requirements.txt (or similar style file),
    returns a list of requirements.
    Assumes anything after a single '#' on a line is a comment, and ignores
    empty lines.

    :param parts: list of filenames which contain the installation "parts",
        i.e. submodule-specific installation requirements
    :returns: A compiled list of requirements.
    """
    requirements = []
    for line in read(*parts).splitlines():
        new_line = re.sub(  # noqa: PD005
            r"(\s*)?#.*$",  # the space immediately before the
            # hash mark, the hash mark, and
            # anything that follows it
            "",  # replace with a blank string
            line,
        )
        new_line = re.sub(  # noqa: PD005
            r"-r.*$",  # link to another requirement file
            "",  # replace with a blank string
            new_line,
        )
        new_line = re.sub(  # noqa: PD005
            r"-e \..*$",  # link to editable install
            "",  # replace with a blank string
            new_line,
        )
        # print(line, "-->", new_line)
        if new_line:  # i.e. we have a non-zero-length string
            requirements.append(new_line)
    return requirements


# pull from requirements.IN, requirements.TXT is generated from this
INSTALL_REQUIRES = read_requirements(".requirements/base.in")

EXTRA_REQUIRES = {
    "dev": read_requirements(".requirements/dev.in"),
    "docs": read_requirements(".requirements/docs.in"),
    "test": read_requirements(".requirements/testing.in"),
    "biology": read_requirements(".requirements/biology.in"),
    "chemistry": read_requirements(".requirements/chemistry.in"),
    "engineering": read_requirements(".requirements/engineering.in"),
    "spark": read_requirements(".requirements/spark.in"),
}

# add 'all' key to EXTRA_REQUIRES
all_requires = []
for k, v in EXTRA_REQUIRES.items():
    all_requires.extend(v)
EXTRA_REQUIRES["all"] = set(all_requires)

for k1 in ["biology", "chemistry", "engineering", "spark"]:
    for v2 in EXTRA_REQUIRES[k1]:
        EXTRA_REQUIRES["docs"].append(v2)

pprint(EXTRA_REQUIRES)


def generate_long_description() -> str:
    """
    Extra chunks from README for PyPI description.

    Target chunks must be contained within `.. pypi-doc` pair comments,
    so there must be an even number of comments in README.

    :returns: Extracted description from README.
    :raises Exception: if odd number of `.. pypi-doc` comments
        in README.
    """
    # Read the contents of README file
    this_directory = Path(__file__).parent
    with open(this_directory / "mkdocs" / "index.md", encoding="utf-8") as f:
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


setup(
    name="pyjanitor",
    version="0.23.1",
    description="Tools for cleaning pandas DataFrames",
    author="pyjanitor devs",
    author_email="ericmajinglong@gmail.com",
    url="https://github.com/pyjanitor-devs/pyjanitor",
    license="MIT",
    # packages=["janitor", "janitor.xarray", "janitor.spark"],
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRA_REQUIRES,
    python_requires=">=3.6",
    long_description=generate_long_description(),
    long_description_content_type="text/x-rst",
)
