from setuptools import setup


def requirements():
    with open("requirements.txt", "r+") as f:
        return f.read()


setup(
    name="pyjanitor",
    version="0.3.2",
    description="Tools for cleaning pandas DataFrames",
    author="Eric J. Ma",
    author_email="ericmajinglong@gmail.com",
    url="https://github.com/ericmjl/pyjanitor",
    packages=["janitor"],
    install_requires=requirements(),
)
