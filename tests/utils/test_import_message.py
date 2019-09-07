import os
import sys

import pytest

from janitor.utils import import_message


@pytest.mark.utils
def test_import_message(capsys):
    is_conda = os.path.exists(os.path.join(sys.prefix, "conda-meta"))
    if is_conda:
        message = (
            "To use the janitor submodule biology, you need to install "
            "biopython.\n\n"
            "To do so, use the following command:\n\n"
            "    conda install -c conda-forge biopython\n"
        )
    else:
        message = (
            "To use the janitor submodule biology, you need to install "
            "biopython.\n\n"
            "To do so, use the following command:\n\n"
            "    pip install biopython\n"
        )
    import_message(
        submodule="biology",
        package="biopython",
        conda_channel="conda-forge",
        pip_install=True,
    )
    captured = capsys.readouterr()
    assert captured.out == message
