import pytest

from janitor.utils import import_message


@pytest.mark.utils
def test_import_message(capsys):
    message = (
        "To use the janitor submodule chemistry, you need to install "
        "RDKit.\n\n"
        "To do so, use the following command:\n\n"
        "    conda install -c rdkit rdkit\n"
    )
    import_message("chemistry", "RDKit", "conda install -c rdkit rdkit")
    captured = capsys.readouterr()
    assert captured.out == message
