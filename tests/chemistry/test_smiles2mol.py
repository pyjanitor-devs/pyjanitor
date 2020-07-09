import importlib

import pytest

from helpers import running_on_ci


@pytest.mark.skipif(
    (importlib.util.find_spec("rdkit") is None) & ~running_on_ci(),
    reason="rdkit tests only required for CI",
)
@pytest.mark.parametrize("progressbar", [None, "terminal"])
@pytest.mark.chemistry
def test_smiles2mol(chemdf, progressbar):
    """Test each SMILES properly converted to Mol object."""
    from rdkit import Chem

    chemdf = chemdf.smiles2mol("smiles", "mol", progressbar)
    assert "mol" in chemdf.columns
    for elem in chemdf["mol"]:
        assert isinstance(elem, Chem.rdchem.Mol)
