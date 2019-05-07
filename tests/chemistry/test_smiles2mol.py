import importlib
import pytest


@pytest.mark.skipif(
    importlib.util.find_spec("rdkit") is None,
    reason="rdkit tests only required for CI",
)
@pytest.mark.parametrize("progressbar", [None, "terminal"])
@pytest.mark.chemistry
def test_smiles2mol(chemdf, progressbar):
    from rdkit import Chem

    chemdf = chemdf.smiles2mol("smiles", "mol", progressbar)
    assert "mol" in chemdf.columns
    for elem in chemdf["mol"]:
        assert isinstance(elem, Chem.rdchem.Mol)
