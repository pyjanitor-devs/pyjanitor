import pytest
import pandas as pd
import janitor.chemistry
from rdkit import Chem


@pytest.fixture
def chemdf():
    df = pd.read_csv(
        "test_data/corrected_smiles.txt", sep="\t", header=None
    ).head(10)
    df.columns = ["id", "smiles"]
    return df


@pytest.mark.parametrize("progressbar", [None, "terminal"])
@pytest.mark.chem
def test_smiles2mol(chemdf, progressbar):
    chemdf = chemdf.smiles2mol("smiles", "mol", progressbar)
    assert "mol" in chemdf.columns
    for elem in chemdf["mol"]:
        assert isinstance(elem, Chem.rdchem.Mol)


@pytest.mark.chem
def test_morganbits(chemdf):
    morgans = chemdf.smiles2mol("smiles", "mol").morganbits("mol")
    assert morgans.shape == (10, 2048)
