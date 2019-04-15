import os
from pathlib import Path

import pandas as pd
import pytest

import janitor.chemistry


@pytest.mark.parametrize("progressbar", [None, "terminal"])
@pytest.mark.chemistry
def test_smiles2mol(chemdf, progressbar):
    from rdkit import Chem

    chemdf = chemdf.smiles2mol("smiles", "mol", progressbar)
    assert "mol" in chemdf.columns
    for elem in chemdf["mol"]:
        assert isinstance(elem, Chem.rdchem.Mol)


@pytest.mark.chemistry
def test_morgan_fingerprint_counts(chemdf):
    morgans = chemdf.smiles2mol("smiles", "mol").morgan_fingerprint(
        "mol", kind="counts"
    )
    assert morgans.shape == (10, 2048)
    assert (morgans.values >= 0).all()


@pytest.mark.chemistry
def test_morgan_fingerprint_bits(chemdf):
    morgans = chemdf.smiles2mol("smiles", "mol").morgan_fingerprint(
        "mol", kind="bits"
    )
    assert morgans.shape == (10, 2048)
    assert set(morgans.values.flatten().tolist()) == set([0, 1])
