import pytest
import janitor.chemistry  # noqa: disable=unused-import


@pytest.mark.chemistry
def test_molecular_descriptors(chemdf):
    mol_desc = chemdf.smiles2mol("smiles", "mol").molecular_descriptors("mol")
    assert mol_desc.shape == (10, 39)
