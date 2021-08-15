import importlib

import pytest
from helpers import running_on_ci

import janitor.chemistry  # noqa: disable=unused-import

# Skip all tests if rdkit not installed
pytestmark = pytest.mark.skipif(
    (importlib.util.find_spec("rdkit") is None) & ~running_on_ci(),
    reason="rdkit tests only required for CI",
)


@pytest.mark.chemistry
def test_molecular_descriptors(chemdf):
    """Test conversion of Mol objects to 39 column molecular descriptors."""
    mol_desc = chemdf.smiles2mol("smiles", "mol").molecular_descriptors("mol")
    assert mol_desc.shape == (10, 39)
