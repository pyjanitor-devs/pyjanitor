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
def test_maccs_keys_fingerprint(chemdf):
    """Test conversion of SMILES strings to MACCS keys fingerprints."""
    maccs_keys = chemdf.smiles2mol("smiles", "mol").maccs_keys_fingerprint(
        "mol"
    )
    assert maccs_keys.shape == (10, 167)
    assert set(maccs_keys.to_numpy().flatten().tolist()) == set([0, 1])
