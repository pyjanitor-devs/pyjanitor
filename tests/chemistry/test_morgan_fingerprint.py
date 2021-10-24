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
def test_morgan_fingerprint_counts(chemdf):
    """Test counts of Morgan Fingerprints converted from Mol objects."""
    morgans = chemdf.smiles2mol("smiles", "mol").morgan_fingerprint(
        "mol", kind="counts"
    )
    assert morgans.shape == (10, 2048)
    assert (morgans.to_numpy() >= 0).all()


@pytest.mark.chemistry
def test_morgan_fingerprint_bits(chemdf):
    """Test bits of Morgan Fingerprints converted from Mol objects."""
    morgans = chemdf.smiles2mol("smiles", "mol").morgan_fingerprint(
        "mol", kind="bits"
    )
    assert morgans.shape == (10, 2048)
    assert set(morgans.to_numpy().flatten().tolist()) == set([0, 1])


@pytest.mark.chemistry
def test_morgan_fingerprint_kind_error(chemdf):
    """Test `morgan_fingerprint` raises exception for invalid `kind`."""
    with pytest.raises(ValueError):
        chemdf.smiles2mol("smiles", "mol").morgan_fingerprint(
            "mol", kind="invalid-kind"
        )
