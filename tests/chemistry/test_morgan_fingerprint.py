import pytest

import janitor.chemistry  # noqa: disable=unused-import


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
