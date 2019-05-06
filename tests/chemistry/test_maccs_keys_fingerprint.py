import importlib
import pytest

import janitor.chemistry  # noqa: disable=unused-import


@pytest.mark.skipif(importlib.util.find_spec('rdkit') is None, reason="rdkit tests only required for CI")
@pytest.mark.chemistry
def test_maccs_keys_fingerprint(chemdf):
    maccs_keys = chemdf.smiles2mol("smiles", "mol").maccs_keys_fingerprint(
        "mol"
    )
    assert maccs_keys.shape == (10, 167)
    assert set(maccs_keys.values.flatten().tolist()) == set([0, 1])
