import importlib
import os

import pytest

import janitor.biology


@pytest.mark.skipif(
    importlib.util.find_spec("Bio") is None,
    reason="Biology tests relying on Biopython only required for CI",
)
@pytest.mark.biology
def test_join_fasta(biodf):
    df = biodf.join_fasta(
        filename=os.path.join(pytest.TEST_DATA_DIR, "sequences.fasta"),
        id_col="sequence_accession",
        column_name="sequence",
    )

    assert "sequence" in df.columns
