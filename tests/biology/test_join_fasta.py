import os

import pytest

import janitor.biology


@pytest.mark.biology
def test_join_fasta(biodf):
    df = biodf.join_fasta(
        filename=os.path.join(pytest.TEST_DATA_DIR, "sequences.fasta"),
        id_col="sequence_accession",
        col_name="sequence",
    )

    assert "sequence" in df.columns
