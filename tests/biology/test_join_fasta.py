import pytest

import janitor.biology
from janitor.testing_utils.fixtures import TEST_DATA_DIR, biodf


@pytest.mark.biology
def test_join_fasta(biodf):
    df = biodf.join_fasta(
        filename=TEST_DATA_DIR / "sequences.fasta",
        id_col="sequence_accession",
        col_name="sequence"
    )

    assert "sequence" in df.columns
