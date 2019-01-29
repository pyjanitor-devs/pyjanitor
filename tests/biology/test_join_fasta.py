import pytest

import janitor.biology
from janitor.testing_utils.fixtures import TEST_DATA_DIR, biodf


@pytest.mark.biology
def test_join_fasta(biodf):
    df = biodf.join_fasta(
        TEST_DATA_DIR / "sequences.fasta", "sequence_accession", "sequence"
    )

    assert "sequence" in df.columns
