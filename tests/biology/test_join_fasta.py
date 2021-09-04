import importlib
import os

import pytest
from helpers import running_on_ci

import janitor.biology  # noqa: F403, F401

# Skip all tests if Biopython not installed
pytestmark = pytest.mark.skipif(
    (importlib.util.find_spec("Bio") is None) & ~running_on_ci(),
    reason="Biology tests relying on Biopython only required for CI",
)


@pytest.mark.biology
def test_join_fasta(biodf):
    """Test adding sequence from FASTA file in `sequence` column."""
    df = biodf.join_fasta(
        filename=os.path.join(pytest.TEST_DATA_DIR, "sequences.fasta"),
        id_col="sequence_accession",
        column_name="sequence",
    )

    assert "sequence" in df.columns
