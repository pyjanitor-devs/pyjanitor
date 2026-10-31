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
def test_to_fasta(biodf, tmp_path):
    """Test writing sequence data to a FASTA file, round-tripped through join_fasta."""
    df = biodf.join_fasta(
        filename=os.path.join(pytest.TEST_DATA_DIR, "sequences.fasta"),
        id_col="sequence_accession",
        column_name="sequence",
    )

    out_file = tmp_path / "output.fasta"
    result = df.to_fasta(
        identifier_column_name="sequence_accession",
        sequence_column_name="sequence",
        filename=str(out_file),
    )

    # to_fasta should return the original dataframe, unmodified
    assert result is df

    from Bio import SeqIO

    written = {
        record.id: str(record.seq)
        for record in SeqIO.parse(str(out_file), "fasta")
    }
    expected = dict(zip(df["sequence_accession"], df["sequence"]))
    assert written == expected
