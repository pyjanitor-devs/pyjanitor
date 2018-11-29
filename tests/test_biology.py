"""
Tests for biology-oriented functions.
"""
import janitor.biology
import pandas as pd
import pytest
import os
from pathlib import Path

test_data_dir = Path(os.path.dirname(os.path.abspath(__file__))) / 'test_data'

@pytest.fixture
def biodf():
    df = (
        pd.read_csv(test_data_dir / 'sequences.tsv', sep='\t')
        .clean_names()
    )
    return df

@pytest.mark.biology
def test_join_fasta(biodf):
    df = (
        biodf
        .join_fasta(test_data_dir / 'sequences.fasta', "sequence_accession", "sequence")
    )

    assert "sequence" in df.columns