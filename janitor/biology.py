"""
pyjanitor functions for biological data processing.
"""

import pandas as pd
import pandas_flavor as pf
from Bio import SeqIO


@pf.register_dataframe_method
def join_fasta(df: pd.DataFrame, filename: str, id_col: str, col_name: str):
    """
    Convenience method to join in a FASTA file as a column.

    This allows us to add the string sequence of a FASTA file as a new column
    of data in the dataframe.

    This function only attaches the string representation of the SeqRecord.Seq
    object from Biopython. Does not attach the full SeqRecord. Alphabet is
    also not stored, under the assumption that the data scientist has domain
    knowledge of what kind of sequence is being read in (nucleotide vs. amino
    acid.)

    For more advanced functions, please use phylopandas.

    :param df: A pandas DataFrame.
    :param filename: Path to the FASTA file.
    :param id_col: The column in the DataFrame that houses sequence IDs.
    :param col_name: The name of the new column.
    """
    seqrecords = {
        x.id: x.seq.__str__() for x in SeqIO.parse(filename, "fasta")
    }
    seq_col = [seqrecords[i] for i in df[id_col]]
    df[col_name] = seq_col
    return df
