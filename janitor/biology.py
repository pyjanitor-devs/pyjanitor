"""
Biology and bioinformatics-oriented data cleaning functions.
"""

import pandas as pd
import pandas_flavor as pf

from .utils import deprecated_alias, import_message

try:
    from Bio import SeqIO
except ImportError:
    import_message(
        submodule="biology",
        package="biopython",
        conda_channel="conda-forge",
        pip_install=True,
    )


@pf.register_dataframe_method
@deprecated_alias(col_name="column_name")
def join_fasta(
    df: pd.DataFrame, filename: str, id_col: str, column_name: str
) -> pd.DataFrame:
    """
    Convenience method to join in a FASTA file as a column.

    This allows us to add the string sequence of a FASTA file as a new column
    of data in the dataframe.

    This method only attaches the string representation of the SeqRecord.Seq
    object from Biopython. Does not attach the full SeqRecord. Alphabet is
    also not stored, under the assumption that the data scientist has domain
    knowledge of what kind of sequence is being read in (nucleotide vs. amino
    acid.)

    This method mutates the original DataFrame.

    For more advanced functions, please use phylopandas.

    Method chaining usage example:

    >>> import tempfile
    >>> import pandas as pd
    >>> import janitor.biology

    >>> tf = tempfile.NamedTemporaryFile()
    >>> tf.write('''>SEQUENCE_1
    ... MTEITAAMVKELRESTGAGMMDCK
    ... >SEQUENCE_2
    ... SATVSEINSETDFVAKN'''.encode('utf8'))
    66
    >>> tf.seek(0)
    0

    >>> df = pd.DataFrame({"sequence_accession":
    ... ["SEQUENCE_1", "SEQUENCE_2", ]})

    >>> df = df.join_fasta(
    ...     filename=tf.name,
    ...     id_col='sequence_accession',
    ...     column_name='sequence',
    ... )

    >>> df.sequence
    0    MTEITAAMVKELRESTGAGMMDCK
    1           SATVSEINSETDFVAKN
    Name: sequence, dtype: object

    :param df: A pandas DataFrame.
    :param filename: Path to the FASTA file.
    :param id_col: The column in the DataFrame that houses sequence IDs.
    :param column_name: The name of the new column.
    :returns: A pandas DataFrame with new FASTA string sequence column.
    """
    seqrecords = {
        x.id: x.seq.__str__() for x in SeqIO.parse(filename, "fasta")
    }
    seq_col = [seqrecords[i] for i in df[id_col]]
    df[column_name] = seq_col
    return df
