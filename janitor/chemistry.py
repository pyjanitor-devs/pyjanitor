import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdmn

import pandas_flavor as pf
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


@pf.register_dataframe_method
def smiles2mol(
    df: pd.DataFrame, smiles_col: str, mols_col: str, drop_nulls: bool = True
):
    """
    Convert a column of SMILES strings into RDKit Mol objects.

    Automatically drops invalid SMILES, as determined by RDKIT.

    Method chaining usage:

    ..code-block:: python

        df = (
            pd.DataFrame(...)
            .smiles2mol(smiles_col='smiles', mols_col='mols')
        )

    :param df: pandas DataFrame.
    :param smiles_col: Name of column that holds the SMILES strings.
    :param mols_col: Name to be given to the new mols column.
    :param drop_nulls: Whether to drop rows whose mols failed to be
        constructed.
    """
    tqdmn().pandas(desc="mol construction")
    df[mols_col] = df[smiles_col].progress_apply(
        lambda x: Chem.MolFromSmiles(x)
    )

    if drop_nulls:
        df.dropna(subset=[mols_col], inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df


@pf.register_dataframe_method
def morganbits(
    df: pd.DataFrame, mols_col: str, radius: int = 3, nbits: int = 2048
):
    """
    Convert a column of RDKIT Mol objects into Morgan Fingerprints.

    Returns a new dataframe without any of the original data. This is
    intentional, as Morgan fingerprints are usually high-dimensional
    features.

    Method chaining usage:

    ..code-block:: python

        df = pd.DataFrame(...)
        morgans = df.morganbits(mols_col='mols', radius=3, nbits=2048)

    If you wish to join the Morgans back into the original dataframe, this
    can be accomplished by doing a `join`, becuase the indices are
    preserved:

    ..code-block:: python

        joined = df.join(morgans)

    :param df: A pandas DataFrame.
    :param mols_col: The name of the column that has the RDKIT mol objects
    :param radius: Radius of Morgan fingerprints. Defaults to 3.
    :param nbits: The length of the fingerprints. Defaults to 2048.
    """
    fps = [
        AllChem.GetMorganFingerprintAsBitVect(m, radius, nbits)
        for m in tqdmn(df[mols_col])
    ]
    np_fps = []
    for fp in tqdmn(fps):
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        np_fps.append(arr)
    np_fps = np.vstack(np_fps)
    fpdf = pd.DataFrame(np_fps)
    fpdf.index = df.index
    return fpdf
