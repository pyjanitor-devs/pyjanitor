from typing import Union

import numpy as np
import pandas as pd
import pandas_flavor as pf
from tqdm import tqdm
from tqdm import tqdm_notebook as tqdmn

from .utils import import_message

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
except ImportError:
    import_message("chemistry", "rdkit", "conda install -c rdkit rdkit")


@pf.register_dataframe_method
def smiles2mol(
    df: pd.DataFrame,
    smiles_col: str,
    mols_col: str,
    drop_nulls: bool = True,
    progressbar: Union[None, str] = None,
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
    valid_progress = ["notebook", "terminal", None]
    if progressbar not in valid_progress:
        raise ValueError(f"progressbar kwarg must be one of {valid_progress}")

    if progressbar is None:
        df[mols_col] = df[smiles_col].apply(lambda x: Chem.MolFromSmiles(x))
    else:
        if progressbar == "notebook":
            tqdmn().pandas(desc="mols")
        elif progressbar == "terminal":
            tqdm.pandas(desc="mols")
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
        for m in df[mols_col]
    ]
    np_fps = []
    for fp in fps:
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        np_fps.append(arr)
    np_fps = np.vstack(np_fps)
    fpdf = pd.DataFrame(np_fps)
    fpdf.index = df.index
    return fpdf
