"""
Chemistry and cheminformatics-oriented data cleaning functions.
"""

from typing import Hashable, Union

import numpy as np
import pandas as pd
import pandas_flavor as pf

from .utils import deprecated_alias, import_message

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem.rdMolDescriptors import (
        GetHashedMorganFingerprint,
        GetMorganFingerprintAsBitVect,
        CalcChi0n,
        CalcChi0v,
        CalcChi1n,
        CalcChi1v,
        CalcChi2n,
        CalcChi2v,
        CalcChi3n,
        CalcChi3v,
        CalcChi4n,
        CalcChi4v,
        CalcExactMolWt,
        CalcFractionCSP3,
        CalcHallKierAlpha,
        CalcKappa1,
        CalcKappa2,
        CalcKappa3,
        CalcLabuteASA,
        CalcNumAliphaticCarbocycles,
        CalcNumAliphaticHeterocycles,
        CalcNumAliphaticRings,
        CalcNumAmideBonds,
        CalcNumAromaticCarbocycles,
        CalcNumAromaticHeterocycles,
        CalcNumAromaticRings,
        CalcNumAtomStereoCenters,
        CalcNumBridgeheadAtoms,
        CalcNumHBA,
        CalcNumHBD,
        CalcNumHeteroatoms,
        CalcNumHeterocycles,
        CalcNumLipinskiHBA,
        CalcNumLipinskiHBD,
        CalcNumRings,
        CalcNumSaturatedCarbocycles,
        CalcNumSaturatedHeterocycles,
        CalcNumSaturatedRings,
        CalcNumSpiroAtoms,
        CalcNumUnspecifiedAtomStereoCenters,
        CalcTPSA,
        GetMACCSKeysFingerprint,
    )
except ImportError:
    import_message(
        submodule="chemistry",
        package="rdkit",
        conda_channel="conda-forge",
        pip_install=False,
    )

try:
    from tqdm import tqdm
    from tqdm import tqdm_notebook as tqdmn
except ImportError:
    import_message(
        submodule="chemistry",
        package="tqdm",
        conda_channel="conda-forge",
        pip_install=True,
    )


@pf.register_dataframe_method
@deprecated_alias(smiles_col="smiles_column_name", mols_col="mols_column_name")
def smiles2mol(
    df: pd.DataFrame,
    smiles_column_name: Hashable,
    mols_column_name: Hashable,
    drop_nulls: bool = True,
    progressbar: Union[None, str] = None,
) -> pd.DataFrame:
    """
    Convert a column of SMILES strings into RDKit Mol objects.

    Automatically drops invalid SMILES, as determined by RDKIT.

    This method mutates the original DataFrame.

    Functional usage example:

    .. code-block:: python

        import pandas as pd
        import janitor.chemistry

        df = pd.DataFrame(...)

        df = janitor.chemistry.smiles2mol(
            df=df,
            smiles_column_name='smiles',
            mols_column_name='mols'
        )

    Method chaining usage example:

    .. code-block:: python

        import pandas as pd
        import janitor.chemistry

        df = pd.DataFrame(...)

        df = df.smiles2mol(smiles_column_name='smiles',
                           mols_column_name='mols')

    A progressbar can be optionally used.

    - Pass in "notebook" to show a tqdm notebook progressbar. (ipywidgets must
      be enabled with your Jupyter installation.)
    - Pass in "terminal" to show a tqdm progressbar. Better suited for use
      with scripts.
    - "none" is the default value - progress bar will be not be shown.

    :param df: pandas DataFrame.
    :param smiles_column_name: Name of column that holds the SMILES strings.
    :param mols_column_name: Name to be given to the new mols column.
    :param drop_nulls: Whether to drop rows whose mols failed to be
        constructed.
    :param progressbar: Whether to show a progressbar or not.
    :returns: A pandas DataFrame with new RDKIT Mol objects column.
    """
    valid_progress = ["notebook", "terminal", None]
    if progressbar not in valid_progress:
        raise ValueError(f"progressbar kwarg must be one of {valid_progress}")

    if progressbar is None:
        df[mols_column_name] = df[smiles_column_name].apply(
            lambda x: Chem.MolFromSmiles(x)
        )
    else:
        if progressbar == "notebook":
            tqdmn().pandas(desc="mols")
        elif progressbar == "terminal":
            tqdm.pandas(desc="mols")
        df[mols_column_name] = df[smiles_column_name].progress_apply(
            lambda x: Chem.MolFromSmiles(x)
        )

    if drop_nulls:
        df = df.dropna(subset=[mols_column_name])
    df = df.reset_index(drop=True)
    return df


@pf.register_dataframe_method
@deprecated_alias(mols_col="mols_column_name")
def morgan_fingerprint(
    df: pd.DataFrame,
    mols_column_name: str,
    radius: int = 3,
    nbits: int = 2048,
    kind: str = "counts",
) -> pd.DataFrame:
    """
    Convert a column of RDKIT Mol objects into Morgan Fingerprints.

    Returns a new dataframe without any of the original data. This is
    intentional, as Morgan fingerprints are usually high-dimensional
    features.

    This method does not mutate the original DataFrame.

    Functional usage example:

    .. code-block:: python

        import pandas as pd
        import janitor.chemistry

        df = pd.DataFrame(...)

        # For "counts" kind
        morgans = janitor.chemistry.morgan_fingerprint(
            df=df.smiles2mol('smiles', 'mols'),
            mols_column_name='mols',
            radius=3,      # Defaults to 3
            nbits=2048,    # Defaults to 2048
            kind='counts'  # Defaults to "counts"
        )

        # For "bits" kind
        morgans = janitor.chemistry.morgan_fingerprint(
            df=df.smiles2mol('smiles', 'mols'),
            mols_column_name='mols',
            radius=3,      # Defaults to 3
            nbits=2048,    # Defaults to 2048
            kind='bits'    # Defaults to "counts"
        )

    Method chaining usage example:

    .. code-block:: python

        import pandas as pd
        import janitor.chemistry

        df = pd.DataFrame(...)

        # For "counts" kind
        morgans = (
            df.smiles2mol('smiles', 'mols')
              .morgan_fingerprint(mols_column_name='mols',
                                  radius=3,      # Defaults to 3
                                  nbits=2048,    # Defaults to 2048
                                  kind='counts'  # Defaults to "counts"
              )
        )

        # For "bits" kind
        morgans = (
            df.smiles2mol('smiles', 'mols')
              .morgan_fingerprint(mols_column_name='mols',
                                  radius=3,    # Defaults to 3
                                  nbits=2048,  # Defaults to 2048
                                  kind='bits'  # Defaults to "counts"
              )
        )

    If you wish to join the morgan fingerprints back into the original
    dataframe, this can be accomplished by doing a `join`,
    because the indices are preserved:

    .. code-block:: python

        joined = df.join(morgans)

    :param df: A pandas DataFrame.
    :param mols_column_name: The name of the column that has the RDKIT
        mol objects
    :param radius: Radius of Morgan fingerprints. Defaults to 3.
    :param nbits: The length of the fingerprints. Defaults to 2048.
    :param kind: Whether to return counts or bits. Defaults to counts.
    :returns: A new pandas DataFrame of Morgan fingerprints.
    """
    acceptable_kinds = ["counts", "bits"]
    if kind not in acceptable_kinds:
        raise ValueError(f"`kind` must be one of {acceptable_kinds}")

    if kind == "bits":
        fps = [
            GetMorganFingerprintAsBitVect(m, radius, nbits, useChirality=True)
            for m in df[mols_column_name]
        ]
    elif kind == "counts":
        fps = [
            GetHashedMorganFingerprint(m, radius, nbits, useChirality=True)
            for m in df[mols_column_name]
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


@pf.register_dataframe_method
@deprecated_alias(mols_col="mols_column_name")
def molecular_descriptors(
    df: pd.DataFrame, mols_column_name: Hashable
) -> pd.DataFrame:
    """
    Convert a column of RDKIT mol objects into a Pandas DataFrame
    of molecular descriptors.

    Returns a new dataframe without any of the original data. This is
    intentional to leave the user only with the data requested.

    This method does not mutate the original DataFrame.

    The molecular descriptors are from the rdkit.Chem.rdMolDescriptors:

        Chi0n, Chi0v, Chi1n, Chi1v, Chi2n, Chi2v, Chi3n, Chi3v,
        Chi4n, Chi4v, ExactMolWt, FractionCSP3, HallKierAlpha, Kappa1,
        Kappa2, Kappa3, LabuteASA, NumAliphaticCarbocycles,
        NumAliphaticHeterocycles, NumAliphaticRings, NumAmideBonds,
        NumAromaticCarbocycles, NumAromaticHeterocycles, NumAromaticRings,
        NumAtomStereoCenters, NumBridgeheadAtoms, NumHBA, NumHBD,
        NumHeteroatoms, NumHeterocycles, NumLipinskiHBA, NumLipinskiHBD,
        NumRings, NumSaturatedCarbocycles, NumSaturatedHeterocycles,
        NumSaturatedRings, NumSpiroAtoms, NumUnspecifiedAtomStereoCenters,
        TPSA.

    Functional usage example:

    .. code-block:: python

        import pandas as pd
        import janitor.chemistry

        df = pd.DataFrame(...)

        mol_desc = janitor.chemistry.molecular_descriptors(
            df=df.smiles2mol('smiles', 'mols'),
            mols_column_name='mols'
        )

    Method chaining usage example:

    .. code-block:: python

        import pandas as pd
        import janitor.chemistry

        df = pd.DataFrame(...)

        mol_desc = (
            df.smiles2mol('smiles', 'mols')
              .molecular_descriptors(mols_column_name='mols')
        )

    If you wish to join the molecular descriptors back into the original
    dataframe, this can be accomplished by doing a `join`,
    because the indices are preserved:

    .. code-block:: python

        joined = df.join(mol_desc)

    :param df: A pandas DataFrame.
    :param mols_column_name: The name of the column that has the RDKIT mol
        objects.
    :returns: A new pandas DataFrame of molecular descriptors.
    """
    descriptors = [
        CalcChi0n,
        CalcChi0v,
        CalcChi1n,
        CalcChi1v,
        CalcChi2n,
        CalcChi2v,
        CalcChi3n,
        CalcChi3v,
        CalcChi4n,
        CalcChi4v,
        CalcExactMolWt,
        CalcFractionCSP3,
        CalcHallKierAlpha,
        CalcKappa1,
        CalcKappa2,
        CalcKappa3,
        CalcLabuteASA,
        CalcNumAliphaticCarbocycles,
        CalcNumAliphaticHeterocycles,
        CalcNumAliphaticRings,
        CalcNumAmideBonds,
        CalcNumAromaticCarbocycles,
        CalcNumAromaticHeterocycles,
        CalcNumAromaticRings,
        CalcNumAtomStereoCenters,
        CalcNumBridgeheadAtoms,
        CalcNumHBA,
        CalcNumHBD,
        CalcNumHeteroatoms,
        CalcNumHeterocycles,
        CalcNumLipinskiHBA,
        CalcNumLipinskiHBD,
        CalcNumRings,
        CalcNumSaturatedCarbocycles,
        CalcNumSaturatedHeterocycles,
        CalcNumSaturatedRings,
        CalcNumSpiroAtoms,
        CalcNumUnspecifiedAtomStereoCenters,
        CalcTPSA,
    ]
    descriptors_mapping = {f.__name__.strip("Calc"): f for f in descriptors}

    feats = dict()
    for name, func in descriptors_mapping.items():
        feats[name] = [func(m) for m in df[mols_column_name]]
    return pd.DataFrame(feats)


@pf.register_dataframe_method
@deprecated_alias(mols_col="mols_column_name")
def maccs_keys_fingerprint(
    df: pd.DataFrame, mols_column_name: Hashable
) -> pd.DataFrame:
    """
    Convert a column of RDKIT mol objects into MACCS Keys Fingerprints.

    Returns a new dataframe without any of the original data.
    This is intentional to leave the user with the data requested.

    This method does not mutate the original DataFrame.

    Functional usage example:

    .. code-block:: python

        import pandas as pd
        import janitor.chemistry

        df = pd.DataFrame(...)

        maccs = janitor.chemistry.maccs_keys_fingerprint(
            df=df.smiles2mol('smiles', 'mols'),
            mols_column_name='mols'
        )

    Method chaining usage example:

    .. code-block:: python

        import pandas as pd
        import janitor.chemistry

        df = pd.DataFrame(...)

        maccs = (
            df.smiles2mol('smiles', 'mols')
              .maccs_keys_fingerprint(mols_column_name='mols')
        )

    If you wish to join the maccs keys fingerprints back into the
    original dataframe, this can be accomplished by doing a `join`,
    because the indices are preserved:

    .. code-block:: python

        joined = df.join(maccs_keys_fingerprint)


    :param df: A pandas DataFrame.
    :param mols_column_name: The name of the column that has the RDKIT mol
        objects.
    :returns: A new pandas DataFrame of MACCS keys fingerprints.
    """

    maccs = [GetMACCSKeysFingerprint(m) for m in df[mols_column_name]]

    np_maccs = []

    for macc in maccs:
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(macc, arr)
        np_maccs.append(arr)
    np_maccs = np.vstack(np_maccs)
    fmaccs = pd.DataFrame(np_maccs)
    fmaccs.index = df.index
    return fmaccs
