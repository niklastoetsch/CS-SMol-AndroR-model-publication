"""
Utility Functions for Molecular Feature Generation and Clustering

This module provides functions for generating molecular fingerprints, calculating
molecular descriptors, and performing clustering analysis on chemical datasets.
"""

import pandas as pd
import numpy as np
from typing import List, Union
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors
from rdkit.SimDivFilters import rdSimDivPickers
import tqdm


# Configuration constants
N_BITS = 1024  # Number of bits for Morgan fingerprints
FINGERPRINT_RADIUS = 3  # Radius for ECFP6 fingerprints (radius=3 → ECFP6)
FP_COLUMNS = [f"fp_{i}" for i in range(N_BITS)]  # Column names for fingerprint features


def assignPointsToClusters(picks: List[int], picks_fps: List, fps: List) -> np.ndarray:
    """
    Assign molecules to clusters based on similarity to cluster centroids.
    
    Parameters
    ----------
    picks : List[int]
        Indices of selected cluster centroids
    picks_fps : List
        Fingerprints of cluster centroids
    fps : List
        All molecular fingerprints to be assigned
        
    Returns
    -------
    np.ndarray
        Cluster assignments for each molecule
    """
    # Initialize the similarity matrix
    num_picks = len(picks)
    sims = np.zeros((num_picks, len(fps)))

    # Calculate similarities only for the selected picks
    for i in range(num_picks):
        sims[i, :] = DataStructs.BulkTanimotoSimilarity(picks_fps[i], fps)

    closest_centroids = np.argmax(sims, axis=0)
    return closest_centroids


def get_cluster_assignments_from_fps(fps: List, tc: float = 0.65, chunk_size: int = 5000) -> List[int]:
    """
    Perform leader clustering on molecular fingerprints.
    
    Uses the leader clustering algorithm to group molecules based on Tanimoto
    similarity, then assigns all molecules to their closest cluster centroid.
    
    Parameters
    ----------
    fps : List
        List of molecular fingerprints (RDKit bit vectors)
    tc : float, default=0.65
        Tanimoto similarity threshold for clustering
    chunk_size : int, default=5000
        Process molecules in chunks to manage memory usage
        
    Returns
    -------
    List[int]
        Cluster assignment for each molecule
        
    Notes
    -----
    The leader clustering algorithm selects cluster centroids greedily,
    choosing molecules that are not similar (below threshold) to any
    existing centroid.
    """
    lp = rdSimDivPickers.LeaderPicker()
    picks = lp.LazyBitVectorPick(fps, len(fps), tc)
    picks_fps = [fps[x] for x in picks]
    results = []

    # assign points to clusters in chunks in order to avoid blowing up the memory
    for i in range(0, len(fps), chunk_size):
        if (i + chunk_size) <= len(fps):
            i_end = i + chunk_size
        else:
            i_end = len(fps)
        chunk = fps[i:i_end]
        result = assignPointsToClusters(picks, picks_fps, chunk) 
        results.extend(result)
    return results


def get_fingerprints(df: pd.DataFrame, smiles_column: str = "flat_smiles") -> List:
    """
    Generate Morgan fingerprints for molecules in a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing SMILES strings
    smiles_column : str, default="flat_smiles"
        Name of column containing SMILES strings
        
    Returns
    -------
    List
        List of RDKit bit vector fingerprints
        
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'flat_smiles': ['CCO', 'CC(=O)O']})
    >>> fps = get_fingerprints(df)
    >>> len(fps) == len(df)
    True
    """
    fingerprints = []
    for smiles in df[smiles_column]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        curr_fps = AllChem.GetMorganFingerprintAsBitVect(mol, FINGERPRINT_RADIUS, N_BITS)
        fingerprints.append(curr_fps)

    return fingerprints


def add_fingerprints_to_df(df: pd.DataFrame, smiles_column: str = "flat_smiles") -> pd.DataFrame:
    """
    Add Morgan fingerprint columns to a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing SMILES strings
    smiles_column : str, default="flat_smiles"
        Name of column containing SMILES strings
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional fingerprint columns (fp_0, fp_1, ..., fp_1023)
        
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'flat_smiles': ['CCO']})
    >>> df_with_fps = add_fingerprints_to_df(df)
    >>> 'fp_0' in df_with_fps.columns
    True
    """
    fingerprints = get_fingerprints(df, smiles_column)
    fingerprints_df = pd.DataFrame(np.array(fingerprints), columns=FP_COLUMNS, index=df.index)
    df = pd.concat([df, fingerprints_df], axis=1)
    return df


def get_rdkit_descriptors(df: pd.DataFrame, SMILES_column: str = "flat_smiles") -> pd.DataFrame:
    """
    Calculate RDKit molecular descriptors for molecules in a DataFrame.
    
    Computes all available RDKit molecular descriptors including physicochemical
    properties, topological indices, and structural features.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing SMILES strings
    SMILES_column : str, default="flat_smiles"
        Name of column containing SMILES strings
        
    Returns
    -------
    pd.DataFrame
        DataFrame with molecular descriptors as columns
        
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'flat_smiles': ['CCO', 'CC(=O)O']})
    >>> descriptors = get_rdkit_descriptors(df)
    >>> 'MolWt' in descriptors.columns
    True
    
    Notes
    -----
    This function calculates over 200 molecular descriptors including:
    - Molecular weight and formula
    - Lipophilicity (LogP)
    - Hydrogen bond donors/acceptors
    - Topological polar surface area
    - Ring counts and aromaticity
    - And many more...
    """

    rdkit_descriptors = {}

    for idx, r in tqdm.tqdm(df.iterrows(), desc="Calculating RDKit descriptors", total=len(df)):
        mol = Chem.MolFromSmiles(r[SMILES_column])
        if mol is None:
            print(f"Error: Invalid SMILES string at index {idx}: {r[SMILES_column]}")
            continue
        rdkit_descriptors[idx] = pd.Series(Chem.Descriptors.CalcMolDescriptors(mol))

    rdkit_descriptors = pd.DataFrame(rdkit_descriptors).T

    return rdkit_descriptors
