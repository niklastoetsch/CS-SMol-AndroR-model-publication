import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.SimDivFilters import rdSimDivPickers


N_BITS = 1024
FINGERPRINT_RADIUS = 3 # for ecfp6 fingerprints
FP_COLUMNS = [f"fp_{i}" for i in range(N_BITS)]


def assignPointsToClusters(picks, picks_fps, fps):
    # Initialize the similarity matrix
    num_picks = len(picks)
    sims = np.zeros((num_picks, len(fps)))

    # Calculate similarities only for the selected picks
    for i in range(num_picks):
        sims[i, :] = DataStructs.BulkTanimotoSimilarity(picks_fps[i], fps)

    closest_centroids = np.argmax(sims, axis=0)
    return closest_centroids


def get_cluster_assignments_from_fps(fps, tc=0.65, chunk_size=5000):
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


def get_fingerprints(df: pd.DataFrame) -> list:
    fingerprints = []
    for smiles in df["flat_smiles"]:
        mol = Chem.MolFromSmiles(smiles)
        curr_fps = AllChem.GetMorganFingerprintAsBitVect(mol, FINGERPRINT_RADIUS, N_BITS)
        fingerprints.append(curr_fps)

    return fingerprints


def add_fingerprints_to_df(df: pd.DataFrame) -> pd.DataFrame:
    fingerprints = get_fingerprints(df)
    fingerprints_df = pd.DataFrame(np.array(fingerprints), columns=FP_COLUMNS, index=df.index)
    df = pd.concat([df, fingerprints_df], axis=1)
    return df
