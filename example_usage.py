#!/usr/bin/env python3
"""
Example script demonstrating the androgen receptor prediction pipeline.

This script shows how to use the ML pipeline with minimal example data.
For full analysis, you will need the complete datasets referenced in the notebooks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report

from ml import run_or_retrieve_from_disc
from utils import add_fingerprints_to_df, FP_COLUMNS, get_fingerprints, get_cluster_assignments_from_fps
from analysis import CV, SHAPAnalyzer


def create_example_data(n_samples=50):
    """
    Create minimal example data for demonstration.
    
    Note: This is synthetic data for testing purposes only.
    Real analysis requires the full datasets.
    """
    np.random.seed(42)
    
    # Example SMILES strings (simple molecules)
    example_smiles = [
        'CCO', 'CC(=O)O', 'CCN', 'CCC', 'CC(C)O', 'CC(=O)N', 'CCC(=O)O',
        'CC(C)(C)O', 'CCCO', 'CC(=O)OC', 'CCN(C)C', 'c1ccccc1', 'CC(C)C',
        'CC(=O)C', 'CCCN', 'CC(O)C', 'CCC(C)O', 'CC(=O)NC', 'CCCC',
        'CC(C)CO'
    ]
    
    # Create dataset
    data = []
    for i in range(n_samples):
        smiles = np.random.choice(example_smiles)
        # Random activity (normally this would be experimental data)
        activity = np.random.choice(['inhibitor', 'inactive'], p=[0.3, 0.7])
        
        data.append({
            'flat_smiles': smiles,
            'activity': activity,
        })
    
    return pd.DataFrame(data)


def main():
    """Demonstrate the ML pipeline with example data."""
    
    print("CS-SMol AndroR Model - Example Usage")
    print("=" * 40)
    
    # Create example data
    print("Creating example dataset...")
    df = create_example_data(n_samples=100)
    print(f"Dataset size: {len(df)} molecules")
    print(f"Activity distribution:\n{df['activity'].value_counts()}")
    
    # Add molecular fingerprints
    df = add_fingerprints_to_df(df)
    print(f"Added {len(FP_COLUMNS)} fingerprint features")
    # Add cluster assignments based on Tanimoto threshold
    tanimoto_threshold = 0.65
    fps = get_fingerprints(df)
    df["cluster_065"] = get_cluster_assignments_from_fps(fps, tanimoto_threshold)
    
    # Prepare data for ML
    X = df[FP_COLUMNS]
    y = df['activity']
    groups = df['cluster_065']
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Number of clusters: {groups.nunique()}")
    
    # Test single model
    print("\nTesting Cross Validation ...")    
    example_splits, example_pipelines = run_or_retrieve_from_disc(
        X, y, groups=groups,
        n_splits=2, n_repetitions=1,
        training_name="example"
    )
    
    cv_obj = CV(example_splits)
    print(f"Cross-validation completed: {len(cv_obj.folds)} folds")
    print(f"PLEASE NOTE: the results below are based on synthetic, randomly generated data and are expected to exhibit poor performance!")
    
    # Simple aggregated performance
    all_y_true = np.concatenate([fold['y'] for fold in example_splits])
    all_y_pred = np.concatenate([fold['y_hat'] for fold in example_splits])
    
    print("\nCross-validation Classification Report:")
    print(classification_report(all_y_true, all_y_pred))
    
    print("\nTop features by mean SHAP value:")
    shap_analysis = SHAPAnalyzer(example_splits, example_pipelines, df[FP_COLUMNS])
    print(shap_analysis.mean_shap_values.sort_values(ascending=False).head())

    print("\n" + "=" * 40)
    print("Example completed successfully!")
    print("\nFor full analysis:")
    print("1. Obtain the complete datasets")
    print("2. Run the Jupyter notebooks")
    print("3. See METHODOLOGY.md or the paper for detailed explanations")

if __name__ == "__main__":
    main()
