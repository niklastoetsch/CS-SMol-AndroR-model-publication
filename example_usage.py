#!/usr/bin/env python3
"""
Example script demonstrating the androgen receptor prediction pipeline.

This script shows how to use the ML pipeline with minimal example data.
For full analysis, you will need the complete datasets referenced in the notebooks.
"""

import pandas as pd
import numpy as np
from ml import create_pipeline, run_cv
from utils import add_fingerprints_to_df, FP_COLUMNS
from analysis import Predictions


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
        # Random cluster (normally based on chemical similarity)
        cluster = np.random.randint(0, 5)
        
        data.append({
            'flat_smiles': smiles,
            'activity': activity,
            'cluster_id': cluster
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
    print("\nGenerating molecular fingerprints...")
    df = add_fingerprints_to_df(df)
    print(f"Added {len(FP_COLUMNS)} fingerprint features")
    
    # Prepare data for ML
    X = df[FP_COLUMNS]
    y = df['activity']
    groups = df['cluster_id']
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Number of clusters: {groups.nunique()}")
    
    # Test single model
    print("\nTesting single model...")
    pipeline = create_pipeline()
    
    # Simple train/test split for demonstration
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)
    
    # Evaluate predictions
    pred_obj = Predictions(y_test, y_pred, y_pred_proba)
    print(f"Test set prevalence: {pred_obj.prevalence:.3f}")
    
    from sklearn.metrics import classification_report, roc_auc_score
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    if len(np.unique(y_test)) > 1:  # Check if both classes present
        auc = roc_auc_score(y_test == 'inhibitor', y_pred_proba[:, 1])
        print(f"\nROC-AUC: {auc:.3f}")
    
    # Demonstrate cross-validation (reduced folds for speed)
    print("\nRunning cross-validation example...")
    print("(Using 2 folds, 1 repetition for speed)")
    
    try:
        cv_results = run_cv(
            X, y, groups=groups,
            n_splits=2, n_repetitions=1,
            training_name="example"
        )
        print(f"Cross-validation completed: {len(cv_results)} folds")
        
        # Simple aggregated performance
        all_y_true = np.concatenate([fold['y'] for fold in cv_results])
        all_y_pred = np.concatenate([fold['y_hat'] for fold in cv_results])
        
        print("\nCross-validation Classification Report:")
        print(classification_report(all_y_true, all_y_pred))
        
    except Exception as e:
        print(f"Cross-validation error (may occur with small example data): {e}")
    
    print("\n" + "=" * 40)
    print("Example completed successfully!")
    print("\nFor full analysis:")
    print("1. Obtain the complete datasets")
    print("2. Run the Jupyter notebooks")
    print("3. See METHODOLOGY.md for detailed explanations")


if __name__ == "__main__":
    main()