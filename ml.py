"""
Machine Learning Pipeline for Androgen Receptor Inhibition Prediction

This module provides the core machine learning functionality for predicting
chemical inhibitory activity against the androgen receptor using gradient
boosting classifiers with stratified group cross-validation.
"""

import os
import numpy as np
import pickle
from typing import Dict, List, Tuple, Any

import pandas as pd
import tqdm

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_class_weight


PIPELINE_REGISTRY = {}


def create_pipeline() -> Pipeline:
    """
    Create a machine learning pipeline for androgen receptor prediction.
    
    Returns
    -------
    Pipeline
        Scikit-learn pipeline with GradientBoostingClassifier
        
    Examples
    --------
    >>> pipeline = create_pipeline()
    >>> pipeline.named_steps.keys()
    dict_keys(['classifier'])
    """
    return Pipeline(steps=[
        ('classifier', GradientBoostingClassifier(verbose=False)),
    ])


def compute_sample_weights(y_train_fold: np.ndarray) -> np.ndarray:
    """
    Compute balanced sample weights for training data to handle class imbalance.
    
    Parameters
    ----------
    y_train_fold : np.ndarray
        Training labels for current cross-validation fold
        
    Returns
    -------
    np.ndarray
        Sample weights for balanced training, same shape as y_train_fold
        
    Examples
    --------
    >>> y = np.array(['inhibitor', 'inactive', 'inhibitor'])
    >>> weights = compute_sample_weights(y)
    >>> len(weights) == len(y)
    True
    """
    classes = np.unique(y_train_fold)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_fold)
    sample_weights = np.ones_like(y_train_fold, dtype=float)
    for idx, cls in enumerate(classes):
        sample_weights[y_train_fold == cls] = weights[idx]
    
    return sample_weights


def run_cv(X, y, groups=None, n_splits: int = 5, n_repetitions: int = 5, training_name: str = "") -> List[Dict[str, Any]]:
    """
    Run stratified group cross-validation for molecular data.
    
    Performs repeated stratified group k-fold cross-validation, ensuring that
    molecules from the same cluster are not split across train/test sets while
    maintaining class balance.
    
    Parameters
    ----------
    X : DataFrame
        Feature matrix (typically molecular fingerprints or descriptors)
    y : Series
        Target labels (e.g., 'inhibitor', 'inactive')
    groups : Series, optional
        Group labels for molecules (e.g., cluster IDs)
    n_splits : int, default=5
        Number of cross-validation folds
    n_repetitions : int, default=5
        Number of repetitions with different random seeds
    training_name : str, default=""
        Name prefix for storing trained pipelines
        
    Returns
    -------
    List[Dict[str, Any]]
        List of dictionaries containing validation results for each fold:
        - 'y': true labels
        - 'y_hat': predicted labels  
        - 'y_hat_proba': predicted probabilities
        - 'val_index': validation set indices
        
    Examples
    --------
    >>> import pandas as pd
    >>> X = pd.DataFrame({'fp_0': [1, 0, 1], 'fp_1': [0, 1, 1]})
    >>> y = pd.Series(['inhibitor', 'inactive', 'inhibitor'])
    >>> groups = pd.Series([0, 1, 0])
    >>> results = run_cv(X, y, groups, n_splits=2, n_repetitions=1)
    >>> len(results) == 2  # n_splits
    True
    """
    splits = []

    for i in tqdm.tqdm(range(n_repetitions), total=n_repetitions, desc=f"Repetitions"):
        cv_spec = StratifiedGroupKFold(n_splits=n_splits, random_state=i, shuffle=True)
        split_obj = cv_spec.split(X, y, groups=groups)
    
        for ii, indeces in tqdm.tqdm(enumerate(split_obj), total=n_splits, desc=f"Splits"):
            train_index, val_index = indeces
            X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
            y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

            sample_weights = compute_sample_weights(y_train_fold)

            # Fit the pipeline
            pipeline = create_pipeline()
            PIPELINE_REGISTRY[f"{training_name}_{i}.{ii}"] = pipeline
            pipeline.fit(X_train_fold, y_train_fold, classifier__sample_weight=sample_weights)
            
            # Make predictions
            y_pred = pipeline.predict(X_val_fold)
            y_hat_proba = pipeline.predict_proba(X_val_fold)

            splits.append(
                {"y": y_val_fold, 
                "y_hat": y_pred, 
                "y_hat_proba": y_hat_proba, 
                "val_index": val_index})

    return splits


def run_or_retrieve_from_disc(X, y, groups=None, n_splits: int = 5, n_repetitions: int = 5, 
                             training_name: str = "", folder: str = ".") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Run cross-validation or load results from disk if they exist.
    
    This function implements caching to avoid re-running expensive cross-validation
    when results already exist on disk.
    
    Parameters
    ----------
    X : DataFrame
        Feature matrix
    y : Series  
        Target labels
    groups : Series, optional
        Group labels for molecules
    n_splits : int, default=5
        Number of cross-validation folds
    n_repetitions : int, default=5
        Number of repetitions with different random seeds
    training_name : str, default=""
        Name prefix for file storage
    folder : str, default="."
        Directory to save/load results
        
    Returns
    -------
    Tuple[List[Dict[str, Any]], Dict[str, Any]]
        Cross-validation results and trained pipelines
        
    Notes
    -----
    Results are saved as pickle files:
    - splits_{training_name}.pkl: Cross-validation results
    - pipelines_{training_name}.pkl: Trained model pipelines
    """
    results_filename = f"{folder}/splits_{training_name}.pkl"
    pipelines_filename = f"{folder}/pipelines_{training_name}.pkl"
    if os.path.exists(results_filename):
        with open(results_filename, 'rb') as f:
            splits = pickle.load(f)
        with open(pipelines_filename, 'rb') as f:
            pipelines_created = pickle.load(f)
    else:
        splits = run_cv(X, y, groups=groups, n_splits=n_splits, n_repetitions=n_repetitions, training_name=training_name)
        pipelines_created = {k: v for k, v in PIPELINE_REGISTRY.items() if k.startswith(training_name)}
        with open(results_filename, 'wb') as f:
            pickle.dump(splits, f)
        with open(pipelines_filename, 'wb') as f:
            pickle.dump(pipelines_created, f)

    return splits, pipelines_created


def fit_model_or_retrieve_from_disc(pipeline_filename: str, X: pd.DataFrame, y: pd.Series):
    if os.path.exists(pipeline_filename):
        with open(pipeline_filename, 'rb') as f:
            fitted_pipeline = pickle.load(f)
    else:
        fitted_pipeline = create_pipeline()
        sample_weights_unrestricted = compute_sample_weights(y)
        fitted_pipeline.fit(
            X=X, 
            y=y, 
            classifier__sample_weight=sample_weights_unrestricted,
        )

        with open(pipeline_filename, 'wb') as f:
            pickle.dump(fitted_pipeline, f)

    return fitted_pipeline
