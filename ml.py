import os
import numpy as np
import pickle

import tqdm

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_class_weight


PIPELINE_REGISTRY = {}


def create_pipeline():
    return Pipeline(steps=[
        ('classifier', GradientBoostingClassifier(verbose=False)),
])


def compute_sample_weights(y_train_fold):
    classes = np.unique(y_train_fold)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_fold)
    sample_weights = np.ones_like(y_train_fold, dtype=float)
    for idx, cls in enumerate(classes):
        sample_weights[y_train_fold == cls] = weights[idx]
    
    return sample_weights


def run_cv(X, y, groups=None, n_splits=5, n_repetitions=5, training_name=""):
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


def run_or_retrieve_from_disc(X, y, groups=None, n_splits=5, n_repetitions=5, training_name="", folder="."):
    """
    Run the cross-validation and save the results to disk.
    If the results already exist on disk, load them instead of running the cross-validation again.
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
