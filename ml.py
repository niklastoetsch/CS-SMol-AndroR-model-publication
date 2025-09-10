import os
import numpy as np
import pickle
import pandas as pd

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


from sklearn.svm import SVC
from catboost import CatBoostClassifier, Pool, cv
from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
#from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler

def create_pipeline_model(model='gbt',class_weights=None):
    
    if model== 'xgb':
        return Pipeline(steps=[('classifier', XGBClassifier(objective='binary:logistic', missing=-999, seed=42, sample_weight=class_weights)),])

    if model== 'svm':
        return Pipeline(steps=[('classifier', SVC(kernel='rbf', class_weight=class_weights,probability=True)),])
    
    if model == 'catboost':
        return Pipeline(steps=[('classifier', CatBoostClassifier(random_seed=42, logging_level="Silent", iterations=150,class_weights=class_weights)),])
    
    if model == 'lr':
        return Pipeline(steps=[('classifier', LogisticRegression(class_weight=class_weights, max_iter=1000)),])
    
    if model == 'rf':
        return Pipeline(steps=[('classifier', RandomForestClassifier(class_weight=class_weights, n_estimators=50, random_state=42)),])

def create_pipeline_model_scaled(model='gbt',class_weights=None):
    
    if model== 'xgb':
        return Pipeline(steps=[('scaler', StandardScaler()),('classifier', XGBClassifier(objective='binary:logistic', missing=-999, seed=42, sample_weight=class_weights)),])

    if model== 'svm':
        return Pipeline(steps=[('scaler', StandardScaler()),('classifier', SVC(kernel='rbf', class_weight=class_weights,probability=True)),])
    
    if model == 'catboost':
        return Pipeline(steps=[('scaler', StandardScaler()),('classifier', CatBoostClassifier(random_seed=42, logging_level="Silent", iterations=150,class_weights=class_weights)),])
    
    if model == 'lr':
        return Pipeline(steps=[('scaler', StandardScaler()),('classifier', LogisticRegression(class_weight=class_weights, max_iter=1000)),])
    
    if model == 'rf':
        return Pipeline(steps=[('scaler', StandardScaler()),('classifier', RandomForestClassifier(class_weight=class_weights, n_estimators=50, random_state=42)),])

    if model == 'gbt':
        return Pipeline(steps=[('scaler', StandardScaler()),('classifier', GradientBoostingClassifier(class_weight=class_weights, random_state=42)),])


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


def run_cv_model(X, y, groups=None, n_splits=5, n_repetitions=5, training_name="",model='gbt'):
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
            if model == 'gbt':
                pipeline = create_pipeline()
                PIPELINE_REGISTRY[f"{training_name}_{i}.{ii}"] = pipeline
                pipeline.fit(X_train_fold, y_train_fold, classifier__sample_weight=sample_weights)
            else:
                classes = np.unique(y_train_fold)
                weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_fold)
                class_weights = dict(zip(classes, weights))
                pipeline = create_pipeline_model(model,class_weights)
                PIPELINE_REGISTRY[f"{training_name}_{i}.{ii}"] = pipeline
                pipeline.fit(X_train_fold, y_train_fold)
            
            
            # Make predictions
            y_pred = pipeline.predict(X_val_fold)
            y_hat_proba = pipeline.predict_proba(X_val_fold)

            splits.append(
                {"y": y_val_fold, 
                "y_hat": y_pred, 
                "y_hat_proba": y_hat_proba, 
                "val_index": val_index})

    return splits


def run_cv_model_scaled(X, y, groups=None, n_splits=5, n_repetitions=5, training_name="",model='gbt'):
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
            if model == 'gbt':
                pipeline = create_pipeline()
                PIPELINE_REGISTRY[f"{training_name}_{i}.{ii}"] = pipeline
                pipeline.fit(X_train_fold, y_train_fold, classifier__sample_weight=sample_weights)
            else:
                classes = np.unique(y_train_fold)
                weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_fold)
                class_weights = dict(zip(classes, weights))
                pipeline = create_pipeline_model_scaled(model,class_weights)
                PIPELINE_REGISTRY[f"{training_name}_{i}.{ii}"] = pipeline
                pipeline.fit(X_train_fold, y_train_fold)
            
            
            # Make predictions
            y_pred = pipeline.predict(X_val_fold)
            y_hat_proba = pipeline.predict_proba(X_val_fold)

            splits.append(
                {"y": y_val_fold, 
                "y_hat": y_pred, 
                "y_hat_proba": y_hat_proba, 
                "val_index": val_index})

    return splits

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, \
    precision_recall_curve, auc, make_scorer, recall_score, balanced_accuracy_score, accuracy_score, precision_score, \
    f1_score, matthews_corrcoef

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import cross_validate

def run_cv_TPO(X, y, groups=None, n_splits=5, n_repetitions=5, training_name=""):
    splits = []

    scoring = {'accuracy': make_scorer(accuracy_score),
                   'balanced_accuracy': make_scorer(balanced_accuracy_score),
                   'MCC': make_scorer(matthews_corrcoef),
                   'f1_score': make_scorer(f1_score),
                   'precision': make_scorer(precision_score),
                   'recall': make_scorer(recall_score),
                   'NPV': make_scorer(precision_score, pos_label=0),
                   'TNR': make_scorer(recall_score, pos_label=0)
                   }

    results_all = pd.DataFrame()

    for i in tqdm.tqdm(range(n_repetitions), total=n_repetitions, desc=f"Repetitions"):
        cv_spec = StratifiedGroupKFold(n_splits=n_splits, random_state=i, shuffle=True)
        #split_obj = cv_spec.split(X, y, groups=groups)

        sample_weights = np.ones_like(y, dtype=float)
        classes = np.unique(y)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)

        for idx, cls in enumerate(classes):
            sample_weights[y == cls] = weights[idx]

        gbt = GradientBoostingClassifier()    

        results = cross_validate(estimator=gbt,
                                    X=X,
                                    y=y,
                                    cv=cv_spec,
                                    scoring=scoring,
                                    params={'sample_weight': sample_weights}, groups=groups)
        
        result_df = pd.DataFrame.from_dict(results)
        result_df['model'] = 'gbt'
        results_all =  pd.concat([results_all,result_df], ignore_index=True)
  

    return results_all

# function to export metrics from each CV run, instead of predictions for validation set
def run_or_retrieve_from_disc_TPO(X, y, groups=None, n_splits=5, n_repetitions=5, training_name="", folder="."):
    """
    Run the cross-validation and save the results to disk.
    If the results already exist on disk, load them instead of running the cross-validation again.
    """

    results = run_cv_TPO(X, y, groups=groups, n_splits=n_splits, n_repetitions=n_repetitions, training_name=training_name)
    return results

def run_or_retrieve_from_disc_model(X, y, groups=None, n_splits=5, n_repetitions=5, training_name="", folder=".",model='gbt'):
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
        splits = run_cv_model(X, y, groups=groups, n_splits=n_splits, n_repetitions=n_repetitions, training_name=training_name,model=model)
        pipelines_created = {k: v for k, v in PIPELINE_REGISTRY.items() if k.startswith(training_name)}
        with open(results_filename, 'wb') as f:
            pickle.dump(splits, f)
        with open(pipelines_filename, 'wb') as f:
            pickle.dump(pipelines_created, f)

    return splits, pipelines_created

def run_or_retrieve_from_disc_model_scaled(X, y, groups=None, n_splits=5, n_repetitions=5, training_name="", folder=".",model='gbt'):
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
        splits = run_cv_model_scaled(X, y, groups=groups, n_splits=n_splits, n_repetitions=n_repetitions, training_name=training_name,model=model)
        pipelines_created = {k: v for k, v in PIPELINE_REGISTRY.items() if k.startswith(training_name)}
        with open(results_filename, 'wb') as f:
            pickle.dump(splits, f)
        with open(pipelines_filename, 'wb') as f:
            pickle.dump(pipelines_created, f)

    return splits, pipelines_created

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
