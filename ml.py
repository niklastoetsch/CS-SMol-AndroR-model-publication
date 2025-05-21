import tqdm

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier


PIPELINE_REGISTRY = {}


def create_pipeline():
    return Pipeline(steps=[
        ('classifier', CatBoostClassifier(auto_class_weights='Balanced', verbose=False)),
])


def run_cv(X, y, groups=None, cv_spec=StratifiedGroupKFold(n_splits=5), training_name=""):
    splits = []

    if isinstance(cv_spec, StratifiedGroupKFold):
        split_obj = cv_spec.split(X, y, groups=groups)
    else:
        split_obj = cv_spec.split(X, y)

    for i, indeces in tqdm.tqdm(enumerate(split_obj), total=5):
        train_index, val_index = indeces
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
        
        # Fit the pipeline
        pipeline = create_pipeline()
        PIPELINE_REGISTRY[f"{training_name}_{i}"] = pipeline
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
