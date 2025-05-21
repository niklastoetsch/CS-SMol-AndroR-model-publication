import tqdm

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier


PIPELINE_REGISTRY = {}


def create_pipeline():
    return Pipeline(steps=[
        ('classifier', CatBoostClassifier(auto_class_weights='Balanced', verbose=False)),
])


def run_cv(X, y, groups=None, n_splits=5, n_repetitions=5, training_name=""):
    splits = []

    for i in tqdm.tqdm(range(n_repetitions), total=n_repetitions, desc=f"Repetitions"):
        cv_spec = StratifiedGroupKFold(n_splits=n_splits)
        split_obj = cv_spec.split(X, y, groups=groups)
    
        for ii, indeces in tqdm.tqdm(enumerate(split_obj), total=n_splits, desc=f"Splits"):
            train_index, val_index = indeces
            X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
            y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
            
            # Fit the pipeline
            pipeline = create_pipeline()
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
