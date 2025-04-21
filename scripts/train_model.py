import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

def train_and_tune_model(data_split_path, model_output_path):
    # Load data splits
    X_train, X_test, y_train, y_test = joblib.load(data_split_path)

    # Define model and hyperparameter search space
    base_model = RandomForestRegressor(random_state=42)
    param_distributions = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_distributions,
        n_iter=10,
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    # Fit and find best model
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    # Save best model
    os.makedirs(model_output_path, exist_ok=True)
    joblib.dump(best_model, os.path.join(model_output_path, 'best_model.joblib'))

if __name__ == "__main__":
    train_and_tune_model("processed/train_test_split.joblib", "models")
