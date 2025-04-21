import joblib
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import os

def load_and_predict(model_path, data_split_path, output_path):
    # Load trained model
    model = joblib.load(model_path)

    # Load data splits
    X_train, X_test, y_train, y_test = joblib.load(data_split_path)

    # Make predictions
    predictions = model.predict(X_test)

    # Save predictions
    os.makedirs(output_path, exist_ok=True)
    predictions_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': predictions
    })
    predictions_df.to_csv(os.path.join(output_path, 'predictions.csv'), index=False)

if __name__ == "__main__":
    load_and_predict("models/best_model.joblib", "processed/train_test_split.joblib", "predictions")
