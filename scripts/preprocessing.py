import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

def preprocess_and_split_data(input_csv_path, output_dir):
    # Read the dataset
    df = pd.read_csv(input_csv_path)

    # Drop unwanted columns (like 'dteday')
    if 'dteday' in df.columns:
        df = df.drop('dteday', axis=1)

    # Separate features and target
    X = df.drop('cnt', axis=1)
    y = df['cnt']

    # One-hot encode categorical features
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols)

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Save the splits
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump((X_train, X_test, y_train, y_test), os.path.join(output_dir, 'train_test_split.joblib'))
    joblib.dump(scaler, os.path.join(output_dir, 'feature_scaler.joblib'))

if __name__ == "__main__":
    preprocess_and_split_data("data/bike_rental.csv", "processed")
