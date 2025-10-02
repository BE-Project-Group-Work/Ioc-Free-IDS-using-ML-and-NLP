import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
import joblib
import os
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

def train_log_model():
    """
    Loads preprocessed log data, trains an Isolation Forest model,
    evaluates it, and saves the model and results.
    """
    print("--- Starting Log Anomaly Detection Model Training ---")

    # --- Define Paths ---
    ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
    PROCESSED_DATA_PATH = os.path.join(ROOT_DIR, 'data', 'processed', 'hdfs_v1_feature_matrix_labeled.parquet')
    MODEL_OUTPUT_DIR = os.path.join(ROOT_DIR, 'initial_models')

    # --- Create Directories ---
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Data
    try:
        df = pd.read_parquet(PROCESSED_DATA_PATH)
        print(f"Successfully loaded data from '{PROCESSED_DATA_PATH}'.")
    except FileNotFoundError:
        print(f"Error: Processed file not found at '{PROCESSED_DATA_PATH}'.")
        print("Please run the 'preprocess' command first.")
        return

    # 2. Prepare Data
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Train only on normal data
    X_train = X[y == 0]
    X_test = X
    y_test = y
    print(f"Training data contains {len(X_train)} 'Normal' samples.")

    # 3. Train Model
    print("Training the Isolation Forest model...")
    iso_forest = IsolationForest(contamination='auto', random_state=42, n_jobs=-1)
    iso_forest.fit(X_train)
    print("Model training complete.")

    # 4. Evaluate Model
    print("\n--- Model Evaluation ---")
    y_pred_raw = iso_forest.predict(X_test)
    y_pred = [0 if p == 1 else 1 for p in y_pred_raw]

    print("Classification Report (0=Normal, 1=Anomaly):")
    print(classification_report(y_test, y_pred, digits=4))

    # 5. Save Model
    model_path = os.path.join(MODEL_OUTPUT_DIR, 'isolation_forest_log_model.joblib')
    joblib.dump(iso_forest, model_path)
    print(f"Model saved to '{model_path}'")

    print("\n--- Log model training finished successfully! ---")

if __name__ == '__main__':
    train_log_model()
