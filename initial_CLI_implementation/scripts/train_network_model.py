import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

def train_network_model():
    """
    Loads preprocessed network data, trains a Random Forest model,
    evaluates it, and saves the model and results to their respective folders.
    """
    print("--- Starting Network Model Training ---")

    # --- Define Paths ---
    ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
    PROCESSED_DATA_PATH = os.path.join(ROOT_DIR, 'data', 'processed', 'cicids2017_cleaned_standardized.parquet')
    MODEL_OUTPUT_DIR = os.path.join(ROOT_DIR, 'initial_models')
    RESULTS_OUTPUT_DIR = os.path.join(ROOT_DIR, 'results')

    # --- Create Directories ---
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)

    # 1. Load Data
    try:
        df = pd.read_parquet(PROCESSED_DATA_PATH)
        print(f"Successfully loaded data from '{PROCESSED_DATA_PATH}'.")
    except FileNotFoundError:
        print(f"Error: Processed file not found at '{PROCESSED_DATA_PATH}'.")
        print("Please run the 'preprocess' command first.")
        return

    # 2. Prepare data
    X = df.drop('label', axis=1)
    y_text = df['label']
    le = LabelEncoder()
    y = le.fit_transform(y_text)

    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    y_test_text = le.inverse_transform(y_test)

    # 4. Train Model
    print("Training the Random Forest model... (This may take a few minutes)")
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=1)
    rf_classifier.fit(X_train, y_train)
    print("Model training complete.")

    # 5. Evaluate Model
    print("\n--- Model Evaluation ---")
    y_pred = rf_classifier.predict(X_test)
    y_pred_text = le.inverse_transform(y_pred)
    
    print("Classification Report:")
    print(classification_report(y_test_text, y_pred_text, digits=4))

    # 6. Save Model Artifacts
    model_path = os.path.join(MODEL_OUTPUT_DIR, 'random_forest_network_model.joblib')
    encoder_path = os.path.join(MODEL_OUTPUT_DIR, 'network_label_encoder.joblib')
    joblib.dump(rf_classifier, model_path)
    joblib.dump(le, encoder_path)
    print(f"Model saved to '{model_path}'")
    print(f"Label encoder saved to '{encoder_path}'")
    
    print("\n--- Network model training finished successfully! ---")

if __name__ == '__main__':
    train_network_model()
