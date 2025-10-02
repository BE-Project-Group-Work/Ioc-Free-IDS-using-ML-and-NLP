import argparse
import os
import pandas as pd
import joblib
import warnings

# --- Library Check ---
try:
    import openpyxl
except ImportError:
    print("Warning: 'openpyxl' is not installed. To read .xlsx files, please run: pip install openpyxl")

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning)

# --- Define Global Paths based on this script's location (project root) ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(ROOT_DIR, 'initial_models')
SCRIPTS_DIR = os.path.join(ROOT_DIR, 'scripts')

def check_and_create_dirs():
    """Ensures all necessary directories exist."""
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.join(ROOT_DIR, 'results'), exist_ok=True)

def preprocess_data():
    """Wrapper to run the entire preprocessing pipeline."""
    print("--- Running the full data preprocessing pipeline ---")
    script_path = os.path.join(SCRIPTS_DIR, 'preprocessing.py')
    if os.path.exists(script_path):
        os.system(f'python "{script_path}"')
    else:
        print(f"Error: '{script_path}' not found.")

def train_model(model_type):
    """Wrapper to run a specified model training script."""
    print(f"--- Training the {model_type} model ---")
    script_name = f'train_{model_type}_model.py'
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    if os.path.exists(script_path):
        os.system(f'python "{script_path}"')
    else:
        print(f"Error: '{script_path}' not found.")

def predict_network(input_file):
    """Makes a prediction on a network flow sample."""
    print("--- Making prediction on new network data ---")
    try:
        model_path = os.path.join(MODELS_DIR, 'random_forest_network_model.joblib')
        encoder_path = os.path.join(MODELS_DIR, 'network_label_encoder.joblib')
        training_data_path = os.path.join(PROCESSED_DATA_DIR, 'cicids2017_cleaned_standardized.parquet')

        model = joblib.load(model_path)
        le = joblib.load(encoder_path)
        
        train_df = pd.read_parquet(training_data_path)
        training_columns = train_df.drop('label', axis=1).columns
        
        if input_file.endswith('.csv'):
            new_data = pd.read_csv(input_file)
        elif input_file.endswith('.xlsx'):
            new_data = pd.read_excel(input_file)
        else:
            print("Error: Unsupported file format. Please provide a .csv or .xlsx file.")
            return
        
        new_data.columns = new_data.columns.str.strip().str.replace(' ', '_').str.lower()
        
        for col in training_columns:
            if col not in new_data.columns:
                new_data[col] = 0 
        
        new_data = new_data[training_columns]
        
        prediction_numeric = model.predict(new_data)
        prediction_proba = model.predict_proba(new_data)
        prediction_label = le.inverse_transform(prediction_numeric)
        confidence = prediction_proba[0][prediction_numeric[0]] * 100
        
        print("\n--- Prediction Result ---")
        print(f"Predicted Attack Type: {prediction_label[0]}")
        print(f"Confidence: {confidence:.2f}%")

    except FileNotFoundError as e:
        print(f"Error: A required file was not found. Please run preprocessing and training first. Details: {e}")
    except Exception as e:
        print(f"An error occurred during prediction: {e}")

def predict_logs(input_file):
    """Makes a prediction on a log event matrix sample."""
    print("--- Making prediction on new log data ---")
    try:
        model_path = os.path.join(MODELS_DIR, 'isolation_forest_log_model.joblib')
        training_data_path = os.path.join(PROCESSED_DATA_DIR, 'hdfs_v1_feature_matrix_labeled.parquet')
        
        model = joblib.load(model_path)
        
        train_df = pd.read_parquet(training_data_path)
        training_columns = train_df.drop('label', axis=1).columns
        
        if input_file.endswith('.csv'):
            new_data = pd.read_csv(input_file)
        elif input_file.endswith('.xlsx'):
            new_data = pd.read_excel(input_file)
        else:
            print("Error: Unsupported file format. Please provide a .csv or .xlsx file.")
            return

        for col in training_columns:
            if col not in new_data.columns:
                new_data[col] = 0
        
        new_data_features = new_data[training_columns]
        
        # --- UPGRADE: Get the raw anomaly score for better insight ---
        # decision_function returns a score. Lower (negative) scores are more anomalous.
        anomaly_score = model.decision_function(new_data_features)
        
        # .predict() is just a threshold on the score (score <= 0 is anomaly)
        prediction = model.predict(new_data_features)
        result = "Anomaly" if prediction[0] == -1 else "Normal"
        
        print("\n--- Prediction Result ---")
        print(f"Predicted Log Status: {result}")
        print(f"Anomaly Score: {anomaly_score[0]:.4f} (Negative scores are more likely to be anomalies)")

    except FileNotFoundError as e:
        print(f"Error: A required file was not found. Please run preprocessing and training first. Details: {e}")
    except Exception as e:
        print(f"An error occurred during prediction: {e}")


if __name__ == '__main__':
    check_and_create_dirs()
    parser = argparse.ArgumentParser(description="IoC-Free IDS using ML and NLP - Main Control")
    
    # --- THIS IS THE FIXED LINE ---
    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')

    parser_preprocess = subparsers.add_parser('preprocess', help='Run the full data preprocessing pipeline.')
    
    parser_train = subparsers.add_parser('train', help='Train a model.')
    parser_train.add_argument('model', choices=['network', 'logs', 'all'], help='Specify which model to train.')
    
    parser_predict = subparsers.add_parser('predict', help='Make a prediction on a new data sample.')
    parser_predict.add_argument('--model', required=True, choices=['network', 'logs'], help='Specify which model to use for prediction.')
    parser_predict.add_argument('--input', required=True, help='Path to the input CSV or XLSX file for prediction.')

    args = parser.parse_args()

    if args.command == 'preprocess':
        preprocess_data()
    elif args.command == 'train':
        if args.model == 'network' or args.model == 'all':
            train_model('network')
        if args.model == 'logs' or args.model == 'all':
            train_model('logs')
    elif args.command == 'predict':
        if not os.path.exists(args.input):
            print(f"Error: Input file not found at '{args.input}'")
        elif args.model == 'network':
            predict_network(args.input)
        elif args.model == 'logs':
            predict_logs(args.input)
























# import argparse
# import os
# import pandas as pd
# import joblib
# import warnings

# warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
# warnings.filterwarnings('ignore', category=FutureWarning)

# # --- Define Global Paths based on this script's location (project root) ---
# ROOT_DIR = os.path.dirname(__file__)
# PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'processed')
# MODELS_DIR = os.path.join(ROOT_DIR, 'initial_models')
# SCRIPTS_DIR = os.path.join(ROOT_DIR, 'scripts')

# def check_and_create_dirs():
#     """Ensures all necessary directories exist."""
#     os.makedirs(os.path.join(ROOT_DIR, 'data', 'processed'), exist_ok=True)
#     os.makedirs(MODELS_DIR, exist_ok=True)
#     os.makedirs(os.path.join(ROOT_DIR, 'results'), exist_ok=True)

# def preprocess_data():
#     """Wrapper to run the entire preprocessing pipeline."""
#     print("--- Running the full data preprocessing pipeline ---")
#     script_path = os.path.join(SCRIPTS_DIR, 'preprocessing.py')
#     if os.path.exists(script_path):
#         os.system(f'python "{script_path}"')
#     else:
#         print(f"Error: '{script_path}' not found.")

# def train_model(model_type):
#     """Wrapper to run a specified model training script."""
#     print(f"--- Training the {model_type} model ---")
#     script_name = f'train_{model_type}_model.py'
#     script_path = os.path.join(SCRIPTS_DIR, script_name)
#     if os.path.exists(script_path):
#         os.system(f'python "{script_path}"')
#     else:
#         print(f"Error: '{script_path}' not found.")

# def predict_network(input_file):
#     """Makes a prediction on a network flow sample."""
#     print("--- Making prediction on new network data ---")
#     try:
#         model_path = os.path.join(MODELS_DIR, 'random_forest_network_model.joblib')
#         encoder_path = os.path.join(MODELS_DIR, 'network_label_encoder.joblib')
#         training_data_path = os.path.join(PROCESSED_DATA_DIR, 'cicids2017_cleaned_standardized.parquet')

#         model = joblib.load(model_path)
#         le = joblib.load(encoder_path)
        
#         train_df = pd.read_parquet(training_data_path)
#         training_columns = train_df.drop('label', axis=1).columns
        
#         if input_file.endswith('.csv'):
#             new_data = pd.read_csv(input_file)
#         elif input_file.endswith('.xlsx'):
#             new_data = pd.read_excel(input_file)
#         else:
#             print("Error: Unsupported file format. Please provide a .csv or .xlsx file.")
#             return
        
#         new_data.columns = new_data.columns.str.strip().str.replace(' ', '_').str.lower()
        
#         for col in training_columns:
#             if col not in new_data.columns:
#                 new_data[col] = 0 
        
#         new_data = new_data[training_columns]
        
#         prediction_numeric = model.predict(new_data)
#         prediction_proba = model.predict_proba(new_data)
#         prediction_label = le.inverse_transform(prediction_numeric)
#         confidence = prediction_proba[0][prediction_numeric[0]] * 100
        
#         print("\n--- Prediction Result ---")
#         print(f"Predicted Attack Type: {prediction_label[0]}")
#         print(f"Confidence: {confidence:.2f}%")

#     except FileNotFoundError as e:
#         print(f"Error: A required file was not found. Please run preprocessing and training first. Details: {e}")
#     except Exception as e:
#         print(f"An error occurred during prediction: {e}")

# def predict_logs(input_file):
#     """Makes a prediction on a log event matrix sample."""
#     print("--- Making prediction on new log data ---")
#     try:
#         model_path = os.path.join(MODELS_DIR, 'isolation_forest_log_model.joblib')
#         model = joblib.load(model_path)
        
#         if input_file.endswith('.csv'):
#             new_data = pd.read_csv(input_file)
#         elif input_file.endswith('.xlsx'):
#             new_data = pd.read_excel(input_file)
#         else:
#             print("Error: Unsupported file format. Please provide a .csv or .xlsx file.")
#             return

#         feature_columns = [col for col in new_data.columns if col.startswith('E')]
#         if not feature_columns:
#             print("Error: Input file must contain event columns (e.g., E1, E2...).")
#             return
            
#         new_data_features = new_data[feature_columns]

#         prediction = model.predict(new_data_features)
#         result = "Anomaly" if prediction[0] == -1 else "Normal"
        
#         print("\n--- Prediction Result ---")
#         print(f"Predicted Log Status: {result}")

#     except FileNotFoundError:
#         print("Error: Model file not found. Please run training first.")
#     except Exception as e:
#         print(f"An error occurred during prediction: {e}")


# if __name__ == '__main__':
#     check_and_create_dirs()
#     parser = argparse.ArgumentParser(description="IoC-Free IDS using ML and NLP - Main Control")
#     subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')

#     parser_preprocess = subparsers.add_parser('preprocess', help='Run the full data preprocessing pipeline.')
    
#     parser_train = subparsers.add_parser('train', help='Train a model.')
#     parser_train.add_argument('model', choices=['network', 'logs', 'all'], help='Specify which model to train.')
    
#     parser_predict = subparsers.add_parser('predict', help='Make a prediction on a new data sample.')
#     parser_predict.add_argument('--model', required=True, choices=['network', 'logs'], help='Specify which model to use for prediction.')
#     parser_predict.add_argument('--input', required=True, help='Path to the input CSV or XLSX file for prediction.')

#     args = parser.parse_args()

#     if args.command == 'preprocess':
#         preprocess_data()
#     elif args.command == 'train':
#         if args.model == 'network' or args.model == 'all':
#             train_model('network')
#         if args.model == 'logs' or args.model == 'all':
#             train_model('logs')
#     elif args.command == 'predict':
#         if not os.path.exists(args.input):
#             print(f"Error: Input file not found at '{args.input}'")
#         elif args.model == 'network':
#             predict_network(args.input)
#         elif args.model == 'logs':
#             predict_logs(args.input)
