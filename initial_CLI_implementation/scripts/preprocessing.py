import pandas as pd
import numpy as np
import os
import glob
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

def process_cicids_dataset(input_dir, output_dir):
    """
    Processes all CICIDS2017 CSV files from the raw data folder and saves a
    single cleaned Parquet file to the processed data folder.
    """
    print("--- Starting CICIDS2017 Data Processing ---")
    
    all_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not all_files:
        print(f"Error: No CSV files found in '{input_dir}'. Please check the path.")
        return
        
    print(f"Found {len(all_files)} CSV files to process.")
    
    list_of_cleaned_dfs = []

    for file_path in all_files:
        print(f"--> Processing: {os.path.basename(file_path)}")
        df = pd.read_csv(file_path, encoding='latin1') # Use latin1 encoding for safety
        
        df.columns = df.columns.str.strip()
        if 'Destination Port' in df.columns:
            df.drop(columns=['Destination Port'], inplace=True)
        
        df = df.loc[:, ~df.columns.duplicated()]
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        list_of_cleaned_dfs.append(df)

    print("\n--> Combining all files into a single master DataFrame...")
    master_df = pd.concat(list_of_cleaned_dfs, ignore_index=True)
    
    master_df.columns = master_df.columns.str.replace(' ', '_').str.lower()
    
    numeric_cols = master_df.select_dtypes(include=np.number).columns
    column_std = master_df[numeric_cols].std()
    zero_std_cols = column_std[column_std == 0].index.tolist()
    if zero_std_cols:
        master_df.drop(columns=zero_std_cols, inplace=True)
        print(f"    - Dropped {len(zero_std_cols)} zero-variance columns.")

    output_path = os.path.join(output_dir, 'cicids2017_cleaned_standardized.parquet')
    master_df.to_parquet(output_path)
    print(f"✅ Success! Network data saved to '{output_path}'")

def process_hdfs_dataset(input_dir, output_dir):
    """
    Processes the HDFS v1 feature matrix and saves a cleaned, labeled
    Parquet file to the processed data folder.
    """
    print("\n--- Starting HDFS Log Data Processing ---")
    matrix_file_path = os.path.join(input_dir, 'Event_occurrence_matrix.csv')
    
    try:
        df_matrix = pd.read_csv(matrix_file_path)
    except FileNotFoundError:
        print(f"Error: Could not find '{matrix_file_path}'. Please check the path.")
        return
        
    feature_columns = [col for col in df_matrix.columns if col.startswith('E')]
    X = df_matrix[feature_columns]
    y_text = df_matrix['Label']
    y = y_text.apply(lambda label: 0 if label == 'Success' else 1)
    
    final_hdfs_data = pd.concat([X, y.rename('label')], axis=1)

    output_path = os.path.join(output_dir, 'hdfs_v1_feature_matrix_labeled.parquet')
    final_hdfs_data.to_parquet(output_path)
    print(f"✅ Success! Log data saved to '{output_path}'")


if __name__ == '__main__':
    # Define relative paths based on the project structure
    # Assumes this script is in a 'scripts' folder at the project root
    ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
    
    NETWORK_DATA_INPUT_DIR = os.path.join(ROOT_DIR, 'data', 'raw', 'CICIDS2017')
    LOG_DATA_INPUT_DIR = os.path.join(ROOT_DIR, 'data', 'raw', 'HDFS_Logs_Dataset', 'HDFS_v1', 'preprocessed')
    PROCESSED_DATA_OUTPUT_DIR = os.path.join(ROOT_DIR, 'data', 'processed')

    # Create the output directory if it doesn't exist
    os.makedirs(PROCESSED_DATA_OUTPUT_DIR, exist_ok=True)
    
    process_cicids_dataset(NETWORK_DATA_INPUT_DIR, PROCESSED_DATA_OUTPUT_DIR)
    process_hdfs_dataset(LOG_DATA_INPUT_DIR, PROCESSED_DATA_OUTPUT_DIR)
