# utils/preprocessing.py
# Data cleaning and preprocessing for ATM cash demand forecasting

import pandas as pd
import numpy as np


def load_nn5_dataset(filepath):
                 
    """
    Load the NN5 dataset from a .ts file.
    
    File format:
        T1:1996-03-18 00-00-00:13.407,14.725,20.564,...
        T2:1996-03-18 00-00-00:11.550,13.591,15.036,...
        ...
    
    Each line = one ATM's complete time series.
    The date at the start is the START date.
    Values after are consecutive daily observations.
    """
    try:
        series_dict = {}
        start_dates = {}
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('#'):   
                            continue  
                
                # Split into 3 parts: ATM_ID, START_DATE, VALUES
                parts = line.split(':', 2)
                if len(parts) != 3:
                    continue
                
                atm_id = parts[0].strip()          # e.g., "T1"
                date_str = parts[1].strip()        # e.g., "1996-03-18 00-00-00"
                values_str = parts[2].strip()      # e.g., "13.407,14.725,..."
                
                # Parse start date (take only the date part)
                start_date = pd.to_datetime(date_str.split()[0])
                start_dates[atm_id] = start_date
                
                # Parse values
                values = []
                for v in values_str.split(','):
                    v = v.strip()
                    if v == '?' or v == '':
                        values.append(np.nan)
                    else:
                        try:
                            values.append(float(v))
                        except ValueError:
                            values.append(np.nan)
                
                series_dict[atm_id] = values
        
        # Find the maximum length across all series
        max_len = max(len(v) for v in series_dict.values())
        
        # Use the earliest start date (they should all be the same)
        common_start = min(start_dates.values())
        
        # Create date index
        date_index = pd.date_range(start=common_start, periods=max_len, freq='D')
        
        # Build DataFrame: each column = one ATM
        df = pd.DataFrame(index=date_index)
        
        for atm_id, values in series_dict.items():
            # Pad with NaN if shorter than max_len
            padded = values + [np.nan] * (max_len - len(values))
            df[atm_id] = padded
        
        print(f"✅ Loaded dataset: {df.shape[0]} days × {df.shape[1]} ATMs")
        print(f"   Start date: {common_start.date()}")
        print(f"   ATM IDs: {list(df.columns[:5])}... (showing first 5)")

        valid_cols = [c for c in df.columns if c.startswith('T') and c[1:].isdigit()]
        df = df[valid_cols] 
        return df
        
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        return None
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None


def handle_missing_values(df):
    """
    Handle missing values using weekday median replacement.
    """
    df_clean = df.copy()
    
    # Group columns by weekday of the index
    for col in df_clean.columns:
        # Create a temporary series with weekday info
        temp = pd.DataFrame({
            'value': df_clean[col],
            'weekday': df_clean.index.dayofweek
        })
        
        # Fill NaN with weekday median
        temp['value'] = temp.groupby('weekday')['value'].transform(
            lambda x: x.fillna(x.median())
        )
        
        # If any NaN remain (entire weekday missing), fill with overall median
        temp['value'] = temp['value'].fillna(df_clean[col].median())
        
        df_clean[col] = temp['value'].values
    
    total_missing = df_clean.isna().sum().sum()
    print(f"✅ Missing values handled. Remaining NaN: {total_missing}")
    return df_clean


def detect_outliers_iqr(df, multiplier=1.5):
    """
    Detect outliers using the IQR method.
    
    IQR = Q3 - Q1
    Lower bound = Q1 - multiplier × IQR
    Upper bound = Q3 + multiplier × IQR
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    multiplier : float
        IQR multiplier (default 1.5, standard)
    
    Returns:
    --------
    outliers : pd.DataFrame
        Boolean DataFrame showing where outliers are
    """
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outliers = (df < lower_bound) | (df > upper_bound)
    
    total_outliers = outliers.sum().sum()
    print(f"✅ Outliers detected: {total_outliers} ({100*total_outliers/df.size:.2f}%)")
    return outliers


def cap_outliers(df, multiplier=1.5):
    """
    Cap outliers to the IQR bounds instead of removing them.
    
    Why cap and not remove?
    -----------------------
    Time series data is sequential. Removing rows breaks the
    timeline. Capping keeps the sequence intact.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    multiplier : float
        IQR multiplier
    
    Returns:
    --------
    df_capped : pd.DataFrame
        DataFrame with outliers capped
    """
    df_capped = df.copy()
    
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    # Clip values to bounds
    df_capped = df_capped.clip(lower=lower_bound, upper=upper_bound, axis=1)
    
    print(f"✅ Outliers capped to IQR bounds")
    return df_capped


def normalize_minmax(df):
    """
    Normalize data to [0, 1] range using Min-Max scaling.
    
    Why normalize?
    --------------
    LSTM and neural networks are sensitive to input scale.
    Different ATMs have different demand ranges. Normalization
    brings everything to the same scale.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    
    Returns:
    --------
    df_norm : pd.DataFrame
        Normalized DataFrame
    scalers : dict
        Dictionary of min/max values for each column (for inverse transform)
    """
    df_norm = df.copy()
    scalers = {}
    
    for col in df.columns:
        col_min = df[col].min()
        col_max = df[col].max()
        
        if col_max - col_min == 0:
            df_norm[col] = 0
        else:
            df_norm[col] = (df[col] - col_min) / (col_max - col_min)
        
        scalers[col] = {'min': col_min, 'max': col_max}
    
    print(f"✅ Data normalized to [0, 1] range")
    return df_norm, scalers


def inverse_normalize(series, scaler):
    """
    Convert normalized values back to original scale.
    
    Parameters:
    -----------
    series : pd.Series or np.array
        Normalized values
    scaler : dict
        Dictionary with 'min' and 'max' keys
    
    Returns:
    --------
    original : pd.Series or np.array
        Values in original scale
    """
    return series * (scaler['max'] - scaler['min']) + scaler['min']


def preprocess_pipeline(filepath, save_clean=True):
    """
    Complete preprocessing pipeline.
    
    Steps:
    1. Load dataset
    2. Handle missing values (weekday median)
    3. Cap outliers (IQR method)
    4. Save cleaned data
    
    Parameters:
    -----------
    filepath : str
        Path to NN5 dataset
    save_clean : bool
        Whether to save cleaned data to CSV
    
    Returns:
    --------
    df_clean : pd.DataFrame
        Cleaned DataFrame
    """
    print("=" * 50)
    print("STARTING PREPROCESSING PIPELINE")
    print("=" * 50)
    
    # Step 1: Load
    df = load_nn5_dataset(filepath)
    if df is None:
        return None
    
    # Step 2: Handle missing values
    df = handle_missing_values(df)
    
    # Step 3: Cap outliers
    df = cap_outliers(df)
    
    # Step 4: Save
    if save_clean:
        output_path = 'data/nn5_cleaned.csv'
        df.to_csv(output_path)
        print(f"✅ Cleaned data saved to: {output_path}")
    
    print("=" * 50)
    print("PREPROCESSING COMPLETE")
    print("=" * 50)
    
    return df

if __name__ == "__main__":
    # Test with sample data
    print("Testing preprocessing with sample data...")
    
    # Create sample data with missing values
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    sample_df = pd.DataFrame({
        'ATM_1': np.random.randint(30000, 80000, 100).astype(float),
        'ATM_2': np.random.randint(40000, 90000, 100).astype(float)
    }, index=dates)
    
    # Introduce some missing values
    sample_df.iloc[10, 0] = np.nan
    sample_df.iloc[25, 1] = np.nan
    sample_df.iloc[50, 0] = 500000  # Outlier
    
    print(f"\nOriginal shape: {sample_df.shape}")
    print(f"Missing values: {sample_df.isna().sum().sum()}")
    
    # Run pipeline steps
    df_clean = handle_missing_values(sample_df)
    df_clean = cap_outliers(df_clean)
    df_norm, scalers = normalize_minmax(df_clean)
    
    print(f"\n✅ Pipeline test successful!")
    print(f"Cleaned shape: {df_clean.shape}")
    print(f"Missing values after: {df_clean.isna().sum().sum()}")

    
# RUN ON REAL NN5 DATA

def run_preprocessing_on_real_data():
    """
    Run the complete preprocessing pipeline on the real NN5 dataset.
    This function should be called after the dataset is downloaded.
    """
    import os
    
    # Path to the NN5 dataset
    dataset_path = 'data/nn5_daily_dataset_without_missing_values.ts'
    
    # Check if file exists
    if not os.path.exists(dataset_path):
        print("❌ Dataset not found!")
        print(f"Please download the NN5 dataset and place it at: {dataset_path}")
        print("\nDownload link: https://zenodo.org/record/4656118")
        return None
    
    # Run the preprocessing pipeline
    df_clean = preprocess_pipeline(dataset_path, save_clean=True)
    
    if df_clean is not None:
        print("\n✅ Preprocessing complete!")
        print(f"Shape: {df_clean.shape}")
        print(f"Date range: {df_clean.index.min()} to {df_clean.index.max()}")
        print(f"\nFirst 5 rows:")
        print(df_clean.head())
        
        # Save a summary
        summary = {
            'total_records': df_clean.shape[0] * df_clean.shape[1],
            'num_atms': df_clean.shape[1],
            'num_days': df_clean.shape[0],
            'date_start': str(df_clean.index.min()),
            'date_end': str(df_clean.index.max()),
            'missing_values': df_clean.isna().sum().sum()
        }
        
        import json
        with open('data/preprocessing_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ Summary saved to: data/preprocessing_summary.json")
    
    return df_clean


if __name__ == "__main__":
    # When run directly, execute the real data preprocessing
    print("=" * 60)
    print("RUNNING PREPROCESSING ON REAL NN5 DATASET")
    print("=" * 60)
    df = run_preprocessing_on_real_data()