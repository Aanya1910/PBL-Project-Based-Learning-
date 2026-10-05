# utils/preprocessing.py
# Data cleaning and preprocessing for ATM cash demand forecasting

import pandas as pd
import numpy as np


def load_nn5_dataset(filepath):
    """
    Load the NN5 dataset from a .ts file.
    
    Parameters:
    -----------
    filepath : str
        Path to the .ts file
    
    Returns:
    --------
    df : pd.DataFrame
        DataFrame with datetime index and ATM columns
    """
    try:
        df = pd.read_csv(filepath, sep='\t', index_col=0)
        df.index = pd.to_datetime(df.index)
        print(f"✅ Loaded dataset: {df.shape[0]} days × {df.shape[1]} ATMs")
        return df
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        return None
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None


def handle_missing_values(df):
    """
    Handle missing values using weekday median replacement.
    
    Why weekday median?
    -------------------
    ATM cash demand follows weekly patterns. A missing Monday
    should be replaced with the median of all other Mondays,
    not the overall median.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with possible NaN values
    
    Returns:
    --------
    df_clean : pd.DataFrame
        DataFrame with no missing values
    """
    df_clean = df.copy()
    
    # Add weekday column temporarily
    df_clean['weekday'] = df_clean.index.dayofweek
    
    # For each ATM column, fill NaN with weekday median
    for col in df.columns:
        if col != 'weekday':
            df_clean[col] = df_clean.groupby('weekday')[col].transform(
                lambda x: x.fillna(x.median())
            )
    
    # Drop the temporary weekday column
    df_clean = df_clean.drop(columns=['weekday'])
    
    # If any NaN still remain (entire weekday missing), fill with overall median
    df_clean = df_clean.fillna(df_clean.median())
    
    print(f"✅ Missing values handled. Remaining NaN: {df_clean.isna().sum().sum()}")
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
