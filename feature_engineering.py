# utils/feature_engineering.py
# Create meaningful features from raw time-series data

import pandas as pd
import numpy as np


def create_temporal_features(df):
    """
    Create calendar-based features from the datetime index.
    
    Features created:
    - day_of_week (0=Monday, 6=Sunday)
    - day_of_month (1-31)
    - month (1-12)
    - quarter (1-4)
    - is_weekend (0 or 1)
    - is_salary_day (1 if day is 1, 7, or >=28)
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with datetime index
    
    Returns:
    --------
    df : pd.DataFrame
        DataFrame with added temporal features
    """
    df = df.copy()
    
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    
    # Salary days in India: 1st, 7th, and last few days of month
    df['is_salary_day'] = df.index.day.isin([1, 7]).astype(int)
    df['is_month_end'] = (df.index.day >= 28).astype(int)
    
    print(f"✅ Temporal features created: day_of_week, day_of_month, month, quarter, is_weekend, is_salary_day, is_month_end")
    return df


def create_lag_features(df, target_col, lags=[1, 7, 14, 30]):
    """
    Create lag features from the target column.
    
    Why lag features?
    -----------------
    Cash demand today is influenced by recent demand.
    Lag-1 captures "yesterday effect", Lag-7 captures
    "same day last week", etc.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    target_col : str
        Column to create lags from (e.g., 'ATM_1')
    lags : list
        List of lag periods
    
    Returns:
    --------
    df : pd.DataFrame
        DataFrame with added lag columns
    """
    df = df.copy()
    
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    
    # Drop rows with NaN created by shifting
    df = df.dropna()
    
    print(f"✅ Lag features created for {target_col}: {lags}")
    return df


def create_rolling_features(df, target_col, windows=[7, 14]):
    """
    Create rolling statistics (average and std deviation).
    
    Why rolling features?
    ---------------------
    Rolling averages smooth out noise and capture recent trends.
    Rolling std dev captures volatility.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    target_col : str
        Column to create rolling features from
    windows : list
        Window sizes (days)
    
    Returns:
    --------
    df : pd.DataFrame
        DataFrame with added rolling features
    """
    df = df.copy()
    
    for window in windows:
        df[f'{target_col}_rolling_mean_{window}'] = (
            df[target_col].rolling(window=window).mean()
        )
        df[f'{target_col}_rolling_std_{window}'] = (
            df[target_col].rolling(window=window).std()
        )
    
    df = df.dropna()
    print(f"✅ Rolling features created for {target_col}: windows={windows}")
    return df


def create_expanding_features(df, target_col):
    """
    Create expanding mean (cumulative average up to that point).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    target_col : str
        Column to create expanding feature from
    
    Returns:
    --------
    df : pd.DataFrame
        DataFrame with added expanding mean
    """
    df = df.copy()
    df[f'{target_col}_expanding_mean'] = df[target_col].expanding().mean()
    print(f"✅ Expanding mean created for {target_col}")
    return df


def create_all_features(df, target_col):
    """
    Master function to create all features for a single ATM.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw DataFrame with datetime index
    target_col : str
        ATM column to forecast (e.g., 'ATM_1')
    
    Returns:
    --------
    df_features : pd.DataFrame
        DataFrame with all engineered features
    feature_cols : list
        List of feature column names (for model training)
    """
    print("=" * 50)
    print(f"CREATING FEATURES FOR: {target_col}")
    print("=" * 50)
    
    # Start with target column only
    df_features = df[[target_col]].copy()
    
    # Add temporal features
    df_features = create_temporal_features(df_features)
    
    # Add lag features
    df_features = create_lag_features(df_features, target_col, lags=[1, 7, 14, 30])
    
    # Add rolling features
    df_features = create_rolling_features(df_features, target_col, windows=[7, 14])
    
    # Add expanding feature
    df_features = create_expanding_features(df_features, target_col)
    
    # List of feature columns (everything except target)
    feature_cols = [col for col in df_features.columns if col != target_col]
    
    print(f"\n✅ Total features created: {len(feature_cols)}")
    print(f"Feature columns: {feature_cols}")
    print(f"Final shape: {df_features.shape}")
    print("=" * 50)
    
    return df_features, feature_cols


def prepare_train_test_split(df_features, target_col, test_size=0.2):
    """
    Split data into training and testing sets.
    
    IMPORTANT: Time series data must NOT be shuffled!
    We use chronological split (first 80% train, last 20% test).
    
    Parameters:
    -----------
    df_features : pd.DataFrame
        DataFrame with features
    target_col : str
        Target column name
    test_size : float
        Fraction of data for testing (default 0.2 = 20%)
    
    Returns:
    --------
    X_train, X_test, y_train, y_test : pd.DataFrame / pd.Series
        Train and test splits
    """
    # Calculate split point
    split_idx = int(len(df_features) * (1 - test_size))
    
    # Features (X) and target (y)
    feature_cols = [col for col in df_features.columns if col != target_col]
    
    X = df_features[feature_cols]
    y = df_features[target_col]
    
    # Chronological split
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    print(f"✅ Train-test split (chronological):")
    print(f"   Training: {len(X_train)} samples ({100*(1-test_size):.0f}%)")
    print(f"   Testing:  {len(X_test)} samples ({100*test_size:.0f}%)")
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    print("Testing feature engineering with sample data...\n")
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    
    sample_df = pd.DataFrame({
        'ATM_1': np.random.randint(30000, 80000, 200).astype(float)
    }, index=dates)
    
    # Create all features
    df_features, feature_cols = create_all_features(sample_df, 'ATM_1')
    
    # Test train-test split
    X_train, X_test, y_train, y_test = prepare_train_test_split(
        df_features, 'ATM_1', test_size=0.2
    )
    
    print(f"\n✅ Feature engineering test successful!")
    print(f"\nSample of features:")
    print(df_features.head(3))
