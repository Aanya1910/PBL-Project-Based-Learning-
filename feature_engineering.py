# utils/feature_engineering.py
# Create meaningful features from raw time-series data

import pandas as pd
import numpy as np

def create_temporal_features(df):
    """Create calendar-based features from the datetime index."""
    df = df.copy()
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    df['is_salary_day'] = df.index.day.isin([1, 7]).astype(int)
    df['is_month_end'] = (df.index.day >= 28).astype(int)
    return df

def create_lag_features(df, target_col, lags=[1, 7, 14, 30]):
    """Create lag features from the target column."""
    df = df.copy()
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    return df.dropna()

def create_rolling_features(df, target_col, windows=[7, 14]):
    """Create rolling statistics."""
    df = df.copy()
    for window in windows:
        df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
        df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
    return df.dropna()

def create_expanding_features(df, target_col):
    """Create expanding mean."""
    df = df.copy()
    df[f'{target_col}_expanding_mean'] = df[target_col].expanding().mean()
    return df

def create_all_features(df, target_col):
    """Master function to create all features for a single ATM."""
    print(f"CREATING FEATURES FOR: {target_col}")
    df_features = df[[target_col]].copy()
    
    df_features = create_temporal_features(df_features)
    df_features = create_lag_features(df_features, target_col, lags=[1, 7, 14, 30])
    df_features = create_rolling_features(df_features, target_col, windows=[7, 14])
    df_features = create_expanding_features(df_features, target_col)
    
    feature_cols = [col for col in df_features.columns if col != target_col]
    print(f"✅ Total features created: {len(feature_cols)}")
    return df_features, feature_cols


def prepare_train_test_split(df_features, target_col, test_size=0.15, val_size=0.15):
    """
    Split data into Train (70%), Validation (15%), and Test (15%).
    """
    total_len = len(df_features)
    
    # Calculate split indices
    test_split_idx = int(total_len * (1 - test_size))
    val_split_idx = int(total_len * (1 - test_size - val_size))
    
    # Feature columns
    feature_cols = [col for col in df_features.columns if col != target_col]
    X = df_features[feature_cols]
    y = df_features[target_col]
    
    # Chronological split (No shuffling!)
    X_train = X.iloc[:val_split_idx]
    y_train = y.iloc[:val_split_idx]
    
    X_val = X.iloc[val_split_idx:test_split_idx]
    y_val = y.iloc[val_split_idx:test_split_idx]
    
    X_test = X.iloc[test_split_idx:]
    y_test = y.iloc[test_split_idx:]
    
    print(f"✅ 3-Way Split Complete:")
    print(f"   Training:   {len(X_train)} samples ({100*(1-test_size-val_size):.0f}%)")
    print(f"   Validation: {len(X_val)} samples ({100*val_size:.0f}%)")
    print(f"   Testing:    {len(X_test)} samples ({100*test_size:.0f}%)")
    
    # Save all three splits
    X_train.to_csv('data/X_train.csv', index=True)
    X_val.to_csv('data/X_val.csv', index=True)
    X_test.to_csv('data/X_test.csv', index=True)
    y_train.to_csv('data/y_train.csv', index=True)
    y_val.to_csv('data/y_val.csv', index=True)
    y_test.to_csv('data/y_test.csv', index=True)
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def run_feature_engineering_on_real_data(atm_column='T1'):
    """Run feature engineering on the cleaned NN5 data."""
    import os
    
    cleaned_path = 'data/nn5_cleaned.csv'
    if not os.path.exists(cleaned_path):
        print("❌ Cleaned data not found! Run preprocessing first.")
        return None, None
    
    df = pd.read_csv(cleaned_path, index_col=0, parse_dates=True)
    
    # Create features
    df_features, feature_cols = create_all_features(df, atm_column)
    
    # Save features
    df_features.to_csv(f'data/features_{atm_column}.csv')
    
    # Split data
    prepare_train_test_split(df_features, atm_column, test_size=0.15, val_size=0.15)
    
    # Create correlation heatmap
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(df_features.corr(), annot=False, cmap='coolwarm', center=0)
    plt.title(f'Feature Correlation Heatmap - {atm_column}')
    plt.tight_layout()
    plt.savefig(f'data/correlation_heatmap_{atm_column}.png', dpi=100)
    
    return df_features, feature_cols


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING FEATURE ENGINEERING ON CLEANED DATA")
    print("=" * 60)
    
    # You can change 'T1' to 'T2', 'T3', etc.
    df_features, feature_cols = run_feature_engineering_on_real_data(atm_column='T1')
    
    print("\n✅ Feature Engineering Complete. Train/Val/Test splits saved to data/")