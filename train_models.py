# Train ARIMA, Random Forest, XGBoost, and LSTM models

import pandas as pd
import numpy as np
import pickle
import os
import json
from datetime import datetime

# ML libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA

# Deep learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
from tensorflow.keras.callbacks import EarlyStopping

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


# UTILITY: CALCULATE sMAPE

def smape(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0
    return 100 * np.mean(diff)


def evaluate_model(y_true, y_pred, model_name):
    """
    Calculate all three metrics for a model.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    smape_val = smape(y_true, y_pred)
    
    results = {
        'model': model_name,
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'sMAPE': round(smape_val, 2)
    }
    
    print(f"\n{'='*40}")
    print(f"MODEL: {model_name}")
    print(f"{'='*40}")
    print(f"MAE:   {mae:,.2f}")
    print(f"RMSE:  {rmse:,.2f}")
    print(f"sMAPE: {smape_val:.2f}%")
    
    return results


# STEP 1: LOAD DATA

def load_prepared_data():
    """Load the train-test splits created by Himani."""
    
    X_train = pd.read_csv('data/X_train.csv', index_col=0, parse_dates=True)
    X_test = pd.read_csv('data/X_test.csv', index_col=0, parse_dates=True)
    y_train = pd.read_csv('data/y_train.csv', index_col=0, parse_dates=True).squeeze()
    y_test = pd.read_csv('data/y_test.csv', index_col=0, parse_dates=True).squeeze()
    
    print(f"✅ Data loaded:")
    print(f"   X_train: {X_train.shape}")
    print(f"   X_test:  {X_test.shape}")
    print(f"   y_train: {y_train.shape}")
    print(f"   y_test:  {y_test.shape}")
    
    return X_train, X_test, y_train, y_test


# MODEL 1: ARIMA (Statistical Baseline)

def train_arima(y_train, y_test, order=(1, 1, 1)):
    """
    Train ARIMA model on the raw time series.
    
    Note: ARIMA doesn't use engineered features — it works
    directly on the target time series.
    
    Parameters:
    -----------
    y_train : pd.Series
        Training target values
    y_test : pd.Series
        Testing target values
    order : tuple
        (p, d, q) parameters for ARIMA
    """
    print("\n" + "="*50)
    print("TRAINING ARIMA MODEL")
    print("="*50)
    
    try:
        # Fit ARIMA
        model = ARIMA(y_train, order=order)
        fitted_model = model.fit()
        
        print(f"✅ ARIMA{order} fitted successfully")
        print(f"   AIC: {fitted_model.aic:.2f}")
        
        # Forecast
        forecast = fitted_model.forecast(steps=len(y_test))
        
        # Evaluate
        results = evaluate_model(y_test, forecast, f'ARIMA{order}')
        
        # Save model
        os.makedirs('models/saved_models', exist_ok=True)
        with open('models/saved_models/arima_model.pkl', 'wb') as f:
            pickle.dump(fitted_model, f)
        print(f"✅ ARIMA model saved")
        
        return fitted_model, forecast, results
        
    except Exception as e:
        print(f"❌ ARIMA training failed: {e}")
        return None, None, None

# MODEL 2: RANDOM FOREST

def train_random_forest(X_train, X_test, y_train, y_test):
    """
    Train Random Forest Regressor.
    
    Uses all engineered features (temporal, lag, rolling).
    """
    print("\n" + "="*50)
    print("TRAINING RANDOM FOREST")
    print("="*50)
    
    # Initialize model
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # Train
    rf.fit(X_train, y_train)
    print("✅ Random Forest trained")
    
    # Predict
    y_pred = rf.predict(X_test)
    
    # Evaluate
    results = evaluate_model(y_test, y_pred, 'Random Forest')
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 5 Features:")
    print(importance.head())
    
    # Save model
    with open('models/saved_models/random_forest.pkl', 'wb') as f:
        pickle.dump(rf, f)
    print(f"✅ Random Forest model saved")
    
    # Save feature importance
    importance.to_csv('models/saved_models/rf_feature_importance.csv', index=False)
    
    return rf, y_pred, results


# MODEL 3: XGBOOST

def train_xgboost(X_train, X_test, y_train, y_test):
    """
    Train XGBoost Regressor.
    
    Expected to be the best performer based on research.
    """
    print("\n" + "="*50)
    print("TRAINING XGBOOST")
    print("="*50)
    
    # Initialize model
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    # Train with early stopping
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    print("✅ XGBoost trained")
    
    # Predict
    y_pred = xgb_model.predict(X_test)
    
    # Evaluate
    results = evaluate_model(y_test, y_pred, 'XGBoost')
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 5 Features:")
    print(importance.head())
    
    # Save model
    with open('models/saved_models/xgboost.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
    print(f"✅ XGBoost model saved")
    
    # Save feature importance
    importance.to_csv('models/saved_models/xgb_feature_importance.csv', index=False)
    
    return xgb_model, y_pred, results

# MODEL 4: LSTM (Deep Learning)

def train_lstm(X_train, X_test, y_train, y_test, 
               sequence_length=14, epochs=50, batch_size=32):
    """
    Train LSTM model for time-series forecasting.
    
    LSTM needs data in 3D shape: (samples, time_steps, features).
    We create sequences of `sequence_length` days to predict next day.
    
    Parameters:
    -----------
    sequence_length : int
        Number of past days to use for prediction (default: 14)
    epochs : int
        Training epochs
    batch_size : int
        Batch size for training
    """
    print("\n" + "="*50)
    print("TRAINING LSTM")
    print("="*50)
    
    # Prepare sequences for LSTM
    def create_sequences(X, y, seq_length):
        X_seq, y_seq = [], []
        for i in range(seq_length, len(X)):
            X_seq.append(X.iloc[i-seq_length:i].values)
            y_seq.append(y.iloc[i])
        return np.array(X_seq), np.array(y_seq)
    
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length)
    
    print(f"✅ Sequences created:")
    print(f"   X_train: {X_train_seq.shape}")
    print(f"   X_test:  {X_test_seq.shape}")
    
    # Build LSTM model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(sequence_length, X_train.shape[1])),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.summary()
    
    # Early stopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_test_seq, y_test_seq),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )
    
    print("✅ LSTM trained")
    
    # Predict
    y_pred = model.predict(X_test_seq).flatten()
    
    # Evaluate
    results = evaluate_model(y_test_seq, y_pred, 'LSTM')
    
    # Save model
    model.save('models/saved_models/lstm_model.h5')
    print(f"✅ LSTM model saved")
    
    return model, y_pred, results


# ============================================================
# MAIN: TRAIN ALL MODELS
# ============================================================

def train_all_models():
    """
    Master function to train all four models and compare.
    """
    print("=" * 60)
    print("TRAINING ALL MODELS")
    print("=" * 60)
    
    # Load data
    X_train, X_test, y_train, y_test = load_prepared_data()
    
    # Store all results
    all_results = []
    
    # Train ARIMA
    _, _, arima_results = train_arima(y_train, y_test)
    if arima_results:
        all_results.append(arima_results)
    
    # Train Random Forest
    _, _, rf_results = train_random_forest(X_train, X_test, y_train, y_test)
    if rf_results:
        all_results.append(rf_results)
    
    # Train XGBoost
    _, _, xgb_results = train_xgboost(X_train, X_test, y_train, y_test)
    if xgb_results:
        all_results.append(xgb_results)
    
    # Train LSTM
    _, _, lstm_results = train_lstm(X_train, X_test, y_train, y_test)
    if lstm_results:
        all_results.append(lstm_results)
    
    # Create comparison table
    comparison = pd.DataFrame(all_results)
    comparison = comparison.sort_values('sMAPE')
    
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(comparison.to_string(index=False))
    
    # Save comparison
    comparison.to_csv('models/saved_models/model_comparison.csv', index=False)
    print(f"\n✅ Comparison saved to: models/saved_models/model_comparison.csv")
    
    return comparison


if __name__ == "__main__":
    comparison = train_all_models()
