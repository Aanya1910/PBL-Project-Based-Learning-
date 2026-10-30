# models/train_models.py
# Train ARIMA, Random Forest, XGBoost, and LSTM models

import pandas as pd
import numpy as np
import pickle
import os
import warnings

# ML libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA

# Deep learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")

# Metrics

def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0
    return 100 * np.mean(diff)

def evaluate_model(y_true, y_pred, model_name):
    """Calculate all three metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    smape_val = smape(y_true, y_pred)
    
    results = {
        'model': model_name,
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'sMAPE': round(smape_val, 2)
    }
    
    print(f"\nMODEL: {model_name}")
    print(f"MAE:   {mae:,.2f}")
    print(f"RMSE:  {rmse:,.2f}")
    print(f"sMAPE: {smape_val:.2f}%")
    
    return results


# DATA LOADING

def load_prepared_data():
    """Load the train-validation-test splits."""
    X_train = pd.read_csv('data/X_train.csv', index_col=0, parse_dates=True)
    X_val = pd.read_csv('data/X_val.csv', index_col=0, parse_dates=True)
    X_test = pd.read_csv('data/X_test.csv', index_col=0, parse_dates=True)
    
    y_train = pd.read_csv('data/y_train.csv', index_col=0, parse_dates=True).squeeze()
    y_val = pd.read_csv('data/y_val.csv', index_col=0, parse_dates=True).squeeze()
    y_test = pd.read_csv('data/y_test.csv', index_col=0, parse_dates=True).squeeze()
    
    print(f"✅ Data loaded:")
    print(f"   Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test

# MODEL 1: ARIMA

def train_arima(y_train, y_test, order=(1, 1, 1)):
    print("\n" + "="*50)
    print("TRAINING ARIMA")
    print("="*50)
    try:
        model = ARIMA(y_train, order=order)
        fitted_model = model.fit()
        forecast = fitted_model.forecast(steps=len(y_test))
        results = evaluate_model(y_test, forecast, f'ARIMA{order}')
        
        os.makedirs('models/saved_models', exist_ok=True)
        with open('models/saved_models/arima_model.pkl', 'wb') as f:
            pickle.dump(fitted_model, f)
        return fitted_model, forecast, results
    except Exception as e:
        print(f"❌ ARIMA failed: {e}")
        return None, None, None

# MODEL 2: RANDOM FOREST

def train_random_forest(X_train, X_val, X_test, y_train, y_val, y_test):
    print("\n" + "="*50)
    print("TRAINING RANDOM FOREST")
    print("="*50)
    
    rf = RandomForestRegressor(
        n_estimators=100, max_depth=15, min_samples_split=5, 
        min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    
    # 1. Validation Performance
    val_pred = rf.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_pred)
    
    # 2. Test Performance
    y_pred = rf.predict(X_test)
    results = evaluate_model(y_test, y_pred, 'Random Forest')
    
    # 3. Calculate Accuracy (R-squared Score)
    accuracy = r2_score(y_test, y_pred)
    
    print(f"Validation MAE: {val_mae:,.2f} (Checking for overfitting)")
    print(f"R² Accuracy Score: {accuracy*100:.2f}%")
    
    # Save model & accuracy
    with open('models/saved_models/random_forest.pkl', 'wb') as f:
        pickle.dump(rf, f)
        
    importance = pd.DataFrame({
        'feature': X_train.columns, 'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    importance.to_csv('models/saved_models/rf_feature_importance.csv', index=False)
    
    return rf, y_pred, results, f"{accuracy*100:.2f}%"


# MODEL 3: XGBOOST

def train_xgboost(X_train, X_val, X_test, y_train, y_val, y_test):
    print("\n" + "="*50)
    print("TRAINING XGBOOST")
    print("="*50)
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
    )
    
    # Uses validation set for early stopping
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    y_pred = xgb_model.predict(X_test)
    results = evaluate_model(y_test, y_pred, 'XGBoost')
    accuracy = r2_score(y_test, y_pred)
    
    print(f"R² Accuracy Score: {accuracy*100:.2f}%")
    
    with open('models/saved_models/xgboost.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
        
    return xgb_model, y_pred, results, f"{accuracy*100:.2f}%"


# MODEL 4: LSTM

def train_lstm(X_train, X_val, X_test, y_train, y_val, y_test, sequence_length=14, epochs=50, batch_size=32):
    print("\n" + "="*50)
    print("TRAINING LSTM")
    print("="*50)
    
    def create_sequences(X, y, seq_length):
        X_seq, y_seq = [], []
        for i in range(seq_length, len(X)):
            X_seq.append(X.iloc[i-seq_length:i].values)
            y_seq.append(y.iloc[i])
        return np.array(X_seq), np.array(y_seq)
    
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length)
    
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(sequence_length, X_train.shape[1])),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Uses validation set for early stopping
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=epochs, batch_size=batch_size,
        callbacks=[early_stop], verbose=0
    )
    
    y_pred = model.predict(X_test_seq).flatten()
    results = evaluate_model(y_test_seq, y_pred, 'LSTM')
    accuracy = r2_score(y_test_seq, y_pred)
    
    print(f"R² Accuracy Score: {accuracy*100:.2f}%")
    
    model.save('models/saved_models/lstm_model.h5')
    
    return model, y_pred, results, f"{accuracy*100:.2f}%"

# MAIN FUNCTION

def train_all_models():
    print("=" * 60)
    print("TRAINING ALL MODELS (WITH VALIDATION SPLIT)")
    print("=" * 60)
    
    X_train, X_val, X_test, y_train, y_val, y_test = load_prepared_data()
    all_results = []
    
    # ARIMA (No features, just time series)
    _, _, arima_results = train_arima(y_train, y_test)
    if arima_results:
        all_results.append(arima_results)
    
    # Random Forest
    _, _, rf_results, rf_acc = train_random_forest(X_train, X_val, X_test, y_train, y_val, y_test)
    if rf_results:
        rf_results['Accuracy_R2'] = rf_acc
        all_results.append(rf_results)
    
    # XGBoost
    _, _, xgb_results, xgb_acc = train_xgboost(X_train, X_val, X_test, y_train, y_val, y_test)
    if xgb_results:
        xgb_results['Accuracy_R2'] = xgb_acc
        all_results.append(xgb_results)
    
    # LSTM
    _, _, lstm_results, lstm_acc = train_lstm(X_train, X_val, X_test, y_train, y_val, y_test)
    if lstm_results:
        lstm_results['Accuracy_R2'] = lstm_acc
        all_results.append(lstm_results)
    
    # Comparison
    comparison = pd.DataFrame(all_results)
    comparison = comparison.sort_values('sMAPE')
    
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(comparison.to_string(index=False))
    
    comparison.to_csv('models/saved_models/model_comparison.csv', index=False)
    print(f"\n✅ Saved to: models/saved_models/model_comparison.csv")

if __name__ == "__main__":
    train_all_models()