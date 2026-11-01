
import pandas as pd
import numpy as np
import pickle
import os
import warnings

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings('ignore')


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


