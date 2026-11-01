
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


def train_random_forest(X_train, X_val, X_test, y_train, y_val, y_test):
    print("\n" + "="*50)
    print("TRAINING RANDOM FOREST")
    print("="*50)
    
    rf = RandomForestRegressor(
        n_estimators=100, max_depth=15, min_samples_split=5, 
        min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    

    val_pred = rf.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_pred)
    

    y_pred = rf.predict(X_test)
    results = evaluate_model(y_test, y_pred, 'Random Forest')
    accuracy = r2_score(y_test, y_pred)
    
    print(f"Validation MAE: {val_mae:,.2f}")
    print(f"R² Accuracy Score: {accuracy*100:.2f}%")
    

    with open('models/saved_models/random_forest.pkl', 'wb') as f:
        pickle.dump(rf, f)
        
    return rf, y_pred, results, f"{accuracy*100:.2f}%"


def train_xgboost(X_train, X_val, X_test, y_train, y_val, y_test):
    print("\n" + "="*50)
    print("TRAINING XGBOOST")
    print("="*50)
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
    )
    
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

# ============================================================
# MODEL 4: HIST GRADIENT BOOSTING (Replaces LSTM)
# ============================================================
