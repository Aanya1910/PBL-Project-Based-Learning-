# models/explain.py
# SHAP values for model explainability

import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt


def load_xgboost_model():
    """Load the trained XGBoost model."""
    with open('models/saved_models/xgboost.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


def compute_shap_values(model, X_test, max_samples=100):
    """
    Compute SHAP values for the XGBoost model.
    
    SHAP explains which features push predictions up or down.
    """
    print("Computing SHAP values...")
    
    # Use a sample for speed
    X_sample = X_test.iloc[:max_samples]
    
    # Create explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    print(f"✅ SHAP values computed for {len(X_sample)} samples")
    
    return explainer, shap_values, X_sample


def plot_shap_summary(shap_values, X_sample):
    """
    Create SHAP summary plot (beeswarm).
    """
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig('models/saved_models/shap_summary.png', dpi=100)
    print("✅ SHAP summary plot saved")
    plt.close()


def plot_shap_bar(shap_values, X_sample):
    """
    Create SHAP bar plot (global feature importance).
    """
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig('models/saved_models/shap_bar.png', dpi=100)
    print("✅ SHAP bar plot saved")
    plt.close()


def save_shap_importance(shap_values, X_sample):
    """
    Save SHAP feature importance as CSV.
    """
    importance = pd.DataFrame({
        'feature': X_sample.columns,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)
    
    importance.to_csv('models/saved_models/shap_importance.csv', index=False)
    print("✅ SHAP importance saved")
    print(importance.head())
    
    return importance


if __name__ == "__main__":
    # Load data
    X_test = pd.read_csv('data/X_test.csv', index_col=0, parse_dates=True)
    
    # Load model
    model = load_xgboost_model()
    
    # Compute SHAP
    explainer, shap_values, X_sample = compute_shap_values(model, X_test)
    
    # Create plots
    plot_shap_summary(shap_values, X_sample)
    plot_shap_bar(shap_values, X_sample)
    
    # Save importance
    importance = save_shap_importance(shap_values, X_sample)