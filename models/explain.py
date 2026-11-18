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

 save_shap_importance(shap_values, X_sample)

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




