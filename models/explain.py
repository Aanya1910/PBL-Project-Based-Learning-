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


