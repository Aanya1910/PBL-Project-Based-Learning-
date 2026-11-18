# models/evaluate.py
# Model evaluation and comparison

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pickle
import os


def load_comparison():
    """Load the model comparison results."""
    path = 'models/saved_models/model_comparison.csv'
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        print("❌ Comparison file not found. Run train_models.py first.")
        return None


