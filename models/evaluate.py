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


def create_comparison_chart(comparison):
    """
    Create a bar chart comparing all models across metrics.
    """
    fig = go.Figure()
    
    metrics = ['MAE', 'RMSE', 'sMAPE']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, metric in enumerate(metrics):
        fig.add_trace(go.Bar(
            name=metric,
            x=comparison['model'],
            y=comparison[metric],
            marker_color=colors[i],
            text=comparison[metric].round(2),
            textposition='auto'
        ))
    
    fig.update_layout(
        title='Model Comparison: MAE, RMSE, and sMAPE',
        xaxis_title='Model',
        yaxis_title='Error',
        barmode='group',
        height=500
    )
    
    fig.write_html('models/saved_models/comparison_chart.html')
    print("✅ Comparison chart saved")
    return fig


def identify_best_model(comparison):
    """
    Identify the best model based on sMAPE.
    """
    best = comparison.loc[comparison['sMAPE'].idxmin()]
    
    print("\n" + "=" * 50)
    print("BEST MODEL")
    print("=" * 50)
    print(f"Model: {best['model']}")
    print(f"MAE:   {best['MAE']:,.2f}")
    print(f"RMSE:  {best['RMSE']:,.2f}")
    print(f"sMAPE: {best['sMAPE']:.2f}%")
    
    # Check if it meets target
    if best['sMAPE'] < 15:
        print(f"\n✅ Target achieved! sMAPE below 15%")
    else:
        print(f"\n⚠️ Target not met. sMAPE above 15%")
    
    return best


if __name__ == "__main__":
    comparison = load_comparison()
    
    if comparison is not None:
        print(comparison.to_string(index=False))
        create_comparison_chart(comparison)
        best = identify_best_model(comparison)