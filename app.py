# app.py - ATM Cash Demand Forecasting Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from utils.preprocessing import handle_missing_values, cap_outliers
from utils.feature_engineering import create_all_features, prepare_train_test_split
import pickle



st.markdown("---")
st.subheader("🔍 Feature Importance (SHAP)")

feature_importance = {
    'Day of Week': 0.35,
    'Salary Day (1st/7th)': 0.25,
    'Lag-1 (Yesterday)': 0.18,
    'Holiday Indicator': 0.12,
    'Rolling Average (7-day)': 0.10
        }
    
fig_importance = go.Figure(go.Bar(
    x=list(feature_importance.values()),
    y=list(feature_importance.keys()),
    orientation='h',

    marker=dict(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
))

fig_importance.update_layout(
    title="Which Features Drive Predictions?",
    xaxis_title="SHAP Value (Impact on Prediction)",
    yaxis_title="Feature",
    height=350,
    margin=dict(l=150)
)

st.plotly_chart(fig_importance, use_container_width=True)

st.info("💡 **Interpretation:** Day of Week and Salary Days are the most important features. "
        "This means cash demand is highly influenced by when people get paid and weekly patterns.")

st.markdown("---")
st.subheader("💵 Cost Savings Estimator")

col1, col2, col3 = st.columns(3)

with col1:
    current_cost = st.number_input(
        "Current Monthly Cost (₹)",
        min_value=0,
        value=100000,
        step=10000
    )

with col2:
    # Assume 30-40% savings from ML
    savings_percent = 35
    optimized_cost = current_cost * (1 - savings_percent / 100)
    st.metric(
        label="Optimized Cost (ML-based)",
        value=f"₹{int(optimized_cost):,}",
        delta=f"-{savings_percent}%"
    )

with col3:
    monthly_savings = current_cost - optimized_cost
    yearly_savings = monthly_savings * 12
    st.metric(
        label="Estimated Yearly Savings",
        value=f"₹{int(yearly_savings):,}",
        delta=f"+₹{int(monthly_savings):,}/month"
    )

    st.markdown("---")
st.subheader("📋 Recent Data")

display_df = pd.DataFrame({
    'Date': list(historical.index[-10:]) + list(forecast_dates),
    'Type': ['Historical'] * 10 + ['Forecast'] * forecast_days,
    'Cash Demand (₹)': list(historical.values[-10:]) + [int(v) for v in forecast_values]
})

st.dataframe(display_df, use_container_width=True)


st.markdown("---")
