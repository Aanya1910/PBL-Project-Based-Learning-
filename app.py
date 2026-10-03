# app.py - ATM Cash Demand Forecasting Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pickle

st.set_page_config(
    page_title="ATM Cash Demand Forecasting",
    page_icon="🏧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .kpi-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .alert-green {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
    }
    .alert-yellow {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
    }
    .alert-red {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)



@st.cache_data
def load_data():
    """
    Load the NN5 dataset.
    For now, we'll create sample data if the file doesn't exist.
    """
    try:
      
        df = pd.read_csv(
            'data/nn5_daily_dataset_without_missing_values.ts',
            sep='\t',
            index_col=0
        )
        df.index = pd.to_datetime(df.index)
        return df
    except FileNotFoundError:
      
        st.warning("⚠️ NN5 dataset not found. Using sample data for demo.")
        return generate_sample_data()


def generate_sample_data():
    """
    Generate realistic sample ATM data for demonstration.
    """
    np.random.seed(42)
    
   
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')

    data = {}
    for atm_id in range(1, 6):

        base = np.random.randint(30000, 80000)

        values = []
        for date in dates:

            weekend_factor = 0.7 if date.weekday() >= 5 else 1.0

            if date.day in [1, 7] or date.day >= 28:
                salary_factor = 1.4
            else:
                salary_factor = 1.0
            

            month_factor = 1.2 if date.month in [10, 11, 12] else 1.0
 
            noise = np.random.normal(1.0, 0.1)
            
            value = base * weekend_factor * salary_factor * month_factor * noise
            values.append(max(0, int(value)))
        
        data[f'ATM_{atm_id}'] = values
    
    df = pd.DataFrame(data, index=dates)
    return df


st.sidebar.title("🏧 ATM Forecast Controls")
st.sidebar.markdown("---")

df = load_data()

selected_atm = st.sidebar.selectbox(
    "Select ATM",
    options=df.columns.tolist(),
    index=0
)


forecast_days = st.sidebar.slider(
    "Forecast Horizon (days)",
    min_value=1,
    max_value=30,
    value=7
)


st.sidebar.markdown("### 💰 Buffer Calculator")
safety_margin = st.sidebar.slider(
    "Safety Margin (%)",
    min_value=0,
    max_value=50,
    value=20
)

days_until_refill = st.sidebar.slider(
    "Days Until Next Refill",
    min_value=1,
    max_value=14,
    value=3
)


st.sidebar.markdown("### 🤖 Model Selection")
model_choice = st.sidebar.selectbox(
    "Select Model",
    options=['LSTM', 'XGBoost', 'Random Forest', 'ARIMA'],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.info("📊 This dashboard predicts daily ATM cash demand using Machine Learning.")
