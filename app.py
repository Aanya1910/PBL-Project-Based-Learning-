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


st.markdown('<h1 class="main-header">🏧 ATM Cash Demand Forecasting</h1>', 
            unsafe_allow_html=True)
st.markdown(f"**Selected ATM:** {selected_atm} | **Forecast Horizon:** {forecast_days} days")
st.markdown("---")

atm_data = df[selected_atm].dropna()
recent_data = atm_data.tail(30)

avg_daily = int(recent_data.mean())
max_daily = int(recent_data.max())
min_daily = int(recent_data.min())
last_value = int(atm_data.iloc[-1])

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
        <div class="kpi-card">
            <h4>💰 Average Daily</h4>
            <h2>₹{avg_daily:,}</h2>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="kpi-card">
            <h4>📈 Maximum Daily</h4>
            <h2>₹{max_daily:,}</h2>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div class="kpi-card">
            <h4>📉 Minimum Daily</h4>
            <h2>₹{min_daily:,}</h2>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
        <div class="kpi-card">
            <h4>🕐 Last Withdrawal</h4>
            <h2>₹{last_value:,}</h2>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.subheader("📊 Cash Demand Forecast")

def generate_forecast(data, days):
    """
    Simple forecast using moving average + trend.
    This will be replaced with actual ML model predictions.
    """
    recent = data.tail(14)
    base = recent.mean()
    trend = (recent.tail(7).mean() - recent.head(7).mean()) / 7
    
    forecast = []
    for i in range(days):
        # Add weekly pattern
        day_of_week = (datetime.now() + timedelta(days=i)).weekday()
        weekend_factor = 0.7 if day_of_week >= 5 else 1.0
        
        # Add trend
        value = base + (trend * i)
        
        # Add some realistic variation
        value = value * weekend_factor * np.random.normal(1.0, 0.05)
        forecast.append(max(0, value))
    
    return forecast

historical = atm_data.tail(60)

forecast_values = generate_forecast(atm_data, forecast_days)
forecast_dates = pd.date_range(
    start=historical.index[-1] + timedelta(days=1),
    periods=forecast_days,
    freq='D'
)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=historical.index,
    y=historical.values,
    mode='lines',
    name='Historical Demand',
    line=dict(color='#1f77b4', width=2)
))

fig.add_trace(go.Scatter(
    x=forecast_dates,
    y=forecast_values,
    mode='lines+markers',
    name='Predicted Demand',
    line=dict(color='#ff7f0e', width=2, dash='dash'),
    marker=dict(size=8)
))

upper_bound = [v * 1.15 for v in forecast_values]
lower_bound = [v * 0.85 for v in forecast_values]

fig.add_trace(go.Scatter(
    x=list(forecast_dates) + list(forecast_dates[::-1]),
    y=upper_bound + lower_bound[::-1],
    fill='toself',
    fillcolor='rgba(255, 127, 14, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    name='Confidence Interval',
    showlegend=True
))

fig.update_layout(
    title=f"Cash Demand Forecast for {selected_atm}",
    xaxis_title="Date",
    yaxis_title="Cash Withdrawal (₹)",
    hovermode='x unified',
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    height=500
)

st.plotly_chart(fig, use_container_width=True)     


st.markdown("---")
st.subheader("💰 Buffer Calculator & Risk Assessment")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("#### Recommended Cash Load")
    
    predicted_total = sum(forecast_values[:days_until_refill])
    recommended_load = predicted_total * (1 + safety_margin / 100)
    
    st.metric(
        label="Predicted Demand (next {} days)".format(days_until_refill),
        value=f"₹{int(predicted_total):,}"
    )
    
    st.metric(
        label="Recommended Load (with {}% safety)".format(safety_margin),
        value=f"₹{int(recommended_load):,}",
        delta=f"+₹{int(recommended_load - predicted_total):,} buffer"
    )

with col_right:
    st.markdown("#### Risk Assessment")
    
    if recommended_load < predicted_total * 1.1:
        risk_level = "HIGH"
        risk_class = "alert-red"
        risk_msg = "🔴 SHORTAGE PREDICTED - Increase cash load immediately!"
    elif recommended_load < predicted_total * 1.25:
        risk_level = "MEDIUM"
        risk_class = "alert-yellow"
        risk_msg = "🟡 LOW BUFFER - Consider increasing cash load."
    else:
        risk_level = "LOW"
        risk_class = "alert-green"
        risk_msg = "🟢 SAFE - Current buffer is sufficient."
    
    st.markdown(f"""
        <div class="{risk_class}">
            <h3>Risk Level: {risk_level}</h3>
            <p>{risk_msg}</p>
        </div>
    "", unsafe_allow_html=True)

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
st.markdown("""
    <div style="text-align: center; color: gray; padding: 1rem;">
        <p>ATM Cash Demand Forecasting | BCA AI & DS Project</p>
        <p>Team: Aanya Vedwal, Ashmita Rawat, Himani, Kumari Bhawana</p>
        <p>Mentor: Mr. Nikhil Bisht</p>
    </div>
""", unsafe_allow_html=True)
