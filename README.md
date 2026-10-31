ATM Cash Demand Forecasting

An AI and Machine Learning-based project for predicting the future cash requirements of ATMs using historical cash withdrawal data, time-based patterns, and relevant features such as weekdays, holidays, salary days, and previous demand.

The project aims to help improve ATM cash management by forecasting cash demand and providing useful recommendations through an interactive dashboard.


📌 Project Overview

ATM cash management is an important challenge for banks. Keeping too much cash in an ATM can increase operational and holding costs, while insufficient cash can lead to cash-outs and inconvenience for customers.

ATM Cash Demand Forecasting uses historical ATM withdrawal data and machine learning/time-series forecasting techniques to estimate future cash requirements.

The project includes data preprocessing, exploratory data analysis, feature engineering, model development, model comparison, and a web-based dashboard for displaying predictions.


🎯 Objectives

The main objectives of this project are:

- Analyze historical ATM transaction and cash withdrawal data.
- Identify factors that influence ATM cash demand.
- Clean and preprocess the dataset for machine learning.
- Perform exploratory data analysis to identify withdrawal patterns.
- Create useful time-based and historical features.
- Develop models for predicting future ATM cash demand.
- Compare different forecasting and machine learning models.
- Evaluate model performance using appropriate metrics.
- Develop a user-friendly web dashboard for predictions.
- Provide insights and cash-load recommendations based on predicted demand.

The project goals and intended dashboard functionality are defined in the original proposal.


🔄 Project Workflow

                 ┌─────────────────────┐
                 │     ATM Dataset     │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │ Data Preprocessing  │
                 │ • Missing Values    │
                 │ • Duplicates        │
                 │ • Data Cleaning     │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │       EDA           │
                 │ • Trends            │
                 │ • Weekdays          │
                 │ • Holidays          │
                 │ • Seasonal Patterns  │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │ Feature Engineering │
                 │ • Day of Week       │
                 │ • Month             │
                 │ • Holiday Indicator │
                 │ • Lag Features      │
                 └──────────┬──────────┘
                            ↓
              ┌───────────────────────────┐
              │ Forecasting / ML Models   │
              │ • ARIMA                   │
              │ • Random Forest           │
              │ • XGBoost                 │
              │ • LSTM                    │
              └─────────────┬─────────────┘
                            ↓
                 ┌─────────────────────┐
                 │ Model Comparison    │
                 │ • MAE               │
                 │ • RMSE              │
                 │ • R²                │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │ Best Model          │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │ Streamlit Dashboard │
                 │ • Predictions       │
                 │ • Insights          │
                 │ • Recommendations   │
                 └─────────────────────┘


🛠️ Technologies Used

Programming Language

- Python

Machine Learning & Data Science

- Pandas
- NumPy
- Scikit-learn
- XGBoost
- TensorFlow / Keras

Time-Series Forecasting

- ARIMA

Visualization

- Matplotlib
- Seaborn
- Plotly

Web Application

- Streamlit

Development Environment

- Jupyter Notebook
- VS Code


🤖 Forecasting Models

The project proposes comparison of multiple forecasting approaches to determine the most suitable model for ATM cash-demand prediction.

1. ARIMA

A statistical time-series forecasting model used to analyze historical demand and predict future values.

2. Random Forest

An ensemble machine learning algorithm that combines multiple decision trees to generate predictions.

3. XGBoost

A gradient-boosting algorithm that can model complex relationships between input features and cash demand.

4. LSTM

A deep-learning model designed for sequential and time-series data. It can learn patterns from historical demand sequences.

The proposal specifies ARIMA, Random Forest, XGBoost, and LSTM as the four forecasting models for the final project.


📊 Dataset

The project uses historical ATM cash withdrawal/transaction data.

The dataset is expected to contain information useful for understanding ATM cash-demand patterns, including:

- Transaction information
- Cash withdrawal/demand
- Date and time information
- ATM-related information
- Previous demand
- Calendar-related features

The proposal identifies transaction, cash, time, location, weather, and previous-demand information as potential data inputs.


🧹 Data Preprocessing

Before training the models, the dataset will be prepared through several preprocessing steps:

- Handling missing values
- Removing duplicate records
- Checking data types
- Detecting data inconsistencies
- Preparing time-based information
- Converting data into a suitable format for modeling

These steps are part of the proposed data-cleaning and preprocessing phase.


🔍 Exploratory Data Analysis

Exploratory Data Analysis will be performed to understand patterns in ATM cash demand.

The analysis will focus on:

- Weekday vs. weekend demand
- Holiday effects
- Salary-day patterns
- Seasonal trends
- Previous-day demand
- Withdrawal trends over time
- Relationships between relevant features and cash demand

Visualizations will be used to make these patterns easier to understand.


⚙️ Feature Engineering

Important features will be selected and created to improve prediction performance.

Possible features include:

Feature| Description
Day of Week| Identifies the weekday
Month| Captures monthly patterns
Holiday Indicator| Identifies holidays
Salary Day| Represents expected salary-related withdrawal patterns
Previous Demand| Historical demand used for prediction
Lag Features| Previous observations of cash demand

Feature engineering is included as a separate project milestone in the proposal.


📈 Model Evaluation

The models will be compared using performance metrics such as:

MAE — Mean Absolute Error

Measures the average absolute difference between actual and predicted values.

RMSE — Root Mean Squared Error

Measures prediction error while giving greater weight to larger errors.

R² Score

Measures how well the model explains the variation in the target variable.

The model with the most suitable performance will be selected based on the comparison.


🖥️ Dashboard

A web-based dashboard will be developed using Streamlit.

The dashboard will allow users to:

- Enter or upload relevant ATM data.
- Generate cash-demand predictions.
- View prediction results.
- Analyze important factors affecting demand.
- View cash-load recommendations.
- Understand historical and predicted demand patterns.

The proposed deliverables specifically include a Python/Streamlit dashboard for predictions and CSV-based cash-load recommendations.


📁 Project Structure

ATM-Cash-Demand-Forecasting/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── exploratory_data_analysis.ipynb
│   ├── feature_engineering.ipynb
│   └── model_comparison.ipynb
│
├── models/
│   ├── arima/
│   ├── random_forest/
│   ├── xgboost/
│   └── lstm/
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── prediction.py
│
├── dashboard/
│   └── app.py
│
├── results/
│   ├── graphs/
│   └── model_comparison/
│
├── requirements.txt
├── README.md

«The folder structure can be updated as the project implementation progresses.»


📌 Expected Outcomes

The completed project is expected to provide:

- Four trained forecasting models.
- A comparison of model performance.
- A selected model for ATM cash-demand forecasting.
- An interactive Streamlit dashboard.
- Cash-demand predictions.
- Cash-load recommendations.
- Insights into factors affecting ATM cash demand.
- Project documentation and final presentation.
- A GitHub repository containing the project code and supporting files.


🔮 Future Scope

Possible future improvements include:

- Real-time ATM transaction integration.
- Integration with live holiday and calendar APIs.
- Weather-based demand analysis.
- Automated ATM cash-replenishment alerts.
- More advanced deep-learning models.
- Deployment of the dashboard as a cloud application.
- Integration with multiple ATM locations and banks.


⚠️ Assumptions & Limitations

The project assumes that historical withdrawal behavior can provide useful information for predicting future demand.

It also assumes that ATM withdrawal patterns are influenced by factors such as the day of the week, holidays, and salary days. It further assumes that the available historical dataset represents typical ATM withdrawal behavior and that real-time transaction processing is outside the current scope.


👩‍💻 Team Members

Name| Role
Aanya Vedwal| Team Lead
Ashmita Rawat| Team Member
Himani| Team Member
Kumari Bhawana| Team Member


📚 References

The project references research on ATM cash-demand forecasting using statistical, machine-learning, and deep-learning approaches, including ARIMA, Random Forest, SVR, MLP, LSTM, and neural-network methods.


📄 Project Status

Status: 🚧 In Development

The project is being developed phase-by-phase, beginning with dataset understanding and preprocessing and progressing toward model development, comparison, dashboard development, and final integration.


⭐ Acknowledgement

This project is developed as part of the BCA (AI & DS) academic project.

Project: ATM Cash Demand Forecasting
Team: Aanya Vedwal, Ashmita Rawat, Himani, Kumari Bhawana
