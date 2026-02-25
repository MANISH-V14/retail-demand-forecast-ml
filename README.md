# 📊 Retail Store Sales Forecasting Platform

End-to-end multi-store demand forecasting system using machine learning.

## 🔍 Overview

This project builds 45 independent time-series forecasting models to predict weekly retail sales for Walmart stores.

Features:
- Store-level forecasting
- Hyperparameter tuning using TimeSeriesSplit
- Feature engineering with lag variables
- Macroeconomic and holiday signals
- Model evaluation (MAE, RMSE, MAPE)
- Performance categorization
- Feature importance analysis
- Heatmap visualization
- Downloadable forecasts

## 🧠 Modeling Strategy

- Aggregated department sales into total weekly store sales
- Created lag features (1, 4, 12 weeks)
- Time-aware cross-validation
- Tuned RandomForest hyperparameters
- Trained separate model per store

## 📊 Results

- Most stores achieved 2%–6% MAPE
- Some stores showed structural volatility (e.g., Store 14)
- Lag features were strongest predictors globally

## 🚀 Dashboard

Interactive Streamlit dashboard allows:

- Store selection
- Backtest visualization
- Forecast next N weeks
- Performance comparison across stores
- Global feature importance analysis

## 🛠 Tech Stack

Python  
Pandas  
Scikit-learn  
RandomForest  
TimeSeriesSplit  
Streamlit  
Matplotlib  
Seaborn  

## ▶ Run Locally
