# Retail Store Sales Forecasting Platform

An end-to-end multi-store demand forecasting project that predicts weekly retail sales and exposes the results through an interactive Streamlit dashboard.

## Live Demo

**Streamlit app:** https://retail-demand-forecast-ml-7bska8u5rzhxx9drjv63jm.streamlit.app/

## Overview

The project builds independent time-series forecasting models for 45 Walmart stores. It combines historical sales with lag, holiday, and macroeconomic signals to estimate future weekly demand.

### Key Capabilities

- Store-level sales forecasting
- Lag-based feature engineering
- Time-aware cross-validation with `TimeSeriesSplit`
- Random Forest hyperparameter tuning
- MAE, RMSE, and MAPE evaluation
- Performance comparison across stores
- Feature importance analysis
- Interactive forecast visualizations
- Downloadable forecast results

## Modeling Strategy

1. Aggregate department-level sales into weekly store-level totals.
2. Create lag features for 1, 4, and 12 weeks.
3. Include available holiday and macroeconomic signals.
4. Use time-aware cross-validation to avoid leakage from future observations.
5. Tune Random Forest parameters and train a separate model for each store.
6. Evaluate each store using MAE, RMSE, and MAPE.

## Results

- Most stores achieved approximately 2% to 6% MAPE in the reported evaluation.
- Some stores, including Store 14, showed greater structural volatility.
- Lag features were among the strongest predictors across the models.

## Interactive Dashboard

The Streamlit dashboard supports:

- Store selection
- Historical backtest visualization
- Forecasting the next N weeks
- Performance comparison across stores
- Global feature importance analysis
- Forecast downloads

## Tech Stack

- Python
- Pandas
- Scikit-learn
- Random Forest
- TimeSeriesSplit
- Streamlit
- Matplotlib
- Seaborn

## What This Project Demonstrates

This project focuses on practical forecasting rather than a single global accuracy number. Training and evaluating models at the store level makes it easier to identify locations with stable demand as well as stores where additional features or a different modeling strategy may be needed.
