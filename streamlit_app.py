import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import timedelta
import os
from train import run_training
import subprocess


# -------------------------------------------------
# PAGE CONFIG FIRST (Must be first Streamlit call)
# -------------------------------------------------
st.set_page_config(page_title="Retail Demand Forecasting Platform", layout="wide")

st.title("📊 Retail Store Sales Forecasting Platform")
st.markdown("Multi-store ML forecasting system using Random Forest models.")

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
df = pd.read_csv("data/processed/processed_data.csv")
df["Date"] = pd.to_datetime(df["Date"])



if not os.path.exists("models/model_performance_summary.csv"):
    st.warning("Models not found. Training models now...")
    run_training()

performance_df = pd.read_csv("models/model_performance_summary.csv")

import os
from sklearn.ensemble import RandomForestRegressor

BEST_PARAMS = {
    "n_estimators": 400,
    "max_depth": 10,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": 42
}

if not os.path.exists("models"):
    os.makedirs("models")

if len(os.listdir("models")) == 0:
    st.write("Training models... Please wait.")

    for store_id in sorted(df["Store"].unique()):
        df_store = df[df["Store"] == store_id]

        X = df_store.drop(columns=["Weekly_Sales", "Date", "Store"])
        y = df_store["Weekly_Sales"]

        model = RandomForestRegressor(**BEST_PARAMS)
        model.fit(X, y)

        joblib.dump(model, f"models/store_{store_id}_model.pkl")

    st.write("Models trained successfully.")

# -------------------------------------------------
# PERFORMANCE CATEGORIZATION
# -------------------------------------------------
def categorize(mape):
    if mape < 3:
        return "Excellent"
    elif mape < 6:
        return "Good"
    else:
        return "Needs Improvement"

performance_df["Category"] = performance_df["MAPE"].apply(categorize)

# -------------------------------------------------
# GLOBAL KPI
# -------------------------------------------------
st.markdown("### 📌 Model Summary Across All Stores")
st.metric("Average MAPE", f"{performance_df['MAPE'].mean():.2f}%")

# -------------------------------------------------
# SIDEBAR CONFIG
# -------------------------------------------------
st.sidebar.header("⚙ Configuration")
store_id = st.sidebar.selectbox("Select Store", sorted(df["Store"].unique()))
forecast_weeks = st.sidebar.slider("Forecast Weeks", 4, 12, 8)

df_store = df[df["Store"] == store_id].copy()

model = joblib.load(f"models/store_{store_id}_model.pkl")

# -------------------------------------------------
# HISTORICAL SALES
# -------------------------------------------------
st.subheader(f"📈 Historical Sales - Store {store_id}")

fig1, ax1 = plt.subplots(figsize=(10,4))
ax1.plot(df_store["Date"], df_store["Weekly_Sales"])
ax1.set_title("Weekly Sales Over Time")
st.pyplot(fig1)

# -------------------------------------------------
# BACKTEST SECTION
# -------------------------------------------------
split_index = int(len(df_store) * 0.8)

X = df_store.drop(columns=["Weekly_Sales", "Date", "Store"])
y = df_store["Weekly_Sales"]

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

test_preds = model.predict(X_test)

backtest_df = pd.DataFrame({
    "Date": df_store["Date"].iloc[split_index:].values,
    "Actual": y_test.values,
    "Predicted": test_preds
})

st.subheader("📉 Backtest: Actual vs Predicted")

fig_bt, ax_bt = plt.subplots(figsize=(10,4))
ax_bt.plot(backtest_df["Date"], backtest_df["Actual"], label="Actual")
ax_bt.plot(backtest_df["Date"], backtest_df["Predicted"], label="Predicted")
ax_bt.legend()
st.pyplot(fig_bt)

# -------------------------------------------------
# STORE PERFORMANCE
# -------------------------------------------------
store_perf = performance_df[performance_df["Store"] == store_id]

st.subheader("📊 Model Performance")
col1, col2, col3 = st.columns(3)
col1.metric("MAE", f"{store_perf['MAE'].values[0]:,.0f}")
col2.metric("RMSE", f"{store_perf['RMSE'].values[0]:,.0f}")
col3.metric("MAPE (%)", f"{store_perf['MAPE'].values[0]:.2f}")

# -------------------------------------------------
# FEATURE IMPORTANCE
# -------------------------------------------------
st.subheader("📌 Feature Importance")

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

fig_imp, ax_imp = plt.subplots(figsize=(8,4))
ax_imp.barh(feature_importance["Feature"], feature_importance["Importance"])
ax_imp.invert_yaxis()
st.pyplot(fig_imp)

# -------------------------------------------------
# FORECAST SECTION
# -------------------------------------------------
st.subheader(f"🔮 {forecast_weeks}-Week Forecast")

last_row = df_store.iloc[-1]
forecast_data = []

current_date = last_row["Date"]
lag_1 = last_row["Weekly_Sales"]
lag_4 = df_store.iloc[-4]["Weekly_Sales"]
lag_12 = df_store.iloc[-12]["Weekly_Sales"]

for i in range(forecast_weeks):
    current_date += timedelta(weeks=1)

    features = np.array([[
        last_row["IsHoliday"],
        last_row["Temperature"],
        last_row["Fuel_Price"],
        last_row["CPI"],
        last_row["Unemployment"],
        current_date.year,
        current_date.month,
        current_date.isocalendar()[1],
        lag_1,
        lag_4,
        lag_12
    ]])

    prediction = model.predict(features)[0]

    forecast_data.append((current_date, prediction))

    lag_12 = lag_4
    lag_4 = lag_1
    lag_1 = prediction

forecast_df = pd.DataFrame(forecast_data, columns=["Date", "Forecast"])

fig2, ax2 = plt.subplots(figsize=(10,4))
ax2.plot(forecast_df["Date"], forecast_df["Forecast"])
ax2.set_title("Future Forecast")
st.pyplot(fig2)

st.download_button(
    label="📥 Download Forecast CSV",
    data=forecast_df.to_csv(index=False),
    file_name=f"store_{store_id}_forecast.csv",
    mime="text/csv"
)

# -------------------------------------------------
# STORE RANKING + HEATMAP
# -------------------------------------------------
st.subheader("🏆 Store Performance Ranking")

sorted_perf = performance_df.sort_values("MAPE")
st.dataframe(sorted_perf)

st.subheader("📌 Store Performance Categories")
st.dataframe(performance_df.sort_values("MAPE"))

import seaborn as sns

st.subheader("🔥 Store MAPE Heatmap")

fig_heat, ax_heat = plt.subplots(figsize=(6,10))
sns.heatmap(performance_df.set_index("Store")[["MAPE"]], 
            annot=True, 
            fmt=".1f",
            cmap="RdYlGn_r",
            ax=ax_heat)

st.pyplot(fig_heat)

st.subheader("🌍 Global Feature Importance (All Stores)")

import glob

feature_list = []
feature_names = X.columns

for file in glob.glob("models/store_*_model.pkl"):
    m = joblib.load(file)
    feature_list.append(m.feature_importances_)

avg_importance = np.mean(feature_list, axis=0)

global_imp_df = pd.DataFrame({
    "Feature": feature_names,
    "Avg Importance": avg_importance
}).sort_values("Avg Importance", ascending=False)

fig_global, ax_global = plt.subplots(figsize=(8,4))
ax_global.barh(global_imp_df["Feature"], global_imp_df["Avg Importance"])
ax_global.invert_yaxis()
st.pyplot(fig_global)