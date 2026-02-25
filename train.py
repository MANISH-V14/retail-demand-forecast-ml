import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import joblib

# Tuned Parameters
BEST_PARAMS = {
    "n_estimators": 400,
    "max_depth": 10,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": 42
}

# Load processed data
df = pd.read_csv("data/processed/processed_data.csv")

os.makedirs("models", exist_ok=True)

store_results = []

for store_id in sorted(df["Store"].unique()):
    df_store = df[df["Store"] == store_id].copy()

    X = df_store.drop(columns=["Weekly_Sales", "Date", "Store"])
    y = df_store["Weekly_Sales"]

    split_index = int(len(df_store) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    model = RandomForestRegressor(**BEST_PARAMS)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

    store_results.append({
        "Store": store_id,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    })

    joblib.dump(model, f"models/store_{store_id}_model.pkl")

    print(f"Store {store_id} trained | MAE: {mae:.2f} | MAPE: {mape:.2f}%")

results_df = pd.DataFrame(store_results)
results_df.to_csv("models/model_performance_summary.csv", index=False)

print("\nAll models retrained with tuned parameters.")