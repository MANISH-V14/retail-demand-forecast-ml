import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

def run_training():

    os.makedirs("models", exist_ok=True)

    df = pd.read_csv("data/processed/processed_data.csv")

    performance_list = []

    for store in df["Store"].unique():

        df_store = df[df["Store"] == store].copy()

        X = df_store.drop(columns=["Weekly_Sales", "Date", "Store"])
        y = df_store["Weekly_Sales"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        model = RandomForestRegressor(
            n_estimators=400,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)

        performance_list.append({
            "Store": store,
            "MAE": mae
        })

        joblib.dump(model, f"models/store_{store}_model.pkl")

    performance_df = pd.DataFrame(performance_list)
    performance_df.to_csv("models/model_performance_summary.csv", index=False)

    print("Training complete.")


if __name__ == "__main__":
    run_training()