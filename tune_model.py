import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Load processed data
df = pd.read_csv("data/processed/processed_data.csv")

# Use Store 1 for tuning
df_store = df[df["Store"] == 1].copy()

X = df_store.drop(columns=["Weekly_Sales", "Date", "Store"])
y = df_store["Weekly_Sales"]

tscv = TimeSeriesSplit(n_splits=5)

param_grid = {
    "n_estimators": [200, 400],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

model = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(
    model,
    param_grid,
    cv=tscv,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

grid_search.fit(X, y)

print("Best Parameters:")
print(grid_search.best_params_)
print("Best Score (MAE):", -grid_search.best_score_)