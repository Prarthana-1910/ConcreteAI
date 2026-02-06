import pandas as pd
import numpy as np
import time
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score

# Load Data
df = pd.read_csv(r"data\features.csv")

X = df[['Cement', 'GGBS', 'FlyAsh', 'Water', 'CoarseAggregate', 'Sand', 'Admixture', 'WBRatio', 'age']]
y = df['Strength']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 1. Initial Model Comparison ---
models = {
    "RandomForest": Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ]),
    "XGBoost": Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42))
    ]),
    "AdaBoost": Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', AdaBoostRegressor(n_estimators=100))
    ]),
    "GaussianProcess": Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', GaussianProcessRegressor())
    ]),
    "CatBoost": Pipeline([ 
        ('scaler', StandardScaler()), 
        ('regressor', CatBoostRegressor(verbose=0, iterations=500)) 
    ])
}

results = []
print("Training Initial Models...")

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    results.append({"Model": name, "R2": round(r2, 4), "MAE": round(mae, 4), "RMSE": round(rmse, 4)})
    joblib.dump(model, f"models/{name.replace(' ', '_')}_pipeline.joblib")

    print(f" > {name} trained.")

results_df = pd.DataFrame(results).sort_values(by="R2", ascending=False)
print("\n--- Model Performance Comparison ---")
print(results_df)

# --- 2. Hyperparameter Tuning (FOR CATBOOST ONLY) ---

param_grid = {
    'regressor__iterations': [300, 500, 800],
    'regressor__depth': [4, 6, 8],
    'regressor__learning_rate': [0.01, 0.05, 0.1],
    'regressor__l2_leaf_reg': [1, 3, 5, 7]
}

def create_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', CatBoostRegressor(verbose=0, random_state=42))
    ])

results_comparison = []

# Grid Search
print("\nStarting Grid Search...")
start = time.time()

grid = GridSearchCV(create_pipeline(), param_grid, cv=3, n_jobs=-1, scoring='r2')
grid.fit(X_train, y_train)

results_comparison.append({
    'Method': 'Grid Search',
    'Best R2': grid.best_score_,
    'Time (s)': time.time() - start
})

# Random Search
print("Starting Random Search...")
start = time.time()

random = RandomizedSearchCV(create_pipeline(), param_grid, n_iter=15, cv=3, n_jobs=-1, scoring='r2', random_state=42)
random.fit(X_train, y_train)

results_comparison.append({
    'Method': 'Random Search',
    'Best R2': random.best_score_,
    'Time (s)': time.time() - start
})

comparison_df = pd.DataFrame(results_comparison)
print("\n--- Hyperparameter Tuning Performance ---")
print(comparison_df)

# Save Best CatBoost Model
best_model = grid.best_estimator_
model_path = "models/Tuned_CatBoost.joblib"
joblib.dump(best_model, model_path)

print(f"Tuned model saved successfully to: {model_path}")
print(f"Best Parameters found: \n{grid.best_params_}")

# --- 2. Hyperparameter Tuning (FOR XGBOOST ONLY) ---

param_grid = {
    'regressor__n_estimators': [300, 600],
    'regressor__max_depth': [4, 6],
    'regressor__learning_rate': [0.03, 0.1],
    'regressor__subsample': [0.8, 1.0],
    'regressor__colsample_bytree': [0.8, 1.0]
}

def create_xgb_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),   # You can remove this; XGBoost doesn't need scaling
        ('regressor', XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        ))
    ])

results_comparison = []

# 🔹 Grid Search
print("\nStarting Grid Search (XGBoost)...")
start = time.time()

grid = GridSearchCV(
    create_xgb_pipeline(),
    param_grid,
    cv=3,
    n_jobs=-1,
    scoring='r2',
    verbose=1
)
grid.fit(X_train, y_train)

results_comparison.append({
    'Method': 'Grid Search',
    'Best R2': grid.best_score_,
    'Time (s)': time.time() - start
})

# 🔹 Random Search
print("Starting Random Search (XGBoost)...")
start = time.time()

random = RandomizedSearchCV(
    create_xgb_pipeline(),
    param_grid,
    n_iter=25,
    cv=3,
    n_jobs=-1,
    scoring='r2',
    random_state=42,
    verbose=1
)
random.fit(X_train, y_train)

results_comparison.append({
    'Method': 'Random Search',
    'Best R2': random.best_score_,
    'Time (s)': time.time() - start
})

comparison_df = pd.DataFrame(results_comparison)
print("\n--- XGBoost Hyperparameter Tuning Performance ---")
print(comparison_df)

# ✅ Save Best Model
best_model = grid.best_estimator_
model_path = "models/Tuned_XGBoost.joblib"
joblib.dump(best_model, model_path)

print(f"\nTuned XGBoost model saved to: {model_path}")
print("Best Parameters:\n", grid.best_params_)
