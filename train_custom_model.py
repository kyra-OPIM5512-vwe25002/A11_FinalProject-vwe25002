import sys, os, json, joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

BITTBRIDGE_DIR = Path('/home/kyralim007/bittbridge')
REPO_DIR = Path('/home/kyralim007/A11_FinalProject-vwe25002')
sys.path.insert(0, str(BITTBRIDGE_DIR))
os.chdir(BITTBRIDGE_DIR)

from miner_model_energy.ml_config import load_model_config
from miner_model_energy.pipeline import prepare_training_data

TARGET_COL = 'target_load_horizon'

config = load_model_config(str(BITTBRIDGE_DIR / 'model_params.yaml'))

print("Loading data (using cache if available)...")
train_df, test_df, feature_cols = prepare_training_data(config)
print(f"Rows: {len(train_df):,} | Features: {len(feature_cols)}")

# Temporal split — last 20% for validation (no shuffling: time series!)
split = int(len(train_df) * 0.8)
train = train_df.iloc[:split]
val   = train_df.iloc[split:]

X_train = train[feature_cols].values
y_train = train[TARGET_COL].values
X_val   = val[feature_cols].values
y_val   = val[TARGET_COL].values

print(f"Train: {len(train):,} rows | Val: {len(val):,} rows")
print("\nTraining HistGradientBoostingRegressor...")

model = HistGradientBoostingRegressor(
    max_iter=500,
    max_depth=6,
    learning_rate=0.05,
    min_samples_leaf=20,
    l2_regularization=0.1,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42,
    verbose=1,
)
model.fit(X_train, y_train)

train_mae = mean_absolute_error(y_train, model.predict(X_train))
val_mae   = mean_absolute_error(y_val,   model.predict(X_val))
train_r2  = r2_score(y_train, model.predict(X_train))
val_r2    = r2_score(y_val,   model.predict(X_val))

print(f"\n{'='*45}")
print(f"Train  MAE: {train_mae:.1f} MW   R²: {train_r2:.4f}")
print(f"Val    MAE: {val_mae:.1f} MW   R²: {val_r2:.4f}")
print(f"{'='*45}")

model_path    = REPO_DIR / 'model_custom.joblib'
contract_path = REPO_DIR / 'feature_contract.json'

joblib.dump(model, model_path)
print(f"\nModel saved   : {model_path}")

contract = {
    "feature_columns": feature_cols,
    "target_column":   TARGET_COL,
    "n_features":      len(feature_cols),
    "model_type":      "HistGradientBoostingRegressor",
    "val_mae":         round(float(val_mae), 2),
    "val_r2":          round(float(val_r2), 4),
}
with open(contract_path, 'w') as f:
    json.dump(contract, f, indent=2)
print(f"Contract saved: {contract_path}")
print("\nDone!")
