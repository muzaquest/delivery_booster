"""Model training for sales forecasting and SHAP explainability.

- Reads merged_dataset.csv (per-restaurant daily dataset)
- Trains LightGBM regressor to predict total_sales
- Saves model, feature list, and SHAP background sample
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DEFAULT_DATASET = "/workspace/data/merged_dataset.csv"
DEFAULT_MODEL_DIR = "/workspace/ml/artifacts"


def build_preprocessor(df: pd.DataFrame) -> Tuple[Pipeline, List[str]]:
    target = "total_sales"
    drop_cols = [
        target,
        "date",
        # drop leakage-like columns if any
    ]
    numeric_cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if c not in drop_cols + numeric_cols]

    ct = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            ("cat", Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    return ct, numeric_cols + categorical_cols


def train_model(csv_path: str = DEFAULT_DATASET, model_dir: str = DEFAULT_MODEL_DIR) -> dict:
    os.makedirs(model_dir, exist_ok=True)
    df = pd.read_csv(csv_path, parse_dates=["date"]) if os.path.exists(csv_path) else pd.DataFrame()
    if df.empty:
        raise FileNotFoundError(f"Dataset not found or empty: {csv_path}")

    # Define target and filter rows with valid targets
    target = "total_sales"
    df = df.copy()
    df = df[pd.notnull(df[target])]

    # Build features
    preprocessor, feat_cols = build_preprocessor(df)

    # Split by time for validation (last 10%)
    df = df.sort_values(["restaurant_id", "date"])  # stable order
    split_idx = int(len(df) * 0.9)
    train_df = df.iloc[:split_idx]
    valid_df = df.iloc[split_idx:]

    X_train = train_df[feat_cols]
    y_train = train_df[target]
    X_valid = valid_df[feat_cols]
    y_valid = valid_df[target]

    # Model
    model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(steps=[("pre", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    # Eval
    pred = pipeline.predict(X_valid)
    metrics = {
        "mae": float(mean_absolute_error(y_valid, pred)),
        "r2": float(r2_score(y_valid, pred)),
        "n_train": int(len(train_df)),
        "n_valid": int(len(valid_df)),
    }

    # SHAP background sample for explainability
    # Use a small random subset for efficiency
    background_idx = np.random.RandomState(42).choice(len(X_train), size=min(500, len(X_train)), replace=False)
    background_sample = X_train.iloc[background_idx]

    # Save artifacts
    joblib.dump(pipeline, os.path.join(model_dir, "model.joblib"))
    with open(os.path.join(model_dir, "features.json"), "w", encoding="utf-8") as f:
        json.dump(feat_cols, f)
    background_sample.to_csv(os.path.join(model_dir, "shap_background.csv"), index=False)
    with open(os.path.join(model_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGBM model for sales forecasting")
    parser.add_argument("--csv", type=str, default=DEFAULT_DATASET, help="Path to merged_dataset.csv")
    parser.add_argument("--out", type=str, default=DEFAULT_MODEL_DIR, help="Artifacts output directory")
    args = parser.parse_args()

    metrics = train_model(args.csv, args.out)
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()