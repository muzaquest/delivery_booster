"""Inference utilities: predictions and SHAP factor attributions."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ARTIFACT_DIR = "/workspace/ml/artifacts"


def load_artifacts(artifact_dir: str = ARTIFACT_DIR):
    model = joblib.load(os.path.join(artifact_dir, "model.joblib"))
    with open(os.path.join(artifact_dir, "features.json"), "r", encoding="utf-8") as f:
        features = json.load(f)
    background_path_parquet = os.path.join(artifact_dir, "shap_background.parquet")
    background_path_csv = os.path.join(artifact_dir, "shap_background.csv")
    if os.path.exists(background_path_parquet):
        background = pd.read_parquet(background_path_parquet)
    elif os.path.exists(background_path_csv):
        background = pd.read_csv(background_path_csv)
    else:
        background = None
    return model, features, background


def load_model_pickle(artifact_dir: str = ARTIFACT_DIR):
    path = os.path.join(artifact_dir, "model.pkl")
    if not os.path.exists(path):
        return None
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_and_explain(df: pd.DataFrame, artifact_dir: str = ARTIFACT_DIR, top_k: int = 10) -> Dict[str, object]:
    model_pickle = load_model_pickle(artifact_dir)
    if model_pickle is not None:
        model = model_pickle
        with open(os.path.join(artifact_dir, "features.json"), "r", encoding="utf-8") as f:
            features = json.load(f)
        background = pd.read_csv(os.path.join(artifact_dir, "shap_background.csv")) if os.path.exists(os.path.join(artifact_dir, "shap_background.csv")) else None
    else:
        model, features, background = load_artifacts(artifact_dir)
    X = df[features]
    preds = model.predict(X)
    # SHAP
    model_step = model.named_steps["model"]
    pre_step: ColumnTransformer = model.named_steps["pre"]
    X_pre = pre_step.transform(X)
    if background is not None and not background.empty:
        bg_pre = pre_step.transform(background[features])
        explainer = shap.TreeExplainer(model_step, data=bg_pre, feature_perturbation="interventional")
    else:
        explainer = shap.TreeExplainer(model_step, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_pre)
    # Map back
    _, groups = _resolve_preprocessed_feature_groups(pre_step)
    abs_shap = np.abs(shap_values)
    feature_imp: Dict[str, float] = {}
    for orig_feat, indices in groups.items():
        if len(indices) == 0:
            continue
        feature_imp[orig_feat] = float(abs_shap[:, indices].mean())
    return {
        "preds": preds,
        "top_factors": top_factors(feature_imp, top_k=top_k),
    }


def _resolve_preprocessed_feature_groups(pre: ColumnTransformer) -> Tuple[List[str], Dict[str, List[int]]]:
    """Return names of preprocessed columns and a mapping from original feature -> indices in preprocessed matrix."""
    names_out: List[str] = []
    groups: Dict[str, List[int]] = {}
    idx = 0
    for name, transformer, cols in pre.transformers_:
        if name == "remainder" and transformer == "drop":
            continue
        if hasattr(transformer, "get_feature_names_out"):
            try:
                local_names = list(transformer.get_feature_names_out(cols))
            except Exception:
                local_names = [str(c) for c in (cols if isinstance(cols, list) else [cols])]
        elif isinstance(transformer, Pipeline):
            last = transformer.steps[-1][1]
            if isinstance(last, OneHotEncoder):
                # OHE: expand per category
                if hasattr(last, "get_feature_names_out"):
                    local_names = list(last.get_feature_names_out(cols))
                else:
                    # Build names manually
                    local_names = []
                    cats = last.categories_
                    for col, cat_vals in zip(cols, cats):
                        local_names.extend([f"{col}_{v}" for v in cat_vals])
            else:
                local_names = [str(c) for c in (cols if isinstance(cols, list) else [cols])]
        else:
            local_names = [str(c) for c in (cols if isinstance(cols, list) else [cols])]

        names_out.extend(local_names)
        # Build groups by original feature name
        if isinstance(cols, list):
            # For OHE, local_names may contain prefixes like feature_xxx
            for c in cols:
                # collect indices for entries starting with "c_" or exactly c
                for j, nm in enumerate(local_names):
                    if nm == c or nm.startswith(f"{c}_"):
                        groups.setdefault(c, []).append(idx + j)
        else:
            c = cols
            for j, nm in enumerate(local_names):
                if nm == c or nm.startswith(f"{c}_"):
                    groups.setdefault(c, []).append(idx + j)
        idx += len(local_names)

    return names_out, groups


def predict_with_shap(df: pd.DataFrame, artifact_dir: str = ARTIFACT_DIR) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, float]]:
    model: Pipeline
    model, features, background = load_artifacts(artifact_dir)

    X = df[features]
    preds = model.predict(X)

    # Build SHAP explainer on the underlying model in pipeline
    model_step = model.named_steps["model"]
    pre_step: ColumnTransformer = model.named_steps["pre"]

    # Preprocess
    X_pre = pre_step.transform(X)

    # Background for SHAP (use interventional to avoid leaf coverage errors)
    try:
        if background is not None and not background.empty:
            bg_pre = pre_step.transform(background[features])
            explainer = shap.TreeExplainer(model_step, data=bg_pre, feature_perturbation="interventional")
        else:
            explainer = shap.TreeExplainer(model_step, feature_perturbation="interventional")
        shap_values = explainer.shap_values(X_pre)
    except Exception:
        # Fallback: let SHAP infer background automatically
        explainer = shap.TreeExplainer(model_step)
        shap_values = explainer.shap_values(X_pre)

    # Map preprocessed columns back to original features
    _, groups = _resolve_preprocessed_feature_groups(pre_step)

    # Aggregate absolute shap by original feature
    abs_shap = np.abs(shap_values)
    feature_imp: Dict[str, float] = {}
    for orig_feat, indices in groups.items():
        if len(indices) == 0:
            continue
        # mean absolute contribution over samples and columns
        feature_imp[orig_feat] = float(abs_shap[:, indices].mean())

    shap_total = np.sum(shap_values, axis=1)
    shap_df = pd.DataFrame({"shap_total": shap_total})
    return preds, shap_df, feature_imp


def top_factors(feature_imp: Dict[str, float], top_k: int = 10, exclude_patterns: Optional[List[str]] = None) -> List[Dict[str, float]]:
    if not feature_imp:
        return []
    # Exclude trivial features by default
    default_exclude = [
        r"^orders_count(?!.*conversion).*",  # orders_count and its lags/rolling
        r"^total_sales.*",                   # total_sales lags/rolling
        r"^restaurant_id$",                 # identity
    ]
    import re
    patterns = [re.compile(p) for p in (exclude_patterns or default_exclude)]
    def is_excluded(name: str) -> bool:
        return any(p.search(name) for p in patterns)

    filtered = {k: v for k, v in feature_imp.items() if not is_excluded(k)}
    if not filtered:
        filtered = feature_imp
    items = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
    total = sum(v for _, v in items) or 1.0
    result = [
        {"feature": k, "impact": v, "impact_percent": round(100.0 * v / total, 2)}
        for k, v in items[:top_k]
    ]
    return result