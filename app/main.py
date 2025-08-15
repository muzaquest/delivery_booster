from fastapi import FastAPI, Query
from typing import Optional
import pandas as pd
from datetime import datetime

from ml.inference import predict_with_shap

app = FastAPI(title="Restaurant Sales Analytics API", version="0.1.0")


@app.get("/health")
async def health() -> dict:
    """Healthcheck endpoint returning service status."""
    return {"status": "ok"}


@app.get("/report")
async def report(period: str = Query(..., description="YYYY-MM-DD_YYYY-MM-DD"), restaurant_id: Optional[int] = None) -> dict:
    # Parse period
    try:
        start_str, end_str = period.split("_")
        start = datetime.strptime(start_str, "%Y-%m-%d").date()
        end = datetime.strptime(end_str, "%Y-%m-%d").date()
    except Exception:
        return {"error": "Invalid period format. Use YYYY-MM-DD_YYYY-MM-DD"}

    # Load merged dataset
    csv_path = "/workspace/data/merged_dataset.csv"
    try:
        df = pd.read_csv(csv_path, parse_dates=["date"])
    except Exception:
        return {"error": "merged_dataset.csv not available. Run ETL first."}

    mask = (df["date"].dt.date >= start) & (df["date"].dt.date <= end)
    if restaurant_id is not None:
        mask &= df["restaurant_id"] == restaurant_id
    period_df = df.loc[mask].copy()

    if period_df.empty:
        return {"error": "No data for requested period/restaurant"}

    # Basic aggregates
    total_sales = float(period_df["total_sales"].sum())
    orders = int(period_df["orders_count"].sum()) if "orders_count" in period_df.columns else None
    aov = float(total_sales / orders) if orders and orders > 0 else None

    preds, shap_df = predict_with_shap(period_df)
    pred_sales_total = float(preds.sum())

    return {
        "period": period,
        "restaurant_id": restaurant_id,
        "actual_total_sales": total_sales,
        "predicted_total_sales": pred_sales_total,
        "orders": orders,
        "aov": aov,
    }


@app.get("/factors")
async def factors(period: str = Query(..., description="YYYY-MM-DD_YYYY-MM-DD"), restaurant_id: Optional[int] = None) -> dict:
    try:
        start_str, end_str = period.split("_")
        start = datetime.strptime(start_str, "%Y-%m-%d").date()
        end = datetime.strptime(end_str, "%Y-%m-%d").date()
    except Exception:
        return {"error": "Invalid period format. Use YYYY-MM-DD_YYYY-MM-DD"}

    csv_path = "/workspace/data/merged_dataset.csv"
    try:
        df = pd.read_csv(csv_path, parse_dates=["date"])
    except Exception:
        return {"error": "merged_dataset.csv not available. Run ETL first."}

    mask = (df["date"].dt.date >= start) & (df["date"].dt.date <= end)
    if restaurant_id is not None:
        mask &= df["restaurant_id"] == restaurant_id
    period_df = df.loc[mask].copy()

    if period_df.empty:
        return {"error": "No data for requested period/restaurant"}

    # Placeholder: return global SHAP total distribution for now
    preds, shap_df = predict_with_shap(period_df)
    shap_abs_mean = float(shap_df["shap_total"].abs().mean()) if not shap_df.empty else 0.0

    return {
        "period": period,
        "restaurant_id": restaurant_id,
        "factors": [
            {"feature": "composite", "impact": shap_abs_mean, "note": "aggregate placeholder"}
        ],
    }