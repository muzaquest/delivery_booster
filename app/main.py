from fastapi import FastAPI, Query
from typing import Optional
import pandas as pd
from datetime import datetime
import numpy as np

from ml.inference import predict_with_shap, top_factors
from app.report_text import generate_full_report
from app.report_basic import build_basic_report

app = FastAPI(title="Restaurant Sales Analytics API", version="0.1.0")


@app.get("/health")
async def health() -> dict:
    """Healthcheck endpoint returning service status."""
    return {"status": "ok"}


@app.get("/report-basic")
async def report_basic(period: str = Query(..., description="YYYY-MM-DD_YYYY-MM-DD"), restaurant_id: Optional[int] = None) -> dict:
    try:
        return build_basic_report(period, restaurant_id)
    except Exception as e:
        return {"error": str(e)}


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

    preds, shap_df, feat_imp = predict_with_shap(period_df)
    pred_sales_total = float(preds.sum())
    top = top_factors(feat_imp, top_k=10)

    return {
        "period": period,
        "restaurant_id": restaurant_id,
        "actual_total_sales": total_sales,
        "predicted_total_sales": pred_sales_total,
        "orders": orders,
        "aov": aov,
        "top_factors": top,
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

    preds, shap_df, feat_imp = predict_with_shap(period_df)
    factors_list = top_factors(feat_imp, top_k=20)

    return {
        "period": period,
        "restaurant_id": restaurant_id,
        "factors": factors_list,
    }


@app.get("/report-text")
async def report_text(period: str = Query(..., description="YYYY-MM-DD_YYYY-MM-DD"), restaurant_id: Optional[int] = None) -> dict:
    try:
        # Validate period format
        start_str, end_str = period.split("_")
        datetime.strptime(start_str, "%Y-%m-%d")
        datetime.strptime(end_str, "%Y-%m-%d")
    except Exception:
        return {"error": "Invalid period format. Use YYYY-MM-DD_YYYY-MM-DD"}

    if restaurant_id is None:
        return {"error": "restaurant_id is required for /report-text"}

    try:
        text = generate_full_report(period=period, restaurant_id=int(restaurant_id))
        return {"report": text}
    except Exception as e:
        return {"error": str(e)}