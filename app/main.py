from fastapi import FastAPI, Query
from typing import Optional
import pandas as pd
from datetime import datetime
import numpy as np

from ml.inference import predict_and_explain, top_factors, load_artifacts
from app.report_text import generate_full_report
from app.report_basic import build_basic_report
import json
import os
from sqlalchemy import text
from etl.data_loader import get_engine

app = FastAPI(title="Restaurant Sales Analytics API", version="0.1.0")


@app.get("/health")
async def health() -> dict:
    """Healthcheck endpoint returning service status and DB connectivity."""
    try:
        eng = get_engine()
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "healthy", "connected": True}
    except Exception as e:
        return {"status": "degraded", "connected": False, "error": str(e)}


@app.on_event("startup")
async def _warmup_model() -> None:
    try:
        load_artifacts()
    except Exception:
        # Warmup best-effort; service still responds with graceful errors
        pass


@app.get("/ml/status")
async def ml_status() -> dict:
    try:
        # Load metrics
        project_root = os.getenv("PROJECT_ROOT", os.getcwd())
        default_artifacts = os.path.join(project_root, "ml", "artifacts")
        metrics_path = os.getenv("ML_ARTIFACT_DIR", default_artifacts)
        mfile = os.path.join(metrics_path, "metrics.json")
        cfile = os.path.join(metrics_path, "champion.json")
        metrics = {}
        champion = {}
        if os.path.exists(mfile):
            with open(mfile, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        if os.path.exists(cfile):
            with open(cfile, "r", encoding="utf-8") as f:
                champion = json.load(f)
        # Ensure we can load model
        load_artifacts()
        return {"ready": True, "metrics": metrics, "champion": champion}
    except Exception as e:
        return {"ready": False, "error": str(e)}


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
    project_root = os.getenv("PROJECT_ROOT", os.getcwd())
    csv_path = os.getenv("ML_DATASET_CSV", os.path.join(project_root, "data", "merged_dataset.csv"))
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

    try:
        result = predict_and_explain(period_df)
        pred_sales_total = float(result["preds"].sum())
        top = result["top_factors"]
    except Exception as e:
        return {"error": f"Model inference failed: {e}"}

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

    project_root = os.getenv("PROJECT_ROOT", os.getcwd())
    csv_path = os.getenv("ML_DATASET_CSV", os.path.join(project_root, "data", "merged_dataset.csv"))
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

    try:
        result = predict_and_explain(period_df, top_k=20)
        factors_list = result["top_factors"]
    except Exception as e:
        return {"error": f"Model inference failed: {e}"}

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


@app.get("/restaurants")
async def restaurants() -> dict:
    """Return list of restaurants with id, name, first_date, last_date, days_count."""
    try:
        eng = get_engine()
        with eng.connect() as conn:
            # Determine available schema
            has_restaurants = False
            has_mapping = False
            try:
                conn.execute(text("SELECT 1 FROM restaurants LIMIT 1"))
                has_restaurants = True
            except Exception:
                has_restaurants = False
            try:
                conn.execute(text("SELECT 1 FROM restaurant_mapping LIMIT 1"))
                has_mapping = True
            except Exception:
                has_mapping = False

            if has_restaurants:
                q1 = text(
                    """
                    SELECT r.id AS id,
                           r.name AS name,
                           MIN(COALESCE(g.stat_date, j.stat_date)) AS first_date,
                           MAX(COALESCE(g.stat_date, j.stat_date)) AS last_date,
                           COUNT(DISTINCT COALESCE(g.stat_date, j.stat_date)) AS days_count
                    FROM restaurants r
                    LEFT JOIN grab_stats g ON g.restaurant_id = r.id
                    LEFT JOIN gojek_stats j ON j.restaurant_id = r.id
                    GROUP BY r.id, r.name
                    ORDER BY name
                    """
                )
                rows = [dict(r._mapping) for r in conn.execute(q1)]
            elif has_mapping:
                q2 = text(
                    """
                    SELECT rm.restaurant_id AS id,
                           rm.restaurant_name AS name,
                           MIN(rs.stat_date) AS first_date,
                           MAX(rs.stat_date) AS last_date,
                           COUNT(DISTINCT rs.stat_date) AS days_count
                    FROM restaurant_mapping rm
                    LEFT JOIN raw_stats rs ON rs.restaurant_name = rm.restaurant_name
                    GROUP BY rm.restaurant_id, rm.restaurant_name
                    ORDER BY name
                    """
                )
                rows = [dict(r._mapping) for r in conn.execute(q2)]
            else:
                rows = []
        return {"restaurants": rows}
    except Exception as e:
        return {"error": str(e)}


@app.get("/coverage")
async def coverage() -> dict:
    """Return suggested_since and suggested_until based on available dates across stats tables."""
    try:
        eng = get_engine()
        q = text(
            """
            SELECT MIN(d) AS first_date, MAX(d) AS last_date
            FROM (
              SELECT MIN(stat_date) AS d FROM grab_stats
              UNION ALL
              SELECT MIN(stat_date) AS d FROM gojek_stats
              UNION ALL
              SELECT MAX(stat_date) AS d FROM grab_stats
              UNION ALL
              SELECT MAX(stat_date) AS d FROM gojek_stats
            ) AS bounds
            """
        )
        with eng.connect() as conn:
            row = conn.execute(q).first()
            if row is None:
                return {"suggested_since": None, "suggested_until": None}
            first_date = row[0]
            last_date = row[1]
        return {"suggested_since": str(first_date) if first_date else None, "suggested_until": str(last_date) if last_date else None}
    except Exception as e:
        return {"error": str(e)}


@app.get("/report-test")
async def report_test(restaurant_name: str = Query(...), period: str = Query(...)) -> dict:
    """Generate basic report text for smoke test; no recursion."""
    try:
        # Resolve restaurant_id by name from available tables
        eng = get_engine()
        with eng.connect() as conn:
            rid = None
            try:
                row = conn.execute(text("SELECT id FROM restaurants WHERE name = :n"), {"n": restaurant_name}).first()
                if row:
                    rid = int(row[0])
            except Exception:
                pass
            if rid is None:
                try:
                    row = conn.execute(text("SELECT restaurant_id FROM restaurant_mapping WHERE restaurant_name = :n"), {"n": restaurant_name}).first()
                    if row:
                        rid = int(row[0])
                except Exception:
                    pass
        if rid is None:
            return {"error": f"Restaurant not found: {restaurant_name}"}
        text_report = generate_full_report(period=period, restaurant_id=rid)
        return {"report": text_report}
    except Exception as e:
        return {"error": str(e)}