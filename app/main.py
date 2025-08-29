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
        df = pd.read_csv(csv_path, parse_dates=["date"]) if os.path.exists(csv_path) else pd.DataFrame()
    except Exception:
        df = pd.DataFrame()

    mask = (df["date"].dt.date >= start) & (df["date"].dt.date <= end)
    if restaurant_id is not None:
        mask &= df["restaurant_id"] == restaurant_id
    period_df = df.loc[mask].copy()

    if period_df.empty:
        return {
            "period": period,
            "restaurant_id": restaurant_id,
            "actual_total_sales": 0.0,
            "predicted_total_sales": 0.0,
            "orders": 0,
            "aov": 0.0,
            "top_factors": []
        }

    # Basic aggregates
    total_sales = float(period_df["total_sales"].sum())
    orders = int(period_df["orders_count"].sum()) if "orders_count" in period_df.columns else None
    aov = float(total_sales / orders) if orders and orders > 0 else None

    try:
        result = predict_and_explain(period_df)
        pred_sales_total = float(result["preds"].sum())
        top = result["top_factors"]
    except Exception:
        pred_sales_total = 0.0
        top = []

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
        df = pd.read_csv(csv_path, parse_dates=["date"]) if os.path.exists(csv_path) else pd.DataFrame()
    except Exception:
        df = pd.DataFrame()

    mask = (df["date"].dt.date >= start) & (df["date"].dt.date <= end)
    if restaurant_id is not None:
        mask &= df["restaurant_id"] == restaurant_id
    period_df = df.loc[mask].copy()

    if period_df.empty:
        return {"period": period, "restaurant_id": restaurant_id, "factors": []}

    try:
        result = predict_and_explain(period_df, top_k=20)
        factors_list = result["top_factors"]
    except Exception:
        factors_list = []

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
        # Try to infer from CSV demo: pick first
        try:
            project_root = os.getenv("PROJECT_ROOT", os.getcwd())
            csv_path = os.getenv("ML_DATASET_CSV", os.path.join(project_root, "data", "merged_dataset.csv"))
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                rid = int(df["restaurant_id"].iloc[0]) if not df.empty else None
                if rid is not None:
                    restaurant_id = rid
        except Exception:
            pass
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
        if not rows:
            # CSV fallback
            try:
                project_root = os.getenv("PROJECT_ROOT", os.getcwd())
                csv_path = os.getenv("ML_DATASET_CSV", os.path.join(project_root, "data", "merged_dataset.csv"))
                df = pd.read_csv(csv_path, parse_dates=["date"]) if os.path.exists(csv_path) else pd.DataFrame()
                if not df.empty:
                    grp = df.groupby(["restaurant_id", "restaurant_name"], as_index=False)["date"].agg(["min", "max", "count"]).reset_index()
                    grp.columns = ["restaurant_id", "restaurant_name", "first_date", "last_date", "days_count"]
                    rows = [
                        {
                            "id": int(r.restaurant_id),
                            "name": str(r.restaurant_name),
                            "first_date": str(pd.to_datetime(r.first_date).date()),
                            "last_date": str(pd.to_datetime(r.last_date).date()),
                            "days_count": int(r.days_count),
                        }
                        for r in grp.itertuples(index=False)
                    ]
            except Exception:
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
            first_date = row[0] if row else None
            last_date = row[1] if row else None
        if not first_date or not last_date:
            # CSV fallback
            project_root = os.getenv("PROJECT_ROOT", os.getcwd())
            csv_path = os.getenv("ML_DATASET_CSV", os.path.join(project_root, "data", "merged_dataset.csv"))
            try:
                df = pd.read_csv(csv_path, parse_dates=["date"]) if os.path.exists(csv_path) else pd.DataFrame()
                if not df.empty:
                    first_date = df["date"].min().date()
                    last_date = df["date"].max().date()
            except Exception:
                pass
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
        if rid is not None:
            text_report = generate_full_report(period=period, restaurant_id=rid)
            return {"report": text_report}
        # CSV fallback demo
        project_root = os.getenv("PROJECT_ROOT", os.getcwd())
        csv_path = os.getenv("ML_DATASET_CSV", os.path.join(project_root, "data", "merged_dataset.csv"))
        try:
            df = pd.read_csv(csv_path, parse_dates=["date"]) if os.path.exists(csv_path) else pd.DataFrame()
        except Exception:
            df = pd.DataFrame()
        if df.empty:
            return {"report": f"Demo report for {restaurant_name} {period}: no data"}
        try:
            start_str, end_str = period.split("_")
            mask = (df["date"] >= start_str) & (df["date"] <= end_str)
            sub = df.loc[mask & ((df.get("restaurant_name") == restaurant_name) | (df.get("restaurant_id") == restaurant_name))].copy()
            if sub.empty and "restaurant_name" in df.columns:
                # try any restaurant
                sub = df.loc[mask].copy()
            total_sales = float(pd.to_numeric(sub.get("total_sales"), errors="coerce").fillna(0).sum()) if not sub.empty else 0.0
            orders = int(pd.to_numeric(sub.get("orders_count"), errors="coerce").fillna(0).sum()) if not sub.empty else 0
            aov = (total_sales / orders) if orders else 0.0
            rain = float(pd.to_numeric(sub.get("rain"), errors="coerce").fillna(0).mean()) if not sub.empty else 0.0
            temp = float(pd.to_numeric(sub.get("temp"), errors="coerce").fillna(0).mean()) if not sub.empty else 0.0
            text_report = (
                f"Demo report for {restaurant_name} {period}\n"
                f"Total sales: {int(total_sales)} IDR\n"
                f"Orders: {orders}\n"
                f"AOV: {int(aov) if orders else 0} IDR\n"
                f"Avg temp: {temp:.1f}°C, Avg rain: {rain:.1f}mm\n"
            )
            return {"report": text_report}
        except Exception as e:
            return {"report": f"Demo report for {restaurant_name} {period}: error {e}"}
    except Exception as e:
        return {"error": str(e)}