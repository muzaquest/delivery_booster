from __future__ import annotations

from typing import Dict, Optional, Tuple
from datetime import date
import pandas as pd
import numpy as np

from etl.data_loader import get_engine


def _read_stats(table: str, restaurant_id: Optional[int], start: date, end: date) -> pd.DataFrame:
    eng = get_engine()
    df = pd.read_sql_query(f"SELECT * FROM {table}", eng, parse_dates=["stat_date", "created_at"]) if table else pd.DataFrame()
    if df.empty:
        return df
    df.columns = [c.lower() for c in df.columns]
    df = df[(df["stat_date"].dt.date >= start) & (df["stat_date"].dt.date <= end)]
    if restaurant_id is not None and "restaurant_id" in df.columns:
        df = df[df["restaurant_id"] == restaurant_id]
    return df.reset_index(drop=True)


def _safe_sum(s):
    return float(pd.to_numeric(s, errors="coerce").fillna(0).sum())


def _safe_mean(s):
    s = pd.to_numeric(s, errors="coerce")
    return float(s.mean()) if len(s) else 0.0


def _build_platform_block(df: pd.DataFrame, platform_name: str) -> Dict:
    block: Dict = {"platform": platform_name}
    block["sales"] = _safe_sum(df.get("sales"))
    block["orders"] = int(pd.to_numeric(df.get("orders"), errors="coerce").fillna(0).sum())
    block["cancelled_orders"] = int(pd.to_numeric(df.get("cancelled_orders"), errors="coerce").fillna(0).sum()) if "cancelled_orders" in df.columns else None
    block["lost_orders"] = int(pd.to_numeric(df.get("lost_orders"), errors="coerce").fillna(0).sum()) if "lost_orders" in df.columns else None
    block["rating_avg"] = _safe_mean(df.get("rating"))
    block["ads_spend"] = _safe_sum(df.get("ads_spend"))
    block["ads_sales"] = _safe_sum(df.get("ads_sales"))
    block["impressions"] = int(pd.to_numeric(df.get("impressions"), errors="coerce").fillna(0).sum()) if "impressions" in df.columns else None
    # ROAS
    block["roas"] = (block["ads_sales"] / block["ads_spend"]) if block["ads_spend"] else None
    # Operations
    block["driver_waiting_minutes"] = _safe_mean(df.get("driver_waiting")) if "driver_waiting" in df.columns else None
    block["accepting_time_min"] = _safe_mean(df.get("accepting_time")) if "accepting_time" in df.columns else None
    block["preparation_time_min"] = _safe_mean(df.get("preparation_time")) if "preparation_time" in df.columns else None
    block["delivery_time_min"] = _safe_mean(df.get("delivery_time")) if "delivery_time" in df.columns else None
    # Outages
    block["offline_rate_avg"] = _safe_mean(df.get("offline_rate")) if "offline_rate" in df.columns else None
    block["close_time_total_min"] = _safe_sum(df.get("close_time")) if "close_time" in df.columns else None
    # Funnel (Grab only typically)
    if platform_name.lower() == "grab":
        for k in [
            "unique_impressions_reach",
            "unique_menu_visits",
            "unique_add_to_carts",
            "unique_conversion_reach",
        ]:
            if k in df.columns:
                block[k] = int(pd.to_numeric(df.get(k), errors="coerce").fillna(0).sum())
        # CTR and conversions if possible
        impr = block.get("impressions") or block.get("unique_impressions_reach") or 0
        clicks = block.get("unique_menu_visits") or 0
        add_to_cart = block.get("unique_add_to_carts") or 0
        conversions = block.get("unique_conversion_reach") or 0
        block["ctr"] = (clicks / impr) if impr else None
        block["conv_click_to_order"] = (conversions / clicks) if clicks else None
        block["conv_cart_to_order"] = (conversions / add_to_cart) if add_to_cart else None
    return block


def _weekend_weekday(df_all: pd.DataFrame) -> Dict:
    if df_all.empty:
        return {"weekend_avg": 0.0, "weekday_avg": 0.0, "effect_pct": 0.0}
    tmp = df_all.copy()
    tmp["date"] = pd.to_datetime(tmp["stat_date"]).dt.normalize()
    agg = tmp.groupby("date", as_index=False)["sales"].sum()
    agg["dow"] = agg["date"].dt.weekday
    weekend = agg[agg["dow"].isin([5, 6])]["sales"].mean()
    weekday = agg[~agg["dow"].isin([5, 6])]["sales"].mean()
    effect = (weekend - weekday) / weekday * 100.0 if pd.notnull(weekday) and weekday else 0.0
    return {"weekend_avg": float(weekend or 0.0), "weekday_avg": float(weekday or 0.0), "effect_pct": float(effect)}


def build_basic_report(period: str, restaurant_id: Optional[int]) -> Dict:
    start_str, end_str = period.split("_")
    start = pd.to_datetime(start_str).date()
    end = pd.to_datetime(end_str).date()

    grab = _read_stats("grab_stats", restaurant_id, start, end)
    gojek = _read_stats("gojek_stats", restaurant_id, start, end)

    # Executive summary
    grab_block = _build_platform_block(grab, "GRAB") if not grab.empty else {}
    gojek_block = _build_platform_block(gojek, "GOJEK") if not gojek.empty else {}

    total_sales = (grab_block.get("sales", 0.0) + gojek_block.get("sales", 0.0))
    total_orders = (grab_block.get("orders", 0) + gojek_block.get("orders", 0))
    successful_orders = None
    # Prefer accepted_orders if present (gojek)
    if "accepted_orders" in gojek.columns:
        successful_orders = int(pd.to_numeric(gojek["accepted_orders"], errors="coerce").fillna(0).sum())
        # if grab has accepted proxy, can adjust here if available later
    aov = (total_sales / successful_orders) if successful_orders else (total_sales / total_orders if total_orders else None)

    # Weekend vs Weekday
    both = pd.concat([grab[["stat_date", "sales"]]] if not grab.empty else [] + [gojek[["stat_date", "sales"]]] if not gojek.empty else [], ignore_index=True) if (not grab.empty or not gojek.empty) else pd.DataFrame(columns=["stat_date","sales"])
    weekend_info = _weekend_weekday(both)

    # Best/Worst day
    best_day = None
    worst_day = None
    if not both.empty:
        day_sales = both.copy()
        day_sales["date"] = pd.to_datetime(day_sales["stat_date"]).dt.normalize()
        day_sales = day_sales.groupby("date", as_index=False)["sales"].sum()
        if not day_sales.empty:
            best = day_sales.sort_values("sales", ascending=False).iloc[0]
            worst = day_sales.sort_values("sales", ascending=True).iloc[0]
            best_day = {"date": str(best["date"].date()), "sales": float(best["sales"]) }
            worst_day = {"date": str(worst["date"].date()), "sales": float(worst["sales"]) }

    # Marketing totals
    ads_spend_total = grab_block.get("ads_spend", 0.0) + gojek_block.get("ads_spend", 0.0)
    ads_sales_total = grab_block.get("ads_sales", 0.0) + gojek_block.get("ads_sales", 0.0)
    roas_total = (ads_sales_total / ads_spend_total) if ads_spend_total else None

    return {
        "period": period,
        "restaurant_id": restaurant_id,
        "executive_summary": {
            "revenue_total": total_sales,
            "orders_total": total_orders,
            "successful_orders": successful_orders,
            "aov": aov,
            "by_platform": {
                "grab": grab_block,
                "gojek": gojek_block,
            },
            "roas_total": roas_total,
            "ads_spend_total": ads_spend_total,
        },
        "sales_trends": {
            "weekend_vs_weekday": weekend_info,
            "best_day": best_day,
            "worst_day": worst_day,
        },
    }