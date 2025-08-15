from __future__ import annotations

from typing import Dict, Optional, Tuple
from datetime import date
import pandas as pd
import numpy as np
import calendar

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
    # Funnel (GRAB only typically)
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


def _month_dynamics(df_all: pd.DataFrame, start: date, end: date) -> Dict[str, Dict[str, float]]:
    if df_all.empty:
        return {}
    tmp = df_all.copy()
    tmp["date"] = pd.to_datetime(tmp["stat_date"]).dt.normalize()
    tmp["ym"] = tmp["date"].dt.to_period("M")
    res: Dict[str, Dict[str, float]] = {}
    for ym, group in tmp.groupby("ym"):
        month_str = str(ym)
        # days in period within that month (inclusive)
        first = max(pd.Timestamp(str(ym) + "-01").date(), start)
        y, m = int(str(ym).split('-')[0]), int(str(ym).split('-')[1])
        last_day = calendar.monthrange(y, m)[1]
        last_month_date = pd.Timestamp(f"{y:04d}-{m:02d}-{last_day:02d}").date()
        last = min(last_month_date, end)
        num_days = max(0, (pd.to_datetime(last) - pd.to_datetime(first)).days + 1)
        total_sales = float(pd.to_numeric(group["sales"], errors="coerce").fillna(0).sum())
        avg_per_day = total_sales / num_days if num_days > 0 else 0.0
        res[month_str] = {"total_sales": total_sales, "days": int(num_days), "avg_per_day": avg_per_day}
    return res


def _best_worst_with_platforms(grab: pd.DataFrame, gojek: pd.DataFrame) -> Tuple[Optional[Dict], Optional[Dict]]:
    if grab.empty and gojek.empty:
        return None, None
    both = []
    if not grab.empty:
        g = grab[["stat_date", "sales", "orders"]].copy()
        g["platform"] = "GRAB"
        both.append(g)
    if not gojek.empty:
        j = gojek[["stat_date", "sales", "orders"]].copy()
        j["platform"] = "GOJEK"
        both.append(j)
    both_df = pd.concat(both, ignore_index=True)
    both_df["date"] = pd.to_datetime(both_df["stat_date"]).dt.normalize()
    day_sales = both_df.groupby(["date"], as_index=False)["sales"].sum()
    if day_sales.empty:
        return None, None
    best = day_sales.sort_values("sales", ascending=False).iloc[0]
    worst = day_sales.sort_values("sales", ascending=True).iloc[0]
    def breakdown(d):
        ddf = both_df[both_df["date"] == d][["platform", "sales", "orders"]].groupby("platform", as_index=False).sum()
        parts = {row["platform"].lower(): {"sales": float(row["sales"]), "orders": int(row["orders"]) } for _, row in ddf.iterrows()}
        return parts
    best_day = {"date": str(best["date"].date()), "total_sales": float(best["sales"]), "by_platform": breakdown(best["date"]) }
    worst_day = {"date": str(worst["date"].date()), "total_sales": float(worst["sales"]), "by_platform": breakdown(worst["date"]) }
    return best_day, worst_day


def _workday_stats(df_all: pd.DataFrame) -> Dict[str, float]:
    if df_all.empty:
        return {"avg_workdays": 0.0, "spread_pct": 0.0, "cv_pct": 0.0, "avg_all_days": 0.0}
    tmp = df_all.copy()
    tmp["date"] = pd.to_datetime(tmp["stat_date"]).dt.normalize()
    daily = tmp.groupby("date", as_index=False)["sales"].sum()
    # All days average
    avg_all = float(daily["sales"].mean()) if not daily.empty else 0.0
    daily["dow"] = daily["date"].dt.weekday
    wd = daily[~daily["dow"].isin([5, 6])]
    if wd.empty:
        return {"avg_workdays": 0.0, "spread_pct": 0.0, "cv_pct": 0.0, "avg_all_days": avg_all}
    mean = wd["sales"].mean()
    std = wd["sales"].std(ddof=0)
    minv = wd["sales"].min()
    maxv = wd["sales"].max()
    spread = (maxv / minv - 1.0) * 100.0 if minv else 0.0
    cv = (std / mean) * 100.0 if mean else 0.0
    return {"avg_workdays": float(mean), "spread_pct": float(spread), "cv_pct": float(cv), "avg_all_days": float(avg_all)}


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
    # Successful orders by platform if fake known (handled elsewhere); here keep placeholders, can be passed in later
    # If accepted_orders column exists in platform tables, we can use it (present in gojek)
    successful_grab = None
    successful_gojek = None
    if not grab.empty and "orders" in grab.columns and "cancelled_orders" in grab.columns:
        # fake will be incorporated at ETL time; here just orders - cancelled if no fake data
        pass
    if not gojek.empty and "accepted_orders" in gojek.columns:
        successful_gojek = int(pd.to_numeric(gojek["accepted_orders"], errors="coerce").fillna(0).sum())
    # AOVs (require successful counts); will be computed by caller if necessary

    # Weekend vs Weekday
    parts = []
    if not grab.empty:
        parts.append(grab[["stat_date", "sales"]])
    if not gojek.empty:
        parts.append(gojek[["stat_date", "sales"]])
    both_all = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["stat_date","sales"])
    weekend_info = _weekend_weekday(both_all)

    # Best/Worst with platform breakdown
    best_day, worst_day = _best_worst_with_platforms(grab, gojek)

    # Monthly dynamics
    month_dyn = _month_dynamics(both_all, start, end)

    # Workday stats
    wd_stats = _workday_stats(both_all)

    result = {
        "period": period,
        "restaurant_id": restaurant_id,
        "executive_summary": {
            "revenue_total": total_sales,
            "orders_total": total_orders,
            "by_platform": {
                "grab": grab_block,
                "gojek": gojek_block,
            },
            "daily_revenue_workdays_avg": wd_stats.get("avg_workdays", 0.0),
            "daily_revenue_all_avg": wd_stats.get("avg_all_days", 0.0),
        },
        "sales_trends": {
            "weekend_vs_weekday": weekend_info,
            "best_day": best_day,
            "worst_day": worst_day,
            "monthly": month_dyn,
            "spread_workdays_pct": wd_stats.get("spread_pct", 0.0),
            "cv_workdays_pct": wd_stats.get("cv_pct", 0.0),
        },
    }
    return result