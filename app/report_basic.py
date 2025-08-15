from __future__ import annotations

from typing import Dict, Optional, Tuple
from datetime import date
import pandas as pd
import numpy as np
import calendar

from etl.data_loader import get_engine, load_fake_orders
from etl.feature_engineering import parse_tourist_flow


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


def _build_marketing_section(restaurant_id: Optional[int], start: date, end: date) -> Dict:
    eng = get_engine()
    # GRAB stats (funnel available)
    qg = (
        "SELECT stat_date, impressions, unique_impressions_reach, unique_menu_visits, "
        "unique_add_to_carts, unique_conversion_reach, ads_orders, ads_spend, ads_sales "
        "FROM grab_stats WHERE stat_date BETWEEN ? AND ?"
        + (" AND restaurant_id=?" if restaurant_id is not None else "")
    )
    params = tuple([str(start), str(end)] + ([restaurant_id] if restaurant_id is not None else []))
    grab = pd.read_sql_query(qg, eng, params=params, parse_dates=["stat_date"]) if eng else pd.DataFrame()
    grab.columns = [c.lower() for c in grab.columns]
    funnel = {}
    if not grab.empty:
        impr = int(pd.to_numeric(grab.get("impressions"), errors="coerce").fillna(0).sum())
        reach = int(pd.to_numeric(grab.get("unique_impressions_reach"), errors="coerce").fillna(0).sum())
        menu = int(pd.to_numeric(grab.get("unique_menu_visits"), errors="coerce").fillna(0).sum())
        cart = int(pd.to_numeric(grab.get("unique_add_to_carts"), errors="coerce").fillna(0).sum())
        # Orders from ads: prefer ads_orders; fallback to unique_conversion_reach
        ads_orders = int(pd.to_numeric(grab.get("ads_orders"), errors="coerce").fillna(0).sum())
        if ads_orders == 0 and "unique_conversion_reach" in grab.columns:
            ads_orders = int(pd.to_numeric(grab.get("unique_conversion_reach"), errors="coerce").fillna(0).sum())
        spend = float(pd.to_numeric(grab.get("ads_spend"), errors="coerce").fillna(0).sum())
        ads_sales = float(pd.to_numeric(grab.get("ads_sales"), errors="coerce").fillna(0).sum())

        ctr = (menu / impr) if impr else None
        conv_click_to_order = (ads_orders / menu) if menu else None
        conv_cart_to_order = (ads_orders / cart) if cart else None
        show_to_order = (ads_orders / impr) if impr else None
        bounce_rate = (1 - (cart / menu)) if menu else None
        abandoned_carts = (1 - (ads_orders / cart)) if cart else None

        avg_ads_order_value = (ads_sales / ads_orders) if ads_orders else None
        # Potential uplift estimates
        uplift = {}
        if avg_ads_order_value is not None:
            # Reduce bounce by 10% of bouncers
            bouncers = max(menu - cart, 0)
            extra_clicks = 0.1 * bouncers
            extra_orders_from_bounce = extra_clicks * (conv_cart_to_order or 0.0)
            uplift_bounce = extra_orders_from_bounce * avg_ads_order_value
            # Eliminate abandoned carts
            missed_orders = max(cart - ads_orders, 0)
            uplift_abandoned = missed_orders * avg_ads_order_value
            uplift = {
                "reduce_bounce_10_pct_revenue": float(uplift_bounce),
                "eliminate_abandoned_revenue": float(uplift_abandoned),
                "total_uplift": float(uplift_bounce + uplift_abandoned),
            }

        funnel = {
            "impressions": impr,
            "unique_reach": reach,
            "menu_visits": menu,
            "add_to_cart": cart,
            "ads_orders": ads_orders,
            "ctr": float(ctr) if ctr is not None else None,
            "conv_click_to_order": float(conv_click_to_order) if conv_click_to_order is not None else None,
            "conv_cart_to_order": float(conv_cart_to_order) if conv_cart_to_order is not None else None,
            "show_to_order": float(show_to_order) if show_to_order is not None else None,
            "bounce_rate": float(bounce_rate) if bounce_rate is not None else None,
            "abandoned_carts_rate": float(abandoned_carts) if abandoned_carts is not None else None,
            "ads_spend": float(spend),
            "ads_sales": float(ads_sales),
            "avg_ads_order_value": float(avg_ads_order_value) if avg_ads_order_value is not None else None,
            "uplift_estimations": uplift,
            "cpc": float(spend / menu) if menu else None,
            "cpa": float(spend / ads_orders) if ads_orders else None,
        }

    # ROAS by month per platform
    def roas_month(table: str) -> Dict[str, float]:
        q = (
            "SELECT strftime('%Y-%m', stat_date) ym, SUM(ads_sales) s, SUM(ads_spend) b "
            f"FROM {table} WHERE stat_date BETWEEN ? AND ?"
            + (" AND restaurant_id=?" if restaurant_id is not None else "")
            + " GROUP BY ym"
        )
        params = tuple([str(start), str(end)] + ([restaurant_id] if restaurant_id is not None else []))
        df = pd.read_sql_query(q, eng, params=params, parse_dates=["ym"]) if eng else pd.DataFrame()
        res = {}
        if not df.empty:
            for _, row in df.iterrows():
                b = float(row["b"]) if row["b"] else 0.0
                s = float(row["s"]) if row["s"] else 0.0
                res[str(row["ym"])[:7]] = (s / b) if b else None
        return res

    roas = {"grab": roas_month("grab_stats"), "gojek": roas_month("gojek_stats")}

    # Seasonal context: tourist flow (use last full year available in Excel)
    seasonal = {}
    try:
        tf = parse_tourist_flow([
            "/workspace/1.-Data-Kunjungan-2025-3.xls",
            "/workspace/Table-1-7-Final-1-1.xls",
        ])
        if not tf.empty:
            tf["date"] = pd.to_datetime(tf["date"]).dt.normalize()
            tf2024 = tf[(tf["date"] >= "2024-01-01") & (tf["date"] < "2025-01-01")].copy()
            if not tf2024.empty:
                tf2024["ym"] = tf2024["date"].dt.to_period("M")
                monthly = tf2024.groupby("ym")["tourist_flow"].sum()
                if not monthly.empty:
                    seasonal = {
                        "2024-04": float(monthly.get(pd.Period("2024-04"), 0.0)),
                        "2024-05": float(monthly.get(pd.Period("2024-05"), 0.0)),
                        "peak_month": str(monthly.idxmax()),
                        "peak_value": float(monthly.max()),
                        "low_month": str(monthly.idxmin()),
                        "low_value": float(monthly.min()),
                    }
    except Exception:
        seasonal = {}

    return {
        "funnel_grab": funnel,
        "roas_by_month": roas,
        "seasonal_context": seasonal,
    }

def _sum_platform(eng, table: str, restaurant_id: Optional[int], start: date, end: date) -> Dict[str, float]:
    base = f"SELECT SUM(sales) sales, SUM(payouts) payouts, SUM(ads_spend) ads_spend, SUM(ads_sales) ads_sales"
    if table == "grab_stats":
        base += ", SUM(ads_orders) ads_orders"
    base += f" FROM {table} WHERE stat_date BETWEEN ? AND ?"
    params = [str(start), str(end)]
    if restaurant_id is not None:
        base += " AND restaurant_id=?"
        params.append(restaurant_id)
    df = pd.read_sql_query(base, eng, params=tuple(params))
    rec = df.iloc[0].to_dict() if not df.empty else {}
    out = {k: float(rec.get(k) or 0.0) for k in ["sales", "payouts", "ads_spend", "ads_sales"]}
    if table == "grab_stats":
        out["ads_orders"] = float(rec.get("ads_orders") or 0.0)
    return out


def _monthly_platform(eng, table: str, restaurant_id: Optional[int], start: date, end: date) -> pd.DataFrame:
    q = (
        f"SELECT strftime('%Y-%m', stat_date) ym, SUM(payouts) payouts, SUM(ads_spend) ads_spend, SUM(ads_sales) ads_sales "
        f"FROM {table} WHERE stat_date BETWEEN ? AND ?"
        + (" AND restaurant_id=?" if restaurant_id is not None else "")
        + " GROUP BY ym"
    )
    params = [str(start), str(end)] + ([restaurant_id] if restaurant_id is not None else [])
    df = pd.read_sql_query(q, eng, params=tuple(params))
    df["ym"] = df["ym"].astype(str)
    return df


def _build_financial_section(restaurant_id: Optional[int], start: date, end: date) -> Dict:
    eng = get_engine()
    grab = _sum_platform(eng, "grab_stats", restaurant_id, start, end)
    gojek = _sum_platform(eng, "gojek_stats", restaurant_id, start, end)
    totals = {k: grab.get(k, 0.0) + gojek.get(k, 0.0) for k in ["sales", "payouts", "ads_spend", "ads_sales"]}

    # Take rate (effective commission share) per platform
    def take_rate(p):
        sales = p.get("sales", 0.0)
        fees = p.get("sales", 0.0) - p.get("payouts", 0.0) - p.get("ads_spend", 0.0)
        return (fees / sales) if sales else None, fees

    grab_tr, grab_comm = take_rate(grab)
    gojek_tr, gojek_comm = take_rate(gojek)
    total_tr, total_comm = take_rate(totals)

    # Net ROAS: attribute commission proportionally to ad sales
    def net_roas(p, comm):
        sales = p.get("sales", 0.0)
        ads_sales = p.get("ads_sales", 0.0)
        ads_spend = p.get("ads_spend", 0.0)
        if ads_spend == 0:
            return None
        comm_ads = (comm * (ads_sales / sales)) if sales else 0.0
        return ((ads_sales - comm_ads) / ads_spend) if ads_spend else None

    grab_net_roas = net_roas(grab, grab_comm)
    gojek_net_roas = net_roas(gojek, gojek_comm)

    # Contribution per ad order (when ads_orders available)
    grab_cpa = (grab["ads_spend"] / grab["ads_orders"]) if grab.get("ads_orders", 0.0) else None
    grab_avg_ads_order = (grab["ads_sales"] / grab["ads_orders"]) if grab.get("ads_orders", 0.0) else None
    grab_comm_rate = (grab_comm / grab["sales"]) if grab.get("sales", 0.0) else None
    grab_contribution = None
    if grab_avg_ads_order is not None and grab_comm_rate is not None and grab_cpa is not None:
        grab_contribution = grab_avg_ads_order * (1.0 - grab_comm_rate) - grab_cpa

    # Waterfall components per platform
    def waterfall(p, comm):
        return {
            "sales": p.get("sales", 0.0),
            "commission": comm,
            "ads_spend": p.get("ads_spend", 0.0),
            "payouts": p.get("payouts", 0.0),
        }

    waterfall_total = waterfall(totals, total_comm)
    waterfall_grab = waterfall(grab, grab_comm)
    waterfall_gojek = waterfall(gojek, gojek_comm)

    # MoM comparisons for payouts and ROAS
    mg = _monthly_platform(eng, "grab_stats", restaurant_id, start, end)
    mj = _monthly_platform(eng, "gojek_stats", restaurant_id, start, end)
    mom = {}
    for ym in sorted(set(mg.get("ym", pd.Series()).tolist() + mj.get("ym", pd.Series()).tolist())):
        mom[ym] = {
            "payouts": float((mg.loc[mg["ym"] == ym, "payouts"].sum() + mj.loc[mj["ym"] == ym, "payouts"].sum())),
            "roas_grab": float((mg.loc[mg["ym"] == ym, "ads_sales"].sum() / mg.loc[mg["ym"] == ym, "ads_spend"].sum())) if mg.loc[mg["ym"] == ym, "ads_spend"].sum() else None,
            "roas_gojek": float((mj.loc[mj["ym"] == ym, "ads_sales"].sum() / mj.loc[mj["ym"] == ym, "ads_spend"].sum())) if mj.loc[mj["ym"] == ym, "ads_spend"].sum() else None,
        }

    return {
        "payouts": {
            "grab": grab.get("payouts", 0.0),
            "gojek": gojek.get("payouts", 0.0),
            "total": totals.get("payouts", 0.0),
        },
        "ad_sales": totals.get("ads_sales", 0.0),
        "ad_sales_share": (totals.get("ads_sales", 0.0) / totals.get("sales", 0.0)) if totals.get("sales", 0.0) else None,
        "roas": {
            "grab": (grab.get("ads_sales", 0.0) / grab.get("ads_spend", 0.0)) if grab.get("ads_spend", 0.0) else None,
            "gojek": (gojek.get("ads_sales", 0.0) / gojek.get("ads_spend", 0.0)) if gojek.get("ads_spend", 0.0) else None,
        },
        "take_rate": {
            "grab": float(grab_tr) if grab_tr is not None else None,
            "gojek": float(gojek_tr) if gojek_tr is not None else None,
            "total": float(total_tr) if total_tr is not None else None,
        },
        "net_roas": {
            "grab": float(grab_net_roas) if grab_net_roas is not None else None,
            "gojek": float(gojek_net_roas) if gojek_net_roas is not None else None,
        },
        "contribution_per_ad_order_grab": float(grab_contribution) if grab_contribution is not None else None,
        "waterfall": {
            "total": waterfall_total,
            "grab": waterfall_grab,
            "gojek": waterfall_gojek,
        },
        "monthly": mom,
        "explain": {
            "take_rate": "Доля комиссий и удержаний: (выручка − выплаты − рекламный бюджет) ÷ выручка",
            "net_roas": "Чистый ROAS: (рекламные продажи − доля комиссий, приходящаяся на рекламные продажи) ÷ рекламный бюджет",
            "contribution": "Юнит‑экономика рекламы (GRAB): средний чек рекламного заказа × (1 − take rate) − CPA",
            "waterfall": "Водопад денежных потоков: выручка → комиссии → рекламный бюджет → выплаты",
        }
    }


def build_marketing_report(period: str, restaurant_id: Optional[int]) -> Dict:
    start_str, end_str = period.split("_")
    start = pd.to_datetime(start_str).date()
    end = pd.to_datetime(end_str).date()
    return _build_marketing_section(restaurant_id, start, end)


def _get_restaurant_name(eng, restaurant_id: int) -> Optional[str]:
    try:
        df = pd.read_sql_query("SELECT name FROM restaurants WHERE id = ?", eng, params=(restaurant_id,))
        if not df.empty:
            return str(df.iloc[0]["name"]) if df.iloc[0]["name"] is not None else None
    except Exception:
        return None
    return None


def _fake_orders_for_platform_period(restaurant_name: Optional[str], platform_key: str, start: date, end: date) -> int:
    try:
        df = load_fake_orders()
        if df is None or df.empty:
            return 0
        cols = {c.lower(): c for c in df.columns}
        # Expected Russian headers per sample
        col_name = cols.get('какой ресторан?') or cols.get('restaurant') or None
        col_date = cols.get('дата накрутки') or cols.get('date') or None
        col_cnt  = cols.get('количество сделанных фейк заказов?') or cols.get('count') or None
        col_plat = cols.get('какая платформа?') or cols.get('platform') or None
        if not (col_date and col_cnt and col_plat):
            return 0
        ff = df.copy()
        # Filter by platform
        ff[col_plat] = ff[col_plat].astype(str).str.lower()
        key = platform_key.lower()
        # Allow aliases
        if key == 'gojek':
            plat_mask = ff[col_plat].str.contains('gojek', na=False)
        elif key == 'grab':
            plat_mask = ff[col_plat].str.contains('grab', na=False)
        else:
            plat_mask = ff[col_plat].str.contains(key, na=False)
        ff = ff[plat_mask]
        # Filter by restaurant name if available
        if restaurant_name and col_name in ff.columns:
            ff[col_name] = ff[col_name].astype(str)
            ff = ff[ff[col_name].str.contains(restaurant_name, case=False, na=False)]
        # Filter by date range
        ff[col_date] = pd.to_datetime(ff[col_date], errors='coerce').dt.date
        ff = ff[(ff[col_date] >= start) & (ff[col_date] <= end)]
        return int(pd.to_numeric(ff[col_cnt], errors='coerce').fillna(0).sum())
    except Exception:
        return 0


def build_quality_report(period: str, restaurant_id: int) -> Dict:
    eng = get_engine()
    start_str, end_str = period.split("_")
    start = pd.to_datetime(start_str).date()
    end = pd.to_datetime(end_str).date()
    # Ratings from GOJEK
    q = (
        "SELECT SUM(five_star_ratings) five, SUM(four_star_ratings) four, "
        "SUM(three_star_ratings) three, SUM(two_star_ratings) two, SUM(one_star_ratings) one, "
        "SUM(orders) orders, SUM(accepted_orders) accepted, SUM(cancelled_orders) cancelled, SUM(lost_orders) lost "
        "FROM gojek_stats WHERE restaurant_id=? AND stat_date BETWEEN ? AND ?"
    )
    df = pd.read_sql_query(q, eng, params=(restaurant_id, str(start), str(end)))
    row = df.iloc[0] if not df.empty else pd.Series()
    five = int(row.get('five') or 0)
    four = int(row.get('four') or 0)
    three = int(row.get('three') or 0)
    two = int(row.get('two') or 0)
    one = int(row.get('one') or 0)
    orders = int(row.get('orders') or 0)
    cancelled = int(row.get('cancelled') or 0)
    lost = int(row.get('lost') or 0)
    # Fake from sheet
    rest_name = _get_restaurant_name(eng, restaurant_id)
    fake = _fake_orders_for_platform_period(rest_name, 'gojek', start, end)

    total_ratings = five + four + three + two + one
    idx = (5*five + 4*four + 3*three + 2*two + 1*one)/total_ratings if total_ratings else None
    neg_1_2 = one + two
    bad_not5 = total_ratings - five
    # Successful orders base = orders − cancelled − lost − fake
    successful = max(orders - cancelled - lost - fake, 0)
    per_bad = (successful / bad_not5) if bad_not5 else None

    return {
        "ratings": {
            "total": total_ratings,
            "five": five,
            "four": four,
            "three": three,
            "two": two,
            "one": one,
            "satisfaction_index": round(idx, 2) if idx is not None else None,
            "negative_1_2": {
                "count": neg_1_2,
                "percent": round(neg_1_2/total_ratings*100, 1) if total_ratings else None,
            },
            "not_five": {
                "count": bad_not5,
                "percent": round(bad_not5/total_ratings*100, 1) if total_ratings else None,
            },
        },
        "successful_orders_gojek": successful,
        "orders_per_not_five_rating": round(per_bad, 1) if per_bad is not None else None,
        "explain": {
            "successful_orders": "Успешные = заказы − отменённые − потерянные − fake (по данным GOJEK и таблицы fake)",
            "orders_per_not_five": "Сколько успешных заказов приходится на одну оценку ниже 5★",
        }
    }


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
        "marketing": _build_marketing_section(restaurant_id, start, end),
        "finance": _build_financial_section(restaurant_id, start, end),
    }
    return result