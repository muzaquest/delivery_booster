"""ETL Feature Engineering to build merged daily dataset.

This module merges aggregated orders, daily weather (cached via Open-Meteo), and
parsed tourist flow into a per-restaurant, per-day dataset, and generates
standard temporal and lag features required for modeling.
"""

from __future__ import annotations

from typing import List, Optional
import pandas as pd
from sqlalchemy.engine import Engine

from etl.data_loader import (
    load_orders,
    load_restaurants,
    parse_tourist_flow,
    get_weather_series_for_restaurant,
    load_operations,
    load_marketing,
    load_platform_outages,
)


# Placeholder: holidays loader function signature (will implement in holidays_loader.py)
def load_holidays_df(start_date: str, end_date: str) -> pd.DataFrame:
    try:
        from etl.holidays_loader import load_holidays_df as _impl

        return _impl(start_date, end_date)
    except Exception:
        return pd.DataFrame(columns=["date", "holiday_name", "region"])  # empty if unavailable


def _generate_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_of_week"] = df["date"].dt.weekday
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df


def _generate_lags(
    df: pd.DataFrame,
    group_cols: List[str],
    target_cols: List[str],
    lags: List[int],
    prefix: Optional[str] = None,
) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(group_cols + ["date"])  # ensure chronological order
    grouped = df.groupby(group_cols, group_keys=False)
    for col in target_cols:
        for lag in lags:
            new_col = f"{prefix + '_' if prefix else ''}{col}_lag_{lag}"
            df[new_col] = grouped[col].shift(lag)
    return df


def _generate_rolling_means(
    df: pd.DataFrame,
    group_cols: List[str],
    target_cols: List[str],
    windows: List[int],
    prefix: Optional[str] = None,
) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(group_cols + ["date"])  # ensure chronological order
    grouped = df.groupby(group_cols, group_keys=False)
    for col in target_cols:
        for win in windows:
            new_col = f"{prefix + '_' if prefix else ''}{col}_rolling_mean_{win}"
            df[new_col] = grouped[col].transform(lambda s: s.rolling(window=win, min_periods=1).mean())
    return df


def build_and_save_dataset(
    engine: Engine,
    start_date: str = "2024-01-01",
    end_date: str = "2025-12-31",
    output_csv_path: str = "/workspace/data/merged_dataset.csv",
    excel_paths: Optional[list[str]] = None,
    fake_orders_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build merged dataset and save to CSV.

    Steps:
      1) Load orders and restaurants from SQLite (excluding fake orders)
      2) Aggregate orders to daily per restaurant
      3) Parse tourist flow from Excel and resample to daily
      4) Pull weather per (restaurant, date) from cache/API
      5) Merge holidays and generate features
      6) Save to CSV
    """
    if excel_paths is None:
        excel_paths = [
            "/workspace/1.-Data-Kunjungan-2025-3.xls",
            "/workspace/Table-1-7-Final-1-1.xls",
        ]

    restaurants_df = load_restaurants(engine)
    orders_daily = load_orders(engine, fake_orders_df=fake_orders_df)

    # Restrict to requested period
    orders_daily = orders_daily[(orders_daily["date"] >= pd.to_datetime(start_date)) & (orders_daily["date"] <= pd.to_datetime(end_date))]

    # Tourist flow (date-level), resample to daily and forward-fill
    tourist_flow = parse_tourist_flow(excel_paths)
    if tourist_flow.empty:
        tourist_flow = pd.DataFrame({"date": pd.date_range(start_date, end_date), "tourist_flow": 0.0})
    else:
        tourist_flow = tourist_flow.copy()
        tourist_flow["date"] = pd.to_datetime(tourist_flow["date"]).dt.normalize()
        tourist_flow = (
            tourist_flow.set_index("date")["tourist_flow"].resample("D").mean().ffill().bfill().reset_index()
        )

    # Holidays
    holidays_df = load_holidays_df(start_date, end_date)
    holidays_df = holidays_df.copy()
    if not holidays_df.empty:
        holidays_df["date"] = pd.to_datetime(holidays_df["date"]).dt.normalize()
        holidays_flags = holidays_df.drop_duplicates(["date"]).assign(is_holiday=1)[["date", "is_holiday"]]
    else:
        holidays_flags = pd.DataFrame({"date": pd.date_range(start_date, end_date), "is_holiday": 0})

    # Build weather dataset for each (restaurant_id, date) present in orders
    unique_restaurant_ids = orders_daily["restaurant_id"].unique().tolist()
    all_weather_parts: list[pd.DataFrame] = []
    for restaurant_id in unique_restaurant_ids:
        dates_for_restaurant = orders_daily.loc[orders_daily["restaurant_id"] == restaurant_id, "date"].sort_values().unique()
        if len(dates_for_restaurant) == 0:
            continue
        start_d = pd.to_datetime(dates_for_restaurant.min()).date()
        end_d = pd.to_datetime(dates_for_restaurant.max()).date()
        weather_df = get_weather_series_for_restaurant(restaurant_id, start_d, end_d, engine)
        all_weather_parts.append(weather_df)

    if len(all_weather_parts) == 0:
        weather_daily = pd.DataFrame(columns=["restaurant_id", "date", "temp", "rain", "wind", "humidity"])
    else:
        weather_daily = pd.concat(all_weather_parts, ignore_index=True)

    # Merge
    merged = orders_daily.merge(weather_daily, on=["restaurant_id", "date"], how="left")
    merged = merged.merge(tourist_flow, on="date", how="left")
    merged = merged.merge(holidays_flags, on="date", how="left")
    merged["is_holiday"] = merged["is_holiday"].fillna(0).astype(int)

    # Bring in platform-level operations and marketing, outages
    ops = load_operations(engine)
    mkt = load_marketing(engine)
    outages = load_platform_outages(engine)

    # Pivot platform metrics into columns (Grab/Gojek), suffix per metric
    def _pivot_platform(df: pd.DataFrame, value_cols: list[str], prefix: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["restaurant_id", "date"])
        df = df.copy()
        df["platform"] = df["platform"].fillna("").str.lower()
        # keep only existing value columns
        present_vals = [c for c in value_cols if c in df.columns]
        if not present_vals:
            return pd.DataFrame(columns=["restaurant_id", "date"])
        keep_cols = ["restaurant_id", "date", "platform"] + present_vals
        df = df[keep_cols]
        wide = df.pivot_table(index=["restaurant_id", "date"], columns="platform", values=present_vals, aggfunc="mean")
        # Flatten multiindex columns: (col, platform)
        wide.columns = [f"{prefix}_{col}_{plat}" for (col, plat) in wide.columns.to_flat_index()]
        wide = wide.reset_index()
        return wide

    ops_wide = _pivot_platform(ops, ["accepting_time", "delivery_time", "preparation_time", "rating"], prefix="ops")
    mkt_wide = _pivot_platform(mkt, ["ads_spend", "roas", "impressions", "clicks"], prefix="mkt")
    out_wide = _pivot_platform(outages, ["offline_minutes", "offline_rate", "close_time"], prefix="outage")

    for extra in (ops_wide, mkt_wide, out_wide):
        merged = merged.merge(extra, on=["restaurant_id", "date"], how="left")

    # Create combined features per spec
    # Marketing totals
    merged["ads_spend_total"] = merged[[c for c in merged.columns if c.startswith("mkt_ads_spend_")]].sum(axis=1)
    merged["impressions_total"] = merged[[c for c in merged.columns if c.startswith("mkt_impressions_")]].sum(axis=1)
    # ROAS platform-specific preserved: mkt_roas_grab, mkt_roas_gojek (если есть такие платформы в данных)

    # Ratings per platform kept: ops_rating_grab/gojek
    # Operational times
    for base in ("accepting_time", "delivery_time", "preparation_time"):
        cols = [c for c in merged.columns if c.startswith(f"ops_{base}_")]
        if cols:
            merged[f"{base}_mean"] = merged[cols].mean(axis=1)

    # Outages short-hands
    for base in ("offline_minutes", "offline_rate", "close_time"):
        cols = [c for c in merged.columns if c.startswith(f"outage_{base}_")]
        if cols:
            merged[f"{base}_sum"] = merged[cols].sum(axis=1)

    # Temporal features
    merged["date"] = pd.to_datetime(merged["date"])  # ensure dtype
    merged = _generate_temporal_features(merged)

    # Lags and rolling
    merged = _generate_lags(
        merged,
        group_cols=["restaurant_id"],
        target_cols=["total_sales", "orders_count"],
        lags=[1, 3, 7],
    )

    weather_cols = ["temp", "rain", "wind", "humidity", "tourist_flow"]
    merged = _generate_lags(
        merged,
        group_cols=["restaurant_id"],
        target_cols=weather_cols,
        lags=list(range(1, 8)),
    )

    merged = _generate_rolling_means(
        merged,
        group_cols=["restaurant_id"],
        target_cols=["total_sales", "orders_count"],
        windows=[7],
    )
    merged = _generate_rolling_means(
        merged,
        group_cols=["restaurant_id"],
        target_cols=weather_cols,
        windows=[7],
    )

    merged.sort_values(["restaurant_id", "date"]).to_csv(output_csv_path, index=False)
    return merged