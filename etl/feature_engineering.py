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
)


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
) -> pd.DataFrame:
    """Build merged dataset and save to CSV.

    Steps:
      1) Load orders and restaurants from SQLite
      2) Aggregate orders to daily per restaurant
      3) Parse tourist flow from Excel and resample to daily
      4) Pull weather per (restaurant, date) from cache/API
      5) Merge and generate features
      6) Save to CSV
    """
    if excel_paths is None:
        excel_paths = [
            "/workspace/1.-Data-Kunjungan-2025-3.xls",
            "/workspace/Table-1-7-Final-1-1.xls",
        ]

    restaurants_df = load_restaurants(engine)
    orders_daily = load_orders(engine)

    # Restrict to requested period
    orders_daily = orders_daily[(orders_daily["date"] >= pd.to_datetime(start_date)) & (orders_daily["date"] <= pd.to_datetime(end_date))]

    # Tourist flow (date-level), resample to daily and forward-fill
    tourist_flow = parse_tourist_flow(excel_paths)
    if tourist_flow.empty:
        # Create an empty frame with date to allow merge without error
        tourist_flow = pd.DataFrame({"date": pd.date_range(start_date, end_date), "tourist_flow": 0.0})
    else:
        tourist_flow = tourist_flow.copy()
        tourist_flow["date"] = pd.to_datetime(tourist_flow["date"]).dt.normalize()
        tourist_flow = (
            tourist_flow.set_index("date")["tourist_flow"].resample("D").mean().ffill().bfill().reset_index()
        )

    # Build weather dataset for each (restaurant_id, date) present in orders
    # This limits API calls to dates with sales activity
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
        # fallback to empty weather
        weather_daily = pd.DataFrame(columns=["restaurant_id", "date", "temp", "rain", "wind", "humidity"])
    else:
        weather_daily = pd.concat(all_weather_parts, ignore_index=True)

    # Merge (orders x weather on restaurant_id+date) then add tourist flow on date
    merged = orders_daily.merge(weather_daily, on=["restaurant_id", "date"], how="left")
    merged = merged.merge(tourist_flow, on="date", how="left")

    # Temporal features
    merged["date"] = pd.to_datetime(merged["date"])  # ensure dtype
    merged = _generate_temporal_features(merged)

    # Lags for sales
    merged = _generate_lags(
        merged,
        group_cols=["restaurant_id"],
        target_cols=["total_sales", "orders_count"],
        lags=[1, 3, 7],
    )

    # Lags for weather and tourist flow (1–7)
    weather_cols = ["temp", "rain", "wind", "humidity", "tourist_flow"]
    merged = _generate_lags(
        merged,
        group_cols=["restaurant_id"],
        target_cols=weather_cols,
        lags=list(range(1, 8)),
    )

    # Rolling means (7-day)
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

    # Save
    merged.sort_values(["restaurant_id", "date"]).to_csv(output_csv_path, index=False)
    return merged