"""Utilities to write ETL outputs into PostgreSQL tables.

This module contains functions that take pandas DataFrames and write them into
PostgreSQL using SQLAlchemy ORM models.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from sqlalchemy.dialects.postgresql import insert

from db.models import Base, Restaurant, Sale, Operation, Marketing, Weather, Holiday
from db.session import get_engine, session_scope


def upsert_dataframe(df: pd.DataFrame, model, conflict_cols: list[str], update_cols: Optional[list[str]] = None) -> int:
    if df is None or df.empty:
        return 0
    records = df.to_dict(orient="records")
    if not records:
        return 0
    engine = get_engine()
    total = 0
    with session_scope() as session:
        table = model.__table__
        stmt = insert(table).values(records)
        if update_cols:
            update_map = {c: getattr(stmt.excluded, c) for c in update_cols}
            stmt = stmt.on_conflict_do_update(index_elements=conflict_cols, set_=update_map)
        else:
            stmt = stmt.on_conflict_do_nothing(index_elements=conflict_cols)
        result = session.execute(stmt)
        total += result.rowcount or 0
    return total


def write_restaurants(df: pd.DataFrame) -> int:
    return upsert_dataframe(
        df[["id", "name", "latitude", "longitude", "location_type"]],
        Restaurant,
        conflict_cols=[Restaurant.id.name],
        update_cols=["name", "latitude", "longitude", "location_type"],
    )


def write_sales(df: pd.DataFrame) -> int:
    return upsert_dataframe(
        df[["restaurant_id", "date", "platform", "total_sales", "orders_count"]],
        Sale,
        conflict_cols=[Sale.restaurant_id.name, Sale.date.name, Sale.platform.name],
        update_cols=["total_sales", "orders_count"],
    )


def write_operations(df: pd.DataFrame) -> int:
    return upsert_dataframe(
        df[["restaurant_id", "date", "accepting_time", "delivery_time", "preparation_time", "rating", "repeat_customers"]],
        Operation,
        conflict_cols=[Operation.restaurant_id.name, Operation.date.name],
        update_cols=["accepting_time", "delivery_time", "preparation_time", "rating", "repeat_customers"],
    )


def write_marketing(df: pd.DataFrame) -> int:
    return upsert_dataframe(
        df[["restaurant_id", "date", "ads_spend", "roas", "impressions", "clicks"]],
        Marketing,
        conflict_cols=[Marketing.restaurant_id.name, Marketing.date.name],
        update_cols=["ads_spend", "roas", "impressions", "clicks"],
    )


def write_weather(df: pd.DataFrame) -> int:
    return upsert_dataframe(
        df[["restaurant_id", "date", "temp", "rain", "wind", "humidity"]],
        Weather,
        conflict_cols=[Weather.restaurant_id.name, Weather.date.name],
        update_cols=["temp", "rain", "wind", "humidity"],
    )


def write_holidays(df: pd.DataFrame) -> int:
    return upsert_dataframe(
        df[["date", "holiday_name", "region"]],
        Holiday,
        conflict_cols=[Holiday.date.name, Holiday.holiday_name.name, Holiday.region.name],
        update_cols=[],
    )