"""Data loading utilities (CSV, databases, APIs) and ETL runner.

- Connects to SQLite database via SQLAlchemy
- Loads orders, restaurants, clients from SQLite
- Parses Excel files with tourist flow
- Retrieves daily weather from Open-Meteo with caching in SQLite (weather_cache)
- Exposes a CLI: `python etl/data_loader.py --run` to build merged dataset
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
import re
import math
from sqlalchemy import (
    Column,
    Date,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    create_engine,
    inspect,
    select,
    Text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.sql import and_

from config import get_env


DEFAULT_SQLITE_PATH = get_env("SQLITE_PATH", "/workspace/database.sqlite")
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/era5"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def get_engine(sqlite_path: Optional[str] = None) -> Engine:
    """Create SQLAlchemy engine for SQLite."""
    db_path = sqlite_path or DEFAULT_SQLITE_PATH
    conn_str = f"sqlite:///{db_path}"
    engine = create_engine(conn_str, future=True)
    return engine


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _find_first_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    cols = set(df.columns)
    for name in candidates:
        if name in cols:
            return name
    return None


def ensure_weather_cache_table(engine: Engine) -> Table:
    """Ensure `weather_cache` table exists in SQLite with appropriate schema."""
    metadata = MetaData()
    weather_cache = Table(
        "weather_cache",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("restaurant_id", Integer, nullable=False),
        Column("date", Date, nullable=False),
        Column("temp", Float),
        Column("rain", Float),
        Column("wind", Float),
        Column("humidity", Float),
        Column("fetched_at", String(32)),
        UniqueConstraint("restaurant_id", "date", name="ux_weather_restaurant_date"),
    )
    metadata.create_all(engine, tables=[weather_cache])
    return weather_cache


def ensure_geocode_cache_table(engine: Engine) -> Table:
    metadata = MetaData()
    geocode_cache = Table(
        "geocode_cache",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("restaurant_id", Integer, nullable=False),
        Column("name", Text, nullable=False),
        Column("latitude", Float, nullable=False),
        Column("longitude", Float, nullable=False),
        Column("fetched_at", String(32)),
        UniqueConstraint("restaurant_id", name="ux_geocode_restaurant"),
    )
    metadata.create_all(engine, tables=[geocode_cache])
    return geocode_cache


def _read_sql_table(engine: Engine, table_name: str) -> pd.DataFrame:
    try:
        df = pd.read_sql_table(table_name, con=engine)
        return _normalize_columns(df)
    except Exception:
        return pd.DataFrame()


def _resolve_table_name(engine: Engine, preferred: str, aliases: Iterable[str]) -> Optional[str]:
    inspector = inspect(engine)
    all_tables = set(inspector.get_table_names())
    if preferred in all_tables:
        return preferred
    for a in aliases:
        if a in all_tables:
            return a
    return None


def load_restaurants(engine: Engine) -> pd.DataFrame:
    table = _resolve_table_name(engine, "restaurants", aliases=["restaurant", "stores"])
    if not table:
        return pd.DataFrame(columns=["id", "name", "latitude", "longitude"])  # empty placeholder
    df = _read_sql_table(engine, table)
    id_col = _find_first_column(df, ["id", "restaurant_id", "rest_id", "store_id"]) or "id"
    name_col = _find_first_column(df, ["name", "restaurant_name", "title"]) or "name"
    lat_col = _find_first_column(df, ["latitude", "lat", "y"]) or "latitude"
    lon_col = _find_first_column(df, ["longitude", "lon", "lng", "x"]) or "longitude"
    out = df[[id_col, name_col, lat_col, lon_col]].rename(
        columns={id_col: "id", name_col: "name", lat_col: "latitude", lon_col: "longitude"}
    )
    out = out.dropna(subset=["id", "latitude", "longitude"]).copy()
    out["id"] = out["id"].astype(int)
    return out


def load_clients(engine: Engine) -> pd.DataFrame:
    table = _resolve_table_name(engine, "clients", aliases=["customers", "users", "customer"])
    if not table:
        return pd.DataFrame()
    return _read_sql_table(engine, table)


def _read_first_line(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            line = f.readline().strip()
            return line if line else None
    except Exception:
        return None


def _extract_google_sheet_id(sheet_url_or_id: str) -> Optional[str]:
    # Accept either full URL or bare spreadsheetId
    if re.fullmatch(r"[A-Za-z0-9_-]{40,}", sheet_url_or_id):
        return sheet_url_or_id
    m = re.search(r"/spreadsheets/d/([A-Za-z0-9_-]+)", sheet_url_or_id)
    if m:
        return m.group(1)
    return None


def _google_api_key() -> Optional[str]:
    # Use env GOOGLE_API_KEY, or fallback to provided key as default
    return get_env("GOOGLE_API_KEY", "AIzaSyA5sYxoUNI0hlsa20lMqDHHLAL80qRIn0w")


def _fetch_google_sheet_first_sheet_title(spreadsheet_id: str, api_key: str) -> Optional[str]:
    url = f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}?fields=sheets(properties(title))&key={api_key}"
    resp = requests.get(url, timeout=30)
    if not resp.ok:
        return None
    data = resp.json()
    sheets = data.get("sheets") or []
    if not sheets:
        return None
    title = ((sheets[0] or {}).get("properties") or {}).get("title")
    return title


def _fetch_google_sheet_values(spreadsheet_id: str, range_a1: str, api_key: str) -> Optional[pd.DataFrame]:
    url = f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values/{requests.utils.quote(range_a1, safe='')}?majorDimension=ROWS&key={api_key}"
    resp = requests.get(url, timeout=30)
    if not resp.ok:
        return None
    data = resp.json()
    values = data.get("values") or []
    if not values:
        return None
    header = [str(h).strip().lower() for h in values[0]]
    rows = values[1:]
    # Normalize row lengths
    normalized_rows = [r + [None] * (len(header) - len(r)) for r in rows]
    df = pd.DataFrame(normalized_rows, columns=header)
    return df


def load_fake_orders(sheet_url_or_id: Optional[str] = None) -> pd.DataFrame:
    """Load fake orders list from a Google Sheet.

    The sheet is expected to include at least one of: order_id, order_number.
    Additional columns (restaurant_id, date, platform) are optional.
    """
    if not sheet_url_or_id:
        # Try to read from file '/workspace/Fake orders'
        sheet_url_or_id = _read_first_line("/workspace/Fake orders") or ""
    spreadsheet_id = _extract_google_sheet_id(sheet_url_or_id)
    api_key = _google_api_key()
    if not spreadsheet_id or not api_key:
        return pd.DataFrame()

    sheet_title = _fetch_google_sheet_first_sheet_title(spreadsheet_id, api_key) or "Sheet1"
    df = _fetch_google_sheet_values(spreadsheet_id, f"{sheet_title}!A:Z", api_key)
    if df is None or df.empty:
        # Try fallback title guesses
        for guess in ("Sheet1", "Fake Orders", "Sheet", "Лист1"):
            df = _fetch_google_sheet_values(spreadsheet_id, f"{guess}!A:Z", api_key)
            if df is not None and not df.empty:
                break
    if df is None or df.empty:
        return pd.DataFrame()

    df = _normalize_columns(df)
    # Normalize common id column names
    if "order_id" not in df.columns:
        oid = _find_first_column(df, ["id", "order", "order_number", "orderid", "номер заказа"])
        if oid:
            df.rename(columns={oid: "order_id"}, inplace=True)
    return df


def load_orders_raw(engine: Engine) -> pd.DataFrame:
    # Try canonical orders table first; else use platform stats
    inspector = inspect(engine)
    orders_table = _resolve_table_name(
        engine, "orders", aliases=["order", "sales", "transactions", "fake_orders", "orders_table"]
    )
    if orders_table and orders_table in inspector.get_table_names():
        return _read_sql_table(engine, orders_table)
    # Fallback to platform stats as raw
    return _load_platforms_combined(engine)


def load_orders(engine: Engine, fake_orders_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Load orders, exclude fake orders if provided, and aggregate to daily per restaurant.

    Supports canonical orders table or platform stats (grab_stats/gojek_stats).
    Output: restaurant_id, date, total_sales, orders_count
    """
    df = load_orders_raw(engine)
    if df.empty:
        return pd.DataFrame(columns=["restaurant_id", "date", "total_sales", "orders_count"])
    df = _normalize_columns(df)

    if {"platform", "total_sales", "orders_count", "date", "restaurant_id"}.issubset(df.columns):
        # Already platform-stats shaped
        agg = (
            df.dropna(subset=["date"]).assign(date=pd.to_datetime(df["date"], errors="coerce").dt.normalize())
            .groupby(["restaurant_id", "date"], as_index=False)
            .agg(total_sales=("total_sales", "sum"), orders_count=("orders_count", "sum"))
            .sort_values(["restaurant_id", "date"]).reset_index(drop=True)
        )
        return agg

    # else fallback to canonical reader (previous implementation)
    # Identify columns
    date_col = _find_first_column(df, ["date", "order_date", "created_at", "created", "time", "datetime"]) or "date"
    rest_col = _find_first_column(df, ["restaurant_id", "rest_id", "store_id", "restaurant"]) or "restaurant_id"
    amount_col = _find_first_column(
        df,
        [
            "total_sales",
            "total",
            "amount",
            "revenue",
            "grand_total",
            "price",
            "order_amount",
            "subtotal",
        ],
    )
    qty_col = _find_first_column(df, ["quantity", "qty", "count", "order_count"])  # optional
    order_id_col = _find_first_column(df, ["order_id", "id", "order", "order_number"])  # for fake filter

    # Filter fake orders by order_id if possible
    if fake_orders_df is not None and not fake_orders_df.empty and order_id_col:
        fod = _normalize_columns(fake_orders_df)
        if "order_id" in fod.columns:
            try:
                mask = ~df[order_id_col].astype(str).isin(fod["order_id"].astype(str))
                df = df[mask].copy()
            except Exception:
                pass

    if date_col not in df.columns:
        return pd.DataFrame(columns=["restaurant_id", "date", "total_sales", "orders_count"])

    out = df.copy()
    out["date"] = pd.to_datetime(out[date_col], errors="coerce").dt.normalize()
    out = out.dropna(subset=[rest_col, "date"]).copy()
    out["restaurant_id"] = out[rest_col].astype(int)

    if amount_col and amount_col in out.columns:
        out["_amount"] = pd.to_numeric(out[amount_col], errors="coerce")
    else:
        price_col = _find_first_column(out, ["price", "unit_price", "avg_price"])  # optional
        if price_col and qty_col and price_col in out.columns and qty_col in out.columns:
            out["_amount"] = pd.to_numeric(out[price_col], errors="coerce") * pd.to_numeric(
                out[qty_col], errors="coerce"
            )
        else:
            out["_amount"] = 0.0

    if qty_col and qty_col in out.columns:
        out["_qty"] = pd.to_numeric(out[qty_col], errors="coerce").fillna(0)
    else:
        out["_qty"] = 1

    daily = (
        out.groupby(["restaurant_id", "date"], as_index=False)
        .agg(total_sales=("_amount", "sum"), orders_count=("_qty", "sum"))
        .sort_values(["restaurant_id", "date"]).reset_index(drop=True)
    )
    return daily


def parse_tourist_flow(excel_paths: List[str]) -> pd.DataFrame:
    """Parse multiple Excel files and return a unified daily tourist_flow DataFrame.

    Heuristics:
      - Identify a date column (or year+month(+day) combo)
      - Identify a numeric visitors column
      - Aggregate to daily frequency
    """
    frames: List[pd.DataFrame] = []
    for path in excel_paths:
        try:
            # Read all sheets; some files might contain the series in later sheets
            sheets = pd.read_excel(path, sheet_name=None, engine="xlrd")
        except Exception:
            # Try default engine for compatibility
            try:
                sheets = pd.read_excel(path, sheet_name=None)
            except Exception:
                continue
        for _, df in (sheets or {}).items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            df = _normalize_columns(df)
            # Try direct date column
            date_col = _find_first_column(df, ["date", "tanggal", "day", "tgl", "periode", "period"])
            year_col = _find_first_column(df, ["year", "tahun", "yr"])
            month_col = _find_first_column(df, ["month", "bulan", "mo"])  # 1..12 or names
            day_col = _find_first_column(df, ["day", "hari"])  # optional

            # Try to pick a value column by typical names
            value_col = _find_first_column(
                df,
                [
                    "visitors",
                    "arrivals",
                    "kunjungan",
                    "jumlah",
                    "total",
                    "count",
                    "wisatawan",
                    "visitor",
                    "tourists",
                ],
            )

            work: Optional[pd.DataFrame] = None
            if date_col and date_col in df.columns:
                work = df[[date_col, value_col]].copy() if value_col else df[[date_col]].copy()
                work.rename(columns={date_col: "date", value_col: "tourist_flow" if value_col else date_col}, inplace=True)
                work["date"] = pd.to_datetime(work["date"], errors="coerce")
            elif year_col and month_col:
                cols = [year_col, month_col]
                if day_col:
                    cols.append(day_col)
                work = df[cols + ([value_col] if value_col else [])].copy()
                work.rename(columns={year_col: "year", month_col: "month", day_col: "day" if day_col else month_col}, inplace=True)
                # Map month names to numbers if needed
                if work["month"].dtype == object:
                    month_map = {m.lower(): i for i, m in enumerate([
                        "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"
                    ], start=1)}
                    work["month"] = work["month"].astype(str).str[:3].str.lower().map(month_map)
                work["day"] = work["day"].fillna(1) if "day" in work.columns else 1
                work["date"] = pd.to_datetime(
                    dict(year=pd.to_numeric(work["year"], errors="coerce"),
                         month=pd.to_numeric(work["month"], errors="coerce"),
                         day=pd.to_numeric(work["day"], errors="coerce")),
                    errors="coerce"
                )
                if value_col:
                    work.rename(columns={value_col: "tourist_flow"}, inplace=True)
            else:
                # Could not parse this sheet
                continue

            if work is None or work.empty:
                continue
            work = work.dropna(subset=["date"]).copy()
            if "tourist_flow" not in work.columns:
                # Try to infer numeric column if not set
                numeric_cols = [c for c in work.columns if c != "date" and pd.api.types.is_numeric_dtype(work[c])]
                if numeric_cols:
                    work["tourist_flow"] = work[numeric_cols[0]]
                else:
                    work["tourist_flow"] = float("nan")
            work = work[["date", "tourist_flow"]]
            frames.append(work)

    if not frames:
        return pd.DataFrame(columns=["date", "tourist_flow"])

    all_flow = pd.concat(frames, ignore_index=True)
    all_flow = all_flow.dropna(subset=["date"]).copy()

    # If multiple entries per date, sum them
    all_flow = (
        all_flow.groupby(all_flow["date"].dt.normalize(), as_index=False)["tourist_flow"].sum()
        .rename(columns={"date": "date"})
        .sort_values("date")
        .reset_index(drop=True)
    )
    return all_flow


def _select_daily_from_open_meteo(
    latitude: float,
    longitude: float,
    date: dt.date,
) -> Dict[str, Optional[float]]:
    """Query Open-Meteo for a single date and return daily variables."""
    start = end = date.strftime("%Y-%m-%d")
    today = dt.date.today()

    if date <= today:
        base_url = OPEN_METEO_ARCHIVE_URL
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start,
            "end_date": end,
            "daily": "temperature_2m_mean,precipitation_sum,windspeed_10m_max,relative_humidity_2m_mean",
            "timezone": "auto",
        }
    else:
        base_url = OPEN_METEO_FORECAST_URL
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start,
            "end_date": end,
            "daily": "temperature_2m_max,precipitation_sum,windspeed_10m_max,relative_humidity_2m_mean",
            "timezone": "auto",
        }

    resp = requests.get(base_url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    daily = data.get("daily") or {}
    dates = daily.get("time") or []
    if not dates:
        return {"temp": None, "rain": None, "wind": None, "humidity": None}

    # Extract first (and only) entry for the date range
    idx = 0
    temp_key = "temperature_2m_mean" if "temperature_2m_mean" in daily else "temperature_2m_max"
    temp = (daily.get(temp_key) or [None])[idx]
    rain = (daily.get("precipitation_sum") or [None])[idx]
    wind = (daily.get("windspeed_10m_max") or [None])[idx]
    humidity = (daily.get("relative_humidity_2m_mean") or [None])[idx]

    return {"temp": temp, "rain": rain, "wind": wind, "humidity": humidity}


def get_restaurant_coordinates(restaurant_id: int, restaurant_name: str, engine: Engine) -> Optional[Tuple[float, float]]:
    # 1) Try restaurants table for lat/lon if exists
    inspector = inspect(engine)
    if "restaurants" in inspector.get_table_names():
        df_rest = pd.read_sql_query("SELECT * FROM restaurants WHERE id = :rid", engine, params={"rid": restaurant_id})
        df_norm = _normalize_columns(df_rest)
        for lat_col in ("latitude", "lat"):
            if lat_col in df_norm.columns:
                lon_col = _find_first_column(df_norm, ["longitude", "lon", "lng"]) or None
                if lon_col and lon_col in df_norm.columns:
                    try:
                        return float(df_norm.iloc[0][lat_col]), float(df_norm.iloc[0][lon_col])
                    except Exception:
                        pass
    # 2) Try geocode cache
    geo_tbl = ensure_geocode_cache_table(engine)
    with engine.begin() as conn:
        row = conn.execute(select(geo_tbl.c.latitude, geo_tbl.c.longitude).where(geo_tbl.c.restaurant_id == restaurant_id)).fetchone()
        if row:
            return float(row.latitude), float(row.longitude)
    # 3) Geocode via Google
    api_key = _google_api_key()
    if not api_key:
        return None
    coords = _google_geocode(restaurant_name, api_key)
    if coords is None:
        return None
    with engine.begin() as conn:
        conn.execute(
            geo_tbl.insert().values(
                restaurant_id=restaurant_id,
                name=restaurant_name,
                latitude=coords[0],
                longitude=coords[1],
                fetched_at=dt.datetime.utcnow().isoformat(timespec="seconds"),
            )
        )
    return coords


def get_weather_for_restaurant(restaurant_id: int, date: dt.date, engine: Optional[Engine] = None) -> Dict[str, Optional[float]]:
    """Get daily weather for a restaurant/date, using SQLite cache to avoid repeated API calls.

    If engine is not provided, a default SQLite engine is created using env SQLITE_PATH or /workspace/database.sqlite.
    """
    if engine is None:
        engine = get_engine()
    inspector = inspect(engine)
    restaurants_table = _resolve_table_name(engine, "restaurants", aliases=["restaurant", "stores"]) or "restaurants"

    # Ensure cache table exists
    weather_cache = ensure_weather_cache_table(engine)

    # Fetch restaurant coordinates (from restaurants table or geocode cache)
    with engine.begin() as conn:
        if restaurants_table in inspector.get_table_names():
            df_rest = pd.read_sql_query(f"SELECT * FROM {restaurants_table} WHERE id = :rid", conn, params={"rid": restaurant_id})
        else:
            df_rest = pd.DataFrame()
    rest_name = None
    if not df_rest.empty:
        rest_norm = _normalize_columns(df_rest)
        rest_name = rest_norm.iloc[0].get("name")
    coords = None
    if not df_rest.empty:
        # try direct columns
        rest_norm = _normalize_columns(df_rest)
        lat_col = _find_first_column(rest_norm, ["latitude", "lat"]) or None
        lon_col = _find_first_column(rest_norm, ["longitude", "lon", "lng"]) or None
        if lat_col and lon_col and lat_col in rest_norm.columns and lon_col in rest_norm.columns:
            try:
                coords = (float(rest_norm.iloc[0][lat_col]), float(rest_norm.iloc[0][lon_col]))
            except Exception:
                coords = None
    if coords is None:
        coords = get_restaurant_coordinates(restaurant_id, rest_name or f"restaurant_{restaurant_id}", engine)
    if coords is None:
        return {"temp": None, "rain": None, "wind": None, "humidity": None}
    latitude, longitude = coords

    # Check cache first
    with engine.begin() as conn:
        stmt = select(weather_cache.c.temp, weather_cache.c.rain, weather_cache.c.wind, weather_cache.c.humidity).where(
            and_(weather_cache.c.restaurant_id == restaurant_id, weather_cache.c.date == date)
        )
        row = conn.execute(stmt).fetchone()
        if row:
            return {"temp": row.temp, "rain": row.rain, "wind": row.wind, "humidity": row.humidity}

    # Fetch from API and cache
    daily = _select_daily_from_open_meteo(latitude, longitude, date)

    with engine.begin() as conn:
        conn.execute(
            weather_cache.insert().values(
                restaurant_id=restaurant_id,
                date=date,
                temp=daily.get("temp"),
                rain=daily.get("rain"),
                wind=daily.get("wind"),
                humidity=daily.get("humidity"),
                fetched_at=dt.datetime.utcnow().isoformat(timespec="seconds"),
            )
        )

    return daily


def _select_daily_range_from_open_meteo(
    latitude: float,
    longitude: float,
    start_date: dt.date,
    end_date: dt.date,
) -> pd.DataFrame:
    """Query Open-Meteo daily variables for a date range and return a DataFrame."""
    start = start_date.strftime("%Y-%m-%d")
    end = end_date.strftime("%Y-%m-%d")

    # Use archive endpoint for historical range (ERA5)
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start,
        "end_date": end,
        "daily": "temperature_2m_mean,precipitation_sum,windspeed_10m_max,relative_humidity_2m_mean",
        "timezone": "auto",
    }
    resp = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json() or {}
    daily = data.get("daily") or {}

    times = daily.get("time") or []
    if not times:
        return pd.DataFrame(columns=["date", "temp", "rain", "wind", "humidity"])

    temp_key = "temperature_2m_mean" if "temperature_2m_mean" in daily else "temperature_2m_max"
    temp_vals = daily.get(temp_key) or [None] * len(times)
    rain_vals = daily.get("precipitation_sum") or [None] * len(times)
    wind_vals = daily.get("windspeed_10m_max") or [None] * len(times)
    hum_vals = daily.get("relative_humidity_2m_mean") or [None] * len(times)

    df = pd.DataFrame({
        "date": pd.to_datetime(times),
        "temp": temp_vals,
        "rain": rain_vals,
        "wind": wind_vals,
        "humidity": hum_vals,
    })
    return df


def get_weather_series_for_restaurant(
    restaurant_id: int, start_date: dt.date, end_date: dt.date, engine: Engine
) -> pd.DataFrame:
    """Return daily weather series for the specified date range using cache+API.

    This optimized version fetches the entire range in a single Open-Meteo request
    when cache misses are detected, then persists results into SQLite cache.
    """
    # Ensure cache table exists
    weather_cache = ensure_weather_cache_table(engine)

    # Fetch restaurant coordinates
    inspector = inspect(engine)
    restaurants_table = _resolve_table_name(engine, "restaurants", aliases=["restaurant", "stores"]) or "restaurants"
    with engine.begin() as conn:
        if restaurants_table in inspector.get_table_names():
            df_rest = pd.read_sql_query(f"SELECT * FROM {restaurants_table} WHERE id = :rid", conn, params={"rid": restaurant_id})
        else:
            df_rest = pd.DataFrame()
    if df_rest.empty:
        return pd.DataFrame(columns=["restaurant_id", "date", "temp", "rain", "wind", "humidity"])

    rest_norm = _normalize_columns(df_rest)
    lat_col = _find_first_column(rest_norm, ["latitude", "lat"]) or "latitude"
    lon_col = _find_first_column(rest_norm, ["longitude", "lon", "lng"]) or "longitude"
    latitude = float(rest_norm.iloc[0][lat_col])
    longitude = float(rest_norm.iloc[0][lon_col])

    # Read what we already have in cache for the range
    with engine.begin() as conn:
        cached_df = pd.read_sql_query(
            """
            SELECT restaurant_id, date, temp, rain, wind, humidity
            FROM weather_cache
            WHERE restaurant_id = :rid AND date BETWEEN :start AND :end
            ORDER BY date
            """,
            conn,
            params={"rid": restaurant_id, "start": start_date, "end": end_date},
            parse_dates=["date"],
        )

    # Determine missing dates
    all_dates = pd.date_range(start=start_date, end=end_date, freq="D").normalize()
    if not cached_df.empty:
        cached_dates = pd.to_datetime(cached_df["date"]).dt.normalize().unique()
    else:
        cached_dates = np.array([], dtype="datetime64[ns]")
    missing = all_dates.difference(pd.DatetimeIndex(cached_dates))

    # If missing, fetch in a single range and upsert into cache
    if len(missing) > 0:
        fetched = _select_daily_range_from_open_meteo(latitude, longitude, start_date, end_date)
        if not fetched.empty:
            # Insert all fetched; ON CONFLICT ignore duplicates
            with engine.begin() as conn:
                for _, row in fetched.iterrows():
                    conn.execute(
                        weather_cache.insert().prefix_with("OR IGNORE").values(
                            restaurant_id=restaurant_id,
                            date=row["date"].date(),
                            temp=row["temp"],
                            rain=row["rain"],
                            wind=row["wind"],
                            humidity=row["humidity"],
                            fetched_at=dt.datetime.utcnow().isoformat(timespec="seconds"),
                        )
                    )
        # Reload cache after insert
        with engine.begin() as conn:
            cached_df = pd.read_sql_query(
                """
                SELECT restaurant_id, date, temp, rain, wind, humidity
                FROM weather_cache
                WHERE restaurant_id = :rid AND date BETWEEN :start AND :end
                ORDER BY date
                """,
                conn,
                params={"rid": restaurant_id, "start": start_date, "end": end_date},
                parse_dates=["date"],
            )

    if cached_df.empty:
        return pd.DataFrame(columns=["restaurant_id", "date", "temp", "rain", "wind", "humidity"])

    cached_df["restaurant_id"] = restaurant_id
    cached_df["date"] = pd.to_datetime(cached_df["date"]).dt.normalize()
    return cached_df[["restaurant_id", "date", "temp", "rain", "wind", "humidity"]]


def _time_to_minutes(val: Any) -> Optional[float]:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    try:
        # If already number (seconds or minutes?)
        if isinstance(val, (int, float)):
            # assume minutes
            return float(val)
        s = str(val)
        if not s or s.lower() in ("nan", "none"):
            return None
        parts = s.split(":")
        if len(parts) == 3:
            h, m, sec = parts
            return int(h) * 60 + int(m) + int(sec) / 60.0
        if len(parts) == 2:
            h, m = parts
            return int(h) * 60 + int(m)
        # Fallback try parse float minutes
        return float(s)
    except Exception:
        return None


def _load_platform_stats(engine: Engine, table_name: str, platform: str) -> pd.DataFrame:
    inspector = inspect(engine)
    if table_name not in inspector.get_table_names():
        return pd.DataFrame()
    df = pd.read_sql_table(table_name, engine)
    df = _normalize_columns(df)
    date_col = _find_first_column(df, ["stat_date", "date"]) or "stat_date"
    rest_col = _find_first_column(df, ["restaurant_id", "rest_id", "store_id"]) or "restaurant_id"
    out = pd.DataFrame()
    if platform == "grab":
        out = pd.DataFrame({
            "restaurant_id": pd.to_numeric(df[rest_col], errors="coerce").astype("Int64"),
            "date": pd.to_datetime(df[date_col], errors="coerce").dt.normalize(),
            "platform": platform,
            "total_sales": pd.to_numeric(df.get("sales"), errors="coerce"),
            "orders_count": pd.to_numeric(df.get("orders"), errors="coerce"),
            "ads_spend": pd.to_numeric(df.get("ads_spend"), errors="coerce"),
            "ads_sales": pd.to_numeric(df.get("ads_sales"), errors="coerce"),
            "impressions": pd.to_numeric(df.get("impressions"), errors="coerce"),
            "rating": pd.to_numeric(df.get("rating"), errors="coerce"),
            "offline_rate": pd.to_numeric(df.get("offline_rate"), errors="coerce"),
            # driver_waiting_time is JSON; attempt to parse avg minutes
            "driver_waiting_time": df.get("driver_waiting_time")
        })
        # Parse driver waiting json → minutes if possible
        def _parse_driver_wait(v):
            try:
                if isinstance(v, str) and v.strip().startswith("{"):
                    d = json.loads(v)
                    # try fields like avg, mean, minutes
                    for k in ("avg", "average", "minutes", "mean"):
                        if k in d:
                            return float(d[k])
                elif isinstance(v, (int, float)):
                    return float(v)
            except Exception:
                return None
            return None
        out["driver_waiting_minutes"] = out["driver_waiting_time"].apply(_parse_driver_wait)
    elif platform == "gojek":
        out = pd.DataFrame({
            "restaurant_id": pd.to_numeric(df[rest_col], errors="coerce").astype("Int64"),
            "date": pd.to_datetime(df[date_col], errors="coerce").dt.normalize(),
            "platform": platform,
            "total_sales": pd.to_numeric(df.get("sales"), errors="coerce"),
            "orders_count": pd.to_numeric(df.get("orders"), errors="coerce"),
            "ads_spend": pd.to_numeric(df.get("ads_spend"), errors="coerce"),
            "ads_sales": pd.to_numeric(df.get("ads_sales"), errors="coerce"),
            "impressions": pd.NA,  # not available in gojek_stats
            "rating": pd.to_numeric(df.get("rating"), errors="coerce"),
            "accepting_time": df.get("accepting_time").apply(_time_to_minutes) if "accepting_time" in df.columns else None,
            "preparation_time": df.get("preparation_time").apply(_time_to_minutes) if "preparation_time" in df.columns else None,
            "delivery_time": df.get("delivery_time").apply(_time_to_minutes) if "delivery_time" in df.columns else None,
            "close_time": pd.to_numeric(df.get("close_time"), errors="coerce"),
        })
    else:
        return pd.DataFrame()

    out = out.dropna(subset=["restaurant_id", "date"]).copy()
    out["restaurant_id"] = out["restaurant_id"].astype(int)
    # Compute ROAS where possible
    if "ads_spend" in out.columns and "ads_sales" in out.columns:
        with np.errstate(divide='ignore', invalid='ignore'):
            out["roas"] = (out["ads_sales"] / out["ads_spend"]).replace([np.inf, -np.inf], np.nan)
    return out


def _load_platforms_combined(engine: Engine) -> pd.DataFrame:
    grab = _load_platform_stats(engine, "grab_stats", "grab")
    gojek = _load_platform_stats(engine, "gojek_stats", "gojek")
    if grab.empty and gojek.empty:
        return pd.DataFrame()
    return pd.concat([d for d in [grab, gojek] if not d.empty], ignore_index=True)


def load_operations(engine: Engine) -> pd.DataFrame:
    # Prefer platform stats
    platforms = _load_platforms_combined(engine)
    if not platforms.empty:
        cols = [c for c in [
            "restaurant_id", "date", "platform", "accepting_time", "delivery_time", "preparation_time", "rating", "driver_waiting_minutes"
        ] if c in platforms.columns]
        return platforms[cols]
    # Fallback to generic operations table
    table = _resolve_table_name(engine, "operations", aliases=["ops", "operation_metrics"])
    if not table:
        return pd.DataFrame(columns=["restaurant_id", "date", "platform", "accepting_time", "delivery_time", "preparation_time", "rating", "repeat_customers"])
    df = _read_sql_table(engine, table)
    df = _normalize_columns(df)
    date_col = _find_first_column(df, ["date", "day", "recorded_at"]) or "date"
    rest_col = _find_first_column(df, ["restaurant_id", "rest_id", "store_id"]) or "restaurant_id"
    platform_col = _find_first_column(df, ["platform", "source", "channel"])  # optional

    out = pd.DataFrame({
        "restaurant_id": pd.to_numeric(df[rest_col], errors="coerce").astype("Int64"),
        "date": pd.to_datetime(df[date_col], errors="coerce").dt.normalize(),
        "accepting_time": pd.to_numeric(df.get("accepting_time"), errors="coerce"),
        "delivery_time": pd.to_numeric(df.get("delivery_time"), errors="coerce"),
        "preparation_time": pd.to_numeric(df.get("preparation_time"), errors="coerce"),
        "rating": pd.to_numeric(df.get("rating"), errors="coerce"),
        "repeat_customers": pd.to_numeric(df.get("repeat_customers"), errors="coerce"),
        "platform": (df[platform_col].astype(str) if platform_col and platform_col in df.columns else pd.Series([None]*len(df))),
    })
    out = out.dropna(subset=["restaurant_id", "date"]).copy()
    out["restaurant_id"] = out["restaurant_id"].astype(int)
    return out


def load_marketing(engine: Engine) -> pd.DataFrame:
    # Prefer platform stats
    platforms = _load_platforms_combined(engine)
    if not platforms.empty:
        return platforms[["restaurant_id", "date", "platform", "ads_spend", "roas", "impressions"]]
    # Fallback to generic marketing table
    table = _resolve_table_name(engine, "marketing", aliases=["ads", "adspend", "campaigns"])
    if not table:
        return pd.DataFrame(columns=["restaurant_id", "date", "platform", "ads_spend", "roas", "impressions", "clicks"])
    df = _read_sql_table(engine, table)
    df = _normalize_columns(df)
    date_col = _find_first_column(df, ["date", "day", "recorded_at"]) or "date"
    rest_col = _find_first_column(df, ["restaurant_id", "rest_id", "store_id"]) or "restaurant_id"
    platform_col = _find_first_column(df, ["platform", "source", "channel"])  # optional
    spend_col = _find_first_column(df, ["ads_spend", "ad_spend", "spend", "budget"]) or "ads_spend"
    roas_col = _find_first_column(df, ["roas"]) or "roas"
    impr_col = _find_first_column(df, ["impressions", "impr"]) or "impressions"
    clicks_col = _find_first_column(df, ["clicks"]) or "clicks"

    out = pd.DataFrame({
        "restaurant_id": pd.to_numeric(df[rest_col], errors="coerce").astype("Int64"),
        "date": pd.to_datetime(df[date_col], errors="coerce").dt.normalize(),
        "ads_spend": pd.to_numeric(df.get(spend_col), errors="coerce"),
        "roas": pd.to_numeric(df.get(roas_col), errors="coerce"),
        "impressions": pd.to_numeric(df.get(impr_col), errors="coerce"),
        "clicks": pd.to_numeric(df.get(clicks_col), errors="coerce"),
        "platform": (df[platform_col].astype(str) if platform_col and platform_col in df.columns else pd.Series([None]*len(df))),
    })
    out = out.dropna(subset=["restaurant_id", "date"]).copy()
    out["restaurant_id"] = out["restaurant_id"].astype(int)
    return out


def load_platform_outages(engine: Engine) -> pd.DataFrame:
    # Prefer platform stats
    platforms = _load_platforms_combined(engine)
    if not platforms.empty:
        cols = [c for c in ["restaurant_id", "date", "platform", "offline_rate", "close_time"] if c in platforms.columns]
        res = platforms[cols].copy()
        # offline_minutes derive from close_time if present
        if "close_time" in res.columns:
            res["offline_minutes"] = pd.to_numeric(res["close_time"], errors="coerce")
        return res
    # Fallback to generic outages loader
    table = _resolve_table_name(engine, "platform_outages", aliases=["outages", "downtime", "platform_downtime"])
    if not table:
        return pd.DataFrame(columns=["restaurant_id", "date", "platform", "offline_minutes", "offline_rate", "close_time"])
    df = _read_sql_table(engine, table)
    df = _normalize_columns(df)
    date_col = _find_first_column(df, ["date", "day", "recorded_at"]) or "date"
    rest_col = _find_first_column(df, ["restaurant_id", "rest_id", "store_id"]) or "restaurant_id"
    platform_col = _find_first_column(df, ["platform", "source", "channel"]) or "platform"
    minutes_col = _find_first_column(df, ["offline_minutes", "downtime_minutes", "close_time", "minutes"])
    rate_col = _find_first_column(df, ["offline_rate", "downtime_rate"]) 

    out = pd.DataFrame({
        "restaurant_id": pd.to_numeric(df[rest_col], errors="coerce").astype("Int64"),
        "date": pd.to_datetime(df[date_col], errors="coerce").dt.normalize(),
        "platform": df[platform_col].astype(str) if platform_col in df.columns else pd.Series([None]*len(df)),
        "offline_minutes": pd.to_numeric(df.get(minutes_col), errors="coerce") if minutes_col else None,
        "offline_rate": pd.to_numeric(df.get(rate_col), errors="coerce") if rate_col else None,
    })
    if "offline_minutes" in out.columns and not isinstance(out["offline_minutes"], type(None)):
        out["close_time"] = out["offline_minutes"]
    else:
        out["close_time"] = None
    out = out.dropna(subset=["restaurant_id", "date"]).copy()
    out["restaurant_id"] = out["restaurant_id"].astype(int)
    return out


def load_orders_platform_daily(engine: Engine, fake_orders_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Load orders, exclude fake orders if provided, and aggregate to daily per restaurant and platform."""
    df = load_orders_raw(engine)
    if df.empty:
        return pd.DataFrame(columns=["restaurant_id", "date", "platform", "total_sales", "orders_count"])
    df = _normalize_columns(df)

    # Identify columns
    date_col = _find_first_column(df, ["date", "order_date", "created_at", "created", "time", "datetime"]) or "date"
    rest_col = _find_first_column(df, ["restaurant_id", "rest_id", "store_id", "restaurant"]) or "restaurant_id"
    platform_col = _find_first_column(df, ["platform", "source", "channel"]) or None
    amount_col = _find_first_column(
        df,
        ["total_sales", "total", "amount", "revenue", "grand_total", "price", "order_amount", "subtotal"],
    )
    qty_col = _find_first_column(df, ["quantity", "qty", "count", "order_count"])  # optional
    order_id_col = _find_first_column(df, ["order_id", "id", "order", "order_number"])  # for fake filter

    # Filter fake orders
    if fake_orders_df is not None and not fake_orders_df.empty and order_id_col:
        fod = _normalize_columns(fake_orders_df)
        if "order_id" in fod.columns:
            try:
                mask = ~df[order_id_col].astype(str).isin(fod["order_id"].astype(str))
                df = df[mask].copy()
            except Exception:
                pass

    if date_col not in df.columns:
        return pd.DataFrame(columns=["restaurant_id", "date", "platform", "total_sales", "orders_count"])

    out = df.copy()
    out["date"] = pd.to_datetime(out[date_col], errors="coerce").dt.normalize()
    out = out.dropna(subset=[rest_col, "date"]).copy()
    out["restaurant_id"] = out[rest_col].astype(int)

    if amount_col and amount_col in out.columns:
        out["_amount"] = pd.to_numeric(out[amount_col], errors="coerce")
    else:
        price_col = _find_first_column(out, ["price", "unit_price", "avg_price"])
        if price_col and qty_col and price_col in out.columns and qty_col in out.columns:
            out["_amount"] = pd.to_numeric(out[price_col], errors="coerce") * pd.to_numeric(out[qty_col], errors="coerce")
        else:
            out["_amount"] = 0.0

    if qty_col and qty_col in out.columns:
        out["_qty"] = pd.to_numeric(out[qty_col], errors="coerce").fillna(0)
    else:
        out["_qty"] = 1

    if platform_col and platform_col in out.columns:
        out["platform"] = out[platform_col].astype(str)
    else:
        out["platform"] = None

    daily = (
        out.groupby(["restaurant_id", "date", "platform"], dropna=False, as_index=False)
        .agg(total_sales=("_amount", "sum"), orders_count=("_qty", "sum"))
        .sort_values(["restaurant_id", "date", "platform"])
        .reset_index(drop=True)
    )
    # Normalize None platform to empty string for portability
    daily["platform"] = daily["platform"].fillna("")
    return daily


def run_full_build(
    sqlite_path: Optional[str] = None,
    start_date: str = "2024-01-01",
    end_date: str = "2025-12-31",
    output_csv_path: str = "/workspace/data/merged_dataset.csv",
    excel_paths: Optional[List[str]] = None,
    fake_orders_sheet: Optional[str] = None,
    write_postgres: bool = False,
) -> pd.DataFrame:
    """Run the full ETL to build merged dataset and save to CSV."""
    engine = get_engine(sqlite_path)

    # Late import to avoid circular
    from etl.feature_engineering import build_and_save_dataset

    fake_df = load_fake_orders(fake_orders_sheet)

    merged = build_and_save_dataset(
        engine=engine,
        start_date=start_date,
        end_date=end_date,
        output_csv_path=output_csv_path,
        excel_paths=excel_paths,
        fake_orders_df=fake_df,
    )

    if write_postgres:
        # Write to PostgreSQL
        from etl.data_to_postgres import (
            write_restaurants, write_sales, write_operations, write_marketing, write_weather, write_holidays
        )
        restaurants_df = load_restaurants(engine)
        write_restaurants(restaurants_df)

        sales_platform = load_orders_platform_daily(engine, fake_orders_df=fake_df)
        write_sales(sales_platform)

        ops_df = load_operations(engine)
        write_operations(ops_df)

        mkt_df = load_marketing(engine)
        write_marketing(mkt_df)

        # Weather: aggregate from cache for days needed
        if not merged.empty:
            weather_needed = merged[["restaurant_id", "date", "temp", "rain", "wind", "humidity"]].dropna(how="all", subset=["temp", "rain", "wind", "humidity"]).drop_duplicates()
            write_weather(weather_needed)

        # Holidays for period
        from etl.holidays_loader import load_holidays_df as _load_holidays
        holidays_df = _load_holidays(start_date, end_date)
        write_holidays(holidays_df)

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="ETL runner to build merged dataset")
    parser.add_argument("--run", action="store_true", help="Run full dataset build")
    parser.add_argument("--sqlite", type=str, default=None, help="Path to SQLite database (default: env SQLITE_PATH or /workspace/database.sqlite)")
    parser.add_argument("--start", type=str, default="2024-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default="2025-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--out", type=str, default="/workspace/data/merged_dataset.csv", help="Output CSV path")
    parser.add_argument(
        "--excel",
        type=str,
        nargs="*",
        default=[
            "/workspace/1.-Data-Kunjungan-2025-3.xls",
            "/workspace/Table-1-7-Final-1-1.xls",
        ],
        help="Paths to Excel files with tourist flow",
    )
    parser.add_argument(
        "--fake-orders-sheet",
        type=str,
        default=None,
        help="Google Sheet URL or ID containing fake orders to exclude (defaults to reading '/workspace/Fake orders')",
    )
    parser.add_argument(
        "--write-postgres",
        action="store_true",
        help="If set, writes normalized tables to PostgreSQL (requires DATABASE_URL or POSTGRES_* envs)",
    )

    args = parser.parse_args()

    if args.run:
        merged = run_full_build(
            sqlite_path=args.sqlite,
            start_date=args.start,
            end_date=args.end,
            output_csv_path=args.out,
            excel_paths=args.excel,
            fake_orders_sheet=args.fake_orders_sheet,
            write_postgres=args.write_postgres,
        )
        print(f"Built dataset with {len(merged)} rows -> {args.out}")
    else:
        print("Nothing to do. Use --run to execute ETL.")


if __name__ == "__main__":
    main()