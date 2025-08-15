"""Holidays loader for Indonesia (ID) with international add-ons.

Data sources:
- Nager.Date public API for ID public holidays
- Additional well-known international observances (limited set)
"""

from __future__ import annotations

import datetime as dt
from typing import List

import pandas as pd
import requests

NAGER_API = "https://date.nager.at/api/v3/PublicHolidays/{year}/{country}"
COUNTRY = "ID"

INTERNATIONAL_OBS = [
    ("-01-01", "New Year's Day", "International"),
    ("-03-08", "International Women's Day", "International"),
    ("-05-01", "Labour Day", "International"),
    ("-06-01", "World Parents Day", "International"),
    ("-12-25", "Christmas Day", "International"),
]


def _fetch_year(year: int) -> pd.DataFrame:
    url = NAGER_API.format(year=year, country=COUNTRY)
    resp = requests.get(url, timeout=30)
    if not resp.ok:
        return pd.DataFrame(columns=["date", "holiday_name", "region"])
    data = resp.json() or []
    rows = []
    for item in data:
        date_str = item.get("date")
        local_name = item.get("localName")
        name = item.get("name")
        counties = item.get("counties") or []
        rows.append({
            "date": pd.to_datetime(date_str, errors="coerce").date() if date_str else None,
            "holiday_name": local_name or name,
            "region": ",".join(counties) if counties else "ID",
        })
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["date"]).copy()
    return df


def _international_for_year(year: int) -> pd.DataFrame:
    rows = []
    for suffix, name, region in INTERNATIONAL_OBS:
        date = dt.date.fromisoformat(f"{year}{suffix}")
        rows.append({"date": date, "holiday_name": name, "region": region})
    return pd.DataFrame(rows)


def load_holidays_df(start_date: str, end_date: str) -> pd.DataFrame:
    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()
    years = list(range(start.year, end.year + 1))

    frames: List[pd.DataFrame] = []
    for y in years:
        frames.append(_fetch_year(y))
        frames.append(_international_for_year(y))

    if not frames:
        return pd.DataFrame(columns=["date", "holiday_name", "region"]) 

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df[(df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))]
    df = df.drop_duplicates(["date", "holiday_name", "region"]).sort_values("date").reset_index(drop=True)
    return df