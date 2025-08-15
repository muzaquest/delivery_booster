"""Holidays loader for Indonesia (ID) with international add-ons and Bali local holidays.

Data sources:
- Nager.Date public API for ID public holidays (includes Islamic/Buddhist/Christian national/regional)
- Additional well-known international observances (limited set)
- Bali local holidays scraped from provided sources for 2024/2025
"""

from __future__ import annotations

import datetime as dt
from typing import List, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup

NAGER_API = "https://date.nager.at/api/v3/PublicHolidays/{year}/{country}"
COUNTRY = "ID"

INTERNATIONAL_OBS = [
    ("-01-01", "New Year's Day", "International"),
    ("-03-08", "International Women's Day", "International"),
    ("-05-01", "Labour Day", "International"),
    ("-12-25", "Christmas Day", "International"),
]

BALI_SOURCES: List[Tuple[int, str]] = [
    (2025, "https://www.bali-gid.com/otdyh-na-bali-sovety-turistam/prazdniki-na-bali/kalendar-prazdnikov-na-bali-2025-god/"),
    (2024, "https://balitime.info/putevoditel/kalendar-prazdnikov-na-2024/"),
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


def _parse_bali_local(year: int, url: str) -> pd.DataFrame:
    resp = requests.get(url, timeout=30)
    if not resp.ok:
        return pd.DataFrame(columns=["date", "holiday_name", "region"]) 
    soup = BeautifulSoup(resp.text, "html.parser")

    # Heuristic: tables or list items with date and holiday name
    rows: List[dict] = []

    # Look for table rows first
    for tr in soup.find_all("tr"):
        cells = [c.get_text(strip=True) for c in tr.find_all(["td", "th"]) if c.get_text(strip=True)]
        if len(cells) < 2:
            continue
        text = " ".join(cells)
        # Expect formats like "12 Мая 2025 — Galungan" or "2025-05-12 ..."
        date = _try_parse_date_from_text(text, year)
        if date is None:
            continue
        name = cells[-1]
        rows.append({"date": date, "holiday_name": name, "region": "Bali (local)"})

    # If not enough rows found, also scan list items
    if len(rows) < 3:
        for li in soup.find_all("li"):
            text = li.get_text(" ", strip=True)
            date = _try_parse_date_from_text(text, year)
            if date is None:
                continue
            # Name is the remainder of the line after date
            parts = text.split(" ", 3)
            name = text
            rows.append({"date": date, "holiday_name": name, "region": "Bali (local)"})

    if not rows:
        return pd.DataFrame(columns=["date", "holiday_name", "region"]) 

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["date"]).copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.drop_duplicates(["date", "holiday_name", "region"]).sort_values("date").reset_index(drop=True)
    return df


def _try_parse_date_from_text(text: str, year: int):
    # Try multiple date formats commonly appearing on Russian-language Bali sites
    import re
    import locale

    # Attempt to parse Russian month names by a simple mapping
    ru_months = {
        "января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6,
        "июля": 7, "августа": 8, "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
        "январь": 1, "февраль": 2, "март": 3, "апрель": 4, "май": 5, "июнь": 6,
        "июль": 7, "август": 8, "сентябрь": 9, "октябрь": 10, "ноябрь": 11, "декабрь": 12,
        "янв": 1, "фев": 2, "мар": 3, "апр": 4, "май": 5, "мая": 5, "июн": 6, "июл": 7,
        "авг": 8, "сен": 9, "окт": 10, "ноя": 11, "дек": 12,
    }

    # Pattern like "12 мая 2025" or "12 мая"
    m = re.search(r"(\d{1,2})\s+([A-Za-zА-Яа-яёЁ]+)(?:\s+(\d{4}))?", text)
    if m:
        day = int(m.group(1))
        month_name = m.group(2).lower()
        month = ru_months.get(month_name)
        if month:
            y = int(m.group(3)) if m.group(3) else year
            try:
                return dt.date(y, month, day)
            except ValueError:
                pass

    # ISO-like date "2025-05-12" or "05-12"
    m2 = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
    if m2:
        try:
            return dt.date(int(m2.group(1)), int(m2.group(2)), int(m2.group(3)))
        except ValueError:
            pass
    m3 = re.search(r"(\d{2})-(\d{2})", text)
    if m3:
        try:
            return dt.date(year, int(m3.group(1)), int(m3.group(2)))
        except ValueError:
            pass

    return None


def load_holidays_df(start_date: str, end_date: str) -> pd.DataFrame:
    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()
    years = list(range(start.year, end.year + 1))

    frames: List[pd.DataFrame] = []
    for y in years:
        frames.append(_fetch_year(y))
        frames.append(_international_for_year(y))
        # Bali local, if we have a known source
        for (yy, url) in BALI_SOURCES:
            if yy == y:
                frames.append(_parse_bali_local(yy, url))

    if not frames:
        return pd.DataFrame(columns=["date", "holiday_name", "region"]) 

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df[(df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))]
    df = df.drop_duplicates(["date", "holiday_name", "region"]).sort_values("date").reset_index(drop=True)
    return df