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
    ("-02-14", "Valentine's Day", "International"),
    ("-10-31", "Halloween", "International"),
]

# Мусульманские праздники 2025 (фиксированные даты для Индонезии)
MUSLIM_HOLIDAYS_2025 = [
    ("2025-04-01", "Eid al-Fitr", "Muslim"),  # Окончание Рамадана
    ("2025-06-07", "Eid al-Adha", "Muslim"),  # Курбан-байрам
    ("2025-07-07", "Islamic New Year", "Muslim"),  # Исламский новый год
    ("2025-09-15", "Maulid Nabi", "Muslim"),  # День рождения Пророка
]

# Балийские праздники 2025 (фиксированные даты)
BALINESE_HOLIDAYS_2025 = [
    ("2025-03-31", "Nyepi", "Balinese"),  # День тишины
    ("2025-05-29", "Galungan", "Balinese"),  # Победа добра над злом
    ("2025-06-08", "Kuningan", "Balinese"),  # Завершение Galungan
    ("2025-09-25", "Galungan", "Balinese"),  # Второй Galungan в году
    ("2025-10-05", "Kuningan", "Balinese"),  # Второй Kuningan
]

# Индонезийские национальные праздники
INDONESIAN_HOLIDAYS = [
    ("-08-17", "Independence Day", "Indonesian"),  # День независимости
    ("-06-01", "Pancasila Day", "Indonesian"),  # День Панчасила
    ("-04-21", "Kartini Day", "Indonesian"),  # День Картини
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
        frames.append(_indonesian_for_year(y))
        
        # Добавляем фиксированные праздники для 2025
        if y == 2025:
            frames.append(_muslim_holidays_2025())
            frames.append(_balinese_holidays_2025())
        
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


def _indonesian_for_year(year: int) -> pd.DataFrame:
    """Индонезийские национальные праздники"""
    rows = []
    for suffix, name, region in INDONESIAN_HOLIDAYS:
        date = dt.date.fromisoformat(f"{year}{suffix}")
        rows.append({"date": date, "holiday_name": name, "region": region})
    return pd.DataFrame(rows)


def _muslim_holidays_2025() -> pd.DataFrame:
    """Мусульманские праздники 2025"""
    rows = []
    for date_str, name, region in MUSLIM_HOLIDAYS_2025:
        date = dt.date.fromisoformat(date_str)
        rows.append({"date": date, "holiday_name": name, "region": region})
    return pd.DataFrame(rows)


def _balinese_holidays_2025() -> pd.DataFrame:
    """Балийские праздники 2025"""
    rows = []
    for date_str, name, region in BALINESE_HOLIDAYS_2025:
        date = dt.date.fromisoformat(date_str)
        rows.append({"date": date, "holiday_name": name, "region": region})
    return pd.DataFrame(rows)


def get_holiday_info_for_date(date_str: str) -> dict:
    """Получение информации о празднике для конкретной даты"""
    try:
        from datetime import datetime
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        month_day = date_obj.strftime('%m-%d')
        
        # Полный список всех праздников с влиянием на бизнес
        all_holidays = {
            # Мусульманские праздники (сильное влияние на курьеров)
            '04-01': {
                'name': 'Eid al-Fitr (окончание Рамадана)',
                'type': 'Muslim',
                'impact': 'Крупнейший мусульманский праздник — курьеры отдыхают, семейные застолья дома',
                'sales_effect': '-30 до -50%',
                'recommendations': 'Увеличить бюджет на 100%, таргетинг на немусульман и туристов'
            },
            '06-07': {
                'name': 'Eid al-Adha (Курбан-байрам)',
                'type': 'Muslim', 
                'impact': 'Мусульманский праздник жертвоприношения — многие курьеры не работают',
                'sales_effect': '-20 до -35%',
                'recommendations': 'Увеличить бюджет на 50%, бонусы курьерам'
            },
            '07-07': {
                'name': 'Islamic New Year',
                'type': 'Muslim',
                'impact': 'Исламский новый год — умеренное влияние',
                'sales_effect': '-10 до -20%',
                'recommendations': 'Небольшое увеличение бюджета'
            },
            '09-15': {
                'name': 'Maulid Nabi (День рождения Пророка)',
                'type': 'Muslim',
                'impact': 'Религиозный праздник — умеренное влияние',
                'sales_effect': '-15 до -25%',
                'recommendations': 'Учесть в планировании бюджета'
            },
            
            # Балийские праздники (региональное влияние)
            '03-31': {
                'name': 'Nyepi (День тишины)',
                'type': 'Balinese',
                'impact': 'Балийский новый год — остров полностью закрыт, запрет на работу',
                'sales_effect': '-80 до -100%',
                'recommendations': 'Не планировать активность, предупредить клиентов заранее'
            },
            '05-29': {
                'name': 'Galungan',
                'type': 'Balinese',
                'impact': 'Балийский праздник победы добра над злом — семейные церемонии',
                'sales_effect': '-20 до -30%',
                'recommendations': 'Снизить ожидания, промо для туристов'
            },
            '06-08': {
                'name': 'Kuningan',
                'type': 'Balinese',
                'impact': 'Завершение Galungan — религиозные церемонии',
                'sales_effect': '-15 до -25%',
                'recommendations': 'Умеренное снижение бюджета'
            },
            '09-25': {
                'name': 'Galungan (второй)',
                'type': 'Balinese',
                'impact': 'Второй Galungan в году',
                'sales_effect': '-20 до -30%',
                'recommendations': 'Аналогично первому Galungan'
            },
            '10-05': {
                'name': 'Kuningan (второй)',
                'type': 'Balinese',
                'impact': 'Второй Kuningan в году',
                'sales_effect': '-15 до -25%',
                'recommendations': 'Умеренное снижение активности'
            },
            
            # Индонезийские национальные праздники
            '08-17': {
                'name': 'День независимости Индонезии',
                'type': 'Indonesian',
                'impact': 'Национальный праздник — государственные учреждения закрыты',
                'sales_effect': '-15 до -25%',
                'recommendations': 'Патриотические промо, семейные комбо'
            },
            '06-01': {
                'name': 'Pancasila Day',
                'type': 'Indonesian',
                'impact': 'День государственной идеологии',
                'sales_effect': '-10 до -15%',
                'recommendations': 'Минимальное влияние'
            },
            '04-21': {
                'name': 'Kartini Day',
                'type': 'Indonesian',
                'impact': 'День эмансипации женщин',
                'sales_effect': '0 до -5%',
                'recommendations': 'Промо для женщин'
            },
            
            # Международные праздники (позитивное влияние)
            '01-01': {
                'name': 'Новый год',
                'type': 'International',
                'impact': 'Празднование, больше заказов еды домой',
                'sales_effect': '+15 до +25%',
                'recommendations': 'Увеличить бюджет, новогодние промо'
            },
            '12-25': {
                'name': 'Рождество',
                'type': 'International',
                'impact': 'Христианский праздник — смешанное влияние',
                'sales_effect': '-5 до +10%',
                'recommendations': 'Рождественские промо'
            },
            '02-14': {
                'name': 'День святого Валентина',
                'type': 'International',
                'impact': 'Романтические ужины — увеличение заказов',
                'sales_effect': '+10 до +20%',
                'recommendations': 'Романтические комбо, промо для пар'
            },
            '05-01': {
                'name': 'День труда',
                'type': 'International',
                'impact': 'Выходной день — больше домашних заказов',
                'sales_effect': '+5 до +15%',
                'recommendations': 'Дневные промо'
            },
        }
        
        holiday_info = all_holidays.get(month_day)
        if holiday_info:
            return {
                'is_holiday': True,
                'name': holiday_info['name'],
                'type': holiday_info['type'],
                'description': f"{holiday_info['name']} — {holiday_info['impact']}",
                'sales_effect': holiday_info['sales_effect'],
                'recommendations': holiday_info['recommendations']
            }
        else:
            weekday = date_obj.strftime('%A')
            weekday_ru = {
                'Monday': 'понедельник', 'Tuesday': 'вторник', 'Wednesday': 'среда',
                'Thursday': 'четверг', 'Friday': 'пятница', 'Saturday': 'суббота', 'Sunday': 'воскресенье'
            }
            return {
                'is_holiday': False,
                'description': f'обычный {weekday_ru.get(weekday, weekday.lower())}, не праздник'
            }
    except:
        return {'is_holiday': False, 'description': 'обычный день'}