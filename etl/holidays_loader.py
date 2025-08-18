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
    ("-01-26", "Australia Day", "International"),
    ("-02-14", "Valentine's Day", "International"),
    ("-03-08", "International Women's Day", "International"),
    ("-03-17", "St. Patrick's Day", "International"),
    ("-04-01", "April Fool's Day", "International"),
    ("-04-22", "Earth Day", "International"),
    ("-05-01", "Labour Day", "International"),
    ("-05-12", "Mother's Day", "International"),
    ("-06-16", "Father's Day", "International"),
    ("-07-04", "Independence Day (USA)", "International"),
    ("-08-15", "Assumption Day", "International"),
    ("-09-21", "International Peace Day", "International"),
    ("-10-31", "Halloween", "International"),
    ("-11-11", "Veterans Day", "International"),
    ("-11-28", "Thanksgiving", "International"),
    ("-12-25", "Christmas Day", "International"),
    ("-12-26", "Boxing Day", "International"),
    ("-12-31", "New Year's Eve", "International"),
]

# Христианские праздники 2025 (для католиков и протестантов в Индонезии)
CHRISTIAN_HOLIDAYS_2025 = [
    ("2025-01-06", "Epiphany", "Christian"),  # Богоявление
    ("2025-02-17", "Ash Wednesday", "Christian"),  # Пепельная среда
    ("2025-03-09", "Palm Sunday", "Christian"),  # Вербное воскресенье
    ("2025-04-18", "Good Friday", "Christian"),  # Страстная пятница
    ("2025-04-20", "Easter Sunday", "Christian"),  # Пасха
    ("2025-04-21", "Easter Monday", "Christian"),  # Пасхальный понедельник
    ("2025-05-29", "Ascension Day", "Christian"),  # Вознесение
    ("2025-06-08", "Pentecost", "Christian"),  # Троица
    ("2025-08-15", "Assumption of Mary", "Christian"),  # Успение Богородицы
    ("2025-11-01", "All Saints Day", "Christian"),  # День всех святых
    ("2025-11-02", "All Souls Day", "Christian"),  # День поминовения усопших
    ("2025-12-08", "Immaculate Conception", "Christian"),  # Непорочное зачатие
    ("2025-12-24", "Christmas Eve", "Christian"),  # Сочельник
    ("2025-12-25", "Christmas Day", "Christian"),  # Рождество
]

# Буддистские праздники 2025 (для буддистов в Индонезии)
BUDDHIST_HOLIDAYS_2025 = [
    ("2025-02-12", "Magha Puja", "Buddhist"),  # Магха Пуджа
    ("2025-04-13", "Songkran", "Buddhist"),  # Сонгкран (Тайский новый год)
    ("2025-05-12", "Vesak Day", "Buddhist"),  # Весак (День Будды)
    ("2025-05-13", "Visakha Puja", "Buddhist"),  # Висакха Пуджа
    ("2025-07-11", "Asalha Puja", "Buddhist"),  # Асалха Пуджа
    ("2025-07-12", "Khao Phansa", "Buddhist"),  # Кхао Пханса
    ("2025-10-08", "Ok Phansa", "Buddhist"),  # Ок Пханса
    ("2025-11-15", "Loy Krathong", "Buddhist"),  # Лой Кратонг
]

# Китайские традиционные праздники 2025 (для китайской общины в Индонезии)
CHINESE_HOLIDAYS_2025 = [
    ("2025-01-29", "Chinese New Year", "Chinese"),  # Китайский новый год
    ("2025-02-12", "Lantern Festival", "Chinese"),  # Фестиваль фонарей
    ("2025-04-04", "Qingming Festival", "Chinese"),  # Цинмин
    ("2025-06-11", "Dragon Boat Festival", "Chinese"),  # Праздник драконьих лодок
    ("2025-08-10", "Qixi Festival", "Chinese"),  # Циси (День влюбленных)
    ("2025-08-29", "Ghost Festival", "Chinese"),  # Праздник голодных духов
    ("2025-09-06", "Mid-Autumn Festival", "Chinese"),  # Праздник середины осени
    ("2025-10-11", "Double Ninth Festival", "Chinese"),  # Праздник двойной девятки
]

# ========== ИСТОРИЧЕСКИЕ ДАННЫЕ ДЛЯ ОБУЧЕНИЯ ML ==========

# Мусульманские праздники 2024 (для обучения ML)
MUSLIM_HOLIDAYS_2024 = [
    ("2024-01-28", "Maulid Nabi Muhammad", "Muslim"),
    ("2024-02-08", "Isra Miraj", "Muslim"),
    ("2024-03-11", "Ramadan begins", "Muslim"),
    ("2024-04-10", "Eid al-Fitr", "Muslim"),
    ("2024-04-11", "Eid al-Fitr", "Muslim"),
    ("2024-04-12", "Eid al-Fitr", "Muslim"),
    ("2024-06-17", "Eid al-Adha", "Muslim"),
    ("2024-06-18", "Eid al-Adha", "Muslim"),
    ("2024-07-07", "Islamic New Year", "Muslim"),
    ("2024-09-16", "Maulid Nabi", "Muslim"),
    ("2024-12-31", "Islamic Year End", "Muslim"),
]

# Мусульманские праздники 2023 (для обучения ML)
MUSLIM_HOLIDAYS_2023 = [
    ("2023-02-18", "Isra Miraj", "Muslim"),
    ("2023-03-23", "Ramadan begins", "Muslim"),
    ("2023-04-22", "Eid al-Fitr", "Muslim"),
    ("2023-04-23", "Eid al-Fitr", "Muslim"),
    ("2023-04-24", "Eid al-Fitr", "Muslim"),
    ("2023-06-29", "Eid al-Adha", "Muslim"),
    ("2023-06-30", "Eid al-Adha", "Muslim"),
    ("2023-07-19", "Islamic New Year", "Muslim"),
    ("2023-09-28", "Maulid Nabi", "Muslim"),
]

# Балийские праздники 2024 (ключевые для обучения ML)
BALINESE_HOLIDAYS_2024 = [
    ("2024-03-11", "Nyepi", "Balinese"),  # День тишины
    ("2024-03-12", "Ngembak Geni", "Balinese"),
    ("2024-04-10", "Galungan", "Balinese"),
    ("2024-04-20", "Kuningan", "Balinese"),
    ("2024-06-05", "Galungan", "Balinese"),
    ("2024-06-15", "Kuningan", "Balinese"),
    ("2024-07-31", "Galungan", "Balinese"),
    ("2024-08-10", "Kuningan", "Balinese"),
    ("2024-09-25", "Galungan", "Balinese"),
    ("2024-10-05", "Kuningan", "Balinese"),
    ("2024-11-20", "Galungan", "Balinese"),
    ("2024-11-30", "Kuningan", "Balinese"),
]

# Балийские праздники 2023 (ключевые для обучения ML)
BALINESE_HOLIDAYS_2023 = [
    ("2023-03-22", "Nyepi", "Balinese"),  # День тишины
    ("2023-03-23", "Ngembak Geni", "Balinese"),
    ("2023-04-22", "Galungan", "Balinese"),
    ("2023-05-02", "Kuningan", "Balinese"),
    ("2023-06-17", "Galungan", "Balinese"),
    ("2023-06-27", "Kuningan", "Balinese"),
    ("2023-08-12", "Galungan", "Balinese"),
    ("2023-08-22", "Kuningan", "Balinese"),
    ("2023-10-07", "Galungan", "Balinese"),
    ("2023-10-17", "Kuningan", "Balinese"),
    ("2023-12-02", "Galungan", "Balinese"),
    ("2023-12-12", "Kuningan", "Balinese"),
]

# Основные международные праздники (исторические)
INTERNATIONAL_HOLIDAYS_HISTORICAL = [
    # 2024
    ("2024-01-01", "New Year's Day", "International"),
    ("2024-02-14", "Valentine's Day", "International"),
    ("2024-03-08", "International Women's Day", "International"),
    ("2024-05-01", "Labour Day", "International"),
    ("2024-10-31", "Halloween", "International"),
    ("2024-12-25", "Christmas Day", "International"),
    ("2024-12-31", "New Year's Eve", "International"),
    # 2023
    ("2023-01-01", "New Year's Day", "International"),
    ("2023-02-14", "Valentine's Day", "International"),
    ("2023-03-08", "International Women's Day", "International"),
    ("2023-05-01", "Labour Day", "International"),
    ("2023-10-31", "Halloween", "International"),
    ("2023-12-25", "Christmas Day", "International"),
    ("2023-12-31", "New Year's Eve", "International"),
]

# Мусульманские праздники 2025 (максимально полный список)
MUSLIM_HOLIDAYS_2025 = [
    ("2025-01-13", "Maulid Nabi Muhammad", "Muslim"),  # День рождения Пророка
    ("2025-01-27", "Isra Miraj", "Muslim"),  # Вознесение Пророка (официальная дата)
    ("2025-02-28", "Isra Miraj", "Muslim"),  # Вознесение Пророка (альт. дата)
    ("2025-03-01", "Ramadan preparation", "Muslim"),  # Подготовка к Рамадану
    ("2025-03-11", "Ramadan begins", "Muslim"),  # Начало Рамадана
    ("2025-03-20", "Ramadan mid-point", "Muslim"),  # Середина Рамадана
    ("2025-03-29", "Lailat al-Qadr", "Muslim"),  # Ночь предопределения
    ("2025-03-30", "Eid al-Fitr", "Muslim"),  # Окончание Рамадана (день 1)
    ("2025-03-31", "Eid al-Fitr", "Muslim"),  # Окончание Рамадана (день 2)
    ("2025-04-01", "Eid al-Fitr", "Muslim"),  # Окончание Рамадана (день 3)
    ("2025-04-02", "Eid celebration", "Muslim"),  # Продолжение празднования
    ("2025-04-03", "Eid celebration", "Muslim"),  # Продолжение празднования
    ("2025-04-04", "Eid celebration", "Muslim"),  # Продолжение празднования
    ("2025-04-07", "Eid celebration", "Muslim"),  # Продолжение празднования
    ("2025-04-09", "Ramadan Kareem", "Muslim"),  # Рамадан благословенный
    ("2025-05-15", "Lailat al-Qadr", "Muslim"),  # Ночь предопределения
    ("2025-06-05", "Day of Arafah", "Muslim"),  # День Арафат
    ("2025-06-06", "Eid al-Adha", "Muslim"),  # Курбан-байрам (день 1)  
    ("2025-06-07", "Eid al-Adha", "Muslim"),  # Курбан-байрам (день 2)
    ("2025-06-08", "Eid al-Adha celebration", "Muslim"),  # Продолжение празднования
    ("2025-06-09", "Eid al-Adha celebration", "Muslim"),  # Продолжение празднования
    ("2025-06-27", "Islamic New Year", "Muslim"),  # Исламский новый год (Muharram)
    ("2025-07-15", "Muharram", "Muslim"),  # Месяц Мухаррам
    ("2025-07-25", "Muharram 10", "Muslim"),  # 10-й день Мухаррама
    ("2025-08-23", "Day of Ashura", "Muslim"),  # День Ашура
    ("2025-09-05", "Maulid Nabi", "Muslim"),  # День рождения Пророка
    ("2025-09-15", "Maulid celebration", "Muslim"),  # Празднование Маулида
    ("2025-10-12", "Islamic Calendar Event", "Muslim"),  # Исламский календарь
    ("2025-10-22", "Rajab month", "Muslim"),  # Месяц Раджаб
    ("2025-11-15", "Lailat al-Miraj", "Muslim"),  # Ночь вознесения
    ("2025-11-25", "Shaban month", "Muslim"),  # Месяц Шабан
    ("2025-12-09", "Day of Arafah", "Muslim"),  # День Арафат
    ("2025-12-19", "Hijri celebration", "Muslim"),  # Хиджри празднование
    ("2025-12-31", "Islamic Year End", "Muslim"),  # Конец исламского года
]

# Балийские праздники 2025 (максимально полный календарь с циклами)
BALINESE_HOLIDAYS_2025 = [
    ("2025-01-07", "Hari Raya Siwaratri", "Balinese"),  # Шиваратри
    ("2025-01-14", "Tilem Kepitu", "Balinese"),  # Новолуние 7-го месяца
    ("2025-01-23", "Saraswati", "Balinese"),  # День знаний и мудрости
    ("2025-01-28", "Purnama Kepitu", "Balinese"),  # Полнолуние 7-го месяца
    ("2025-02-06", "Tumpek Landep", "Balinese"),  # Благословение металла
    ("2025-02-08", "Hari Raya Saraswati", "Balinese"),  # Сарасвати
    ("2025-02-12", "Hari Raya Pagerwesi", "Balinese"),  # Пагервеси
    ("2025-02-13", "Tilem Kedelapan", "Balinese"),  # Новолуние 8-го месяца
    ("2025-02-22", "Melasti", "Balinese"),  # Очищение перед Nyepi
    ("2025-02-27", "Purnama Kedelapan", "Balinese"),  # Полнолуние 8-го месяца
    ("2025-03-14", "Tilem Kesanga", "Balinese"),  # Новолуние 9-го месяца
    ("2025-03-28", "Tawur Kesanga", "Balinese"),  # Тавур Кесанга
    ("2025-03-29", "Nyepi", "Balinese"),  # День тишины (главный праздник)
    ("2025-03-30", "Ngembak Geni", "Balinese"),  # День после Nyepi
    ("2025-04-03", "Rambut Sedana", "Balinese"),  # Балийский новый год
    ("2025-04-12", "Tilem Kedasa", "Balinese"),  # Новолуние 10-го месяца
    ("2025-04-17", "Tumpek Wariga", "Balinese"),  # Благословение растений
    ("2025-04-22", "Penampahan Galungan", "Balinese"),  # Подготовка к Галунган
    ("2025-04-23", "Galungan", "Balinese"),  # Галунган (первый цикл)
    ("2025-04-24", "Umanis Galungan", "Balinese"),  # День после Галунган
    ("2025-04-26", "Purnama Kedasa", "Balinese"),  # Полнолуние 10-го месяца
    ("2025-05-01", "Odalan", "Balinese"),  # Храмовый праздник
    ("2025-05-02", "Penampahan Kuningan", "Balinese"),  # Подготовка к Кунинган
    ("2025-05-03", "Kuningan", "Balinese"),  # Кунинган (первый цикл)
    ("2025-05-11", "Tilem Jiyestha", "Balinese"),  # Новолуние месяца Джиестха
    ("2025-05-15", "Purnama Kapat", "Balinese"),  # Полнолуние 4-го месяца
    ("2025-05-25", "Purnama Jiyestha", "Balinese"),  # Полнолуние Джиестха
    ("2025-05-29", "Tumpek Uduh", "Balinese"),  # Тумпек Удух
    ("2025-06-08", "Kuningan", "Balinese"),  # Завершение Galungan
    ("2025-06-10", "Tilem Asadha", "Balinese"),  # Новолуние Асадха
    ("2025-06-22", "Tumpek Uduh", "Balinese"),  # Благословение кокосов
    ("2025-06-23", "Purnama Asadha", "Balinese"),  # Полнолуние Асадха
    ("2025-07-09", "Tilem Srawana", "Balinese"),  # Новолуние Сравана
    ("2025-07-13", "Purnama Kelima", "Balinese"),  # Полнолуние 5-го месяца
    ("2025-07-17", "Purnama", "Balinese"),  # Полнолуние
    ("2025-07-22", "Purnama Srawana", "Balinese"),  # Полнолуние Сравана
    ("2025-08-03", "Tumpek Kandang", "Balinese"),  # Благословение животных
    ("2025-08-07", "Tilem Bhadrawada", "Balinese"),  # Новолуние Бхадрапада
    ("2025-08-15", "Pitra Paksa", "Balinese"),  # Поминовение предков
    ("2025-08-20", "Purnama Bhadrawada", "Balinese"),  # Полнолуние Бхадрапада
    ("2025-08-29", "Purnama Keenam", "Balinese"),  # Полнолуние 6-го месяца
    ("2025-09-05", "Tilem Aswina", "Balinese"),  # Новолуние Асвина
    ("2025-09-06", "Hari Raya Saraswati", "Balinese"),  # Сарасвати (второй цикл)
    ("2025-09-10", "Hari Raya Pagerwesi", "Balinese"),  # Пагервеси (второй цикл)
    ("2025-09-14", "Tumpek Kuningan", "Balinese"),  # Благословение книг
    ("2025-09-18", "Penampahan Galungan", "Balinese"),  # Подготовка ко второму Галунган
    ("2025-09-19", "Purnama Aswina", "Balinese"),  # Полнолуние Асвина
    ("2025-09-25", "Galungan", "Balinese"),  # Второй Galungan в году
    ("2025-09-26", "Umanis Galungan", "Balinese"),  # День после второго Галунган
    ("2025-10-04", "Tilem Kartika", "Balinese"),  # Новолуние Картика
    ("2025-10-05", "Kuningan", "Balinese"),  # Второй Kuningan
    ("2025-10-18", "Purnama Kartika", "Balinese"),  # Полнолуние Картика
    ("2025-10-19", "Tumpek Wayang", "Balinese"),  # Благословение искусства
    ("2025-11-02", "Diwali", "Balinese"),  # Праздник огней
    ("2025-11-03", "Tilem Margasirsa", "Balinese"),  # Новолуние Маргасирса
    ("2025-11-16", "Purnama Kedelapan", "Balinese"),  # Полнолуние 8-го месяца
    ("2025-11-17", "Purnama Margasirsa", "Balinese"),  # Полнолуние Маргасирса
    ("2025-11-18", "Penampahan Galungan", "Balinese"),  # Подготовка к третьему Галунган
    ("2025-11-19", "Galungan", "Balinese"),  # Третий Галунган в году
    ("2025-11-20", "Umanis Galungan", "Balinese"),  # День после третьего Галунган
    ("2025-11-29", "Kuningan", "Balinese"),  # Третий Кунинган
    ("2025-11-30", "Tumpek Krulut", "Balinese"),  # Благословение музыки
    ("2025-12-02", "Tilem Pusya", "Balinese"),  # Новолуние Пушья
    ("2025-12-14", "Purnama Kesembilan", "Balinese"),  # Полнолуние 9-го месяца
    ("2025-12-16", "Purnama Pusya", "Balinese"),  # Полнолуние Пушья
    ("2025-12-22", "Dharma Shanti", "Balinese"),  # Религиозный праздник
    ("2025-12-28", "Galungan", "Balinese"),  # Четвертый Galungan в году
    ("2025-12-31", "Tilem Magha", "Balinese"),  # Новолуние Магха
]

# Индонезийские национальные праздники (полный календарь 2025)
INDONESIAN_HOLIDAYS = [
    ("-01-01", "New Year's Day", "Indonesian"),  # Новый год
    ("-01-27", "National Education Day", "Indonesian"),  # День образования
    ("-02-10", "Chinese New Year", "Indonesian"),  # Китайский новый год
    ("-02-28", "National Awakening Day", "Indonesian"),  # День национального пробуждения
    ("-03-11", "Ramadan begins", "Indonesian"),  # Начало Рамадана
    ("-04-18", "Good Friday", "Indonesian"),  # Страстная пятница
    ("-04-21", "Kartini Day", "Indonesian"),  # День Картини
    ("-05-01", "Labour Day", "Indonesian"),  # День труда
    ("-05-02", "National Education Day", "Indonesian"),  # День национального образования
    ("-05-09", "Vesak Day", "Indonesian"),  # День Будды
    ("-05-20", "National Awakening Day", "Indonesian"),  # День национального пробуждения
    ("-05-29", "Ascension Day", "Indonesian"),  # Вознесение Господне
    ("-06-01", "Pancasila Day", "Indonesian"),  # День Панчасила
    ("-07-17", "National Constitution Day", "Indonesian"),  # День конституции
    ("-08-17", "Independence Day", "Indonesian"),  # День независимости
    ("-09-30", "G30S/PKI Day", "Indonesian"),  # День памяти G30S
    ("-10-02", "Batik Day", "Indonesian"),  # День батика
    ("-10-05", "Armed Forces Day", "Indonesian"),  # День вооруженных сил
    ("-10-28", "Youth Pledge Day", "Indonesian"),  # День клятвы молодежи
    ("-11-10", "Heroes' Day", "Indonesian"),  # День героев
    ("-12-22", "Mother's Day", "Indonesian"),  # День матери
    ("-12-25", "Christmas Day", "Indonesian"),  # Рождество
    ("-12-26", "Boxing Day", "Indonesian"),  # День подарков
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
    """Загрузка всех праздников для ML обучения и аналитики"""
    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()
    years = list(range(start.year, end.year + 1))

    frames: List[pd.DataFrame] = []
    for y in years:
        # Официальные индонезийские праздники (Nager API)
        frames.append(_fetch_year(y))
        
        # Международные праздники
        frames.append(_international_for_year(y))
        
        # Национальные индонезийские праздники
        frames.append(_indonesian_for_year(y))
        
        # Исторические данные для обучения ML (2023-2024)
        if y in [2023, 2024]:
            frames.append(_historical_holidays(y))
        
        # Полные праздники для 2025
        if y == 2025:
            frames.append(_muslim_holidays_2025())
            frames.append(_balinese_holidays_2025())
            frames.append(_christian_holidays_2025())
            frames.append(_buddhist_holidays_2025())
            frames.append(_chinese_holidays_2025())
        
        # Локальные балийские праздники (скрейпинг)
        for (yy, url) in BALI_SOURCES:
            if yy == y:
                frames.append(_parse_bali_local(yy, url))

    if not frames:
        return pd.DataFrame(columns=["date", "holiday_name", "region"]) 

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df[(df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))]
    df = df.drop_duplicates(["date", "holiday_name", "region"]).sort_values("date").reset_index(drop=True)
    
    # Добавляем информацию о типе праздника для ML
    df["holiday_type"] = df["region"]
    df["is_major_holiday"] = df["holiday_name"].str.contains(
        "Nyepi|Eid al-Fitr|Eid al-Adha|Galungan|Christmas|New Year", 
        case=False, na=False
    )
    
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


def _historical_holidays(year: int) -> pd.DataFrame:
    """Исторические праздники для обучения ML (2023-2024)"""
    rows = []
    
    # Мусульманские праздники
    if year == 2024:
        for date_str, name, region in MUSLIM_HOLIDAYS_2024:
            date = dt.date.fromisoformat(date_str)
            rows.append({"date": date, "holiday_name": name, "region": region})
    elif year == 2023:
        for date_str, name, region in MUSLIM_HOLIDAYS_2023:
            date = dt.date.fromisoformat(date_str)
            rows.append({"date": date, "holiday_name": name, "region": region})
    
    # Балийские праздники
    if year == 2024:
        for date_str, name, region in BALINESE_HOLIDAYS_2024:
            date = dt.date.fromisoformat(date_str)
            rows.append({"date": date, "holiday_name": name, "region": region})
    elif year == 2023:
        for date_str, name, region in BALINESE_HOLIDAYS_2023:
            date = dt.date.fromisoformat(date_str)
            rows.append({"date": date, "holiday_name": name, "region": region})
    
    # Международные исторические праздники
    for date_str, name, region in INTERNATIONAL_HOLIDAYS_HISTORICAL:
        if date_str.startswith(str(year)):
            date = dt.date.fromisoformat(date_str)
            rows.append({"date": date, "holiday_name": name, "region": region})
    
    return pd.DataFrame(rows)


def _christian_holidays_2025() -> pd.DataFrame:
    """Христианские праздники 2025"""
    rows = []
    for date_str, name, region in CHRISTIAN_HOLIDAYS_2025:
        date = dt.date.fromisoformat(date_str)
        rows.append({"date": date, "holiday_name": name, "region": region})
    return pd.DataFrame(rows)


def _buddhist_holidays_2025() -> pd.DataFrame:
    """Буддистские праздники 2025"""
    rows = []
    for date_str, name, region in BUDDHIST_HOLIDAYS_2025:
        date = dt.date.fromisoformat(date_str)
        rows.append({"date": date, "holiday_name": name, "region": region})
    return pd.DataFrame(rows)


def _chinese_holidays_2025() -> pd.DataFrame:
    """Китайские праздники 2025"""
    rows = []
    for date_str, name, region in CHINESE_HOLIDAYS_2025:
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