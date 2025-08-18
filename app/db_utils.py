"""
Утилиты для работы с базой данных
Обеспечивают единый интерфейс для SQLite и PostgreSQL
"""

import os
import pandas as pd
from etl.data_loader import get_engine


def get_universal_engine():
    """Получение движка БД (PostgreSQL или SQLite fallback)"""
    
    db_url = os.getenv("DATABASE_URL")
    
    if db_url and "postgresql" in db_url:
        # Используем PostgreSQL
        import psycopg2
        return psycopg2.connect(db_url)
    else:
        # Fallback к SQLite
        return get_engine()


def execute_query(query: str, params: tuple = None, use_postgres_syntax: bool = None) -> pd.DataFrame:
    """
    Выполнение запроса с автоматическим определением синтаксиса
    
    Args:
        query: SQL запрос
        params: Параметры запроса
        use_postgres_syntax: Принудительное использование PostgreSQL синтаксиса
    
    Returns:
        DataFrame с результатами
    """
    
    db_url = os.getenv("DATABASE_URL")
    is_postgres = bool(db_url and "postgresql" in db_url)
    
    if use_postgres_syntax is None:
        use_postgres_syntax = is_postgres
    
    try:
        if is_postgres:
            # PostgreSQL
            engine = get_universal_engine()
            return pd.read_sql_query(query, engine, params=params)
        else:
            # SQLite
            engine = get_engine()
            return pd.read_sql_query(query, engine, params=params)
            
    except Exception as e:
        # Fallback к SQLite если PostgreSQL недоступен
        if is_postgres:
            engine = get_engine()
            # Конвертируем PostgreSQL синтаксис в SQLite если нужно
            sqlite_query = _convert_postgres_to_sqlite(query)
            return pd.read_sql_query(sqlite_query, engine, params=params)
        else:
            raise e


def _convert_postgres_to_sqlite(query: str) -> str:
    """Простая конвертация PostgreSQL запросов в SQLite"""
    
    # Заменяем %s на ? для параметров
    query = query.replace('%s', '?')
    
    # Заменяем PostgreSQL функции на SQLite аналоги
    replacements = {
        'EXTRACT(DOW FROM': 'strftime(\'%w\',',
        'EXTRACT(HOUR FROM': 'strftime(\'%H\',',
        'EXTRACT(MINUTE FROM': 'strftime(\'%M\',',
        'now()': 'datetime(\'now\')',
        'COALESCE': 'IFNULL',
    }
    
    for pg_syntax, sqlite_syntax in replacements.items():
        query = query.replace(pg_syntax, sqlite_syntax)
    
    return query


def get_restaurants_with_data() -> pd.DataFrame:
    """Получение списка ресторанов с данными"""
    
    if os.getenv("DATABASE_URL") and "postgresql" in os.getenv("DATABASE_URL"):
        # PostgreSQL - из новых таблиц
        query = """
            SELECT DISTINCT 
                rm.restaurant_id as id,
                rm.restaurant_name as name,
                COUNT(df.stat_date) as days_with_data,
                MIN(df.stat_date) as first_date,
                MAX(df.stat_date) as last_date
            FROM restaurant_mapping rm
            LEFT JOIN daily_facts df ON rm.restaurant_name = df.restaurant_name
            WHERE rm.is_active = true
            GROUP BY rm.restaurant_id, rm.restaurant_name
            HAVING COUNT(df.stat_date) > 0
            ORDER BY rm.restaurant_name
        """
        
        return execute_query(query, use_postgres_syntax=True)
    else:
        # SQLite - старые таблицы
        query = """
            SELECT DISTINCT 
                r.id,
                r.name,
                COUNT(g.stat_date) as grab_days,
                COUNT(j.stat_date) as gojek_days
            FROM restaurants r
            LEFT JOIN grab_stats g ON r.id = g.restaurant_id
            LEFT JOIN gojek_stats j ON r.id = j.restaurant_id
            GROUP BY r.id, r.name
            HAVING COUNT(g.stat_date) > 0 OR COUNT(j.stat_date) > 0
            ORDER BY r.name
        """
        
        return execute_query(query, use_postgres_syntax=False)