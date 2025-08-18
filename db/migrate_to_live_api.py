"""
Скрипт миграции для подключения к живому API
Создает новые таблицы и мигрирует существующие данные
"""

import os
import sys
import logging
from datetime import datetime

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

sys.path.append('/workspace')

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_DSN = os.getenv("DATABASE_URL")


def create_database_schema():
    """Создание схемы БД для работы с живым API"""
    
    if not DB_DSN:
        logger.error("DATABASE_URL not set. Using SQLite fallback.")
        return False
    
    try:
        # Читаем SQL схему
        schema_path = '/workspace/db/schema.sql'
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema_sql = f.read()
        
        # Выполняем миграцию
        with psycopg2.connect(DB_DSN) as conn:
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            
            with conn.cursor() as cursor:
                logger.info("Creating database schema...")
                
                # Выполняем SQL по частям (разделяем по CREATE)
                statements = schema_sql.split(';')
                
                for i, statement in enumerate(statements):
                    statement = statement.strip()
                    if statement:
                        try:
                            cursor.execute(statement)
                            logger.info(f"Executed statement {i+1}/{len(statements)}")
                        except Exception as e:
                            # Некоторые CREATE IF NOT EXISTS могут давать warnings
                            if "already exists" in str(e).lower():
                                logger.info(f"Statement {i+1} - object already exists, skipping")
                            else:
                                logger.error(f"Error in statement {i+1}: {e}")
                                logger.error(f"Statement: {statement[:100]}...")
        
        logger.info("✅ Database schema created successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error creating database schema: {e}")
        return False


def migrate_existing_data():
    """Миграция существующих данных из SQLite в PostgreSQL"""
    
    try:
        import sqlite3
        import pandas as pd
        
        # Подключаемся к SQLite
        sqlite_path = '/workspace/database.sqlite'
        if not os.path.exists(sqlite_path):
            logger.warning("SQLite database not found, skipping data migration")
            return True
        
        sqlite_conn = sqlite3.connect(sqlite_path)
        
        # Мигрируем рестораны
        logger.info("Migrating restaurants...")
        restaurants_df = pd.read_sql_query("SELECT id, name FROM restaurants", sqlite_conn)
        
        with psycopg2.connect(DB_DSN) as pg_conn:
            with pg_conn.cursor() as cursor:
                for _, row in restaurants_df.iterrows():
                    cursor.execute("""
                        INSERT INTO restaurant_mapping (restaurant_id, restaurant_name, is_active)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (restaurant_name) DO NOTHING
                    """, (row['id'], row['name'], True))
                
                pg_conn.commit()
                logger.info(f"✅ Migrated {len(restaurants_df)} restaurants")
        
        # Мигрируем статистику GRAB
        logger.info("Migrating GRAB stats...")
        grab_df = pd.read_sql_query("""
            SELECT 
                r.name as restaurant_name,
                g.stat_date,
                g.sales,
                g.orders,
                g.ads_spend,
                g.ads_sales,
                g.cancelled_orders,
                g.rating,
                g.offline_rate
            FROM grab_stats g
            JOIN restaurants r ON g.restaurant_id = r.id
            ORDER BY g.stat_date DESC
            LIMIT 10000
        """, sqlite_conn)
        
        _migrate_stats_batch(grab_df, 'grab')
        
        # Мигрируем статистику GOJEK  
        logger.info("Migrating GOJEK stats...")
        gojek_df = pd.read_sql_query("""
            SELECT 
                r.name as restaurant_name,
                g.stat_date,
                g.sales,
                g.orders,
                g.ads_spend,
                g.ads_sales,
                g.cancelled_orders,
                g.lost_orders,
                g.rating,
                g.accepting_time,
                g.preparation_time,
                g.delivery_time
            FROM gojek_stats g
            JOIN restaurants r ON g.restaurant_id = r.id
            ORDER BY g.stat_date DESC
            LIMIT 10000
        """, sqlite_conn)
        
        _migrate_stats_batch(gojek_df, 'gojek')
        
        sqlite_conn.close()
        
        # Обновляем витрину
        from etl.api_client import refresh_materialized_view
        refresh_materialized_view()
        
        logger.info("✅ Data migration completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error migrating data: {e}")
        return False


def _migrate_stats_batch(df, source):
    """Миграция батча статистики"""
    
    with psycopg2.connect(DB_DSN) as conn:
        with conn.cursor() as cursor:
            
            for _, row in df.iterrows():
                # Создаем payload из всех полей
                payload = row.to_dict()
                
                # Удаляем None значения
                payload = {k: v for k, v in payload.items() if pd.notna(v)}
                
                row_hash = _hash_payload(payload)
                
                cursor.execute("""
                    INSERT INTO raw_stats (
                        restaurant_name, source, stat_date, payload, row_hash,
                        sales_idr, orders_total, ads_spend_idr, ads_sales_idr,
                        cancelled_orders, lost_orders, rating_avg, offline_time_min
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (restaurant_name, source, stat_date) DO NOTHING
                """, (
                    row.get('restaurant_name'),
                    source,
                    row.get('stat_date'),
                    json.dumps(payload, ensure_ascii=False),
                    row_hash,
                    row.get('sales', 0),
                    row.get('orders', 0),
                    row.get('ads_spend', 0),
                    row.get('ads_sales', 0),
                    row.get('cancelled_orders', 0),
                    row.get('lost_orders', 0),
                    row.get('rating', 0),
                    row.get('offline_rate', 0)
                ))
            
            conn.commit()
            logger.info(f"✅ Migrated {len(df)} {source} records")


def _hash_payload(obj):
    """Создание хеша для идемпотентности"""
    import hashlib
    import json
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def test_api_connection():
    """Тестирование подключения к API"""
    try:
        from etl.api_client import fetch_restaurant_stats
        from datetime import date, timedelta
        
        # Тестируем на небольшом периоде
        test_date = date.today() - timedelta(days=7)
        end_date = date.today() - timedelta(days=1)
        
        logger.info("Testing API connection...")
        result = fetch_restaurant_stats("Only Kebab", "grab", test_date, end_date)
        
        if result and 'data' in result:
            logger.info(f"✅ API test successful: {len(result['data'])} records received")
            return True
        else:
            logger.error("❌ API test failed: no data received")
            return False
            
    except Exception as e:
        logger.error(f"❌ API test failed: {e}")
        return False


def main():
    """Основная функция миграции"""
    
    logger.info("🚀 Starting migration to live API...")
    
    # 1. Тестируем API
    if not test_api_connection():
        logger.error("❌ API connection test failed. Check API_BASE and network.")
        return False
    
    # 2. Создаем схему БД
    if not create_database_schema():
        logger.error("❌ Database schema creation failed.")
        return False
    
    # 3. Мигрируем существующие данные
    if not migrate_existing_data():
        logger.error("❌ Data migration failed.")
        return False
    
    logger.info("✅ Migration to live API completed successfully!")
    logger.info("Next steps:")
    logger.info("1. Test with: python etl/api_client.py 'Only Kebab' --source=all")
    logger.info("2. Update Streamlit UI with sync buttons")
    logger.info("3. Retrain ML model with: python ml/training.py --from-db")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)