"""
API клиент для получения статистики ресторанов из живой БД
Основано на спецификации: GET /api/v1/getRestaurantStats
"""

import os
import time
import hashlib
import json
import logging
from datetime import date, timedelta
from typing import Dict, Any, Optional, List, Iterable

import requests
import psycopg2
from psycopg2.extras import execute_values, RealDictCursor

# Конфигурация
API_BASE = os.getenv("STATS_API_BASE", "http://5.187.7.140:3000")
DB_DSN = os.getenv("DATABASE_URL")
PROJECT_ROOT = os.getenv("PROJECT_ROOT", os.getcwd())
TIMEOUT = 15  # секунд
RETRIES = 3
CHUNK_DAYS = 14

# Настройка логирования
os.makedirs(os.path.join(PROJECT_ROOT, 'logs'), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PROJECT_ROOT, 'logs', 'api_client.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def _hash_payload(obj: Any) -> str:
    """Создание хеша для идемпотентности"""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _make_request(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Выполнение HTTP запроса с ретраями"""
    url = f"{API_BASE}{path}"
    last_exception = None
    
    for attempt in range(RETRIES):
        try:
            logger.info(f"API request attempt {attempt + 1}: {url} with params {params}")
            response = requests.get(url, params=params, timeout=TIMEOUT)
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"API success: {len(result.get('data', []))} records received")
            return result
            
        except Exception as e:
            last_exception = e
            wait_time = 1.5 ** (attempt + 1)  # Экспоненциальная пауза
            logger.warning(f"API request failed (attempt {attempt + 1}): {e}. Retrying in {wait_time:.1f}s")
            time.sleep(wait_time)
    
    raise RuntimeError(f"API request failed after {RETRIES} retries: {last_exception}")


def fetch_restaurant_stats(restaurant_name: str, source: str, start_date: date, end_date: date) -> Dict[str, Any]:
    """
    Получение статистики ресторана из API
    
    Args:
        restaurant_name: Точное название ресторана
        source: 'grab' или 'gojek'
        start_date: Начало периода
        end_date: Конец периода
    
    Returns:
        JSON ответ с данными
    """
    params = {
        "restaurant_name": restaurant_name,
        "source": source,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }
    
    return _make_request("/api/v1/getRestaurantStats", params)


def _get_db_connection():
    """Создание подключения к PostgreSQL"""
    if not DB_DSN:
        raise RuntimeError("DATABASE_URL environment variable is not set")
    
    return psycopg2.connect(DB_DSN)


def _parse_time_field(time_value) -> Optional[float]:
    """Парсинг временных полей в минуты"""
    if not time_value:
        return None
    
    try:
        if isinstance(time_value, str) and ':' in time_value:
            # Формат HH:MM:SS
            parts = time_value.split(':')
            hours = int(parts[0])
            minutes = int(parts[1])
            return hours * 60 + minutes
        else:
            # Числовое значение
            return float(time_value)
    except:
        return None


def _normalize_api_data(api_response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Нормализация данных из API в формат для БД
    
    Args:
        api_response: Ответ от API
    
    Returns:
        Список нормализованных записей
    """
    data = api_response.get("data", [])
    restaurant_name = api_response.get("restaurant_name")
    source = api_response.get("source")
    
    normalized_rows = []
    
    for record in data:
        # Извлекаем дату
        stat_date = record.get("stat_date") or record.get("date")
        if not stat_date:
            logger.warning(f"Skipping record without date: {record}")
            continue
        
        # Нормализуем основные поля
        normalized_record = {
            "restaurant_name": restaurant_name,
            "source": source,
            "stat_date": stat_date,
            "payload": record,
            "row_hash": _hash_payload(record),
            
            # Основные метрики
            "sales_idr": record.get("sales", 0),
            "orders_total": record.get("orders", 0),
            "ads_spend_idr": record.get("ads_spend", 0),
            "ads_sales_idr": record.get("ads_sales", 0),
            "cancelled_orders": record.get("cancelled_orders", 0),
            "lost_orders": record.get("lost_orders", 0),
            "rating_avg": record.get("rating", 0),
            
            # Временные метрики (приводим к минутам)
            "prep_time_min": _parse_time_field(record.get("preparation_time")),
            "confirm_time_min": _parse_time_field(record.get("accepting_time")),
            "delivery_time_min": _parse_time_field(record.get("delivery_time")),
            "offline_time_min": record.get("offline_rate", 0),
        }
        
        normalized_rows.append(normalized_record)
    
    logger.info(f"Normalized {len(normalized_rows)} records for {restaurant_name} ({source})")
    return normalized_rows


def ensure_restaurant_exists(restaurant_name: str) -> int:
    """
    Автоматически добавляет новый ресторан в restaurant_mapping если его нет
    
    Args:
        restaurant_name: Название ресторана
    
    Returns:
        restaurant_id (новый или существующий)
    """
    with _get_db_connection() as conn:
        with conn.cursor() as cursor:
            # Проверяем, существует ли ресторан
            cursor.execute(
                "SELECT restaurant_id FROM restaurant_mapping WHERE restaurant_name = %s",
                (restaurant_name,)
            )
            result = cursor.fetchone()
            
            if result:
                return result[0]
            
            # Добавляем новый ресторан
            cursor.execute(
                """
                INSERT INTO restaurant_mapping (restaurant_name, is_active, created_at)
                VALUES (%s, TRUE, now())
                RETURNING restaurant_id
                """,
                (restaurant_name,)
            )
            new_id = cursor.fetchone()[0]
            logger.info(f"🆕 Автоматически добавлен новый ресторан: {restaurant_name} (ID: {new_id})")
            return new_id


def upsert_stats_data(rows: List[Dict[str, Any]]) -> int:
    """
    UPSERT данных в raw_stats с проверкой изменений по хешу
    Автоматически добавляет новые рестораны в restaurant_mapping
    
    Args:
        rows: Список нормализованных записей
    
    Returns:
        Количество фактически обновленных записей
    """
    if not rows:
        return 0
    
    # Автоматически добавляем новые рестораны
    unique_restaurants = set(row["restaurant_name"] for row in rows)
    for restaurant_name in unique_restaurants:
        ensure_restaurant_exists(restaurant_name)
    
    upsert_sql = """
        INSERT INTO raw_stats (
            restaurant_name, source, stat_date, payload, row_hash,
            sales_idr, orders_total, ads_spend_idr, ads_sales_idr,
            cancelled_orders, lost_orders, rating_avg,
            prep_time_min, confirm_time_min, delivery_time_min, offline_time_min
        )
        VALUES %s
        ON CONFLICT (restaurant_name, source, stat_date) 
        DO UPDATE SET
            payload = EXCLUDED.payload,
            row_hash = EXCLUDED.row_hash,
            sales_idr = EXCLUDED.sales_idr,
            orders_total = EXCLUDED.orders_total,
            ads_spend_idr = EXCLUDED.ads_spend_idr,
            ads_sales_idr = EXCLUDED.ads_sales_idr,
            cancelled_orders = EXCLUDED.cancelled_orders,
            lost_orders = EXCLUDED.lost_orders,
            rating_avg = EXCLUDED.rating_avg,
            prep_time_min = EXCLUDED.prep_time_min,
            confirm_time_min = EXCLUDED.confirm_time_min,
            delivery_time_min = EXCLUDED.delivery_time_min,
            offline_time_min = EXCLUDED.offline_time_min,
            updated_at = now()
        WHERE raw_stats.row_hash <> EXCLUDED.row_hash
    """
    
    with _get_db_connection() as conn:
        with conn.cursor() as cursor:
            # Подготавливаем данные для bulk insert
            values = []
            for row in rows:
                values.append((
                    row["restaurant_name"],
                    row["source"],
                    row["stat_date"],
                    json.dumps(row["payload"], ensure_ascii=False),
                    row["row_hash"],
                    row["sales_idr"],
                    row["orders_total"],
                    row["ads_spend_idr"],
                    row["ads_sales_idr"],
                    row["cancelled_orders"],
                    row["lost_orders"],
                    row["rating_avg"],
                    row["prep_time_min"],
                    row["confirm_time_min"],
                    row["delivery_time_min"],
                    row["offline_time_min"]
                ))
            
            # Выполняем UPSERT
            execute_values(cursor, upsert_sql, values, page_size=500)
            
            # Получаем количество измененных записей
            updated_count = cursor.rowcount
            conn.commit()
            
            logger.info(f"UPSERT completed: {updated_count} records updated")
            return updated_count


def get_last_loaded_date(restaurant_name: str, source: str) -> Optional[date]:
    """Получение последней загруженной даты для ресторана и источника"""
    
    query = """
        SELECT MAX(stat_date) 
        FROM raw_stats 
        WHERE restaurant_name = %s AND source = %s
    """
    
    try:
        with _get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (restaurant_name, source))
                result = cursor.fetchone()
                return result[0] if result and result[0] else None
    except Exception as e:
        logger.error(f"Error getting last loaded date: {e}")
        return None


def refresh_materialized_view():
    """Обновление материализованной витрины"""
    try:
        with _get_db_connection() as conn:
            with conn.cursor() as cursor:
                logger.info("Refreshing materialized view mv_daily_metrics...")
                cursor.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY mv_daily_metrics")
                conn.commit()
                logger.info("Materialized view refreshed successfully")
    except Exception as e:
        logger.error(f"Error refreshing materialized view: {e}")
        raise


def add_ml_job(job_type: str, restaurant_name: str = None, payload: Dict = None):
    """Добавление задачи в очередь ML"""
    try:
        with _get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO ml_jobs (job_type, restaurant_name, payload) 
                    VALUES (%s, %s, %s)
                    """,
                    (job_type, restaurant_name, json.dumps(payload) if payload else None)
                )
                conn.commit()
                logger.info(f"ML job added: {job_type} for {restaurant_name}")
    except Exception as e:
        logger.error(f"Error adding ML job: {e}")


def run_incremental_sync(restaurant_name: str, source: str, start_date: Optional[date] = None, end_date: Optional[date] = None) -> Dict[str, Any]:
    """
    Инкрементальная синхронизация данных ресторана
    
    Args:
        restaurant_name: Название ресторана
        source: 'grab' или 'gojek'
        start_date: Начало периода (по умолчанию - от последней загруженной даты)
        end_date: Конец периода (по умолчанию - вчера)
    
    Returns:
        Статистика синхронизации
    """
    today = date.today()
    yesterday = today - timedelta(days=1)
    
    # Определяем период загрузки
    if end_date is None:
        end_date = yesterday
    
    if start_date is None:
        last_date = get_last_loaded_date(restaurant_name, source)
        if last_date:
            start_date = last_date + timedelta(days=1)
        else:
            # Если данных нет, загружаем последние 90 дней
            start_date = end_date - timedelta(days=90)
    
    if start_date > end_date:
        logger.info(f"No new data to sync for {restaurant_name} ({source})")
        return {"status": "up_to_date", "records_updated": 0}
    
    logger.info(f"Starting incremental sync for {restaurant_name} ({source}): {start_date} to {end_date}")
    
    total_records_updated = 0
    current_start = start_date
    
    # Загружаем чанками по 14 дней
    while current_start <= end_date:
        current_end = min(current_start + timedelta(days=CHUNK_DAYS - 1), end_date)
        
        try:
            # Получаем данные из API
            api_response = fetch_restaurant_stats(restaurant_name, source, current_start, current_end)
            
            # Нормализуем данные
            normalized_rows = _normalize_api_data(api_response)
            
            # Сохраняем в БД
            updated_count = upsert_stats_data(normalized_rows)
            total_records_updated += updated_count
            
            logger.info(f"Chunk {current_start} to {current_end}: {updated_count} records updated")
            
        except Exception as e:
            logger.error(f"Error processing chunk {current_start} to {current_end}: {e}")
            # Продолжаем со следующим чанком
        
        current_start = current_end + timedelta(days=1)
    
    # Обновляем витрину данных
    if total_records_updated > 0:
        refresh_materialized_view()
        
        # Добавляем задачу на переобучение ML если много новых данных
        if total_records_updated >= 30:
            add_ml_job("retrain", restaurant_name, {
                "reason": "significant_data_update",
                "new_records": total_records_updated,
                "period": f"{start_date} to {end_date}"
            })
    
    result = {
        "status": "success",
        "records_updated": total_records_updated,
        "period": f"{start_date} to {end_date}",
        "restaurant_name": restaurant_name,
        "source": source
    }
    
    logger.info(f"Sync completed: {result}")
    return result


def sync_all_sources(restaurant_name: str, start_date: Optional[date] = None, end_date: Optional[date] = None) -> Dict[str, Any]:
    """Синхронизация данных для всех источников (GRAB + GOJEK)"""
    
    results = {}
    total_updated = 0
    
    for source in ['grab', 'gojek']:
        try:
            result = run_incremental_sync(restaurant_name, source, start_date, end_date)
            results[source] = result
            total_updated += result.get('records_updated', 0)
        except Exception as e:
            logger.error(f"Error syncing {source} for {restaurant_name}: {e}")
            results[source] = {"status": "error", "error": str(e)}
    
    return {
        "restaurant_name": restaurant_name,
        "total_records_updated": total_updated,
        "sources": results,
        "period": f"{start_date or 'auto'} to {end_date or 'yesterday'}"
    }


def get_available_restaurants() -> List[str]:
    """Получение списка ресторанов из БД"""
    try:
        with _get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT DISTINCT restaurant_name 
                    FROM raw_stats 
                    WHERE restaurant_name IS NOT NULL
                    ORDER BY restaurant_name
                """)
                return [row[0] for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Error getting restaurants list: {e}")
        return []


def get_data_quality_report(restaurant_name: str = None) -> Dict[str, Any]:
    """Отчет о качестве данных"""
    try:
        with _get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                where_clause = "WHERE restaurant_name = %s" if restaurant_name else ""
                params = (restaurant_name,) if restaurant_name else ()
                
                cursor.execute(f"""
                    SELECT 
                        restaurant_name,
                        COUNT(*) as total_days,
                        COUNT(CASE WHEN grab_issue IS NOT NULL THEN 1 END) as grab_issues,
                        COUNT(CASE WHEN gojek_issue IS NOT NULL THEN 1 END) as gojek_issues,
                        COUNT(CASE WHEN sales_issue IS NOT NULL THEN 1 END) as sales_issues,
                        MIN(stat_date) as first_date,
                        MAX(stat_date) as last_date
                    FROM data_quality_check
                    {where_clause}
                    GROUP BY restaurant_name
                    ORDER BY restaurant_name
                """, params)
                
                return {"restaurants": [dict(row) for row in cursor.fetchall()]}
    except Exception as e:
        logger.error(f"Error generating data quality report: {e}")
        return {"error": str(e)}


# CLI интерфейс
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="API client for restaurant stats")
    parser.add_argument("restaurant_name", help="Restaurant name")
    parser.add_argument("--source", choices=['grab', 'gojek', 'all'], default='all', help="Data source")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--quality-report", action='store_true', help="Show data quality report")
    
    args = parser.parse_args()
    
    if args.quality_report:
        report = get_data_quality_report(args.restaurant_name)
        print(json.dumps(report, indent=2, default=str))
    else:
        start_date = date.fromisoformat(args.start_date) if args.start_date else None
        end_date = date.fromisoformat(args.end_date) if args.end_date else None
        
        if args.source == 'all':
            result = sync_all_sources(args.restaurant_name, start_date, end_date)
        else:
            result = run_incremental_sync(args.restaurant_name, args.source, start_date, end_date)
        
        print(json.dumps(result, indent=2, default=str))