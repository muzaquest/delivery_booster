"""
Построение витрин данных из raw_stats для ML и отчетов
Заменяет CSV файлы на БД витрины
"""

import os
import sys
import logging
from datetime import date, datetime

# PostgreSQL-only utilities removed for MySQL-first bootstrap

sys.path.append('/workspace')

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_DSN = os.getenv("DATABASE_URL")


def create_daily_facts_view():
    """Создание витрины daily_facts"""
    
    sql = """
    -- MySQL-compatible: use regular view instead of materialized view
    DROP VIEW IF EXISTS daily_facts;
    
    CREATE VIEW daily_facts AS
    SELECT 
        rm.restaurant_id,
        rs.restaurant_name,
        rs.stat_date,
        
        -- Агрегированные метрики по дню
        SUM(CASE WHEN rs.source = 'grab' THEN rs.sales_idr ELSE 0 END) as grab_sales,
        SUM(CASE WHEN rs.source = 'gojek' THEN rs.sales_idr ELSE 0 END) as gojek_sales,
        SUM(rs.sales_idr) as total_sales,
        
        SUM(CASE WHEN rs.source = 'grab' THEN rs.orders_total ELSE 0 END) as grab_orders,
        SUM(CASE WHEN rs.source = 'gojek' THEN rs.orders_total ELSE 0 END) as gojek_orders,
        SUM(rs.orders_total) as total_orders,
        
        SUM(CASE WHEN rs.source = 'grab' THEN rs.ads_spend_idr ELSE 0 END) as grab_ads_spend,
        SUM(CASE WHEN rs.source = 'gojek' THEN rs.ads_spend_idr ELSE 0 END) as gojek_ads_spend,
        SUM(rs.ads_spend_idr) as total_ads_spend,
        
        SUM(CASE WHEN rs.source = 'grab' THEN rs.ads_sales_idr ELSE 0 END) as grab_ads_sales,
        SUM(CASE WHEN rs.source = 'gojek' THEN rs.ads_sales_idr ELSE 0 END) as gojek_ads_sales,
        SUM(rs.ads_sales_idr) as total_ads_sales,
        
        -- Операционные метрики
        SUM(CASE WHEN rs.source = 'grab' THEN rs.cancelled_orders ELSE 0 END) as grab_cancelled,
        SUM(CASE WHEN rs.source = 'gojek' THEN rs.cancelled_orders ELSE 0 END) as gojek_cancelled,
        SUM(rs.cancelled_orders) as total_cancelled,
        
        SUM(CASE WHEN rs.source = 'gojek' THEN rs.lost_orders ELSE 0 END) as gojek_lost,
        
        -- Средние рейтинги
        AVG(CASE WHEN rs.source = 'grab' AND rs.rating_avg > 0 THEN rs.rating_avg END) as grab_rating,
        AVG(CASE WHEN rs.source = 'gojek' AND rs.rating_avg > 0 THEN rs.rating_avg END) as gojek_rating,
        
        -- Операционные времена (только GOJEK)
        AVG(CASE WHEN rs.source = 'gojek' AND rs.prep_time_min > 0 THEN rs.prep_time_min END) as gojek_prep_time,
        AVG(CASE WHEN rs.source = 'gojek' AND rs.confirm_time_min > 0 THEN rs.confirm_time_min END) as gojek_confirm_time,
        AVG(CASE WHEN rs.source = 'gojek' AND rs.delivery_time_min > 0 THEN rs.delivery_time_min END) as gojek_delivery_time,
        
        -- Время оффлайн (GRAB)
        SUM(CASE WHEN rs.source = 'grab' THEN rs.offline_time_min ELSE 0 END) as grab_offline_min,
        
        -- Метаданные
        COUNT(*) as sources_count,  -- Должно быть 2 (grab + gojek)
        MAX(rs.updated_at) as last_updated
        
    FROM raw_stats rs
    LEFT JOIN restaurant_mapping rm ON rs.restaurant_name = rm.restaurant_name
    WHERE rm.is_active = 1 OR rm.is_active IS NULL
    GROUP BY rm.restaurant_id, rs.restaurant_name, rs.stat_date;
    
    -- Index creation should be handled separately in MySQL
    """
    
    logger.warning("create_daily_facts_view is PostgreSQL-specific; for MySQL use SQLAlchemy to execute the SQL if needed.")
    return False


def create_ml_dataset_view():
    """Создание витрины ml_dataset с внешними данными"""
    
    sql = """
    DROP MATERIALIZED VIEW IF EXISTS ml_dataset CASCADE;
    
    CREATE MATERIALIZED VIEW ml_dataset AS
    SELECT 
        df.*,
        
        -- Погодные данные (если есть в кеше)
        wc.temp,
        wc.rain, 
        wc.wind,
        wc.humidity,
        
        -- Праздники (простая проверка по дате)
        CASE 
            WHEN TO_CHAR(df.stat_date, 'MM-DD') IN ('04-01', '06-07', '07-07', '09-15') THEN 1  -- Мусульманские
            WHEN TO_CHAR(df.stat_date, 'MM-DD') IN ('03-31', '05-29', '06-08', '09-25', '10-05') THEN 1  -- Балийские
            WHEN TO_CHAR(df.stat_date, 'MM-DD') IN ('08-17', '06-01', '04-21') THEN 1  -- Индонезийские
            WHEN TO_CHAR(df.stat_date, 'MM-DD') IN ('01-01', '02-14', '05-01', '12-25') THEN 1  -- Международные
            ELSE 0
        END as is_holiday,
        
        -- Временные признаки
        EXTRACT(DOW FROM df.stat_date) as day_of_week,
        CASE WHEN EXTRACT(DOW FROM df.stat_date) IN (0, 6) THEN 1 ELSE 0 END as is_weekend,
        
        -- Лаги (простая реализация через window functions)
        LAG(df.total_sales, 1) OVER (PARTITION BY df.restaurant_id ORDER BY df.stat_date) as total_sales_lag_1,
        LAG(df.total_sales, 3) OVER (PARTITION BY df.restaurant_id ORDER BY df.stat_date) as total_sales_lag_3,
        LAG(df.total_sales, 7) OVER (PARTITION BY df.restaurant_id ORDER BY df.stat_date) as total_sales_lag_7,
        
        LAG(df.total_orders, 1) OVER (PARTITION BY df.restaurant_id ORDER BY df.stat_date) as orders_lag_1,
        LAG(df.total_orders, 3) OVER (PARTITION BY df.restaurant_id ORDER BY df.stat_date) as orders_lag_3,
        LAG(df.total_orders, 7) OVER (PARTITION BY df.restaurant_id ORDER BY df.stat_date) as orders_lag_7,
        
        -- Скользящие средние
        AVG(df.total_sales) OVER (
            PARTITION BY df.restaurant_id 
            ORDER BY df.stat_date 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) as total_sales_rolling_7d,
        
        AVG(df.total_orders) OVER (
            PARTITION BY df.restaurant_id 
            ORDER BY df.stat_date 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) as orders_rolling_7d
        
    FROM daily_facts df
    LEFT JOIN weather_cache wc ON wc.restaurant_id = df.restaurant_id AND wc.date = df.stat_date
    ORDER BY df.restaurant_id, df.stat_date;
    
    -- Индексы для ML dataset
    CREATE INDEX idx_ml_dataset_restaurant_date ON ml_dataset(restaurant_id, stat_date);
    CREATE INDEX idx_ml_dataset_date ON ml_dataset(stat_date);
    """
    
    try:
        with psycopg2.connect(DB_DSN) as conn:
            with conn.cursor() as cursor:
                logger.info("Creating ml_dataset materialized view...")
                cursor.execute(sql)
                conn.commit()
                logger.info("✅ ml_dataset view created successfully")
                return True
    except Exception as e:
        logger.error(f"❌ Error creating ml_dataset view: {e}")
        return False


def build_all_views():
    """Построение всех витрин данных"""
    
    logger.info("🏗️ Building data views...")
    
    success = True
    
    # 1. daily_facts
    if not create_daily_facts_view():
        success = False
    
    # 2. ml_dataset
    if not create_ml_dataset_view():
        success = False
    
    if success:
        logger.info("✅ All data views built successfully")
        
        # Статистика
        try:
            with psycopg2.connect(DB_DSN) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    
                    # Статистика daily_facts
                    cursor.execute("SELECT COUNT(*) as total_days, COUNT(DISTINCT restaurant_name) as restaurants FROM daily_facts")
                    daily_stats = cursor.fetchone()
                    
                    # Статистика ml_dataset
                    cursor.execute("SELECT COUNT(*) as total_rows FROM ml_dataset WHERE total_sales > 0")
                    ml_stats = cursor.fetchone()
                    
                    logger.info(f"📊 Data views statistics:")
                    logger.info(f"   daily_facts: {daily_stats['total_days']} days, {daily_stats['restaurants']} restaurants")
                    logger.info(f"   ml_dataset: {ml_stats['total_rows']} valid rows for ML")
                    
        except Exception as e:
            logger.warning(f"Error getting statistics: {e}")
    
    return success


def export_to_csv_for_ml(output_path: str = "/workspace/data/live_dataset.csv"):
    """Экспорт ml_dataset в CSV для совместимости с существующим ML кодом"""
    
    try:
        import pandas as pd
        
        with psycopg2.connect(DB_DSN) as conn:
            # Выбираем нужные колонки для ML
            query = """
                SELECT 
                    restaurant_id,
                    stat_date as date,
                    total_sales,
                    total_orders as orders_count,
                    temp,
                    rain,
                    wind, 
                    humidity,
                    is_holiday,
                    day_of_week,
                    is_weekend,
                    
                    -- Операционные метрики
                    gojek_prep_time as ops_preparation_time_gojek,
                    gojek_confirm_time as ops_accepting_time_gojek,
                    gojek_delivery_time as ops_delivery_time_gojek,
                    gojek_rating as ops_rating_gojek,
                    grab_rating as ops_rating_grab,
                    
                    -- Маркетинговые метрики
                    grab_ads_spend as mkt_ads_spend_grab,
                    gojek_ads_spend as mkt_ads_spend_gojek,
                    grab_ads_sales as mkt_ads_sales_grab,
                    gojek_ads_sales as mkt_ads_sales_gojek,
                    
                    -- Лаги
                    total_sales_lag_1,
                    total_sales_lag_3,
                    total_sales_lag_7,
                    orders_lag_1,
                    orders_lag_3,
                    orders_lag_7,
                    
                    -- Скользящие средние
                    total_sales_rolling_7d,
                    orders_rolling_7d
                    
                FROM ml_dataset
                WHERE total_sales > 0 AND total_orders > 0
                ORDER BY restaurant_id, stat_date
            """
            
            df = pd.read_sql_query(query, conn)
            
            # Создаем директорию если нужно
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Сохраняем CSV
            df.to_csv(output_path, index=False)
            
            logger.info(f"✅ ML dataset exported to {output_path}: {len(df)} rows")
            return True
            
    except Exception as e:
        logger.error(f"❌ Error exporting ML dataset: {e}")
        return False


def get_data_gaps_report(restaurant_name: str = None) -> Dict[str, Any]:
    """Отчет о пропусках в данных"""
    
    try:
        with psycopg2.connect(DB_DSN) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                
                where_clause = "WHERE restaurant_name = %s" if restaurant_name else ""
                params = (restaurant_name,) if restaurant_name else ()
                
                # Проверяем пропуски по дням
                cursor.execute(f"""
                    WITH date_range AS (
                        SELECT generate_series(
                            (SELECT MIN(stat_date) FROM daily_facts {where_clause}),
                            (SELECT MAX(stat_date) FROM daily_facts {where_clause}),
                            '1 day'::interval
                        )::date as expected_date
                    ),
                    missing_days AS (
                        SELECT 
                            dr.expected_date,
                            df.restaurant_name,
                            CASE WHEN df.stat_date IS NULL THEN 'missing_completely'
                                 WHEN df.sources_count < 2 THEN 'missing_platform'
                                 WHEN df.total_sales = 0 THEN 'zero_sales'
                                 ELSE 'ok'
                            END as issue_type
                        FROM date_range dr
                        CROSS JOIN (SELECT DISTINCT restaurant_name FROM daily_facts {where_clause}) r
                        LEFT JOIN daily_facts df ON dr.expected_date = df.stat_date AND r.restaurant_name = df.restaurant_name
                    )
                    SELECT 
                        restaurant_name,
                        issue_type,
                        COUNT(*) as days_count,
                        MIN(expected_date) as first_issue_date,
                        MAX(expected_date) as last_issue_date
                    FROM missing_days
                    WHERE issue_type != 'ok'
                    GROUP BY restaurant_name, issue_type
                    ORDER BY restaurant_name, issue_type
                """, params)
                
                gaps = [dict(row) for row in cursor.fetchall()]
                
                # Общая статистика
                cursor.execute(f"""
                    SELECT 
                        COUNT(DISTINCT restaurant_name) as restaurants,
                        COUNT(*) as total_days,
                        COUNT(CASE WHEN sources_count < 2 THEN 1 END) as incomplete_days,
                        COUNT(CASE WHEN total_sales = 0 THEN 1 END) as zero_sales_days,
                        MIN(stat_date) as first_date,
                        MAX(stat_date) as last_date
                    FROM daily_facts
                    {where_clause}
                """, params)
                
                summary = dict(cursor.fetchone())
                
                return {
                    "summary": summary,
                    "gaps": gaps,
                    "generated_at": datetime.now().isoformat()
                }
                
    except Exception as e:
        logger.error(f"Error generating data gaps report: {e}")
        return {"error": str(e)}


# CLI интерфейс
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build data views from raw_stats")
    parser.add_argument("--build-views", action='store_true', help="Build all materialized views")
    parser.add_argument("--export-csv", type=str, help="Export ML dataset to CSV file")
    parser.add_argument("--gaps-report", action='store_true', help="Generate data gaps report")
    parser.add_argument("--restaurant", type=str, help="Filter by restaurant name")
    
    args = parser.parse_args()
    
    if not DB_DSN:
        logger.error("DATABASE_URL not set")
        sys.exit(1)
    
    if args.build_views:
        success = build_all_views()
        sys.exit(0 if success else 1)
    
    elif args.export_csv:
        success = export_to_csv_for_ml(args.export_csv)
        sys.exit(0 if success else 1)
    
    elif args.gaps_report:
        import json
        report = get_data_gaps_report(args.restaurant)
        print(json.dumps(report, indent=2, default=str))
    
    else:
        parser.print_help()