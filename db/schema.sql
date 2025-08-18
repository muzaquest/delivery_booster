-- Схема БД для работы с живыми данными API
-- Основано на рекомендациях ChatGPT с улучшениями

-- Рестораны (маппинг ID<->Name)
CREATE TABLE IF NOT EXISTS restaurant_mapping (
    restaurant_id SERIAL PRIMARY KEY,
    restaurant_name TEXT UNIQUE NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT now()
);

-- Сырые данные по дням из API
CREATE TABLE IF NOT EXISTS raw_stats (
    restaurant_name TEXT NOT NULL,
    source TEXT NOT NULL CHECK (source IN ('grab','gojek')),
    stat_date DATE NOT NULL,
    payload JSONB NOT NULL,
    row_hash TEXT NOT NULL,
    
    -- Денормализованные поля для быстрых запросов
    sales_idr BIGINT,
    orders_total INT,
    ads_spend_idr BIGINT,
    ads_sales_idr BIGINT,
    cancelled_orders INT,
    lost_orders INT,
    rating_avg REAL,
    
    -- Операционные метрики (в минутах)
    prep_time_min REAL,
    confirm_time_min REAL,
    delivery_time_min REAL,
    offline_time_min REAL,
    
    -- Метаданные
    updated_at TIMESTAMP DEFAULT now(),
    
    PRIMARY KEY (restaurant_name, source, stat_date)
);

-- Индексы для быстрых запросов
CREATE INDEX IF NOT EXISTS idx_raw_stats_date ON raw_stats(stat_date);
CREATE INDEX IF NOT EXISTS idx_raw_stats_restaurant ON raw_stats(restaurant_name);
CREATE INDEX IF NOT EXISTS idx_raw_stats_source_date ON raw_stats(source, stat_date);
CREATE INDEX IF NOT EXISTS idx_raw_stats_updated ON raw_stats(updated_at);

-- Материализованная витрина для ML и отчетов
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_daily_metrics AS
SELECT
    rm.restaurant_id,
    rs.restaurant_name,
    rs.source,
    rs.stat_date,
    
    -- Основные метрики
    COALESCE(rs.sales_idr, (rs.payload->>'sales')::BIGINT, 0) AS sales,
    COALESCE(rs.orders_total, (rs.payload->>'orders')::INT, 0) AS orders,
    COALESCE(rs.ads_spend_idr, (rs.payload->>'ads_spend')::BIGINT, 0) AS ads_spend,
    COALESCE(rs.ads_sales_idr, (rs.payload->>'ads_sales')::BIGINT, 0) AS ads_sales,
    
    -- Операционные метрики
    COALESCE(rs.cancelled_orders, (rs.payload->>'cancelled_orders')::INT, 0) AS cancelled_orders,
    COALESCE(rs.lost_orders, (rs.payload->>'lost_orders')::INT, 0) AS lost_orders,
    COALESCE(rs.rating_avg, (rs.payload->>'rating')::REAL, 0) AS rating,
    
    -- Временные метрики (приводим к минутам)
    COALESCE(rs.prep_time_min, 
        CASE 
            WHEN rs.payload->>'preparation_time' ~ '^[0-9]+:[0-9]+:[0-9]+$' THEN
                EXTRACT(HOUR FROM (rs.payload->>'preparation_time')::TIME) * 60 + 
                EXTRACT(MINUTE FROM (rs.payload->>'preparation_time')::TIME)
            ELSE (rs.payload->>'preparation_time')::REAL
        END
    ) AS preparation_time_min,
    
    COALESCE(rs.delivery_time_min,
        CASE 
            WHEN rs.payload->>'delivery_time' ~ '^[0-9]+:[0-9]+:[0-9]+$' THEN
                EXTRACT(HOUR FROM (rs.payload->>'delivery_time')::TIME) * 60 + 
                EXTRACT(MINUTE FROM (rs.payload->>'delivery_time')::TIME)
            ELSE (rs.payload->>'delivery_time')::REAL
        END
    ) AS delivery_time_min,
    
    COALESCE(rs.offline_time_min, (rs.payload->>'offline_rate')::REAL, 0) AS offline_rate_min,
    
    -- Полный payload для расширенного анализа
    rs.payload,
    rs.updated_at
    
FROM raw_stats rs
LEFT JOIN restaurant_mapping rm ON rs.restaurant_name = rm.restaurant_name
WHERE rm.is_active IS TRUE OR rm.is_active IS NULL;

-- Индексы для витрины
CREATE INDEX IF NOT EXISTS idx_mv_daily_restaurant_date ON mv_daily_metrics(restaurant_id, stat_date);
CREATE INDEX IF NOT EXISTS idx_mv_daily_source_date ON mv_daily_metrics(source, stat_date);

-- Очередь ML заданий
CREATE TABLE IF NOT EXISTS ml_jobs (
    id SERIAL PRIMARY KEY,
    job_type TEXT NOT NULL,  -- 'retrain' | 'refresh_shap' | 'build_dataset'
    status TEXT NOT NULL DEFAULT 'pending',  -- 'pending' | 'running' | 'done' | 'failed'
    restaurant_name TEXT,
    payload JSONB,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT now(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_ml_jobs_status ON ml_jobs(status, created_at);

-- Функция для автоматического обновления витрины
CREATE OR REPLACE FUNCTION refresh_daily_metrics() 
RETURNS VOID AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_daily_metrics;
END;
$$ LANGUAGE plpgsql;

-- Триггер для автообновления витрины при изменении raw_stats
CREATE OR REPLACE FUNCTION trigger_refresh_metrics()
RETURNS TRIGGER AS $$
BEGIN
    -- Добавляем задачу на обновление витрины (не сразу, чтобы не блокировать)
    INSERT INTO ml_jobs(job_type, payload) 
    VALUES ('refresh_metrics', json_build_object('trigger_time', now()));
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Создаем триггер (опционально, для автообновления)
DROP TRIGGER IF EXISTS trigger_raw_stats_refresh ON raw_stats;
CREATE TRIGGER trigger_raw_stats_refresh
    AFTER INSERT OR UPDATE ON raw_stats
    FOR EACH STATEMENT
    EXECUTE FUNCTION trigger_refresh_metrics();

-- Вью для проверки качества данных (data gaps)
CREATE OR REPLACE VIEW data_quality_check AS
SELECT 
    restaurant_name,
    stat_date,
    CASE WHEN grab_sales IS NULL THEN 'missing_grab' ELSE NULL END as grab_issue,
    CASE WHEN gojek_sales IS NULL THEN 'missing_gojek' ELSE NULL END as gojek_issue,
    CASE WHEN grab_sales = 0 AND gojek_sales = 0 THEN 'zero_sales' ELSE NULL END as sales_issue
FROM (
    SELECT 
        restaurant_name,
        stat_date,
        SUM(CASE WHEN source = 'grab' THEN sales END) as grab_sales,
        SUM(CASE WHEN source = 'gojek' THEN sales END) as gojek_sales
    FROM mv_daily_metrics 
    GROUP BY restaurant_name, stat_date
) t
WHERE grab_issue IS NOT NULL OR gojek_issue IS NOT NULL OR sales_issue IS NOT NULL;