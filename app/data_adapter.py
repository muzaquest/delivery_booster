"""
Адаптер данных для работы с SQLite (legacy) и PostgreSQL (live API)
Обеспечивает единый интерфейс для всех отчетов
"""

import os
import pandas as pd
from typing import Optional, Dict, Any, List
from datetime import datetime, date

from etl.data_loader import get_engine


class DataAdapter:
    """Универсальный адаптер для работы с данными"""
    
    def __init__(self):
        self.db_url = os.getenv("DATABASE_URL")
        self.use_postgres = bool(self.db_url and "postgresql" in self.db_url)
        
        if self.use_postgres:
            import psycopg2
            self.engine = psycopg2.connect(self.db_url)
        else:
            self.engine = get_engine()  # SQLite
    
    def get_restaurants_list(self) -> pd.DataFrame:
        """Получение списка ресторанов"""
        
        if self.use_postgres:
            query = """
                SELECT restaurant_id as id, restaurant_name as name 
                FROM restaurant_mapping 
                WHERE is_active = true 
                ORDER BY restaurant_name
            """
        else:
            query = "SELECT id, name FROM restaurants ORDER BY name"
        
        return pd.read_sql_query(query, self.engine)
    
    def get_restaurant_stats(self, restaurant_id: int, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Получение статистики ресторана за период"""
        
        if self.use_postgres:
            # Получаем название ресторана
            name_query = "SELECT restaurant_name FROM restaurant_mapping WHERE restaurant_id = %s"
            with self.engine.cursor() as cursor:
                cursor.execute(name_query, (restaurant_id,))
                result = cursor.fetchone()
                restaurant_name = result[0] if result else None
            
            if not restaurant_name:
                return {"grab": pd.DataFrame(), "gojek": pd.DataFrame()}
            
            # Данные из витрины
            grab_query = """
                SELECT 
                    stat_date,
                    grab_sales as sales,
                    grab_orders as orders,
                    grab_ads_spend as ads_spend,
                    grab_ads_sales as ads_sales,
                    grab_cancelled as cancelled_orders,
                    grab_rating as rating,
                    grab_offline_min as offline_rate
                FROM daily_facts
                WHERE restaurant_name = %s AND stat_date BETWEEN %s AND %s
                AND grab_sales > 0
                ORDER BY stat_date
            """
            
            gojek_query = """
                SELECT 
                    stat_date,
                    gojek_sales as sales,
                    gojek_orders as orders,
                    gojek_ads_spend as ads_spend,
                    gojek_ads_sales as ads_sales,
                    gojek_cancelled as cancelled_orders,
                    gojek_lost as lost_orders,
                    gojek_rating as rating,
                    gojek_prep_time as preparation_time,
                    gojek_confirm_time as accepting_time,
                    gojek_delivery_time as delivery_time
                FROM daily_facts
                WHERE restaurant_name = %s AND stat_date BETWEEN %s AND %s
                AND gojek_sales > 0
                ORDER BY stat_date
            """
            
            grab_df = pd.read_sql_query(grab_query, self.engine, params=(restaurant_name, start_date, end_date))
            gojek_df = pd.read_sql_query(gojek_query, self.engine, params=(restaurant_name, start_date, end_date))
            
        else:
            # Старые запросы к SQLite
            grab_query = """
                SELECT stat_date, sales, orders, ads_spend, ads_sales, cancelled_orders, 
                       rating, offline_rate
                FROM grab_stats 
                WHERE restaurant_id = ? AND stat_date BETWEEN ? AND ?
                ORDER BY stat_date
            """
            
            gojek_query = """
                SELECT stat_date, sales, orders, ads_spend, ads_sales, cancelled_orders,
                       lost_orders, rating, preparation_time, accepting_time, delivery_time
                FROM gojek_stats 
                WHERE restaurant_id = ? AND stat_date BETWEEN ? AND ?
                ORDER BY stat_date
            """
            
            grab_df = pd.read_sql_query(grab_query, self.engine, params=(restaurant_id, start_date, end_date))
            gojek_df = pd.read_sql_query(gojek_query, self.engine, params=(restaurant_id, start_date, end_date))
        
        return {"grab": grab_df, "gojek": gojek_df}
    
    def get_kpi_data(self, start_date: str, end_date: str) -> Dict[str, float]:
        """Получение KPI данных для панели"""
        
        if self.use_postgres:
            query = """
                SELECT 
                    SUM(total_sales) as sales,
                    SUM(total_orders) as orders,
                    SUM(total_ads_spend) as ads_spend,
                    SUM(total_ads_sales) as ads_sales,
                    AVG(CASE WHEN grab_rating > 0 AND gojek_rating > 0 
                        THEN (grab_rating + gojek_rating) / 2
                        WHEN grab_rating > 0 THEN grab_rating
                        WHEN gojek_rating > 0 THEN gojek_rating
                        ELSE NULL END) as rating,
                    SUM(total_cancelled) as cancels
                FROM daily_facts
                WHERE stat_date BETWEEN %s AND %s
            """
            
            with self.engine.cursor() as cursor:
                cursor.execute(query, (start_date, end_date))
                result = cursor.fetchone()
                
                if result:
                    sales, orders, ads_spend, ads_sales, rating, cancels = result
                    return {
                        'sales': float(sales or 0),
                        'orders': float(orders or 0),
                        'aov': float(sales / orders) if orders and orders > 0 else 0.0,
                        'ads_spend': float(ads_spend or 0),
                        'ads_sales': float(ads_sales or 0),
                        'roas': float(ads_sales / ads_spend) if ads_spend and ads_spend > 0 else 0.0,
                        'rating': float(rating or 0),
                        'cancels': float(cancels or 0),
                        'mer': float(sales / ads_spend) if ads_spend and ads_spend > 0 else 0.0,
                    }
        else:
            # Старый способ через SQLite
            return self._get_kpi_sqlite(start_date, end_date)
        
        return {}
    
    def _get_kpi_sqlite(self, start_date: str, end_date: str) -> Dict[str, float]:
        """KPI из SQLite (старый способ)"""
        
        grab_query = """
            SELECT SUM(sales) sales, SUM(orders) orders, SUM(ads_spend) ads_spend, 
                   SUM(ads_sales) ads_sales, AVG(rating) rating, SUM(cancelled_orders) canc 
            FROM grab_stats WHERE stat_date BETWEEN ? AND ?
        """
        
        gojek_query = """
            SELECT SUM(sales) sales, SUM(orders) orders, SUM(ads_spend) ads_spend, 
                   SUM(ads_sales) ads_sales, AVG(rating) rating, SUM(cancelled_orders) canc 
            FROM gojek_stats WHERE stat_date BETWEEN ? AND ?
        """
        
        g = pd.read_sql_query(grab_query, self.engine, params=(start_date, end_date)).iloc[0].fillna(0)
        j = pd.read_sql_query(gojek_query, self.engine, params=(start_date, end_date)).iloc[0].fillna(0)
        
        sales = float(g['sales'] + j['sales'])
        orders = float((g['orders'] or 0) + (j['orders'] or 0))
        ads_spend = float(g['ads_spend'] + j['ads_spend'])
        ads_sales = float(g['ads_sales'] + j['ads_sales'])
        rating = float(((g['rating'] or 0) + (j['rating'] or 0)) / (2 if ((g['rating'] or 0) and (j['rating'] or 0)) else 1) or 0)
        canc = float((g['canc'] or 0) + (j['canc'] or 0))
        
        return {
            'sales': sales,
            'orders': orders,
            'aov': (sales / orders) if orders else 0.0,
            'ads_spend': ads_spend,
            'ads_sales': ads_sales,
            'roas': (ads_sales / ads_spend) if ads_spend else 0.0,
            'rating': rating,
            'cancels': canc,
            'mer': (sales / ads_spend) if ads_spend else 0.0,
        }
    
    def get_ml_dataset(self, restaurant_id: int, start_date: str, end_date: str) -> pd.DataFrame:
        """Получение данных для ML анализа"""
        
        if self.use_postgres:
            # Получаем название ресторана
            name_query = "SELECT restaurant_name FROM restaurant_mapping WHERE restaurant_id = %s"
            with self.engine.cursor() as cursor:
                cursor.execute(name_query, (restaurant_id,))
                result = cursor.fetchone()
                restaurant_name = result[0] if result else None
            
            if not restaurant_name:
                return pd.DataFrame()
            
            query = """
                SELECT * FROM ml_dataset
                WHERE restaurant_name = %s AND stat_date BETWEEN %s AND %s
                ORDER BY stat_date
            """
            
            return pd.read_sql_query(query, self.engine, params=(restaurant_name, start_date, end_date))
        else:
            # Пытаемся использовать существующий CSV
            try:
                import os
                dataset_path = os.getenv("ML_DATASET_CSV", os.path.join(os.getenv("PROJECT_ROOT", os.getcwd()), "data", "merged_dataset.csv"))
                df = pd.read_csv(dataset_path, parse_dates=["date"])
                mask = (df["restaurant_id"] == restaurant_id) & \
                       (df["date"] >= start_date) & (df["date"] <= end_date)
                return df.loc[mask].copy()
            except:
                return pd.DataFrame()
    
    def get_data_status(self) -> Dict[str, Any]:
        """Получение статуса данных"""
        
        if self.use_postgres:
            try:
                with self.engine.cursor() as cursor:
                    # Статистика по данным
                    cursor.execute("""
                        SELECT 
                            COUNT(DISTINCT restaurant_name) as restaurants,
                            COUNT(*) as total_days,
                            MIN(stat_date) as first_date,
                            MAX(stat_date) as last_date,
                            MAX(last_updated) as last_sync
                        FROM daily_facts
                    """)
                    
                    result = cursor.fetchone()
                    
                    if result:
                        restaurants, total_days, first_date, last_date, last_sync = result
                        return {
                            "data_source": "PostgreSQL (Live API)",
                            "restaurants": restaurants,
                            "total_days": total_days,
                            "date_range": f"{first_date} — {last_date}",
                            "last_sync": last_sync,
                            "status": "live"
                        }
            except:
                pass
        
        # Fallback к SQLite
        try:
            query = """
                SELECT 
                    COUNT(DISTINCT restaurant_id) as restaurants,
                    COUNT(*) as grab_records
                FROM grab_stats
            """
            result = pd.read_sql_query(query, self.engine)
            
            if not result.empty:
                return {
                    "data_source": "SQLite (Static)",
                    "restaurants": int(result.iloc[0]['restaurants']),
                    "grab_records": int(result.iloc[0]['grab_records']),
                    "status": "static"
                }
        except:
            pass
        
        return {
            "data_source": "Unknown",
            "status": "error"
        }


# Глобальный экземпляр адаптера
_data_adapter = None

def get_data_adapter() -> DataAdapter:
    """Получение глобального адаптера данных"""
    global _data_adapter
    if _data_adapter is None:
        _data_adapter = DataAdapter()
    return _data_adapter