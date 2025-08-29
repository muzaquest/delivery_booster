"""
Адаптер данных для работы с SQLite (legacy) и MySQL (prod)
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
        self.use_mysql = bool(self.db_url and "mysql" in self.db_url)
        # Use SQLAlchemy engine for both MySQL and SQLite
        self.engine = get_engine()
    
    def get_restaurants_list(self) -> pd.DataFrame:
        """Получение списка ресторанов"""
        
        if self.use_mysql:
            query = """
                SELECT restaurant_id as id, restaurant_name as name 
                FROM restaurant_mapping 
                WHERE is_active = 1 
                ORDER BY restaurant_name
            """
        else:
            query = "SELECT id, name FROM restaurants ORDER BY name"
        
        return pd.read_sql_query(query, self.engine)
    
    def get_restaurant_stats(self, restaurant_id: int, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Получение статистики ресторана за период"""
        
        if self.use_mysql:
            # Получаем название ресторана
            name_query = "SELECT restaurant_name FROM restaurant_mapping WHERE restaurant_id = :rid"
            name_df = pd.read_sql_query(name_query, self.engine, params={"rid": restaurant_id})
            restaurant_name = name_df.iloc[0][0] if not name_df.empty else None
            
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
                WHERE restaurant_name = :rname AND stat_date BETWEEN :start AND :end
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
                WHERE restaurant_name = :rname AND stat_date BETWEEN :start AND :end
                AND gojek_sales > 0
                ORDER BY stat_date
            """
            
            params = {"rname": restaurant_name, "start": start_date, "end": end_date}
            grab_df = pd.read_sql_query(grab_query, self.engine, params=params)
            gojek_df = pd.read_sql_query(gojek_query, self.engine, params=params)
            
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
        
        if self.use_mysql:
            query = """
                SELECT 
                    SUM(total_sales) as sales,
                    SUM(total_orders) as orders,
                    SUM(total_ads_spend) as ads_spend,
                    SUM(total_ads_sales) as ads_sales,
                    AVG(NULLIF(grab_rating,0) + NULLIF(gojek_rating,0)) as rating,
                    SUM(total_cancelled) as cancels
                FROM daily_facts
                WHERE stat_date BETWEEN :start AND :end
            """
            
            df = pd.read_sql_query(query, self.engine, params={"start": start_date, "end": end_date})
            if not df.empty:
                row = df.iloc[0].fillna(0)
                sales = float(row.get('sales', 0))
                orders = float(row.get('orders', 0))
                ads_spend = float(row.get('ads_spend', 0))
                ads_sales = float(row.get('ads_sales', 0))
                rating = float(row.get('rating', 0))
                cancels = float(row.get('cancels', 0))
                return {
                    'sales': sales,
                    'orders': orders,
                    'aov': float(sales / orders) if orders and orders > 0 else 0.0,
                    'ads_spend': ads_spend,
                    'ads_sales': ads_sales,
                    'roas': float(ads_sales / ads_spend) if ads_spend and ads_spend > 0 else 0.0,
                    'rating': rating,
                    'cancels': cancels,
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
        
        if self.use_mysql:
            # Получаем название ресторана
            name_query = "SELECT restaurant_name FROM restaurant_mapping WHERE restaurant_id = :rid"
            name_df = pd.read_sql_query(name_query, self.engine, params={"rid": restaurant_id})
            restaurant_name = name_df.iloc[0][0] if not name_df.empty else None
            
            if not restaurant_name:
                return pd.DataFrame()
            
            query = """
                SELECT * FROM ml_dataset
                WHERE restaurant_name = :rname AND stat_date BETWEEN :start AND :end
                ORDER BY stat_date
            """
            
            return pd.read_sql_query(query, self.engine, params={"rname": restaurant_name, "start": start_date, "end": end_date})
        else:
            # Пытаемся использовать существующий CSV
            try:
                project_root = os.getenv("PROJECT_ROOT", os.getcwd())
                csv_path = os.getenv("ML_DATASET_CSV", os.path.join(project_root, "data", "merged_dataset.csv"))
                df = pd.read_csv(csv_path, parse_dates=["date"])
                mask = (df["restaurant_id"] == restaurant_id) & \
                       (df["date"] >= start_date) & (df["date"] <= end_date)
                return df.loc[mask].copy()
            except:
                return pd.DataFrame()
    
    def get_data_status(self) -> Dict[str, Any]:
        """Получение статуса данных"""
        
        if self.use_mysql:
            try:
                df = pd.read_sql_query(
                    """
                        SELECT 
                            COUNT(DISTINCT restaurant_name) as restaurants,
                            COUNT(*) as total_days,
                            MIN(stat_date) as first_date,
                            MAX(stat_date) as last_date
                        FROM daily_facts
                    """,
                    self.engine,
                )
                if not df.empty:
                    row = df.iloc[0]
                    return {
                        "data_source": "MySQL (Live)",
                        "restaurants": int(row.get("restaurants", 0) or 0),
                        "total_days": int(row.get("total_days", 0) or 0),
                        "date_range": f"{row.get('first_date')} — {row.get('last_date')}",
                        "status": "live",
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