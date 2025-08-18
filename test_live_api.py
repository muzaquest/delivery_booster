"""
Тестирование подключения к живому API
Проверяет доступность API и качество данных
"""

import os
import sys
import json
from datetime import date, timedelta

sys.path.append('/workspace')

# Настройка переменных окружения для тестирования
os.environ.setdefault("STATS_API_BASE", "http://5.187.7.140:3000")

def test_api_connection():
    """Тестирование подключения к API"""
    
    print("🔌 ТЕСТИРОВАНИЕ ПОДКЛЮЧЕНИЯ К ЖИВОМУ API")
    print("=" * 60)
    
    try:
        from etl.api_client import fetch_restaurant_stats
        
        # Тестовые параметры
        restaurant_name = "Only Kebab"
        source = "grab"
        start_date = date.today() - timedelta(days=7)
        end_date = date.today() - timedelta(days=1)
        
        print(f"📊 Тестовый запрос:")
        print(f"   Ресторан: {restaurant_name}")
        print(f"   Источник: {source}")
        print(f"   Период: {start_date} — {end_date}")
        print()
        
        # Выполняем запрос
        result = fetch_restaurant_stats(restaurant_name, source, start_date, end_date)
        
        if result and 'data' in result:
            data = result['data']
            print(f"✅ API РАБОТАЕТ!")
            print(f"📊 Получено записей: {len(data)}")
            print(f"🏪 Ресторан: {result.get('restaurant_name')}")
            print(f"📱 Источник: {result.get('source')}")
            print(f"📅 Период: {result.get('period')}")
            print()
            
            if data:
                # Анализируем первую запись
                first_record = data[0]
                print("🔍 ПРИМЕР ДАННЫХ (первая запись):")
                print("-" * 40)
                
                key_fields = ['stat_date', 'sales', 'orders', 'ads_spend', 'ads_sales', 'rating']
                for field in key_fields:
                    if field in first_record:
                        print(f"   {field}: {first_record[field]}")
                
                print()
                print("📋 ВСЕ ДОСТУПНЫЕ ПОЛЯ:")
                print("-" * 30)
                for i, field in enumerate(sorted(first_record.keys()), 1):
                    print(f"   {i:2d}. {field}")
                
                return True
            else:
                print("⚠️ API работает, но данных нет")
                return False
        else:
            print("❌ API не вернул данные")
            return False
            
    except Exception as e:
        print(f"❌ ОШИБКА ПОДКЛЮЧЕНИЯ К API: {e}")
        return False


def test_multiple_restaurants():
    """Тестирование нескольких ресторанов"""
    
    print("\n🏪 ТЕСТИРОВАНИЕ НЕСКОЛЬКИХ РЕСТОРАНОВ")
    print("=" * 60)
    
    restaurants = [
        "Only Kebab",
        "Ika Canggu", 
        "Asai Cafe",
        "Rasgulai"
    ]
    
    sources = ['grab', 'gojek']
    test_date = date.today() - timedelta(days=3)
    
    results = {}
    
    for restaurant in restaurants:
        results[restaurant] = {}
        
        for source in sources:
            try:
                from etl.api_client import fetch_restaurant_stats
                
                result = fetch_restaurant_stats(restaurant, source, test_date, test_date)
                
                if result and 'data' in result and result['data']:
                    record_count = len(result['data'])
                    first_record = result['data'][0]
                    sales = first_record.get('sales', 0)
                    orders = first_record.get('orders', 0)
                    
                    results[restaurant][source] = {
                        'status': '✅',
                        'records': record_count,
                        'sales': sales,
                        'orders': orders
                    }
                    print(f"✅ {restaurant} ({source}): {record_count} записей, {sales:,} IDR, {orders} заказов")
                else:
                    results[restaurant][source] = {'status': '❌', 'error': 'No data'}
                    print(f"❌ {restaurant} ({source}): нет данных")
                    
            except Exception as e:
                results[restaurant][source] = {'status': '❌', 'error': str(e)}
                print(f"❌ {restaurant} ({source}): ошибка - {e}")
    
    return results


def test_database_setup():
    """Тестирование настройки БД"""
    
    print("\n🗄️ ТЕСТИРОВАНИЕ НАСТРОЙКИ БАЗЫ ДАННЫХ")
    print("=" * 60)
    
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("⚠️ DATABASE_URL не настроен, будет использоваться SQLite")
        return False
    
    try:
        import psycopg2
        
        # Тестируем подключение
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT version()")
                version = cursor.fetchone()[0]
                print(f"✅ PostgreSQL подключение успешно")
                print(f"📊 Версия: {version}")
                
                # Проверяем существование таблиц
                cursor.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name IN ('raw_stats', 'restaurant_mapping', 'ml_jobs')
                """)
                
                tables = [row[0] for row in cursor.fetchall()]
                
                if tables:
                    print(f"✅ Найдены таблицы: {', '.join(tables)}")
                else:
                    print("⚠️ Таблицы для живого API не созданы")
                    print("💡 Запустите: python db/migrate_to_live_api.py")
                
                return True
                
    except Exception as e:
        print(f"❌ Ошибка подключения к PostgreSQL: {e}")
        return False


def main():
    """Основная функция тестирования"""
    
    print("🧪 ТЕСТИРОВАНИЕ ИНТЕГРАЦИИ С ЖИВЫМ API")
    print("=" * 70)
    print()
    
    # 1. Тестируем БД
    db_ok = test_database_setup()
    
    # 2. Тестируем API
    api_ok = test_api_connection()
    
    # 3. Если API работает, тестируем несколько ресторанов
    if api_ok:
        restaurants_results = test_multiple_restaurants()
    
    print("\n📋 ИТОГОВЫЙ ОТЧЕТ")
    print("=" * 30)
    print(f"🗄️ База данных: {'✅ OK' if db_ok else '❌ Требует настройки'}")
    print(f"🔌 API подключение: {'✅ OK' if api_ok else '❌ Недоступен'}")
    
    if api_ok and db_ok:
        print()
        print("🚀 ГОТОВО К РАБОТЕ С ЖИВЫМИ ДАННЫМИ!")
        print()
        print("📋 Следующие шаги:")
        print("1. Настроить DATABASE_URL в переменных окружения")
        print("2. Запустить миграцию: python db/migrate_to_live_api.py")
        print("3. Синхронизировать данные: python etl/api_client.py 'Only Kebab'")
        print("4. Переобучить ML: python ml/training.py --from-db")
        print("5. Развернуть на Replit")
    else:
        print()
        print("⚠️ ТРЕБУЕТСЯ НАСТРОЙКА")
        if not db_ok:
            print("- Настроить PostgreSQL и DATABASE_URL")
        if not api_ok:
            print("- Проверить доступность API (5.187.7.140:3000)")


if __name__ == "__main__":
    main()