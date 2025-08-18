"""Улучшенный раздел 8 критических дней"""

import pandas as pd
from etl.data_loader import get_engine


def section8_critical_days_improved(period: str, restaurant_id: int) -> str:
    """Новый улучшенный раздел критических дней"""
    
    try:
        start_str, end_str = period.split("_")
        eng = get_engine()
        
        # Получаем данные за период
        query = """
            SELECT stat_date, 
                   SUM(sales) as total_sales,
                   SUM(orders) as total_orders,
                   SUM(ads_spend) as total_ads_spend
            FROM (
                SELECT stat_date, sales, orders, ads_spend FROM grab_stats WHERE restaurant_id = ?
                UNION ALL
                SELECT stat_date, sales, orders, ads_spend FROM gojek_stats WHERE restaurant_id = ?
            )
            WHERE stat_date BETWEEN ? AND ?
            GROUP BY stat_date
            ORDER BY total_sales ASC
        """
        
        df = pd.read_sql_query(query, eng, params=(restaurant_id, restaurant_id, start_str, end_str))
        
        if df.empty:
            return "8. 🚨 КРИТИЧЕСКИЕ ДНИ\n" + ("═" * 80) + "\n📊 Нет данных за выбранный период."
        
        # Находим критические дни (падение ≥25% от медианы)
        median_sales = df['total_sales'].median()
        threshold = median_sales * 0.75
        critical_days = df[df['total_sales'] <= threshold]
        
        lines = []
        lines.append("8. 🚨 КРИТИЧЕСКИЕ ДНИ")
        lines.append("═" * 80)
        
        if critical_days.empty:
            lines.append("✅ В периоде нет критических провалов продаж (падение >25% от медианы)")
            lines.append(f"📊 Медианные продажи: {median_sales:,.0f} IDR")
            return "\n".join(lines)
        
        # Анализируем каждый критический день (максимум 5)
        for _, day in critical_days.head(5).iterrows():
            lines.append("")
            lines.append(f"🔴 {day['stat_date']}")
            lines.append("")
            
            loss_pct = ((day['total_sales'] - median_sales) / median_sales * 100)
            loss_amount = max(median_sales - day['total_sales'], 0)
            avg_check = day['total_sales'] / day['total_orders'] if day['total_orders'] > 0 else 0
            
            lines.append("### 📊 **КЛЮЧЕВЫЕ ЦИФРЫ**")
            lines.append(f"- **Продажи:** {day['total_sales']:,.0f} IDR (медиана: {median_sales:,.0f} IDR) → **{loss_pct:+.1f}%**")
            lines.append(f"- **Потери:** {loss_amount:,.0f} IDR")
            lines.append(f"- **Заказы:** {day['total_orders']} шт")
            lines.append(f"- **Средний чек:** {avg_check:,.0f} IDR")
            lines.append("")
            
            # Анализ причин
            lines.append("### 🔍 **РЕАЛЬНЫЕ ПРИЧИНЫ**")
            lines.append("")
            
            # Рекламный бюджет
            avg_ads = df['total_ads_spend'].mean()
            if day['total_ads_spend'] < avg_ads * 0.7:
                drop_pct = ((avg_ads - day['total_ads_spend']) / avg_ads * 100)
                lines.append(f"**1. 🔴 УРЕЗАНИЕ РЕКЛАМНОГО БЮДЖЕТА**")
                lines.append(f"- {day['total_ads_spend']:,.0f} IDR против нормы {avg_ads:,.0f} IDR (-{drop_pct:.0f}%)")
                lines.append(f"- **Влияние:** ~{(avg_ads - day['total_ads_spend']) * 12:,.0f} IDR потерь")
                lines.append("")
            
            # Средний чек
            avg_check_period = (df['total_sales'] / df['total_orders']).mean()
            if avg_check < avg_check_period * 0.8:
                check_drop = ((avg_check_period - avg_check) / avg_check_period * 100)
                lines.append(f"**2. 🔴 ПАДЕНИЕ СРЕДНЕГО ЧЕКА**")
                lines.append(f"- {avg_check:,.0f} IDR против нормы {avg_check_period:,.0f} IDR (-{check_drop:.0f}%)")
                lines.append(f"- **Влияние:** {(avg_check_period - avg_check) * day['total_orders']:,.0f} IDR потерь")
                lines.append("")
            
            # Внешние факторы
            lines.append("### 🌍 **ВНЕШНИЕ ФАКТОРЫ**")
            lines.append("")
            
            holiday_info = check_holiday_by_date(day['stat_date'])
            lines.append(f"**🕌 Праздники:** {holiday_info}")
            lines.append(f"**🌧️ Погода:** требует проверки метеоданных")
            lines.append("")
            
            # Рекомендации
            lines.append("### 🎯 **РЕКОМЕНДАЦИИ**")
            lines.append("")
            
            if day['total_ads_spend'] < avg_ads * 0.7:
                potential = min(loss_amount * 0.6, 3000000)
                lines.append(f"**1. 🔴 Восстановить рекламный бюджет до {avg_ads:,.0f} IDR/день**")
                lines.append(f"- **Потенциальный эффект:** {potential:,.0f} IDR")
                lines.append("")
            
            if avg_check < avg_check_period * 0.8:
                potential = min(loss_amount * 0.4, 2000000)
                lines.append(f"**2. 🟡 Запустить upsell: промо при заказе >400K IDR**")
                lines.append(f"- **Потенциальный эффект:** {potential:,.0f} IDR")
                lines.append("")
            
            # Финансовый итог
            total_potential = min(loss_amount * 0.8, 4000000)
            recovery_pct = (total_potential / loss_amount * 100) if loss_amount > 0 else 0
            
            lines.append("### 💰 **ФИНАНСОВЫЙ ИТОГ**")
            lines.append(f"- **Потенциал восстановления:** {total_potential:,.0f} IDR ({recovery_pct:.0f}% от потерь)")
        
        return "\n".join(lines)
        
    except Exception as e:
        return f"8. 🚨 КРИТИЧЕСКИЕ ДНИ\n{'═' * 80}\n❌ Ошибка анализа: {str(e)}"


def check_holiday_by_date(date_str):
    """Проверка праздников по дате"""
    try:
        from datetime import datetime
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        month_day = date_obj.strftime('%m-%d')
        
        holidays = {
            '04-01': 'Eid al-Fitr (окончание Рамадана) — крупнейший мусульманский праздник, курьеры отдыхают',
            '06-07': 'Eid al-Adha (Курбан-байрам) — мусульманский праздник жертвоприношения',
            '03-31': 'Nyepi (День тишины) — балийский новый год, остров закрыт',
            '05-29': 'Galungan — балийский праздник, снижение активности на 20-30%',
            '06-08': 'Kuningan — балийский праздник',
            '01-01': 'Новый год — обычно УВЕЛИЧЕНИЕ заказов на 15-25%',
            '12-25': 'Рождество — смешанное влияние',
            '08-17': 'День независимости Индонезии — национальный праздник',
        }
        
        return holidays.get(month_day, 'обычный день, не праздник')
    except:
        return 'обычный день'
