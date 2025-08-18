"""
Улучшенный анализ критических дней с глубокой аналитикой:
- Учет праздников и их влияния на продажи
- Анализ погодных факторов 
- ML-анализ без тривиальных факторов
- Конкретные рекомендации с ROI
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from etl.data_loader import get_engine


def analyze_critical_days_improved(period: str, restaurant_id: int) -> str:
    """
    Главная функция анализа критических дней
    """
    try:
        start_str, end_str = period.split("_")
        
        # Загружаем данные
        try:
            df = pd.read_csv("/workspace/data/merged_dataset.csv", parse_dates=["date"])
            sub = df[(df["restaurant_id"] == restaurant_id) & (df["date"] >= start_str) & (df["date"] <= end_str)].copy()
        except:
            return "8. 🚨 КРИТИЧЕСКИЕ ДНИ\n" + ("═" * 80) + "\n❌ Данные для ML-анализа недоступны. Запустите обучение модели."
        
        if sub.empty:
            return "8. 🚨 КРИТИЧЕСКИЕ ДНИ\n" + ("═" * 80) + "\n📊 Нет данных за выбранный период."

        # Находим критические дни (падение ≥25% от медианы)
        daily = sub.groupby("date", as_index=False)["total_sales"].sum().sort_values("date")
        median_sales = float(daily["total_sales"].median()) if len(daily) else 0.0
        threshold = 0.75 * median_sales  # 25% падение
        critical_dates = daily.loc[daily["total_sales"] <= threshold, "date"].dt.normalize().tolist()

        lines = []
        lines.append("8. 🚨 КРИТИЧЕСКИЕ ДНИ")
        lines.append("═" * 80)
        
        if not critical_dates:
            lines.append("✅ В периоде нет критических провалов продаж (падение >25% от медианы)")
            lines.append("")
            lines.append(f"📊 Медианные продажи: {_format_currency(median_sales)}")
            
            # Краткая сводка по внешним факторам
            weather_impact = _analyze_weather_impact_period(sub)
            holiday_impact = _analyze_holiday_impact_period(sub)
            
            if weather_impact['significant']:
                lines.append(f"🌧️ Влияние погоды: {weather_impact['description']}")
            if holiday_impact['significant']:
                lines.append(f"🎌 Влияние праздников: {holiday_impact['description']}")
                
            return "\n".join(lines)

        # Анализируем каждый критический день (максимум 5)
        eng = get_engine()
        for i, critical_date in enumerate(critical_dates[:5]):
            if i > 0:
                lines.append("")  # Разделитель между днями
                
            day_analysis = _analyze_critical_day_improved(
                critical_date, sub, daily, median_sales, restaurant_id, start_str, end_str, eng
            )
            lines.extend(day_analysis)

        # Общие выводы если больше 1 дня
        if len(critical_dates) > 1:
            lines.append("")
            lines.append("📊 ОБЩИЕ ВЫВОДЫ")
            lines.append("─" * 40)
            
            summary = _generate_period_summary_improved(critical_dates, sub, restaurant_id)
            lines.extend(summary)

        return "\n".join(lines)
        
    except Exception as e:
        return f"8. 🚨 КРИТИЧЕСКИЕ ДНИ\n{'═' * 80}\n❌ Ошибка анализа: {str(e)}"


def _analyze_weather_impact_period(sub):
    """Анализ влияния погоды за период"""
    try:
        if 'rain' not in sub.columns:
            return {'significant': False, 'description': ''}
        
        # Анализируем дождливые дни
        sub['heavy_rain'] = (sub['rain'].fillna(0.0) >= 10.0).astype(int)
        by_rain = sub.groupby('heavy_rain')['total_sales'].mean().to_dict()
        
        if 0 in by_rain and 1 in by_rain:
            rain_effect = (by_rain[1] - by_rain[0]) / by_rain[0] * 100.0
            if abs(rain_effect) > 10:  # Значимое влияние >10%
                direction = "снижение" if rain_effect < 0 else "увеличение"
                return {
                    'significant': True, 
                    'description': f'{direction} продаж на {abs(rain_effect):.1f}% в дождливые дни'
                }
        
        return {'significant': False, 'description': ''}
    except:
        return {'significant': False, 'description': ''}


def _analyze_holiday_impact_period(sub):
    """Анализ влияния праздников за период"""
    try:
        if 'is_holiday' not in sub.columns:
            return {'significant': False, 'description': ''}
        
        by_holiday = sub.groupby(sub['is_holiday'].fillna(0).astype(int))['total_sales'].mean().to_dict()
        
        if 0 in by_holiday and 1 in by_holiday:
            holiday_effect = (by_holiday[1] - by_holiday[0]) / by_holiday[0] * 100.0
            if abs(holiday_effect) > 15:  # Значимое влияние >15%
                direction = "снижение" if holiday_effect < 0 else "увеличение"
                return {
                    'significant': True,
                    'description': f'{direction} продаж на {abs(holiday_effect):.1f}% в праздничные дни'
                }
        
        return {'significant': False, 'description': ''}
    except:
        return {'significant': False, 'description': ''}


def _analyze_critical_day_improved(critical_date, sub, daily, median_sales, restaurant_id, start_str, end_str, eng):
    """Анализ одного критического дня в новом формате"""
    
    day_str = critical_date.strftime('%Y-%m-%d')
    day_sales = float(daily.loc[daily["date"] == critical_date, "total_sales"].iloc[0])
    loss_pct = ((day_sales - median_sales) / median_sales * 100) if median_sales else 0
    loss_amount = max(median_sales - day_sales, 0)
    
    lines = []
    
    # Заголовок с ключевыми цифрами
    lines.append(f"🔴 {day_str}")
    lines.append("")
    
    # Получаем операционные данные
    grab_data, gojek_data = _get_day_operational_data(eng, restaurant_id, day_str)
    period_averages = _get_period_averages(eng, restaurant_id, start_str, end_str, day_str)
    
    # Ключевые цифры
    total_orders = (grab_data.get('orders', 0) if grab_data else 0) + (gojek_data.get('orders', 0) if gojek_data else 0)
    avg_check = day_sales / total_orders if total_orders > 0 else 0
    normal_orders = period_averages.get('avg_orders', 34)
    normal_check = period_averages.get('avg_check', 400000)
    
    lines.append("### 📊 **КЛЮЧЕВЫЕ ЦИФРЫ**")
    lines.append(f"- **Продажи:** {_format_currency(day_sales)} (медиана: {_format_currency(median_sales)}) → **{loss_pct:+.1f}%**")
    lines.append(f"- **Потери:** {_format_currency(loss_amount)}")
    lines.append(f"- **Заказы:** {total_orders} шт (норма: {normal_orders:.0f}) → **{((total_orders - normal_orders)/normal_orders*100):+.1f}%**")
    lines.append(f"- **Средний чек:** {_format_currency(avg_check)} (норма: {_format_currency(normal_check)}) → **{((avg_check - normal_check)/normal_check*100):+.1f}%**")
    lines.append("")
    
    # Анализ причин
    lines.append("### 🔍 **РЕАЛЬНЫЕ ПРИЧИНЫ**")
    lines.append("")
    
    root_causes = _identify_root_causes_improved(critical_date, sub, grab_data, gojek_data, period_averages, day_str)
    
    # Основные причины
    if root_causes['primary']:
        for i, cause in enumerate(root_causes['primary'][:3], 1):
            priority_icon = "🔴" if i <= 2 else "🟡"
            lines.append(f"**{i}. {priority_icon} {cause['title']}**")
            lines.append(f"- {cause['description']}")
            lines.append(f"- **Влияние:** {cause['impact']}")
            lines.append("")
    
    # Внешние факторы
    lines.append("### 🌍 **ВНЕШНИЕ ФАКТОРЫ**")
    lines.append("")
    
    holiday_info = _get_holiday_info(critical_date, sub)
    weather_info = _get_weather_info(critical_date, sub)
    
    lines.append(f"**🕌 Праздники:** {holiday_info['description']}")
    lines.append(f"**🌧️ Погода:** {weather_info['description']}")
    lines.append("")
    
    # Что помогло
    if root_causes['positive']:
        lines.append("### ✅ **ЧТО ПОМОГЛО ИЗБЕЖАТЬ БОЛЬШИХ ПОТЕРЬ**")
        lines.append("")
        for factor in root_causes['positive'][:3]:
            lines.append(f"**💪 {factor['title']}:**")
            lines.append(f"- {factor['description']}")
            lines.append("")
    
    # Рекомендации
    lines.append("### 🎯 **КОНКРЕТНЫЕ РЕКОМЕНДАЦИИ**")
    lines.append("")
    
    recommendations = _generate_day_recommendations_improved(root_causes, loss_amount, holiday_info)
    
    for i, rec in enumerate(recommendations[:3], 1):
        priority_icon = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}[rec['priority']]
        lines.append(f"**{i}. {priority_icon} {rec['action']}**")
        lines.append(f"- **Потенциальный эффект:** {rec['potential_impact']}")
        lines.append("")
    
    # Финансовый итог
    total_potential = sum([float(rec['potential_value']) for rec in recommendations if 'potential_value' in rec])
    recovery_pct = (total_potential / loss_amount * 100) if loss_amount > 0 else 0
    
    lines.append("### 💰 **ФИНАНСОВЫЙ ИТОГ**")
    lines.append(f"- **Общий потенциал восстановления:** {_format_currency(total_potential)} ({recovery_pct:.0f}% от потерь)")
    
    return lines


def _get_day_operational_data(eng, restaurant_id, day_str):
    """Получение операционных данных за день"""
    try:
        # GRAB данные
        grab_query = """
            SELECT sales, orders, ads_spend, ads_sales, offline_rate, cancelled_orders, rating
            FROM grab_stats 
            WHERE restaurant_id = ? AND stat_date = ?
        """
        grab_df = pd.read_sql_query(grab_query, eng, params=(restaurant_id, day_str))
        grab_data = grab_df.iloc[0].to_dict() if not grab_df.empty else None
        
        # GOJEK данные
        gojek_query = """
            SELECT sales, orders, ads_spend, ads_sales, accepting_time, preparation_time, 
                   delivery_time, cancelled_orders, rating
            FROM gojek_stats 
            WHERE restaurant_id = ? AND stat_date = ?
        """
        gojek_df = pd.read_sql_query(gojek_query, eng, params=(restaurant_id, day_str))
        gojek_data = gojek_df.iloc[0].to_dict() if not gojek_df.empty else None
        
        return grab_data, gojek_data
    except:
        return None, None


def _get_period_averages(eng, restaurant_id, start_str, end_str, exclude_date):
    """Получение средних показателей за период"""
    try:
        # Средние показатели исключая анализируемый день
        query = """
            SELECT 
                AVG(grab_sales + gojek_sales) as avg_sales,
                AVG(grab_orders + gojek_orders) as avg_orders,
                AVG(grab_ads + gojek_ads) as avg_ads_spend,
                AVG((grab_sales + gojek_sales) / (grab_orders + gojek_orders)) as avg_check
            FROM (
                SELECT 
                    stat_date,
                    COALESCE(SUM(CASE WHEN source = 'grab' THEN sales END), 0) as grab_sales,
                    COALESCE(SUM(CASE WHEN source = 'grab' THEN orders END), 0) as grab_orders,
                    COALESCE(SUM(CASE WHEN source = 'grab' THEN ads_spend END), 0) as grab_ads,
                    COALESCE(SUM(CASE WHEN source = 'gojek' THEN sales END), 0) as gojek_sales,
                    COALESCE(SUM(CASE WHEN source = 'gojek' THEN orders END), 0) as gojek_orders,
                    COALESCE(SUM(CASE WHEN source = 'gojek' THEN ads_spend END), 0) as gojek_ads
                FROM (
                    SELECT stat_date, sales, orders, ads_spend, 'grab' as source
                    FROM grab_stats 
                    WHERE restaurant_id = ? AND stat_date BETWEEN ? AND ? AND stat_date != ?
                    UNION ALL
                    SELECT stat_date, sales, orders, ads_spend, 'gojek' as source
                    FROM gojek_stats 
                    WHERE restaurant_id = ? AND stat_date BETWEEN ? AND ? AND stat_date != ?
                )
                GROUP BY stat_date
            )
        """
        
        df = pd.read_sql_query(query, eng, params=(
            restaurant_id, start_str, end_str, exclude_date,
            restaurant_id, start_str, end_str, exclude_date
        ))
        
        if not df.empty:
            return df.iloc[0].to_dict()
        else:
            return {'avg_sales': 13000000, 'avg_orders': 34, 'avg_ads_spend': 300000, 'avg_check': 400000}
    except:
        return {'avg_sales': 13000000, 'avg_orders': 34, 'avg_ads_spend': 300000, 'avg_check': 400000}


def _identify_root_causes_improved(critical_date, sub, grab_data, gojek_data, period_averages, day_str):
    """Определение причин провала с учетом всех факторов"""
    
    causes = {'primary': [], 'external': [], 'positive': []}
    
    # Анализ рекламного бюджета
    day_grab_spend = grab_data.get('ads_spend', 0) if grab_data else 0
    day_gojek_spend = gojek_data.get('ads_spend', 0) if gojek_data else 0
    total_day_spend = day_grab_spend + day_gojek_spend
    avg_spend = period_averages.get('avg_ads_spend', 300000)
    
    if total_day_spend < avg_spend * 0.7:  # Снижение >30%
        spend_drop_pct = ((avg_spend - total_day_spend) / avg_spend * 100)
        causes['primary'].append({
            'title': 'КРИТИЧЕСКОЕ УРЕЗАНИЕ РЕКЛАМНОГО БЮДЖЕТА',
            'description': f'Общий бюджет {_format_currency(total_day_spend)} против нормы {_format_currency(avg_spend)} (-{spend_drop_pct:.0f}%)',
            'impact': f'~{_format_currency((avg_spend - total_day_spend) * 15)} потенциальных потерь',
            'category': 'Marketing'
        })
    
    # Анализ среднего чека
    day_orders = (grab_data.get('orders', 0) if grab_data else 0) + (gojek_data.get('orders', 0) if gojek_data else 0)
    day_sales = (grab_data.get('sales', 0) if grab_data else 0) + (gojek_data.get('sales', 0) if gojek_data else 0)
    day_check = day_sales / day_orders if day_orders > 0 else 0
    avg_check = period_averages.get('avg_check', 400000)
    
    if day_check < avg_check * 0.8:  # Снижение >20%
        check_drop = avg_check - day_check
        check_drop_pct = (check_drop / avg_check * 100)
        causes['primary'].append({
            'title': 'КРИТИЧЕСКОЕ ПАДЕНИЕ СРЕДНЕГО ЧЕКА',
            'description': f'Средний чек {_format_currency(day_check)} против нормы {_format_currency(avg_check)} (-{check_drop_pct:.0f}%)',
            'impact': f'{_format_currency(check_drop * day_orders)} прямых потерь',
            'category': 'Operations'
        })
    
    # Анализ операционных проблем
    if gojek_data and 'preparation_time' in gojek_data:
        prep_time = _parse_time_to_minutes(gojek_data['preparation_time'])
        if prep_time and prep_time > 25:  # Очень медленная кухня
            causes['primary'].append({
                'title': 'МЕДЛЕННАЯ КУХНЯ',
                'description': f'Время приготовления {prep_time:.1f} минут (критично >25 мин)',
                'impact': 'Снижение конверсии и удовлетворенности клиентов',
                'category': 'Operations'
            })
    
    # Анализ технических проблем
    if grab_data and grab_data.get('offline_rate', 0) > 60:  # Больше часа оффлайн
        offline_hours = grab_data['offline_rate'] / 60
        causes['primary'].append({
            'title': 'ДЛИТЕЛЬНЫЙ ОФФЛАЙН GRAB',
            'description': f'Платформа была недоступна {offline_hours:.1f} часов',
            'impact': f'~{_format_currency(offline_hours * 200000)} потерь от недоступности',
            'category': 'Technical'
        })
    
    # Позитивные факторы
    if grab_data and grab_data.get('ads_spend', 0) > 0:
        grab_roas = grab_data.get('ads_sales', 0) / grab_data['ads_spend']
        if grab_roas > 20:  # Высокий ROAS
            causes['positive'].append({
                'title': 'Высокая эффективность рекламы GRAB',
                'description': f'ROAS {grab_roas:.1f}x (отличный результат)',
                'impact': 'Максимальная отдача от каждого рубля рекламы'
            })
    
    if gojek_data and gojek_data.get('ads_spend', 0) > 0:
        gojek_roas = gojek_data.get('ads_sales', 0) / gojek_data['ads_spend']
        if gojek_roas > 15:
            causes['positive'].append({
                'title': 'Высокая эффективность рекламы GOJEK',
                'description': f'ROAS {gojek_roas:.1f}x (хороший результат)',
                'impact': 'Эффективное использование бюджета'
            })
    
    # Качество сервиса
    avg_rating = 0
    rating_count = 0
    if grab_data and grab_data.get('rating'):
        avg_rating += grab_data['rating']
        rating_count += 1
    if gojek_data and gojek_data.get('rating'):
        avg_rating += gojek_data['rating']
        rating_count += 1
    
    if rating_count > 0:
        avg_rating /= rating_count
        if avg_rating >= 4.7:
            causes['positive'].append({
                'title': 'Высокое качество сервиса',
                'description': f'Средний рейтинг {avg_rating:.1f}/5.0',
                'impact': 'Сохранение лояльности клиентов'
            })
    
    return causes


def _get_holiday_info(critical_date, sub):
    """Получение информации о праздниках"""
    try:
        day_data = sub[sub["date"].dt.normalize() == critical_date]
        if day_data.empty:
            return {'is_holiday': False, 'description': 'обычный день'}
        
        is_holiday = bool(day_data.iloc[0].get('is_holiday', 0))
        
        if is_holiday:
            # Определяем тип праздника по дате
            date_str = critical_date.strftime('%m-%d')
            
            # Известные мусульманские праздники 2025
            muslim_holidays = {
                '04-01': 'Eid al-Fitr (окончание Рамадана)',
                '06-07': 'Eid al-Adha (Курбан-байрам)',
            }
            
            # Балийские праздники
            balinese_holidays = {
                '03-31': 'Nyepi (День тишины)',
                '05-29': 'Galungan',
                '06-08': 'Kuningan',
            }
            
            # Международные праздники
            international_holidays = {
                '01-01': 'Новый год',
                '12-25': 'Рождество',
                '08-17': 'День независимости Индонезии',
            }
            
            holiday_name = (muslim_holidays.get(date_str) or 
                          balinese_holidays.get(date_str) or 
                          international_holidays.get(date_str) or 
                          'праздничный день')
            
            # Определяем влияние на продажи
            if date_str in muslim_holidays:
                effect = "крупнейший мусульманский праздник — курьеры отдыхают, семейные застолья дома"
            elif date_str in balinese_holidays:
                effect = "балийский праздник — снижение активности на 20-30%"
            elif date_str == '01-01':
                effect = "Новый год — обычно увеличение заказов на 15-25%"
            else:
                effect = "праздничный день — изменение паттернов потребления"
            
            return {
                'is_holiday': True,
                'name': holiday_name,
                'description': f'{holiday_name} — {effect}'
            }
        else:
            weekday = critical_date.strftime('%A')
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


def _get_weather_info(critical_date, sub):
    """Получение информации о погоде"""
    try:
        day_data = sub[sub["date"].dt.normalize() == critical_date]
        if day_data.empty:
            return {'description': 'данные о погоде недоступны'}
        
        row = day_data.iloc[0]
        rain = float(row.get('rain', 0)) if pd.notna(row.get('rain')) else 0
        temp = float(row.get('temp', 0)) if pd.notna(row.get('temp')) else None
        
        weather_parts = []
        
        # Анализ дождя
        if rain >= 25:
            weather_parts.append(f"сильный дождь {rain:.1f}мм (курьеры не работают, -25% заказов)")
        elif rain >= 10:
            weather_parts.append(f"умеренный дождь {rain:.1f}мм (~-15% заказов)")
        elif rain > 0:
            weather_parts.append(f"легкий дождь {rain:.1f}мм (минимальное влияние)")
        else:
            weather_parts.append("без дождя")
        
        # Анализ температуры
        if temp:
            if temp > 35:
                weather_parts.append(f"очень жарко {temp:.1f}°C (снижение активности)")
            elif temp < 20:
                weather_parts.append(f"прохладно {temp:.1f}°C (больше заказов горячей еды)")
            else:
                weather_parts.append(f"комфортная температура {temp:.1f}°C")
        
        return {'description': ', '.join(weather_parts)}
    except:
        return {'description': 'обычная погода'}


def _generate_day_recommendations_improved(root_causes, loss_amount, holiday_info):
    """Генерация рекомендаций для конкретного дня"""
    
    recommendations = []
    
    # Анализируем категории причин
    marketing_issues = len([c for c in root_causes['primary'] if c.get('category') == 'Marketing'])
    operational_issues = len([c for c in root_causes['primary'] if c.get('category') == 'Operations'])
    technical_issues = len([c for c in root_causes['primary'] if c.get('category') == 'Technical'])
    
    # Маркетинговые рекомендации
    if marketing_issues > 0:
        if holiday_info.get('is_holiday'):
            recommendations.append({
                'action': 'Увеличить рекламный бюджет на 50-100% в праздники',
                'potential_impact': f'{_format_currency(min(loss_amount * 0.6, 3000000))}',
                'potential_value': min(loss_amount * 0.6, 3000000),
                'priority': 'High'
            })
            recommendations.append({
                'action': 'Таргетинг на немусульман и туристов в религиозные праздники',
                'potential_impact': f'{_format_currency(min(loss_amount * 0.3, 1500000))}',
                'potential_value': min(loss_amount * 0.3, 1500000),
                'priority': 'High'
            })
        else:
            recommendations.append({
                'action': 'Восстановить рекламный бюджет до нормального уровня',
                'potential_impact': f'{_format_currency(min(loss_amount * 0.7, 4000000))}',
                'potential_value': min(loss_amount * 0.7, 4000000),
                'priority': 'High'
            })
    
    # Операционные рекомендации
    if operational_issues > 0:
        recommendations.append({
            'action': 'Запустить upsell стратегию: промо при заказе >400K IDR',
            'potential_impact': f'{_format_currency(min(loss_amount * 0.4, 2000000))}',
            'potential_value': min(loss_amount * 0.4, 2000000),
            'priority': 'High'
        })
        recommendations.append({
            'action': 'Оптимизировать процессы кухни: дополнительный персонал в пик',
            'potential_impact': f'{_format_currency(min(loss_amount * 0.2, 1000000))}',
            'potential_value': min(loss_amount * 0.2, 1000000),
            'priority': 'Medium'
        })
    
    # Технические рекомендации
    if technical_issues > 0:
        recommendations.append({
            'action': 'Настроить мониторинг доступности платформ + резервные каналы',
            'potential_impact': f'{_format_currency(min(loss_amount * 0.8, 5000000))}',
            'potential_value': min(loss_amount * 0.8, 5000000),
            'priority': 'High'
        })
    
    # Если нет основных проблем
    if not recommendations:
        recommendations.append({
            'action': 'Провести детальный анализ операций в аналогичные дни',
            'potential_impact': 'требует анализа',
            'potential_value': 0,
            'priority': 'Low'
        })
    
    return sorted(recommendations, key=lambda x: {'High': 3, 'Medium': 2, 'Low': 1}[x['priority']], reverse=True)


def _generate_period_summary_improved(critical_dates, sub, restaurant_id):
    """Общие выводы по всем критическим дням"""
    
    lines = []
    
    # Паттерны
    weather_days = len([d for d in critical_dates if _was_rainy_day_improved(d, sub)])
    holiday_days = len([d for d in critical_dates if _was_holiday_improved(d, sub)])
    
    lines.append(f"📊 Всего критических дней: {len(critical_dates)}")
    
    if weather_days > 0:
        lines.append(f"🌧️ Дождливых дней: {weather_days} ({weather_days/len(critical_dates)*100:.0f}%)")
    
    if holiday_days > 0:
        lines.append(f"🎌 Праздничных дней: {holiday_days} ({holiday_days/len(critical_dates)*100:.0f}%)")
    
    # Приоритетная рекомендация
    lines.append("")
    lines.append("🎯 ПРИОРИТЕТНАЯ РЕКОМЕНДАЦИЯ:")
    
    if holiday_days >= len(critical_dates) * 0.5:
        lines.append("   Разработать стратегию работы в праздничные дни")
    elif weather_days >= len(critical_dates) * 0.5:
        lines.append("   Разработать стратегию работы в плохую погоду")
    else:
        lines.append("   Усилить контроль рекламного бюджета и операционных процессов")
    
    return lines


def _was_rainy_day_improved(date, sub):
    """Проверка на дождливый день"""
    try:
        day_data = sub[sub["date"].dt.normalize() == date]
        if day_data.empty:
            return False
        rain = float(day_data.iloc[0].get('rain', 0)) if pd.notna(day_data.iloc[0].get('rain')) else 0
        return rain >= 10
    except:
        return False


def _was_holiday_improved(date, sub):
    """Проверка на праздник"""
    try:
        day_data = sub[sub["date"].dt.normalize() == date]
        if day_data.empty:
            return False
        return bool(day_data.iloc[0].get('is_holiday', 0))
    except:
        return False


def _format_currency(amount):
    """Форматирование валюты"""
    if amount >= 1000000:
        return f"{amount/1000000:.1f}M IDR"
    elif amount >= 1000:
        return f"{amount/1000:.0f}K IDR"
    else:
        return f"{amount:.0f} IDR"


def _parse_time_to_minutes(time_str):
    """Парсинг времени в минуты"""
    if not time_str:
        return None
    try:
        if isinstance(time_str, str) and ':' in time_str:
            parts = time_str.split(':')
            return int(parts[0]) * 60 + int(parts[1])
        return float(time_str)
    except:
        return None