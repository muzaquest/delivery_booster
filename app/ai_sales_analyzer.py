"""
AI Sales Analyzer - Умный анализатор падения продаж
Отвечает на вопросы владельцев ресторанов человеческим маркетинговым языком
"""

import re
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import os

from app.data_adapter import get_data_adapter
from ml.inference import load_artifacts
import shap
import numpy as np


class SalesAnalyzer:
    """Умный анализатор продаж с ML и человеческим языком"""
    
    def __init__(self):
        self.adapter = get_data_adapter()
        self.restaurants = self._load_restaurants()
        
    def _load_restaurants(self) -> Dict[str, int]:
        """Загружает список ресторанов для поиска по названию"""
        try:
            restaurants_list = self.adapter.get_restaurants_list()
            
            # Проверяем формат данных
            if not restaurants_list:
                return {}
            
            # Если это список словарей
            if isinstance(restaurants_list[0], dict):
                return {r['name'].lower(): r['id'] for r in restaurants_list}
            
            # Если это список кортежей (id, name)
            elif isinstance(restaurants_list[0], (tuple, list)):
                return {r[1].lower(): r[0] for r in restaurants_list}
            
            # Если это строки (только названия)
            elif isinstance(restaurants_list[0], str):
                return {r.lower(): i for i, r in enumerate(restaurants_list, 1)}
            
            return {}
            
        except Exception as e:
            # Fallback: используем базовый список
            return {
                'only eggs': 20,
                'ika canggu': 11,
                'huge': 1,
                'soul kitchen': 6,
                'ika ubud': 7,
                'signa': 9,
                'prana': 12,
                'the room': 15
            }
    
    def parse_question(self, question: str) -> Dict[str, any]:
        """Парсит вопрос пользователя и извлекает ресторан, период, тип анализа"""
        result = {
            'restaurant_name': None,
            'restaurant_id': None,
            'period': None,
            'analysis_type': 'sales_drop',
            'timeframe': 'specific'  # specific, recent, trend
        }
        
        question_lower = question.lower()
        
        # Ищем название ресторана
        for name, restaurant_id in self.restaurants.items():
            if name in question_lower:
                result['restaurant_name'] = name
                result['restaurant_id'] = restaurant_id
                break
        
        # Определяем временной период
        if 'последние' in question_lower or 'последний' in question_lower:
            result['timeframe'] = 'recent'
            
            # Ищем количество месяцев
            months_match = re.search(r'(\d+)\s*месяц', question_lower)
            if months_match:
                months = int(months_match.group(1))
                end_date = datetime.now().date()
                start_date = end_date.replace(day=1) - timedelta(days=30*months)
                result['period'] = f"{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}"
            else:
                # По умолчанию последние 3 месяца
                end_date = datetime.now().date()
                start_date = end_date - timedelta(days=90)
                result['period'] = f"{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}"
        
        # Улучшенный парсинг периодов
        month_mapping = {
            'янв': '01', 'январ': '01',
            'фев': '02', 'феврал': '02', 
            'мар': '03', 'март': '03',
            'апр': '04', 'апрел': '04',
            'май': '05', 'мая': '05', 'мае': '05',
            'июн': '06', 'июня': '06', 'июне': '06',
            'июл': '07', 'июля': '07', 'июле': '07',
            'авг': '08', 'август': '08',
            'сен': '09', 'сентябр': '09',
            'окт': '10', 'октябр': '10',
            'ноя': '11', 'ноябр': '11',
            'дек': '12', 'декабр': '12'
        }
        
        # Ищем год
        year_match = re.search(r'(202[3-9])', question_lower)
        year = year_match.group(1) if year_match else '2025'
        
        # Ищем месяц
        for month_name, month_num in month_mapping.items():
            if month_name in question_lower:
                # Определяем количество дней в месяце
                days_in_month = {
                    '01': 31, '02': 28, '03': 31, '04': 30, '05': 31, '06': 30,
                    '07': 31, '08': 31, '09': 30, '10': 31, '11': 30, '12': 31
                }
                last_day = days_in_month.get(month_num, 30)
                result['period'] = f'{year}-{month_num}-01_{year}-{month_num}-{last_day:02d}'
                break
        
        # Ищем конкретные даты в формате YYYY-MM-DD
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', question_lower)
        if date_match and not result['period']:
            # Если нашли конкретную дату, делаем период на весь месяц этой даты
            date_str = date_match.group(1)
            year, month, day = date_str.split('-')
            days_in_month = {
                '01': 31, '02': 28, '03': 31, '04': 30, '05': 31, '06': 30,
                '07': 31, '08': 31, '09': 30, '10': 31, '11': 30, '12': 31
            }
            last_day = days_in_month.get(month, 30)
            result['period'] = f'{year}-{month}-01_{year}-{month}-{last_day:02d}'
        
        # Определяем тип анализа
        if any(word in question_lower for word in ['падают', 'упали', 'снижение', 'падение']):
            result['analysis_type'] = 'sales_drop'
        elif any(word in question_lower for word in ['растут', 'выросли', 'увеличение', 'рост']):
            result['analysis_type'] = 'sales_growth'
        elif any(word in question_lower for word in ['тренд', 'динамика', 'изменение']):
            result['analysis_type'] = 'trend_analysis'
        
        return result
    
    def analyze_sales_drop(self, restaurant_id: int, period: str, restaurant_name: str) -> str:
        """Анализирует причины падения продаж с ML и человеческим языком"""
        try:
            # Получаем данные через data_adapter
            start_str, end_str = period.split('_')
            kpi_data = self.adapter.get_kpi_data(start_str, end_str)
            if not kpi_data:
                return f"❌ Нет данных по ресторану {restaurant_name} за период {period}"
            
            # Загружаем ML модель
            model, features, background = load_artifacts()
            if not model or not features:
                return "⚠️ ML модель не обучена. Запустите обучение для получения точного анализа причин."
            
            # Получаем данные ресторана
            from etl.data_loader import get_engine
            engine = get_engine()
            
            start_str, end_str = period.split('_')
            
            query = f"""
            SELECT date, total_sales, orders_count, ads_spend, ads_sales, 
                   cancelled_orders, rating, is_holiday, rain, temp
            FROM merged_dataset_view 
            WHERE restaurant_id = {restaurant_id} 
            AND date BETWEEN '{start_str}' AND '{end_str}'
            ORDER BY date
            """
            
            try:
                df = pd.read_sql(query, engine, parse_dates=["date"])
            except:
                # Fallback на базовые данные
                return self._basic_sales_analysis(restaurant_name, period, kpi_data)
            
            if df.empty:
                return f"❌ Нет данных по ресторану {restaurant_name} за {period}"
            
            # Анализируем тренд продаж
            median_sales = df['total_sales'].median()
            avg_sales = df['total_sales'].mean()
            min_sales = df['total_sales'].min()
            max_sales = df['total_sales'].max()
            
            # Находим проблемные дни
            threshold = median_sales * 0.7  # 30% падение
            bad_days = df[df['total_sales'] <= threshold]
            
            response = []
            response.append(f"🔍 **АНАЛИЗ ПАДЕНИЯ ПРОДАЖ: {restaurant_name.upper()}**")
            response.append("=" * 60)
            response.append("")
            
            # Общая картина
            response.append("📊 **ОБЩАЯ КАРТИНА:**")
            response.append(f"• Период анализа: {start_str} — {end_str}")
            response.append(f"• Медианные продажи: {self._format_idr(median_sales)}")
            response.append(f"• Разброс: от {self._format_idr(min_sales)} до {self._format_idr(max_sales)}")
            response.append(f"• Проблемных дней: {len(bad_days)} из {len(df)} ({len(bad_days)/len(df)*100:.1f}%)")
            response.append("")
            
            if len(bad_days) == 0:
                response.append("✅ **ХОРОШИЕ НОВОСТИ:** Критических падений не обнаружено!")
                response.append("📈 Продажи стабильны и выше порога критичности")
                return "\\n".join(response)
            
            # ML анализ причин
            try:
                # Анализируем самый худший день
                worst_day = bad_days.loc[bad_days['total_sales'].idxmin()]
                worst_date = worst_day['date'].strftime('%Y-%m-%d')
                worst_sales = worst_day['total_sales']
                loss = median_sales - worst_sales
                
                response.append(f"🔴 **САМЫЙ КРИТИЧЕСКИЙ ДЕНЬ: {worst_date}**")
                response.append(f"• Продажи: {self._format_idr(worst_sales)} (медиана: {self._format_idr(median_sales)})")
                response.append(f"• Потери: {self._format_idr(loss)} ({((worst_sales - median_sales) / median_sales * 100):+.1f}%)")
                response.append("")
                
                # SHAP анализ для худшего дня
                day_features = worst_day[features] if all(f in worst_day.index for f in features) else None
                
                if day_features is not None:
                    X_day = day_features.values.reshape(1, -1)
                    pre = model.named_steps["pre"]
                    mdl = model.named_steps["model"]
                    X_pre = pre.transform(X_day)
                    
                    if background is not None and not background.empty:
                        bg_pre = pre.transform(background[features])
                        explainer = shap.TreeExplainer(mdl, data=bg_pre)
                    else:
                        explainer = shap.TreeExplainer(mdl)
                    
                    shap_values = explainer.shap_values(X_pre)[0]
                    
                    # Переводим SHAP в человеческий язык
                    causes = self._shap_to_business_language(features, shap_values, worst_day, loss)
                    
                    response.append("🔍 **ГЛАВНЫЕ ПРИЧИНЫ ПАДЕНИЯ:**")
                    for i, cause in enumerate(causes[:3], 1):
                        response.append(f"{i}. {cause}")
                    response.append("")
                
            except Exception as e:
                response.append("⚠️ **ML анализ недоступен**, используем базовый анализ")
                response.append("")
            
            # Анализ общих трендов
            response.append("📈 **ОБЩИЕ ТРЕНДЫ ЗА ПЕРИОД:**")
            
            # Сравниваем первую и вторую половину периода
            mid_point = len(df) // 2
            first_half = df.iloc[:mid_point]['total_sales'].mean()
            second_half = df.iloc[mid_point:]['total_sales'].mean()
            trend_change = ((second_half - first_half) / first_half * 100) if first_half > 0 else 0
            
            if trend_change < -10:
                response.append(f"📉 **Продажи падают:** -{abs(trend_change):.1f}% во второй половине периода")
            elif trend_change > 10:
                response.append(f"📈 **Продажи растут:** +{trend_change:.1f}% во второй половине периода")
            else:
                response.append(f"📊 **Продажи стабильны:** изменение {trend_change:+.1f}%")
            
            # Праздничное влияние
            holiday_days = df[df['is_holiday'] == 1]
            if len(holiday_days) > 0:
                holiday_avg = holiday_days['total_sales'].mean()
                normal_avg = df[df['is_holiday'] == 0]['total_sales'].mean()
                holiday_effect = ((holiday_avg - normal_avg) / normal_avg * 100) if normal_avg > 0 else 0
                
                if holiday_effect < -15:
                    response.append(f"🕌 **Праздники сильно влияют:** -{abs(holiday_effect):.1f}% в праздничные дни")
                elif holiday_effect > 15:
                    response.append(f"🕌 **Праздники помогают:** +{holiday_effect:.1f}% в праздничные дни")
            
            # Погодное влияние
            rainy_days = df[df['rain'] >= 5.0]
            if len(rainy_days) > 0:
                rainy_avg = rainy_days['total_sales'].mean()
                dry_avg = df[df['rain'] < 5.0]['total_sales'].mean()
                rain_effect = ((rainy_avg - dry_avg) / dry_avg * 100) if dry_avg > 0 else 0
                
                if rain_effect < -10:
                    response.append(f"🌧️ **Дождь снижает продажи:** -{abs(rain_effect):.1f}% в дождливые дни")
            
            response.append("")
            
            # Конкретные рекомендации
            response.append("🎯 **КОНКРЕТНЫЕ РЕКОМЕНДАЦИИ:**")
            response.append("1. 🔴 Проанализируйте маркетинговую активность в худшие дни")
            response.append("2. 🟡 Разработайте стратегию для праздничных дней")
            response.append("3. 🟡 Внедрите погодные промо-акции")
            response.append("4. 🟢 Мониторьте операционные показатели (время доставки, рейтинг)")
            
            return "\\n".join(response)
            
        except Exception as e:
            return f"❌ Ошибка анализа: {str(e)}"
    
    def _shap_to_business_language(self, features: List[str], shap_values: np.ndarray, 
                                 day_data: pd.Series, total_loss: float) -> List[str]:
        """Переводит SHAP значения в понятный бизнес-язык"""
        causes = []
        
        # Сортируем по негативному влиянию
        negative_factors = []
        for feature, shap_val in zip(features, shap_values):
            if shap_val < 0:  # Негативное влияние
                contribution = abs(shap_val)
                percentage = (contribution / total_loss * 100) if total_loss > 0 else 0
                negative_factors.append((feature, contribution, percentage))
        
        # Сортируем по вкладу
        negative_factors.sort(key=lambda x: x[1], reverse=True)
        
        for feature, contribution, percentage in negative_factors[:5]:
            cause = self._translate_feature_to_business(feature, day_data, contribution, percentage)
            if cause:
                causes.append(cause)
        
        return causes
    
    def _translate_feature_to_business(self, feature: str, day_data: pd.Series, 
                                     contribution: float, percentage: float) -> Optional[str]:
        """Переводит ML фичу в бизнес-объяснение"""
        
        # Маркетинговые факторы
        if 'ads_spend' in feature.lower():
            platform = "GRAB" if 'grab' in feature.lower() else ("GOJEK" if 'gojek' in feature.lower() else "")
            if platform:
                return f"🔴 **{platform}: урезание рекламного бюджета** — потери {self._format_idr(contribution)} ({percentage:.1f}%)"
            else:
                return f"🔴 **Снижение рекламного бюджета** — потери {self._format_idr(contribution)} ({percentage:.1f}%)"
        
        if 'roas' in feature.lower():
            platform = "GRAB" if 'grab' in feature.lower() else ("GOJEK" if 'gojek' in feature.lower() else "")
            return f"🔴 **{platform}: низкая эффективность рекламы** — потери {self._format_idr(contribution)} ({percentage:.1f}%)"
        
        # Операционные факторы
        if 'prep' in feature.lower() or 'preparation' in feature.lower():
            return f"🟡 **Медленная кухня** — время приготовления повлияло на {self._format_idr(contribution)} ({percentage:.1f}%)"
        
        if 'delivery' in feature.lower():
            return f"🟡 **Медленная доставка** — время доставки снизило продажи на {self._format_idr(contribution)} ({percentage:.1f}%)"
        
        if 'rating' in feature.lower():
            return f"🟡 **Падение рейтинга** — качество сервиса повлияло на {self._format_idr(contribution)} ({percentage:.1f}%)"
        
        if 'cancel' in feature.lower():
            return f"🔴 **Рост отмен заказов** — потери {self._format_idr(contribution)} ({percentage:.1f}%)"
        
        # Внешние факторы
        if 'holiday' in feature.lower():
            return f"🕌 **Праздничный день** — снижение активности на {self._format_idr(contribution)} ({percentage:.1f}%)"
        
        if 'rain' in feature.lower():
            rain_mm = day_data.get('rain', 0) or 0
            if rain_mm >= 10:
                return f"🌧️ **Сильный дождь ({rain_mm:.1f}мм)** — курьеры не работали, потери {self._format_idr(contribution)} ({percentage:.1f}%)"
            elif rain_mm >= 2:
                return f"🌧️ **Дождь ({rain_mm:.1f}мм)** — снижение активности на {self._format_idr(contribution)} ({percentage:.1f}%)"
        
        # Пропускаем незначительные факторы
        if percentage < 5:
            return None
        
        return f"🟢 **{feature}** — влияние {self._format_idr(contribution)} ({percentage:.1f}%)"
    
    def _basic_sales_analysis(self, restaurant_name: str, period: str, kpi_data: Dict) -> str:
        """Базовый анализ без ML когда нет детальных данных"""
        response = []
        response.append(f"📊 **БАЗОВЫЙ АНАЛИЗ: {restaurant_name.upper()}**")
        response.append("=" * 50)
        response.append("")
        
        # Используем доступные KPI
        if "total_sales" in kpi_data:
            response.append(f"💰 Общие продажи: {self._format_idr(kpi_data['total_sales'])}")
        
        if "total_orders" in kpi_data:
            response.append(f"📦 Общие заказы: {kpi_data['total_orders']}")
            
            if "total_sales" in kpi_data and kpi_data['total_orders'] > 0:
                avg_check = kpi_data['total_sales'] / kpi_data['total_orders']
                response.append(f"💵 Средний чек: {self._format_idr(avg_check)}")
        
        response.append("")
        response.append("⚠️ **Для детального анализа причин падения:**")
        response.append("1. Запустите ETL для загрузки полных данных")
        response.append("2. Обучите ML модель")
        response.append("3. Повторите запрос для получения SHAP анализа")
        
        return "\\n".join(response)
    
    def _format_idr(self, amount: float) -> str:
        """Форматирует сумму в IDR"""
        try:
            if amount >= 1_000_000:
                return f"{amount/1_000_000:.1f}M IDR"
            elif amount >= 1_000:
                return f"{amount/1_000:.0f}K IDR"
            else:
                return f"{amount:.0f} IDR"
        except:
            return str(amount)
    
    def answer_question(self, question: str) -> str:
        """Главная функция - отвечает на вопрос пользователя"""
        if not question.strip():
            return "❓ Задайте вопрос о продажах ресторана"
        
        # Парсим вопрос
        parsed = self.parse_question(question)
        
        if not parsed['restaurant_name']:
            # Показываем доступные рестораны
            restaurant_names = list(self.restaurants.keys())[:10]
            return f"""❓ **Не указан ресторан**
            
Доступные рестораны: {', '.join(restaurant_names)}...

**Примеры вопросов:**
• "Почему упали продажи в Only Eggs в мае 2025?"
• "Почему падают продажи в Ika Canggu последние 2 месяца?"
• "Что происходит с продажами в Huge в июле?"
"""
        
        if not parsed['period']:
            return f"""❓ **Не указан период**

**Примеры вопросов:**
• "Почему упали продажи в {parsed['restaurant_name']} в мае 2025?"
• "Почему падают продажи в {parsed['restaurant_name']} последние 3 месяца?"
• "Что происходит с {parsed['restaurant_name']} в июле 2025?"
"""
        
        # Выполняем анализ
        if parsed['analysis_type'] == 'sales_drop':
            return self.analyze_sales_drop(
                parsed['restaurant_id'], 
                parsed['period'], 
                parsed['restaurant_name']
            )
        else:
            return f"🔄 Анализ типа '{parsed['analysis_type']}' в разработке"


# Глобальный экземпляр анализатора
_analyzer = None

def get_sales_analyzer():
    """Получает глобальный экземпляр анализатора"""
    global _analyzer
    if _analyzer is None:
        _analyzer = SalesAnalyzer()
    return _analyzer


def analyze_sales_question(question: str) -> str:
    """Публичная функция для анализа вопросов о продажах"""
    analyzer = get_sales_analyzer()
    return analyzer.answer_question(question)