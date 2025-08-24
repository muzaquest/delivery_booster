import os
import sys
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import streamlit as st

PROJECT_ROOT = os.getenv("PROJECT_ROOT", os.getcwd())
if PROJECT_ROOT not in sys.path:
	sys.path.append(PROJECT_ROOT)

from etl.data_loader import get_engine
from app.report_text import generate_full_report


def _list_restaurants() -> pd.DataFrame:
	"""Получение списка ресторанов через адаптер"""
	try:
		from app.data_adapter import get_data_adapter
		adapter = get_data_adapter()
		return adapter.get_restaurants_list()
	except Exception:
		# Fallback к старому способу
		eng = get_engine()
		df = pd.read_sql_query('SELECT id, name FROM restaurants ORDER BY name', eng)
		return df


def _ensure_reports_dir() -> str:
	dir_path = os.path.join(PROJECT_ROOT, 'reports')
	os.makedirs(dir_path, exist_ok=True)
	return dir_path


def _period_presets() -> dict:
	today = date.today()
	start_month = today.replace(day=1)
	last_month_end = start_month - timedelta(days=1)
	last_month_start = last_month_end.replace(day=1)
	week_start = today - timedelta(days=7)
	return {
		'Последние 7 дней': (week_start, today),
		'Этот месяц': (start_month, today),
		'Прошлый месяц': (last_month_start, last_month_end),
	}


def _format_period(d1: date, d2: date) -> str:
	return f"{d1.strftime('%Y-%m-%d')}_{d2.strftime('%Y-%m-%d')}"


def _sync_restaurant_data():
	"""Синхронизация данных ресторана с живым API"""
	try:
		# Проверяем доступность API клиента
		import os
		if not os.getenv("DATABASE_URL"):
			st.error("❌ DATABASE_URL не настроен. Используется локальная SQLite.")
			return
		
		with st.spinner('Синхронизация данных с API...'):
			from etl.api_client import sync_all_sources
			from datetime import date, timedelta
			
			# Синхронизируем последние 30 дней для всех ресторанов
			restaurants = ['Only Kebab', 'Ika Canggu', 'Asai Cafe']  # Можно расширить
			
			total_updated = 0
			for restaurant in restaurants:
				try:
					result = sync_all_sources(
						restaurant, 
						start_date=date.today() - timedelta(days=30),
						end_date=date.today() - timedelta(days=1)
					)
					total_updated += result.get('total_records_updated', 0)
				except Exception as e:
					st.warning(f"Ошибка синхронизации {restaurant}: {e}")
			
			if total_updated > 0:
				st.success(f"✅ Обновлено {total_updated} записей")
				
				# Проверяем нужно ли переобучение ML
				if total_updated >= 30:
					st.info("🤖 Рекомендуется переобучить ML модель (много новых данных)")
					if st.button("🚀 Переобучить модель"):
						_retrain_ml_model()
			else:
				st.info("ℹ️ Новых данных не найдено")
				
	except ImportError:
		st.error("❌ API клиент не найден. Убедитесь что etl/api_client.py доступен.")
	except Exception as e:
		st.error(f"❌ Ошибка синхронизации: {e}")


def _retrain_ml_model():
	"""Переобучение ML модели"""
	try:
		with st.spinner('Переобучение ML модели...'):
			import subprocess
			
			# Экспортируем данные в CSV
			from etl.build_views import export_to_csv_for_ml
			live_csv = os.getenv('ML_DATASET_LIVE_PATH', os.path.join(PROJECT_ROOT, 'data', 'live_dataset.csv'))
			artifact_dir = os.getenv('ML_ARTIFACT_DIR', os.path.join(PROJECT_ROOT, 'ml', 'artifacts'))
			if export_to_csv_for_ml(live_csv):
				# Запускаем обучение
				result = subprocess.run([
					'python', os.path.join('ml', 'training.py'), 
					'--csv', live_csv,
					'--out', artifact_dir
				], capture_output=True, text=True, cwd=PROJECT_ROOT)
				
				if result.returncode == 0:
					st.success("✅ ML модель переобучена успешно!")
					st.json(result.stdout)
				else:
					st.error(f"❌ Ошибка обучения: {result.stderr}")
			else:
				st.error("❌ Ошибка экспорта данных для ML")
				
	except Exception as e:
		st.error(f"❌ Ошибка переобучения: {e}")


def tab_restaurant_analysis():
	st.header('Анализ ресторана')
	
	# Кнопка обновления данных
	col1, col2 = st.columns([3, 1])
	with col2:
		if st.button('🔄 Обновить данные из API'):
			_sync_restaurant_data()
	
	rest_df = _list_restaurants()
	if rest_df.empty:
		st.warning('Таблица restaurants пуста. Убедитесь, что БД доступна.')
		return
	rest_map = {f"{row['name']} (ID {row['id']})": int(row['id']) for _, row in rest_df.iterrows()}
	
	with col1:
		label = st.selectbox('Ресторан', list(rest_map.keys()))
	rest_id = rest_map[label]
	rest_name = label.split(' (ID')[0]

	presets = _period_presets()
	preset = st.selectbox('Период (пресеты)', list(presets.keys()))
	start_default, end_default = presets[preset]
	col1, col2 = st.columns(2)
	with col1:
		start_date = st.date_input('Начало', start_default)
	with col2:
		end_date = st.date_input('Окончание', end_default)

	period = _format_period(start_date, end_date)

	if st.button('Сформировать отчёт'):
		try:
			text = generate_full_report(period=period, restaurant_id=rest_id)
			st.success('Отчёт сформирован')
			st.text_area('Отчёт', value=text, height=600)
			reports_dir = _ensure_reports_dir()
			fn = os.path.join(reports_dir, f"report_{rest_id}_{period}.md")
			with open(fn, 'w', encoding='utf-8') as f:
				f.write(text)
			st.download_button('Скачать .md', data=text, file_name=os.path.basename(fn), mime='text/markdown')
		except Exception as e:
			st.error(f'Ошибка формирования: {e}')


def _aggregate_kpi(engine, start: date, end: date) -> dict:
	"""Получение KPI через адаптер данных"""
	try:
		from app.data_adapter import get_data_adapter
		adapter = get_data_adapter()
		start_s, end_s = start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')
		return adapter.get_kpi_data(start_s, end_s)
	except Exception:
		# Fallback к старому способу
		start_s, end_s = start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')
		q = lambda t: pd.read_sql_query(
			f"SELECT SUM(sales) sales, SUM(orders) orders, SUM(ads_spend) ads_spend, SUM(ads_sales) ads_sales, AVG(rating) rating, SUM(cancelled_orders) canc FROM {t} WHERE stat_date BETWEEN ? AND ?",
			engine, params=(start_s, end_s)
		)
		g = q('grab_stats').iloc[0].fillna(0)
		j = q('gojek_stats').iloc[0].fillna(0)
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


def _delta(a: float, b: float) -> float:
	if b == 0:
		return 0.0
	return (a - b) / b * 100.0


def tab_base_analysis():
	st.header('Анализ базы (KPI)')
	eng = get_engine()
	presets = _period_presets()
	preset = st.selectbox('Период (пресеты)', list(presets.keys()))
	start_default, end_default = presets[preset]
	col1, col2 = st.columns(2)
	with col1:
		start_date = st.date_input('Начало', start_default)
	with col2:
		end_date = st.date_input('Окончание', end_default)

	n_days = (end_date - start_date).days + 1
	prev_end = start_date - timedelta(days=1)
	prev_start = prev_end - timedelta(days=n_days - 1)

	k_now = _aggregate_kpi(eng, start_date, end_date)
	k_prev = _aggregate_kpi(eng, prev_start, prev_end)

	st.subheader('KPI панель')
	colA, colB, colC = st.columns(3)
	with colA:
		st.metric('Total Sales', f"{int(k_now['sales']):,} IDR".replace(',', ' '), f"{_delta(k_now['sales'], k_prev['sales']):.1f}%")
		st.metric('Orders', int(k_now['orders']), f"{_delta(k_now['orders'], k_prev['orders']):.1f}%")
		st.metric('AOV', f"{int(k_now['aov']):,} IDR".replace(',', ' '), f"{_delta(k_now['aov'], k_prev['aov']):.1f}%")
	with colB:
		st.metric('Ads Spend', f"{int(k_now['ads_spend']):,} IDR".replace(',', ' '), f"{_delta(k_now['ads_spend'], k_prev['ads_spend']):.1f}%")
		st.metric('Ads Sales', f"{int(k_now['ads_sales']):,} IDR".replace(',', ' '), f"{_delta(k_now['ads_sales'], k_prev['ads_sales']):.1f}%")
		st.metric('ROAS', f"{k_now['roas']:.2f}x", f"{_delta(k_now['roas'], k_prev['roas']):.1f}%")
	with colC:
		st.metric('Payouts', '—', '—')
		st.metric('Cancels', int(k_now['cancels']), f"{_delta(k_now['cancels'], k_prev['cancels']):.1f}%")
		st.metric('MER', f"{k_now['mer']:.2f}", f"{_delta(k_now['mer'], k_prev['mer']):.1f}%")

	st.caption('Примечание: выплаты (Payouts) доступны, если источники содержат поле payouts.')


def tab_ai_query():
	st.header('🤖 AI Аналитик продаж')
	st.caption('Умный анализ причин падения продаж с ML и человеческим языком. Задавайте вопросы о конкретных ресторанах и периодах.')
	
	# Примеры вопросов
	with st.expander("💡 Примеры вопросов"):
		st.write("""
		**🔍 Анализ падения продаж:**
		• "Почему упали продажи в Only Eggs в мае 2025?"
		• "Почему падают продажи в Ika Canggu последние 2 месяца?"
		• "Что происходит с продажами в Huge в июле?"
		
		**📈 Анализ трендов:**
		• "Как изменились продажи в Soul Kitchen за квартал?"
		• "Какие факторы влияют на продажи в Prana?"
		• "Сравни эффективность рекламы GRAB vs GOJEK в Signa"
		
		**🎯 Стратегические вопросы:**
		• "Какие рестораны показывают лучший рост?"
		• "Влияние праздников на продажи в Canggu"
		• "Оптимальный рекламный бюджет для The Room"
		""")
	
	# Поле ввода вопроса
	question = st.text_area(
		'Ваш вопрос о продажах:', 
		placeholder='Например: "Почему упали продажи в Only Eggs в мае 2025?"',
		height=100
	)
	
	col1, col2 = st.columns([1, 4])
	with col1:
		analyze_button = st.button('🔍 Анализировать', type='primary')
	
	if analyze_button:
		if not question.strip():
			st.warning('❓ Введите вопрос о продажах ресторана')
			return
		
		# Показываем процесс анализа
		with st.spinner('🤖 Анализирую продажи с помощью ML...'):
			try:
				from app.ai_sales_analyzer import analyze_sales_question
				
				# Получаем ответ от AI анализатора
				answer = analyze_sales_question(question)
				
				# Показываем результат
				st.markdown("### 🎯 Результат анализа:")
				st.markdown(answer)
				
				# Дополнительные действия
				st.markdown("---")
				col1, col2, col3 = st.columns(3)
				
				with col1:
					if st.button("📊 Подробный отчет"):
						st.info("Перейдите на вкладку 'Анализ ресторана' для полного отчета")
				
				with col2:
					if st.button("🔄 Обновить данные"):
						st.info("Перейдите на вкладку 'Анализ ресторана' → 'Обновить данные из API'")
				
				with col3:
					if st.button("🤖 Переобучить ML"):
						st.info("ML модель будет переобучена на актуальных данных")
				
			except Exception as e:
				st.error(f"❌ Ошибка анализа: {str(e)}")
				st.info("💡 Убедитесь, что ML модель обучена и данные загружены")
	
	# Статус системы
	st.markdown("---")
	with st.expander("🔧 Статус AI системы"):
		try:
			from app.data_adapter import get_data_adapter
			adapter = get_data_adapter()
			status = adapter.get_data_status()
			
			st.write(f"📊 **Источник данных:** {status.get('data_source', 'Неизвестно')}")
			st.write(f"🏪 **Доступно ресторанов:** {status.get('restaurants', 0)}")
			
			# Проверяем ML модель
			import os
			artifact_dir = os.getenv('ML_ARTIFACT_DIR', os.path.join(PROJECT_ROOT, 'ml', 'artifacts'))
			if os.path.exists(os.path.join(artifact_dir, 'model.joblib')):
				st.success("✅ ML модель готова для анализа")
			else:
				st.warning("⚠️ ML модель не обучена. Запустите обучение для точного анализа.")
			
			# Проверяем праздники
			if os.path.exists(os.path.join(PROJECT_ROOT, 'etl', 'holidays_loader.py')):
				st.success("✅ Система праздников готова (233 праздника)")
			
		except Exception as e:
			st.error(f"❌ Ошибка проверки статуса: {e}")


def main():
	st.set_page_config(page_title='Food Intelligence', layout='wide')
	st.title('Food Intelligence — Аналитика продаж ресторанов')
	
	# Показываем статус данных
	_show_data_status()
	
	tab1, tab2, tab3 = st.tabs(['Анализ ресторана', 'Анализ базы', 'Свободный запрос (AI)'])
	with tab1:
		tab_restaurant_analysis()
	with tab2:
		tab_base_analysis()
	with tab3:
		tab_ai_query()


def _show_data_status():
	"""Показ статуса данных в шапке"""
	try:
		from app.data_adapter import get_data_adapter
		adapter = get_data_adapter()
		status = adapter.get_data_status()
		
		if status.get("status") == "live":
			st.success(f"🔄 Live данные: {status.get('restaurants')} ресторанов, последняя синхронизация: {status.get('last_sync', 'неизвестно')}")
		elif status.get("status") == "static":
			st.warning(f"📁 Статичные данные: {status.get('restaurants')} ресторанов. Для live данных настройте DATABASE_URL.")
		else:
			st.error("❌ Проблемы с данными. Проверьте подключение к БД.")
			
	except Exception:
		st.info("📁 Используются локальные данные SQLite")


if __name__ == '__main__':
	main()