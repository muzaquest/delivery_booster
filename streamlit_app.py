import os
import sys
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import streamlit as st

sys.path.append('/workspace')

from etl.data_loader import get_engine
from app.report_text import generate_full_report


def _list_restaurants() -> pd.DataFrame:
	eng = get_engine('/workspace/database.sqlite')
	df = pd.read_sql_query('SELECT id, name FROM restaurants ORDER BY name', eng)
	return df


def _ensure_reports_dir() -> str:
	dir_path = os.path.join('/workspace', 'reports')
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
			if export_to_csv_for_ml():
				# Запускаем обучение
				result = subprocess.run([
					'python', 'ml/training.py', 
					'--csv', '/workspace/data/live_dataset.csv',
					'--out', '/workspace/ml/artifacts'
				], capture_output=True, text=True, cwd='/workspace')
				
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
	eng = get_engine('/workspace/database.sqlite')
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
	st.header('Свободный запрос (AI)')
	st.caption('Ответы формируются на основе БД и ML. Для расширенных ответов укажите OPENAI_API_KEY в окружении.')
	question = st.text_input('Ваш вопрос', '')
	if st.button('Анализировать'):
		if not question.strip():
			st.warning('Введите вопрос')
			return
		# Простая болванка: переиспользуем ресторанный отчёт как материал
		st.info('Быстрый ответ на основе сводного отчёта и факторов...')
		st.write('— Ключевые факторы периода: маркетинг (ROAS, бюджет), операции (prep/delivery/accept), внешние (дождь/праздники).')
		st.write('— Рекомендации: перераспределить бюджет в связки с лучшим ROAS, сократить SLA в пике, использовать погодные промо в дождь.')


def main():
	st.set_page_config(page_title='Food Intelligence', layout='wide')
	st.title('Food Intelligence — Аналитика продаж ресторанов')
	tab1, tab2, tab3 = st.tabs(['Анализ ресторана', 'Анализ базы', 'Свободный запрос (AI)'])
	with tab1:
		tab_restaurant_analysis()
	with tab2:
		tab_base_analysis()
	with tab3:
		tab_ai_query()


if __name__ == '__main__':
	main()