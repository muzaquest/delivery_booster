import os
import pytest


def dataset_available() -> bool:
	return os.path.exists('/workspace/data/merged_dataset.csv') and os.path.exists('/workspace/database.sqlite')


@pytest.mark.skipif(not dataset_available(), reason='Dataset/SQLite not available in this env')
def test_full_report_sections_present():
	from app.report_text import generate_full_report
	period = '2025-04-01_2025-05-31'
	restaurant_id = 20
	text = generate_full_report(period=period, restaurant_id=restaurant_id)
	assert '📊 1. ИСПОЛНИТЕЛЬНОЕ РЕЗЮМЕ' in text
	assert '📈 2. АНАЛИЗ ПРОДАЖ И ТРЕНДОВ' in text
	assert '👥 3. ДЕТАЛЬНЫЙ АНАЛИЗ КЛИЕНТСКОЙ БАЗЫ' in text
	assert '📈 4. МАРКЕТИНГОВАЯ ЭФФЕКТИВНОСТЬ И ВОРОНКА' in text
	assert '6. ⏰ ОПЕРАЦИОННЫЕ МЕТРИКИ' in text
	assert '7. ⭐ КАЧЕСТВО ОБСЛУЖИВАНИЯ' in text
	assert '8. 🚨 КРИТИЧЕСКИЕ ДНИ (ML)' in text
	assert '9. 🎯 СТРАТЕГИЧЕСКИЕ РЕКОМЕНДАЦИИ' in text