# 🚀 РУКОВОДСТВО ПО РАЗВЕРТЫВАНИЮ С ЖИВЫМ API

## 📋 **ПЛАН РАЗВЕРТЫВАНИЯ**

### **1. ЛОКАЛЬНОЕ ТЕСТИРОВАНИЕ (сначала здесь)**

#### **Шаг 1: Настройка переменных окружения**
```bash
export STATS_API_BASE="http://5.187.7.140:3000"
export DATABASE_URL="postgresql://user:password@localhost:5432/analytics"
```

#### **Шаг 2: Тестирование API**
```bash
python test_live_api.py
```

#### **Шаг 3: Создание схемы БД**
```bash
python db/migrate_to_live_api.py
```

#### **Шаг 4: Первая синхронизация данных**
```bash
# Синхронизация одного ресторана
python etl/api_client.py "Only Kebab" --source=all --start-date=2025-01-01

# Построение витрин
python etl/build_views.py --build-views

# Экспорт для ML
python etl/build_views.py --export-csv=/workspace/data/live_dataset.csv
```

#### **Шаг 5: Обучение ML на живых данных**
```bash
python ml/training.py --from-db --out=/workspace/ml/artifacts
```

#### **Шаг 6: Тестирование отчетов**
```bash
# Запуск API
uvicorn app.main:app --reload &

# Запуск Streamlit
streamlit run streamlit_app.py

# Тест отчета
curl "http://localhost:8000/report-text?period=2025-01-01_2025-01-31&restaurant_id=11"
```

---

### **2. РАЗВЕРТЫВАНИЕ НА REPLIT**

#### **Шаг 1: Импорт проекта в Replit**
1. Создать новый Python Repl
2. Импортировать репозиторий
3. Настроить Secrets (Environment)

#### **Шаг 2: Настройка Secrets в Replit**
```
STATS_API_BASE = http://5.187.7.140:3000
DATABASE_URL = postgresql://user:pass@host:port/db
OPENAI_API_KEY = sk-... (опционально)
```

#### **Шаг 3: Установка зависимостей**
```bash
pip install -r requirements.txt
```

#### **Шаг 4: Инициализация БД**
```bash
python db/migrate_to_live_api.py
```

#### **Шаг 5: Синхронизация данных**
```bash
# Загружаем данные всех ресторанов за последние 90 дней
python etl/api_client.py "Only Kebab" --source=all
python etl/api_client.py "Ika Canggu" --source=all  
python etl/api_client.py "Asai Cafe" --source=all

# Строим витрины
python etl/build_views.py --build-views
```

#### **Шаг 6: Обучение ML**
```bash
python ml/training.py --from-db --out=/workspace/ml/artifacts
```

#### **Шаг 7: Запуск сервисов**
```bash
# API (в фоне)
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# Streamlit UI
streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
```

---

## 🔄 **АВТОМАТИЗАЦИЯ ОБНОВЛЕНИЙ**

### **Ежедневная синхронизация (cron):**
```bash
# Добавить в crontab Replit
0 6 * * * cd /workspace && python etl/api_client.py "Only Kebab" --source=all
0 6 * * * cd /workspace && python etl/api_client.py "Ika Canggu" --source=all
30 6 * * * cd /workspace && python etl/build_views.py --build-views
```

### **Автопереобучение ML:**
```bash
# Еженедельно по воскресеньям
0 7 * * 0 cd /workspace && python ml/training.py --from-db
```

---

## 🎯 **ЧТО ИЗМЕНИТСЯ В ОТЧЕТАХ**

### **ДО (статичные данные):**
```
🔴 2025-04-01 (исторический анализ)
- Данные из снапшота БД
- Анализ прошлых событий
```

### **ПОСЛЕ (живые данные):**
```
🔴 2025-12-15 (анализ вчерашнего дня!)
- Свежие данные из API
- Реальные причины вчерашнего провала
- Актуальные рекомендации на сегодня
- ML анализ на основе последних трендов
```

### **Раздел 8 станет:**
- **Актуальным** — анализ последних дней
- **Точным** — ML на свежих данных  
- **Полезным** — рекомендации для текущих проблем
- **Детальным** — разбивка по платформам с живыми цифрами

---

## 🧪 **ТЕСТИРОВАНИЕ**

### **Проверка готовности:**
```bash
# Полный тест системы
python test_live_api.py

# Проверка качества данных
python etl/api_client.py "Only Kebab" --quality-report

# Тест отчета с живыми данными
curl "http://localhost:8000/report-text?period=2025-12-01_2025-12-15&restaurant_id=11"
```

### **Мониторинг:**
- Логи в `/workspace/logs/api_client.log`
- Статус ML jobs в таблице `ml_jobs`
- Качество данных в view `data_quality_check`

---

## ✅ **ГОТОВНОСТЬ К РАЗВЕРТЫВАНИЮ**

**Система готова к переходу на живые данные!**

1. ✅ **API клиент** с ретраями и идемпотентностью
2. ✅ **PostgreSQL схема** с витринами данных
3. ✅ **Миграция** существующих данных
4. ✅ **ML интеграция** с БД
5. ✅ **Streamlit UI** с кнопками синхронизации
6. ✅ **Автоматизация** обновлений
7. ✅ **Мониторинг** и логирование

**Можно тестировать локально, а затем развертывать на Replit!**