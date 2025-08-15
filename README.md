# Аналитика продаж ресторанов (FastAPI + PostgreSQL + ETL + ML)

Модульная система аналитики и прогноза продаж с FastAPI, PostgreSQL, ETL, и ML (LightGBM + SHAP).

## Быстрый старт (локально)

Требования: Python 3.11+

1) Виртуальное окружение

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Установка зависимостей

```bash
pip install -r requirements.txt
```

3) Запуск API

```bash
uvicorn app.main:app --reload
```

4) Проверка здоровья

```bash
curl http://127.0.0.1:8000/health
# {"status": "ok"}
```

## PostgreSQL

Переменные окружения (или `DATABASE_URL`):

- `POSTGRES_HOST` (по умолчанию: localhost)
- `POSTGRES_PORT` (по умолчанию: 5432)
- `POSTGRES_DB` (по умолчанию: analytics)
- `POSTGRES_USER` (по умолчанию: postgres)
- `POSTGRES_PASSWORD` (по умолчанию: postgres)

Создание таблиц:

```bash
python db/init_db.py
```

## ETL

- SQLite-источник: `/workspace/database.sqlite`
- Турпоток (Excel): `/workspace/1.-Data-Kunjungan-2025-3.xls`, `/workspace/Table-1-7-Final-1-1.xls`
- Fake orders: файл `/workspace/Fake orders` содержит Google Sheet ссылку; либо передайте `--fake-orders-sheet`.

Сборка объединённого датасета:

```bash
python etl/data_loader.py --run \
  --start 2023-01-01 --end 2025-12-31 \
  --out /workspace/data/merged_dataset.csv \
  --excel /workspace/1.-Data-Kunjungan-2025-3.xls /workspace/Table-1-7-Final-1-1.xls
```

Запись нормализованных таблиц в PostgreSQL (продажи по платформам, операционные, маркетинг, погода, праздники):

```bash
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=analytics
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=postgres

python etl/data_loader.py --run --write-postgres \
  --start 2023-01-01 --end 2025-12-31 \
  --out /workspace/data/merged_dataset.csv
```

## Обучение ML-модели и объяснимость (SHAP)

```bash
python ml/training.py --csv /workspace/data/merged_dataset.csv --out /workspace/ml/artifacts
```

- Модель: LightGBM (регрессия по `total_sales`)
- Артефакты: `ml/artifacts/model.joblib`, `features.json`, `shap_background.csv`, `metrics.json`

## API (после ETL и обучения)

- `GET /report?period=YYYY-MM-DD_YYYY-MM-DD&restaurant_id=...` — сводный отчёт: фактические/прогнозные продажи, средний чек, ТОП‑факторы (SHAP)
- `GET /factors?period=...&restaurant_id=...` — список факторов с влиянием в % (SHAP)
- Все ответы — на русском

## Развёртывание на Replit

1) Импортируйте репозиторий в Replit.
2) В Secrets (Environment) задайте при необходимости:
   - `GOOGLE_API_KEY` — для Google Sheets с фейковыми заказами (если лист приватный — откройте для чтения по ссылке)
   - `SQLITE_PATH` — путь к SQLite (по умолчанию `/workspace/database.sqlite`)
   - `DATABASE_URL`/`POSTGRES_*` — если используете внешнюю PostgreSQL
3) В `replit.nix` не требуется, проект на Python — Replit установит зависимости из `requirements.txt` автоматически.
4) Команда запуска (Run):
   - Первый запуск: выполнить ETL и обучение (можно в Shell или Replit script)
     ```bash
     python etl/data_loader.py --run --start 2023-01-01 --end 2025-12-31 --out /workspace/data/merged_dataset.csv
     python ml/training.py --csv /workspace/data/merged_dataset.csv --out /workspace/ml/artifacts
     ```
   - Затем старт сервера:
     ```bash
     uvicorn app.main:app --host 0.0.0.0 --port 8000
     ```
5) Проверка:
   - `GET /health` → `{ "status": "ok" }`
   - `GET /report?period=2025-04-01_2025-05-31&restaurant_id=<ID>`
   - `GET /factors?period=2025-04-01_2025-05-31&restaurant_id=<ID>`

Примечания честности данных:
- Фейковые заказы вычитаются из заказов и выручки (Google Sheets «Fake orders»)
- Погода — Open‑Meteo по геолокации каждого ресторана (с кэшированием в SQLite)
- Праздники — официальные (Nager.Date) + локальные балийские (скрейпинг 2024/2025)
- Туристический поток — из предоставленных Excel
- Модель обучается на полном периоде (2.5 года), факторы — по SHAP

## Пример подробного отчёта (Ika Kero, апрель–май 2025, без ML)

📊 1. ИСПОЛНИТЕЛЬНОЕ РЕЗЮМЕ
————————————————————————————————————————————————————————————————————————
💰 Общая выручка: 462,089,600 IDR (GRAB: 156,945,600 IDR + GOJEK: 305,144,000 IDR)

📦 Общие заказы: 1,183
   ├── 📱 GRAB: 339
   └── 🛵 GOJEK: 844

📈 2. АНАЛИЗ ПРОДАЖ И ТРЕНДОВ
————————————————————————————————————————————————————————————————————————
📊 Динамика по месяцам:
  2025-04: 225,734,600 IDR (30 дней, 7,524,487 IDR/день)
  2025-05: 236,355,000 IDR (31 дней, 7,624,355 IDR/день)

🗓️ Выходные vs Будни:
  📅 Средние продажи в выходные: 7,852,941 IDR
  📅 Средние продажи в будни: 7,467,945 IDR
  📊 Эффект выходных: 5.2%
📊 АНАЛИЗ РАБОЧИХ ДНЕЙ:
🏆 Лучший день: 2025-04-21 - 16,147,900 IDR
   💰 GRAB: 6,315,500 IDR (6 заказов) | GOJEK: 9,832,400 IDR (27 заказов)
📉 Худший день: 2025-05-02 - 2,987,100 IDR
   💰 GRAB: 769,700 IDR | GOJEK: 2,217,400 IDR

📈 4. МАРКЕТИНГОВАЯ ЭФФЕКТИВНОСТЬ И ВОРОНКА
————————————————————————————————————————————————————————————————————————
📊 Маркетинговая воронка (только GRAB):
  👁️ Показы рекламы: 60,656
  🔗 Посещения меню: 2,294 (CTR: 3.8%)
  🛒 Добавления в корзину: 436 (конверсия: 8.4% от кликов)
  📦 Заказы от рекламы: 193 (конверсия: 44.3% от корзины)
  
  📊 КЛЮЧЕВЫЕ КОНВЕРСИИ:
  • 🎯 Показ → Заказ: 0.3%
  • 🔗 Клик → Заказ: 8.4%
  • 🛒 Корзина → Заказ: 44.3%
  
  📉 ДЕТАЛЬНЫЙ АНАЛИЗ ВОРОНКИ:
  • 💔 Доля ушедших без покупки: 81.0% (1,858 ушли без покупки)
  • 🛒 Доля неоформленных корзин: 55.7% (243 добавили, но не купили)
  
  💰 ПОТЕНЦИАЛ ОПТИМИЗАЦИИ ВОРОНКИ:
  • 📈 Снижение доли ушедших на 10%: 44,446,002 IDR
  • 🛒 Устранение неоформленных корзин: 131,317,452 IDR  
  • 🎯 Общий потенциал: 175,763,454 IDR

💸 Стоимость привлечения (GRAB):
  💰 CPC: 1,879 IDR (расчёт: бюджет ÷ посещения меню)
  💰 CPA: 22,333 IDR (расчёт: бюджет ÷ заказы от рекламы)

7. ⭐ КАЧЕСТВО ОБСЛУЖИВАНИЯ И УДОВЛЕТВОРЕННОСТЬ (GOJEK)
————————————————————————————————————————————————————————————————————————
📊 Распределение оценок (всего: 95):
  ⭐⭐⭐⭐⭐ 5 звезд: 92 (96.8%)
  ⭐⭐⭐⭐ 4 звезды: 0 (0.0%)
  ⭐⭐⭐ 3 звезды: 2 (2.1%)
  ⭐⭐ 2 звезды: 1 (1.1%)
  ⭐ 1 звезда: 0 (0.0%)

📈 Индекс удовлетворенности: 4.93/5.0
🚨 Негативные отзывы (1-2★): 1 (1.1%)

📊 Частота плохих оценок (не 5★):
  📈 Плохих оценок всего: 3 из 95 (3.2%)
  📦 Успешных заказов GOJEK на 1 плохую оценку: 247.7

Примечание: Пример без ML‑раздела. После обучения модели в отчёт добавятся: прогноз vs факт, ТОП‑факторы (SHAP), аномалии, сравнение периодов.