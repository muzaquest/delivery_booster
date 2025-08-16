# Аналитика продаж ресторанов (FastAPI + PostgreSQL + ETL + ML)

Модульная система аналитики и прогноза продаж с FastAPI, PostgreSQL, ETL, и ML (LightGBM/RandomForest + SHAP).

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

- Модели: LightGBM и RandomForest (выбираем чемпиона по MAE)
- Артефакты: `ml/artifacts/model.joblib`, `features.json`, `shap_background.csv`, `metrics.json`

## API (после ETL и обучения)

- `GET /report?period=YYYY-MM-DD_YYYY-MM-DD&restaurant_id=...` — сводка: факт/прогноз, AOV, ТОП‑факторы (SHAP)
- `GET /factors?period=...&restaurant_id=...` — список факторов с влиянием в % (SHAP)
- `GET /report-text?period=...&restaurant_id=...` — полный текстовый отчёт на русском (со всеми разделами и ML)

## Раздел “Критические дни (ML)”

В текстовом отчёте (`/report-text`) есть раздел, который автоматически находит дни с падением продаж ≥ 30% к медиане периода и показывает для каждого дня:
- ТОП‑факторы (SHAP) по всем нетривиальным признакам: категория (Operations/Marketing/External/Quality/Other), направление (↑/↓), вклад в IDR и %
- Вклад групп факторов (%): Operations, Marketing, External (погода/праздники/турпоток), Quality
- Контекст дня: оффлайн платформ (GRAB/GOJEK) с длительностью, погода (дождь мм, температура, ветер, влажность), флаг праздника, маркетинг (spend/ROAS), операции (prep/accept/delivery)
- Рекомендации: что сделать, чтобы нивелировать причину падения

Пример запроса (Only Eggs, апрель–май 2025):

```bash
curl "http://127.0.0.1:8000/report-text?period=2025-04-01_2025-05-31&restaurant_id=20"
```

Фрагмент раздела:

```
8. 🚨 КРИТИЧЕСКИЕ ДНИ (ML)
—
📉 КРИТИЧЕСКИЙ ДЕНЬ: 2025-04-21 (выручка: 1 793 000 IDR; отклонение к медиане: −77.2%)
🔎 ТОП‑факторы (ML):
  • [Marketing] mkt_roas_grab: ↓ вклад ~169 061 IDR (X%)
  • [Operations] preparation_time_mean: ↓ вклад ~233 904 IDR (Y%)
  ...
📊 Вклад групп факторов: Operations 45.0%, Marketing 38.0%, External 10.0%...
📅 Контекст дня:
  • 📱 GRAB оффлайн: 0:00:00
  • 🛵 GOJEK оффлайн: 1:10:00
  • 🎯 ROAS и расходы по каналам, SLA, погода, праздник
💡 Что сделать:
  • Перенастроить кампании и сократить SLA в пике; погодные промо в дождь
```

## Развёртывание на Replit

1) Импортируйте репозиторий в Replit.
2) В Secrets (Environment) задайте при необходимости:
   - `GOOGLE_API_KEY` — для Google Sheets с фейковыми заказами
   - `SQLITE_PATH` — путь к SQLite (по умолчанию `/workspace/database.sqlite`)
   - `DATABASE_URL`/`POSTGRES_*` — если используете внешнюю PostgreSQL
3) Первый запуск: выполните ETL и обучение, затем старт сервера:

```bash
python etl/data_loader.py --run --start 2023-01-01 --end 2025-12-31 --out /workspace/data/merged_dataset.csv
python ml/training.py --csv /workspace/data/merged_dataset.csv --out /workspace/ml/artifacts
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Примечания честности данных
- Фейковые заказы вычитаются из заказов и выручки
- Погода — Open‑Meteo по координатам ресторанов (кэш в SQLite; фолбэк по геокодингу)
- Праздники — Nager.Date + локальные балийские (скрейпинг 2024/2025; при недоступности источника — кеш)
- Туристический поток — из предоставленных Excel
- Модель обучается на всём периоде (2.5 года), факторы — SHAP; тривиальные признаки исключены из объяснений