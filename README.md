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

4) Streamlit UI (3 вкладки)

```bash
streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
```

Вкладки:
- Анализ ресторана: выбор ресторана и периода, генерация полного отчёта (1–9), сохранение в `reports/` и скачивание `.md`.
- Анализ базы: KPI‑панель (Sales, Orders, AOV, ROAS, MER, Cancels) и MoM сравнение.
- Свободный запрос (AI): ответы на основе БД+ML (для расширенных ответов задайте `OPENAI_API_KEY`).

5) Проверка здоровья

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

## Обучение ML-модели и объяснимость (SHAP)

```bash
python ml/training.py --csv /workspace/data/merged_dataset.csv --out /workspace/ml/artifacts
```

- Модели: LightGBM и RandomForest (выбираем чемпиона по MAE)
- Артефакты: `ml/artifacts/model.joblib`, `features.json`, `shap_background.csv`, `metrics.json`
- Версия датасета: `metrics.json` содержит хэш, размер и время обучения; в отчёте печатается краткая версия.

## API (после ETL и обучения)

- `GET /report?period=YYYY-MM-DD_YYYY-MM-DD&restaurant_id=...` — сводка: факт/прогноз, AOV, ТОП‑факторы (SHAP)
- `GET /factors?period=...&restaurant_id=...` — список факторов с влиянием в % (SHAP)
- `GET /report-text?period=...&restaurant_id=...` — полный текстовый отчёт на русском (со всеми разделами и ML)

## Тесты

```bash
pytest -q
```
- Smoke‑тест API: `/health`
- Снапшот‑тест отчёта: проверяет наличие разделов 1–9 для одного ресторана/периода

## Развёртывание на Replit

1) Импортируйте репозиторий в Replit.
2) В Secrets (Environment) задайте при необходимости:
   - `GOOGLE_API_KEY` — для Google Sheets с фейковыми заказами
   - `SQLITE_PATH` — путь к SQLite (по умолчанию `/workspace/database.sqlite`)
   - `DATABASE_URL`/`POSTGRES_*` — если используете внешнюю PostgreSQL
3) Первый запуск: выполните ETL и обучение, затем старт сервера и UI:

```bash
python etl/data_loader.py --run --start 2023-01-01 --end 2025-12-31 --out /workspace/data/merged_dataset.csv
python ml/training.py --csv /workspace/data/merged_dataset.csv --out /workspace/ml/artifacts
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
```

## Примечания честности данных
- Фейковые заказы вычитаются из заказов и выручки
- Погода — Open‑Meteo по координатам ресторанов (кеш в SQLite; фолбэк по геокодингу)
- Праздники — Nager.Date + локальные балийские (скрейпинг 2024/2025; при недоступности источника — кеш)
- Туристический поток — из предоставленных Excel
- Модель обучается на всём периоде (2.5 года), факторы — SHAP; тривиальные признаки исключены из объяснений