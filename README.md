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

## Пример отчёта (Ika Kero, апрель–май 2025, без ML)

- Раздел 1. Исполнительное резюме (сводка):
  - Общая выручка: 462,089,600 IDR (GRAB 156,945,600; GOJEK 305,144,000)
  - Общие заказы: 1,183
  - Маркетинг (ROAS): GRAB 24.20x; GOJEK 18.32x

- Раздел 2. Продажи и тренды:
  - Месячная динамика: апрель 225,734,600 IDR (30 дней, 7,524,487/день); май 236,355,000 IDR (31 день, 7,624,355/день)
  - Выходные vs будни: 7,852,941 vs 7,467,945 IDR (+5.16%)
  - Лучший день: 2025-04-21 — 16,147,900 IDR (GOJEK 9,832,400; GRAB 6,315,500)
  - Худший день: 2025-05-02 — 2,987,100 IDR (GOJEK 2,217,400; GRAB 769,700)

- Раздел 3. Клиентская база (GRAB + GOJEK):
  - Новые: 1,088; Повторные: 784; Реактивированные: 38 (пример из Only Eggs)
  - Для Ika Kero — статистика по сегментам доступна из `grab_stats`/`gojek_stats` и будет выведена аналогично

- Раздел 4. Маркетинговая эффективность и воронка (GRAB):
  - Показы 60,656; Меню 2,294; В корзину 436; Заказы 193
  - CTR 3.78%; Клик→Заказ 8.41%; Корзина→Заказ 44.27%
  - CPC 1,879 IDR (расчёт: бюджет ÷ посещения меню); CPA 22,333 IDR; ROAS GRAB 24.86x (апр), 23.80x (май)

- Раздел 5. Финансовые показатели:
  - Выплаты: GRAB 120,042,508; GOJEK 223,770,266; Итого 343,812,774 IDR
  - Доля рекламных продаж: 72.01%
  - Take rate: GRAB 20.77%; GOJEK 22.58%; Чистый ROAS: GRAB 19.17x; GOJEK 14.19x
  - Водопад: выручка → комиссии → рекламный бюджет → выплаты

- Раздел 6. Операционные метрики:
  - (пример для Only Eggs): GRAB ожидание 9.3 мин; GOJEK приготовление 16.2 мин, доставка 13.5 мин, ожидание 8.9 мин; отмены 0.2%
  - Критические сбои (>1ч) берутся из БД (GRAB в минутах; GOJEK в ч:м:с)

- Раздел 7. Качество обслуживания и удовлетворённость (GOJEK):
  - Распределение оценок, индекс удовлетворённости, негативные 1–2★
  - «Успешных заказов на 1 оценку не 5★» — база: заказы − отменённые − потерянные − fake

Примечание: ML‑раздел (факторы и объяснения SHAP) подключается после обучения модели на всём периоде. Запустите полный ETL и тренинг (см. разделы выше).