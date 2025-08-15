# Restaurant Sales Analytics

A modular system for restaurant sales analytics with FastAPI, PostgreSQL, ETL, and ML (LightGBM + SHAP).

## Quickstart

- Requirements: Python 3.11+

1) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies

```bash
pip install -r requirements.txt
```

3) Run the API server

```bash
uvicorn app.main:app --reload
```

4) Healthcheck

```bash
curl http://127.0.0.1:8000/health
# {"status": "ok"}
```

## PostgreSQL

Set environment variables or a `DATABASE_URL`:

- `POSTGRES_HOST` (default: localhost)
- `POSTGRES_PORT` (default: 5432)
- `POSTGRES_DB` (default: analytics)
- `POSTGRES_USER` (default: postgres)
- `POSTGRES_PASSWORD` (default: postgres)

Create tables:

```bash
python db/init_db.py
```

## ETL

- SQLite source (provided): `/workspace/database.sqlite`
- Tourism Excel: `/workspace/1.-Data-Kunjungan-2025-3.xls`, `/workspace/Table-1-7-Final-1-1.xls`
- Fake orders: file `/workspace/Fake orders` contains a Google Sheet link. Or pass `--fake-orders-sheet`.

Run full build to merged CSV:

```bash
python etl/data_loader.py --run \
  --start 2024-01-01 --end 2025-12-31 \
  --out /workspace/data/merged_dataset.csv \
  --excel /workspace/1.-Data-Kunjungan-2025-3.xls /workspace/Table-1-7-Final-1-1.xls
```

Optional:

```bash
# Use explicit sqlite path
python etl/data_loader.py --run --sqlite /workspace/database.sqlite

# Provide fake orders sheet URL/ID
python etl/data_loader.py --run --fake-orders-sheet "https://docs.google.com/spreadsheets/d/<ID>/edit"
```

The merged dataset contains per-restaurant daily rows with sales, weather, tourism, holiday flags and engineered features (lags and rolling means).

## Project Structure

- `app/` FastAPI application
- `ml/` ML models and training
- `etl/` Data loading and cleaning
- `db/` SQLAlchemy models and DB setup
- `tests/` Pytest tests

Further steps will add database models, ETL scripts, weather client with caching, ML training, and reporting endpoints.