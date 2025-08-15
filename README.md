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

## Project Structure

- `app/` FastAPI application
- `ml/` ML models and training
- `etl/` Data loading and cleaning
- `db/` SQLAlchemy models and DB setup
- `tests/` Pytest tests

Further steps will add database models, ETL scripts, weather client with caching, ML training, and reporting endpoints.