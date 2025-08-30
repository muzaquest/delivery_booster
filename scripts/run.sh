#!/usr/bin/env bash
set -e
pkill -f "uvicorn|streamlit" >/dev/null 2>&1 || true
export ANALYTICS_API_BASE="http://localhost:8000"
export API_BASE="http://localhost:8000"
export API_BASE_URL="http://localhost:8000"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 &
for i in {1..20}; do curl -sf http://localhost:8000/health >/dev/null && break; sleep 1; done
streamlit cache clear >/dev/null 2>&1 || true
streamlit run streamlit_app.py \
  --server.address 0.0.0.0 \
  --server.port 5000 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --server.fileWatcherType none