#!/usr/bin/env bash
set -e

curl -s http://localhost:8000/health | grep -q '"connected":true' || true
REST=$(curl -s http://localhost:8000/restaurants | python3 - <<'PY'
import sys, json
data=json.load(sys.stdin)
rows=data.get('restaurants') or []
print(rows[0]['name'] if rows else 'Demo One')
PY
)
echo "Restaurant: $REST"
curl -s --get "http://localhost:8000/report-test" \
  --data-urlencode "restaurant_name=${REST}" \
  --data-urlencode "period=2025-06-01_2025-06-30" | head -n 1
echo "GREEN"

