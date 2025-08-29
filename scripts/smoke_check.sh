#!/usr/bin/env bash
set -e

curl -s http://localhost:8000/health
curl -s http://localhost:8000/restaurants | head
curl -s --get "http://localhost:8000/report-test" \
  --data-urlencode "restaurant_name=Any" \
  --data-urlencode "period=2025-07-01_2025-07-31" | head
echo "GREEN"

