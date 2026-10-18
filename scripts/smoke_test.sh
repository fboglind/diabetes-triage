#!/usr/bin/env bash
# Acceptance smoke test against a running container.
# Usage: scripts/smoke_test.sh [BASE_URL] [EXPECTED_VERSION]
set -euo pipefail

BASE_URL="${1:-http://localhost:80}"
EXPECTED_VERSION="${2:-}"

SAMPLE='{"age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03, "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02, "s5": 0.02, "s6": -0.001}'
MISSING_S6='{"age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03, "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02, "s5": 0.02}'

echo "Waiting for $BASE_URL/health ..."
for _ in $(seq 1 30); do
  curl -fsS "$BASE_URL/health" > /dev/null 2>&1 && break
  sleep 1
done

echo "1/3 GET /health"
health=$(curl -fsS "$BASE_URL/health")
echo "$health"
echo "$health" | jq -e '.status == "ok"' > /dev/null
if [ -n "$EXPECTED_VERSION" ]; then
  echo "$health" | jq -e --arg v "$EXPECTED_VERSION" '.model_version == $v' > /dev/null
fi

echo "2/3 POST /predict (valid)"
pred=$(curl -fsS -X POST "$BASE_URL/predict" -H "Content-Type: application/json" -d "$SAMPLE")
echo "$pred"
echo "$pred" | jq -e '(keys == ["prediction"]) and (.prediction | type == "number")' > /dev/null

echo "3/3 POST /predict (missing s6)"
code=$(curl -s -o /tmp/err.json -w '%{http_code}' -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" -d "$MISSING_S6")
cat /tmp/err.json; echo
[ "$code" = "422" ]
jq -e '.detail[0].loc[-1] == "s6"' /tmp/err.json > /dev/null

echo "All smoke tests passed."
