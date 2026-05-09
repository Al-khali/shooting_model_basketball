#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# smoke_test.sh — Validate a running shoot-ai API instance
#
# Works for both local Docker and GCP Cloud Run.
#
# Usage:
#   # Local Docker:
#   docker compose up -d
#   bash scripts/smoke_test.sh
#
#   # GCP Cloud Run:
#   export BASE_URL="https://shoot-ai-dev-xxxx-ew.a.run.app"
#   export API_KEY="your-key"
#   bash scripts/smoke_test.sh
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8080}"
API_KEY="${API_KEY:-}"
PASS=0
FAIL=0

green() { echo -e "\033[32m✅ $*\033[0m"; }
red()   { echo -e "\033[31m❌ $*\033[0m"; }

check() {
  local desc="$1"
  local expected_status="$2"
  shift 2
  local actual_status
  actual_status=$(curl -s -o /dev/null -w "%{http_code}" "$@")
  if [ "$actual_status" = "$expected_status" ]; then
    green "${desc} → ${actual_status}"
    PASS=$((PASS + 1))
  else
    red "${desc} → expected ${expected_status}, got ${actual_status}"
    FAIL=$((FAIL + 1))
  fi
}

check_body() {
  local desc="$1"
  local pattern="$2"
  shift 2
  local body
  body=$(curl -s "$@")
  if echo "$body" | python3 -c "import sys,json; d=json.load(sys.stdin); exit(0 if '${pattern}' in str(d) else 1)" 2>/dev/null; then
    green "${desc}"
    PASS=$((PASS + 1))
  else
    red "${desc} — pattern '${pattern}' not found in: ${body:0:200}"
    FAIL=$((FAIL + 1))
  fi
}

echo "==> Smoke tests: ${BASE_URL}"
echo ""

# ── 1. Health check (no auth required) ──────────────────────────────────────
check "GET /health → 200" "200" "${BASE_URL}/health"
check_body "GET /health has status=ok" "ok" "${BASE_URL}/health"
check_body "GET /health has version" "version" "${BASE_URL}/health"

# ── 2. Auth checks (only run if API_KEY is set) ──────────────────────────────
if [ -n "$API_KEY" ]; then
  check "GET /player/smoke-test/history without key → 401" "401" \
    "${BASE_URL}/player/smoke-test/history"

  check "GET /player/smoke-test/history with valid key → 200" "200" \
    -H "X-API-Key: ${API_KEY}" \
    "${BASE_URL}/player/smoke-test/history"

  check_body "GET /player/smoke-test/history returns player_id" "smoke-test" \
    -H "X-API-Key: ${API_KEY}" \
    "${BASE_URL}/player/smoke-test/history"
else
  echo "  (skipping auth checks — set API_KEY to enable)"
fi

# ── 3. CORS preflight (OPTIONS should always pass) ───────────────────────────
check "OPTIONS /health → 200 (CORS preflight)" "200" \
  -X OPTIONS \
  -H "Origin: http://localhost:3000" \
  -H "Access-Control-Request-Method: GET" \
  "${BASE_URL}/health"

# ── 4. 404 on unknown routes ─────────────────────────────────────────────────
check "GET /nonexistent → 404" "404" "${BASE_URL}/nonexistent"

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "==> Results: ${PASS} passed, ${FAIL} failed"
[ "$FAIL" -eq 0 ] || exit 1
