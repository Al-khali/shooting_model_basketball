#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# local_e2e.sh — End-to-end smoke for the AI Shoot stack, on the host machine.
#
# This is NOT scripts/smoke_test.sh — that one only hits /health on a running
# server (post-deploy validation). local_e2e.sh validates the full pipeline:
# spawns uvicorn, generates a stub mp4, runs POST /analyze, polls /session,
# asserts the response shape behaves correctly when GEMINI_API_KEY is missing
# (status=error, no leaked exception text in user-facing fields).
#
# Usage:
#   bash scripts/local_e2e.sh                     # full e2e, no key
#   API_KEY=foo bash scripts/local_e2e.sh         # with API_KEYS configured
#
# Requirements: uv, ffmpeg, curl, python3.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PORT="${PORT:-8088}"               # non-default to avoid colliding with dev servers
BASE_URL="http://127.0.0.1:${PORT}"
STUB_VIDEO="${STUB_VIDEO:-/tmp/shoot_ai_e2e_stub.mp4}"
LOG_FILE="${LOG_FILE:-/tmp/shoot_ai_e2e_uvicorn.log}"
SERVER_PID=""
PASS=0
FAIL=0

green() { printf '\033[32m✅ %s\033[0m\n' "$*"; }
red()   { printf '\033[31m❌ %s\033[0m\n' "$*"; }
info()  { printf '\033[36m▶  %s\033[0m\n' "$*"; }

assert_eq() {
  local desc="$1" expected="$2" actual="$3"
  if [ "$expected" = "$actual" ]; then
    green "${desc} (= ${actual})"
    PASS=$((PASS + 1))
  else
    red "${desc}: expected '${expected}', got '${actual}'"
    FAIL=$((FAIL + 1))
  fi
}

cleanup() {
  if [ -n "${SERVER_PID}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    info "Stopping uvicorn (pid=${SERVER_PID})"
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  rm -f "${STUB_VIDEO}"
}
trap cleanup EXIT

# ────────────────────────────────────────────────────────────────────────────
# Preconditions
# ────────────────────────────────────────────────────────────────────────────
info "Checking required tools"
command -v uv      >/dev/null || { red "uv not found"; exit 1; }
command -v ffmpeg  >/dev/null || { red "ffmpeg not found"; exit 1; }
command -v curl    >/dev/null || { red "curl not found"; exit 1; }
command -v python3 >/dev/null || { red "python3 not found"; exit 1; }
green "Preconditions OK"

# ────────────────────────────────────────────────────────────────────────────
# Generate stub video — 2s of synthetic content, no real player
# ────────────────────────────────────────────────────────────────────────────
info "Generating stub mp4 → ${STUB_VIDEO}"
ffmpeg -y -loglevel error \
  -f lavfi -i "testsrc=duration=2:size=640x480:rate=30" \
  -c:v libx264 -pix_fmt yuv420p -movflags +faststart \
  "${STUB_VIDEO}"
test -s "${STUB_VIDEO}" || { red "ffmpeg produced empty file"; exit 1; }
green "Stub video ready ($(wc -c < "${STUB_VIDEO}") bytes)"

# ────────────────────────────────────────────────────────────────────────────
# Boot uvicorn in background
# ────────────────────────────────────────────────────────────────────────────
info "Starting uvicorn on ${BASE_URL}"
# `set +e` for the background launch — we capture pid manually.
set +e
uv run uvicorn src.api.main:app \
  --host 127.0.0.1 --port "${PORT}" \
  --log-level info > "${LOG_FILE}" 2>&1 &
SERVER_PID=$!
set -e
info "uvicorn pid=${SERVER_PID}, log=${LOG_FILE}"

# Wait for /health to respond — up to 30s.
for i in $(seq 1 30); do
  if curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
    green "Server up after ${i}s"
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    red "uvicorn exited before responding — see ${LOG_FILE}"
    tail -40 "${LOG_FILE}" || true
    exit 1
  fi
  sleep 1
  if [ "$i" -eq 30 ]; then
    red "Server did not respond within 30s"
    tail -40 "${LOG_FILE}" || true
    exit 1
  fi
done

# ────────────────────────────────────────────────────────────────────────────
# Test 1 — /health responds with a version string from package metadata
# ────────────────────────────────────────────────────────────────────────────
info "GET /health"
HEALTH=$(curl -fsS "${BASE_URL}/health")
HEALTH_STATUS=$(echo "${HEALTH}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status', ''))")
HEALTH_VERSION=$(echo "${HEALTH}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('version', ''))")
assert_eq "health.status" "ok" "${HEALTH_STATUS}"
# Version must be non-empty and not the old hardcoded literal "0.5.0".
if [ -z "${HEALTH_VERSION}" ]; then
  red "health.version is empty"
  FAIL=$((FAIL + 1))
elif [ "${HEALTH_VERSION}" = "0.5.0" ]; then
  red "health.version is still the old hardcoded literal '0.5.0' — Bug D not fixed"
  FAIL=$((FAIL + 1))
else
  green "health.version dynamic = ${HEALTH_VERSION}"
  PASS=$((PASS + 1))
fi

# ────────────────────────────────────────────────────────────────────────────
# Test 2 — POST /analyze accepts the upload and returns a task_id
# ────────────────────────────────────────────────────────────────────────────
info "POST /analyze with stub video"
ANALYZE=$(curl -fsS -X POST "${BASE_URL}/analyze" \
  -F "file=@${STUB_VIDEO}" \
  -F "player_id=e2e-local" \
  -F "player_level=intermediate")
TASK_ID=$(echo "${ANALYZE}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('task_id', ''))")
INIT_STATUS=$(echo "${ANALYZE}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status', ''))")
assert_eq "analyze.status (initial)" "processing" "${INIT_STATUS}"
if [ -z "${TASK_ID}" ]; then
  red "analyze did not return task_id"
  FAIL=$((FAIL + 1))
  exit 1
fi
green "task_id = ${TASK_ID}"

# ────────────────────────────────────────────────────────────────────────────
# Test 3 — Poll /session/{id} until terminal state (done | error)
# ────────────────────────────────────────────────────────────────────────────
info "Polling /session/${TASK_ID} (max 20s)"
FINAL_STATUS=""
FINAL_BODY=""
for i in $(seq 1 20); do
  FINAL_BODY=$(curl -fsS "${BASE_URL}/session/${TASK_ID}")
  FINAL_STATUS=$(echo "${FINAL_BODY}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status', ''))")
  if [ "${FINAL_STATUS}" = "done" ] || [ "${FINAL_STATUS}" = "error" ]; then
    break
  fi
  sleep 1
done
info "Final status: ${FINAL_STATUS}"

# ────────────────────────────────────────────────────────────────────────────
# Test 4 — Without GEMINI_API_KEY (and without real pose models), the
# pipeline must terminate with status=error and MUST NOT leak the raw
# exception text into any user-facing field.
# ────────────────────────────────────────────────────────────────────────────
if [ -z "${GEMINI_API_KEY:-}" ]; then
  info "Validating no-key behavior (T0-5 Bug A/B fix)"
  assert_eq "session.status (no key)" "error" "${FINAL_STATUS}"

  # Body must not contain the literal SDK error message — that would be
  # a regression of Bug A (leak into detailed_analysis or anywhere else).
  if echo "${FINAL_BODY}" | grep -q "GEMINI_API_KEY environment variable not set"; then
    red "FAIL: raw SDK error text leaked into response — Bug A regression"
    echo "${FINAL_BODY}" | python3 -m json.tool | head -20
    FAIL=$((FAIL + 1))
  else
    green "No internal exception text in response body"
    PASS=$((PASS + 1))
  fi
fi

# ────────────────────────────────────────────────────────────────────────────
# Summary
# ────────────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════"
echo " e2e summary: ${PASS} passed, ${FAIL} failed"
echo "════════════════════════════════════════"
if [ "${FAIL}" -gt 0 ]; then
  echo ""
  echo "Last response body:"
  echo "${FINAL_BODY}" | python3 -m json.tool 2>/dev/null || echo "${FINAL_BODY}"
  echo ""
  echo "Server tail:"
  tail -40 "${LOG_FILE}" || true
  exit 1
fi
