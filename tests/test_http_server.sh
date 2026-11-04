#!/usr/bin/env bash
set -e

PORT=18080
MODEL="tests/lfm2_toy.gguf"
BIN="./lfm-run"

echo "[Test] Running Embedded HTTP REST server & Web Studio automated integration tests..."

if [ ! -f "$BIN" ]; then
    echo "Binary $BIN not found. Run make all first."
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "Toy model $MODEL not found. Run make fixtures first."
    exit 1
fi

# Launch server in background
$BIN -m "$MODEL" -s --port "$PORT" -j 2 > /dev/null 2>&1 &
SERVER_PID=$!

# Ensure cleanup on exit
cleanup() {
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Wait up to 5 seconds for server to be ready
READY=0
for i in {1..20}; do
    if curl -s "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
        READY=1
        break
    fi
    sleep 0.2
done

if [ "$READY" -ne 1 ]; then
    echo "FAILED: Server failed to start on port $PORT within 5 seconds."
    exit 1
fi
echo "  Server started successfully on port $PORT (PID: $SERVER_PID)."

# 1. Test GET /health
HEALTH_RESP=$(curl -s "http://127.0.0.1:$PORT/health")
echo "  GET /health response: $HEALTH_RESP"
if [[ "$HEALTH_RESP" != *"status"*"ok"* ]] || [[ "$HEALTH_RESP" != *"lfm2"* ]]; then
    echo "FAILED: GET /health returned unexpected payload: $HEALTH_RESP"
    exit 1
fi

# 2. Test GET / (Embedded Web UI)
UI_RESP=$(curl -s "http://127.0.0.1:$PORT/")
if [[ "$UI_RESP" != *"<!DOCTYPE html>"* ]] || [[ "$UI_RESP" != *"Liquid AI"* ]]; then
    echo "FAILED: GET / did not return embedded Web UI."
    exit 1
fi
echo "  GET / verified: Embedded single-page Web UI returned (HTML length: ${#UI_RESP} bytes)."

# 3. Test POST /v1/chat/completions
CHAT_RESP=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"lfm2","messages":[{"role":"user","content":"Liquid Foundation Models"}],"temperature":0.7,"max_tokens":8}')
echo "  POST /v1/chat/completions response received."
if [[ "$CHAT_RESP" != *"chat.completion"* ]] || [[ "$CHAT_RESP" != *"choices"* ]]; then
    echo "FAILED: POST /v1/chat/completions returned unexpected response: $CHAT_RESP"
    exit 1
fi

# 4. Graceful termination
kill "$SERVER_PID"
wait "$SERVER_PID" 2>/dev/null || true
echo "  Server gracefully terminated via SIGTERM."

echo "PASSED: Embedded HTTP REST server & Web Studio tests successful!"
exit 0
