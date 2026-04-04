#!/usr/bin/env bash
# End-to-end integration test for the backtest platform.
#
# This script:
#   1. Builds the Rust CLI
#   2. Generates synthetic test data
#   3. Starts the Python strategy server
#   4. Runs a full backtest
#   5. Verifies results can be listed and shown
#   6. Cleans up

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CLI="$PROJECT_DIR/engine/target/release/backtest"

cd "$PROJECT_DIR"

# Track the strategy server PID so we can clean up on exit
SERVER_PID=""
cleanup() {
    if [ -n "$SERVER_PID" ]; then
        echo "Cleaning up: stopping strategy server (PID $SERVER_PID)"
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    # Remove test data and results generated during this run
    rm -rf "$PROJECT_DIR/data/NSE/TESTSTOCK"
    echo "Cleanup complete."
}
trap cleanup EXIT

echo "=== Building Rust engine ==="
(cd "$PROJECT_DIR/engine" && cargo build --release -p backtest-cli)
echo ""

echo "=== Generating test data ==="
"$CLI" data generate-test-data \
  --symbol TESTSTOCK --from 2023-01-01 --to 2023-12-31 --interval day --start-price 1000
echo ""

echo "=== Listing cached data ==="
"$CLI" data list
echo ""

echo "=== Setting up Python environment ==="
cd "$PROJECT_DIR/strategies"
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -e ".[dev]"
else
  source .venv/bin/activate
fi
./generate_proto.sh
cd "$PROJECT_DIR"
echo ""

echo "=== Starting strategy server ==="
(cd "$PROJECT_DIR/strategies" && python -m server.server) &
SERVER_PID=$!
sleep 3

# Verify the server is actually listening
if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "ERROR: Strategy server failed to start."
    exit 1
fi
echo "Strategy server running (PID $SERVER_PID)"
echo ""

echo "=== Running backtest ==="
OUTPUT=$("$CLI" run \
  --strategy sma_crossover \
  --symbols TESTSTOCK \
  --from 2023-01-01 \
  --to 2023-12-31 \
  --capital 1000000 \
  --interval day \
  --params '{"fast_period": 10, "slow_period": 30}')
echo "$OUTPUT"

# Extract the backtest ID from the output
BACKTEST_ID=$(echo "$OUTPUT" | grep "ID:" | head -1 | awk '{print $2}')
if [ -z "$BACKTEST_ID" ]; then
    echo "ERROR: Could not extract backtest ID from run output."
    exit 1
fi
echo ""

echo "=== Listing results ==="
"$CLI" results list
echo ""

echo "=== Showing detailed results for $BACKTEST_ID ==="
"$CLI" results show "$BACKTEST_ID"
echo ""

echo "=== Stopping strategy server ==="
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
SERVER_PID=""

echo ""
echo "========================================="
echo "  E2E Test PASSED"
echo "========================================="
