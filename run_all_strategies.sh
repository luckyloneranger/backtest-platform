#!/usr/bin/env bash
# run_all_strategies.sh — Run all strategies in parallel using multiple gRPC server instances.
#
# Usage: ./run_all_strategies.sh [--capital 1000000] [--from 2024-01-01] [--to 2024-12-31]
#
# Spawns one Python strategy server per strategy on sequential ports,
# runs all backtests in parallel, collects results.

set -euo pipefail

# --- Config ---
CAPITAL="${CAPITAL:-1000000}"
FROM="${FROM:-2024-01-01}"
TO="${TO:-2024-12-31}"
SYMBOLS="${SYMBOLS:-RELIANCE,INFY,TCS,BAJFINANCE,HINDUNILVR,BHARTIARTL,SBIN,ICICIBANK,HDFCBANK,ITC}"
FLAGS="${FLAGS:---max-drawdown 0.20 --max-volume-pct 0.10 --max-exposure 0.80}"
CLI="./engine/target/release/backtest"
BASE_PORT=50051
MAX_PARALLEL="${MAX_PARALLEL:-6}"

# Strategy name : interval
STRATEGIES=(
    "sma_crossover:day"
    "rsi_daily_trend:15minute"
    "donchian_breakout:15minute"
    "confluence:day"
    "pairs_trading:day"
    "regime_adaptive:15minute"
    "vwap_reversion:5minute"
    "bollinger_squeeze:5minute"
    "orb_breakout:5minute"
    "portfolio_combiner:15minute"
    "intraday_momentum:5minute"
    "time_adaptive:5minute"
    "relative_strength:15minute"
    "multi_tf_confirm:5minute"
)

# Parse CLI args
while [[ $# -gt 0 ]]; do
    case $1 in
        --capital) CAPITAL="$2"; shift 2 ;;
        --from) FROM="$2"; shift 2 ;;
        --to) TO="$2"; shift 2 ;;
        --symbols) SYMBOLS="$2"; shift 2 ;;
        --max-parallel) MAX_PARALLEL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=== Parallel Strategy Runner ==="
echo "  Capital: ₹$(printf '%s' "$CAPITAL" | sed ':a;s/\B[0-9]\{3\}\>/,&/;ta')"
echo "  Period:  $FROM to $TO"
echo "  Symbols: $SYMBOLS"
echo "  Parallel: $MAX_PARALLEL at a time"
echo ""

# --- Kill any existing strategy servers ---
pkill -9 -f "python -m server.server" 2>/dev/null || true
sleep 1

# --- Change to strategies dir for server startup ---
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJ_DIR/strategies"

# Activate venv
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

# --- Spawn strategy servers ---
PIDS=()
PORTS=()
NUM_SERVERS=$MAX_PARALLEL

echo "Starting $NUM_SERVERS strategy servers..."
for i in $(seq 0 $((NUM_SERVERS - 1))); do
    port=$((BASE_PORT + i))
    python -m server.server --port "$port" &>/dev/null &
    PIDS+=($!)
    PORTS+=($port)
done
sleep 3
echo "  Servers running on ports ${PORTS[*]}"
echo ""

cd "$PROJ_DIR"

# --- Run backtests in parallel batches ---
RESULTS_FILE=$(mktemp)
ACTIVE_JOBS=0
PORT_IDX=0

run_backtest() {
    local strat="$1"
    local interval="$2"
    local port="$3"

    local output
    output=$($CLI run --strategy "$strat" --symbols "$SYMBOLS" \
        --from "$FROM" --to "$TO" --capital "$CAPITAL" --interval "$interval" \
        --strategy-port "$port" $FLAGS 2>&1)

    local id
    id=$(echo "$output" | grep "Results saved" | sed 's/.*results\///' | tr -d '/')

    if [ -n "$id" ]; then
        local detail
        detail=$($CLI results show "$id" 2>&1)
        local ret trades sharpe dd
        ret=$(echo "$detail" | grep "Total Return:" | sed 's/.*Return: *//')
        trades=$(echo "$detail" | grep "Total Trades:" | sed 's/.*Trades: *//')
        sharpe=$(echo "$detail" | grep "Sharpe Ratio:" | sed 's/.*Ratio: *//')
        dd=$(echo "$detail" | grep "Max Drawdown:" | sed 's/.*Drawdown: *//')
        printf "  %-22s %8s  Return: %8s  Trades: %5s  Sharpe: %8s  DD: %s\n" \
            "$strat" "$interval" "$ret" "$trades" "$sharpe" "$dd" >> "$RESULTS_FILE"
    else
        printf "  %-22s %8s  FAILED\n" "$strat" "$interval" >> "$RESULTS_FILE"
    fi
}

echo "Running ${#STRATEGIES[@]} strategies..."
echo ""

JOB_PIDS=()
for entry in "${STRATEGIES[@]}"; do
    strat="${entry%%:*}"
    interval="${entry##*:}"
    port=${PORTS[$((PORT_IDX % NUM_SERVERS))]}
    PORT_IDX=$((PORT_IDX + 1))

    run_backtest "$strat" "$interval" "$port" &
    JOB_PIDS+=($!)
    ACTIVE_JOBS=$((ACTIVE_JOBS + 1))

    # Throttle: wait if we've hit max parallel
    if [ "$ACTIVE_JOBS" -ge "$MAX_PARALLEL" ]; then
        wait "${JOB_PIDS[0]}"
        JOB_PIDS=("${JOB_PIDS[@]:1}")
        ACTIVE_JOBS=$((ACTIVE_JOBS - 1))
    fi
done

# Wait for remaining jobs
for pid in "${JOB_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
done

# --- Print results sorted ---
echo "=== Results ($FROM to $TO, ₹$CAPITAL) ==="
sort -t':' -k3 -rn "$RESULTS_FILE" 2>/dev/null || cat "$RESULTS_FILE"
rm -f "$RESULTS_FILE"

# --- Cleanup servers ---
echo ""
echo "Stopping servers..."
for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
done
wait 2>/dev/null || true
echo "Done."
