#!/usr/bin/env bash
# sweep.sh — Run parameter sweeps in parallel with isolated server instances.
# Each backtest gets its own dedicated server on a unique port.
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
CLI="$PROJ_DIR/engine/target/release/backtest"
S="RELIANCE,INFY,TCS,BAJFINANCE,HINDUNILVR,BHARTIARTL,SBIN,ICICIBANK,HDFCBANK,ITC"
CAPITAL="${1:-100000}"
F="--max-drawdown 0.25 --max-volume-pct 0.10 --max-exposure 0.80"
MAX_JOBS=4
OUTFILE=$(mktemp)
NEXT_PORT=51000

cd "$PROJ_DIR/strategies"
source .venv/bin/activate 2>/dev/null || true

run_one() {
    local strat=$1 int=$2 params=$3 label=$4
    local port=$((NEXT_PORT))
    NEXT_PORT=$((NEXT_PORT + 1))

    # Start isolated server
    python -m server.server --port "$port" &>/dev/null &
    local srv_pid=$!
    sleep 2

    cd "$PROJ_DIR"
    local r=$($CLI run --strategy "$strat" --symbols "$S" --from 2024-01-01 --to 2025-12-31 \
        --capital "$CAPITAL" --interval "$int" --strategy-port "$port" $F --params "$params" 2>&1)

    local ret=$(echo "$r" | grep "Return:" | tail -1 | awk '{print $2}')
    local trades=$(echo "$r" | grep "Trades:" | tail -1 | awk '{print $2}')
    local sharpe=$(echo "$r" | grep "Sharpe" | tail -1 | awk '{print $3}')

    kill $srv_pid 2>/dev/null || true
    printf "%-8s %-40s  %10s  %5s trades  Sharpe: %s\n" "$label" "$params" "$ret" "$trades" "$sharpe" >> "$OUTFILE"
}

echo "=== Parameter Sweep @ ₹$(printf '%d' $CAPITAL) (2024-2025) ==="
echo "  Max parallel: $MAX_JOBS"
echo ""

JOBS=()

submit() {
    run_one "$@" &
    JOBS+=($!)
    if [ ${#JOBS[@]} -ge $MAX_JOBS ]; then
        wait "${JOBS[0]}" 2>/dev/null || true
        JOBS=("${JOBS[@]:1}")
    fi
}

# Portfolio Combiner
for risk in 0.01 0.015 0.02 0.03; do
  for atr in 1.5 2.0 2.5; do
    submit portfolio_combiner 15minute "{\"risk_per_trade\":$risk,\"atr_mult\":$atr}" "PC"
  done
done

# RSI
for risk in 0.2 0.3 0.4; do
  for cool in 25 50 100; do
    submit rsi_daily_trend 15minute "{\"risk_pct\":$risk,\"cooldown_bars\":$cool}" "RSI"
  done
done

# Donchian
for risk in 0.005 0.01 0.015; do
  for atr in 1.5 2.0 2.5; do
    submit donchian_breakout 15minute "{\"risk_per_trade\":$risk,\"atr_multiplier\":$atr}" "DON"
  done
done

# SMA
for fast in 10 15; do
  for slow in 30 50; do
    submit sma_crossover day "{\"fast_period\":$fast,\"slow_period\":$slow,\"min_spread\":0.005}" "SMA"
  done
done

# Confluence
for thresh in 2 3; do
  for atr in 1.0 1.5 2.0; do
    submit confluence day "{\"confluence_threshold\":$thresh,\"atr_multiplier\":$atr}" "CONF"
  done
done

# Wait for all
for pid in "${JOBS[@]}"; do
    wait "$pid" 2>/dev/null || true
done

echo ""
for strat in PC RSI DON SMA CONF; do
    echo "--- $strat ---"
    grep "^$strat" "$OUTFILE" | sort -k3 -rn
    echo ""
done

rm -f "$OUTFILE"
pkill -f "server.server" 2>/dev/null || true
echo "Done."
