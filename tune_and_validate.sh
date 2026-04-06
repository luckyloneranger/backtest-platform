#!/usr/bin/env bash
# tune_and_validate.sh — Tune parameters on 2024 training data, test best on 2025.
# Uses isolated server instances for parallel execution.
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
CLI="$PROJ_DIR/engine/target/release/backtest"
S="RELIANCE,INFY,TCS,BAJFINANCE,HINDUNILVR,BHARTIARTL,SBIN,ICICIBANK,HDFCBANK,ITC,KOTAKBANK,LT"
CAPITAL=100000
F="--max-drawdown 0.25 --max-volume-pct 0.10 --max-exposure 0.80"
TRAIN_FROM="2024-01-01"
TRAIN_TO="2024-12-31"
TEST_FROM="2025-01-01"
TEST_TO="2025-12-31"
MAX_JOBS=4
OUTFILE=$(mktemp)
NEXT_PORT=51000

cd "$PROJ_DIR/strategies"
source .venv/bin/activate 2>/dev/null || true

run_one() {
    local strat=$1 int=$2 params=$3 label=$4 period_from=$5 period_to=$6
    local port=$NEXT_PORT
    NEXT_PORT=$((NEXT_PORT + 1))

    python -m server.server --port "$port" &>/dev/null &
    local srv_pid=$!
    sleep 2

    cd "$PROJ_DIR"
    local r=$($CLI run --strategy "$strat" --symbols "$S" --from "$period_from" --to "$period_to" \
        --capital "$CAPITAL" --interval "$int" --strategy-port "$port" $F --params "$params" 2>&1)

    local ret=$(echo "$r" | grep "Return:" | tail -1 | sed 's/.*Return: *//' | tr -d ' ')
    local trades=$(echo "$r" | grep "Trades:" | tail -1 | sed 's/.*Trades: *//' | tr -d ' ')
    local sharpe=$(echo "$r" | grep "Sharpe" | tail -1 | sed 's/.*Ratio: *//' | tr -d ' ')

    kill $srv_pid 2>/dev/null || true
    echo "$label|$ret|$trades|$sharpe" >> "$OUTFILE"
}

JOBS=()
submit() {
    run_one "$@" &
    JOBS+=($!)
    if [ ${#JOBS[@]} -ge $MAX_JOBS ]; then
        wait "${JOBS[0]}" 2>/dev/null || true
        JOBS=("${JOBS[@]:1}")
    fi
}

echo "====================================="
echo "  PARALLEL PARAMETER TUNING ON 2024"
echo "  12 stocks, ₹1L capital"
echo "  Max $MAX_JOBS parallel jobs"
echo "====================================="
echo ""

# --- Portfolio Combiner ---
for risk in 0.01 0.015 0.02 0.025; do
  for atr in 1.5 2.0 2.5 3.0; do
    submit portfolio_combiner 15minute "{\"risk_per_trade\":$risk,\"atr_mult\":$atr}" "PC r=$risk a=$atr" "$TRAIN_FROM" "$TRAIN_TO"
  done
done

# --- RSI ---
for risk in 0.2 0.3 0.4; do
  for cool in 25 50 75; do
    for loss in 0.02 0.03 0.05; do
      submit rsi_daily_trend 15minute "{\"risk_pct\":$risk,\"cooldown_bars\":$cool,\"max_loss_pct\":$loss}" "RSI r=$risk c=$cool l=$loss" "$TRAIN_FROM" "$TRAIN_TO"
    done
  done
done

# --- Donchian ---
for risk in 0.005 0.01 0.015 0.02; do
  for atr in 1.5 2.0 2.5; do
    submit donchian_breakout 15minute "{\"risk_per_trade\":$risk,\"atr_multiplier\":$atr}" "DON r=$risk a=$atr" "$TRAIN_FROM" "$TRAIN_TO"
  done
done

# --- OU Mean Reversion ---
for z_entry in 1.5 2.0 2.5; do
  for risk in 0.02 0.03 0.05; do
    for hl in 20 30 50; do
      submit ou_mean_reversion day "{\"zscore_entry\":$z_entry,\"risk_pct\":$risk,\"max_halflife\":$hl}" "OU z=$z_entry r=$risk hl=$hl" "$TRAIN_FROM" "$TRAIN_TO"
    done
  done
done

# --- Ensemble Meta ---
for conf in 0.55 0.60 0.65 0.70; do
  for ret in 0.005 0.01 0.015; do
    submit ensemble_meta day "{\"confidence_threshold\":$conf,\"target_return\":$ret}" "ENS c=$conf t=$ret" "$TRAIN_FROM" "$TRAIN_TO"
  done
done

# --- SMA ---
for fast in 10 15; do
  for slow in 30 50; do
    for spread in 0.003 0.005 0.01; do
      submit sma_crossover day "{\"fast_period\":$fast,\"slow_period\":$slow,\"min_spread\":$spread}" "SMA f=$fast s=$slow sp=$spread" "$TRAIN_FROM" "$TRAIN_TO"
    done
  done
done

# Wait for all
for pid in "${JOBS[@]}"; do wait "$pid" 2>/dev/null || true; done

echo ""
echo "=== TOP 5 PER STRATEGY (sorted by return on 2024 training data) ==="
echo ""
for prefix in PC RSI DON OU ENS SMA; do
    echo "--- $prefix ---"
    grep "^$prefix" "$OUTFILE" | sort -t'|' -k2 -rn | head -5
    echo ""
done

echo ""
echo "=== OVERALL TOP 10 ==="
sort -t'|' -k2 -rn "$OUTFILE" | head -10

rm -f "$OUTFILE"
pkill -f "server.server" 2>/dev/null || true
echo ""
echo "Done. Use the best params per strategy to run on 2025 test data."
