# Strategy Full Feature Utilization — Design Document

Date: 2026-04-05

## Goal

Upgrade all 4 strategies (3 deterministic + 1 LLM) to use the full engine feature set — limit orders, SL-M engine stops, order cancellation, and dynamic CNC/MIS product type selection — to improve fill quality, stop execution, and overall profitability.

## Problem

The engine supports 4 actions (BUY/SELL/HOLD/CANCEL), 4 order types (MARKET/LIMIT/SL/SL-M), and 3 product types (CNC/MIS/NRML). Currently:

| Feature | Engine | SMA | RSI | Donchian | LLM |
|---------|--------|-----|-----|----------|-----|
| BUY/SELL | Yes | Yes | Yes | Yes | Yes |
| CANCEL | Yes | No | No | No | No |
| MARKET | Yes | Yes | Yes | Yes | Yes |
| LIMIT | Yes | No | No | No | No |
| SL/SL-M | Yes | No | No | No | No |
| CNC | Yes | Yes | No | Yes | Yes |
| MIS | Yes | No | Yes | No | No |
| Dynamic CNC/MIS | Yes | No | No | No | No |

Strategies use only MARKET orders and manage stops in Python code (software stops). This means:
- Stops can only trigger at the next `on_bar` call (up to 15 min late)
- Gap handling for stops is done in Python rather than the engine's proven gap fill logic
- No limit orders for better entry prices
- No cancellation for managing order lifecycle

## Design

### Strategy 1: SMA Crossover

**Limit entries:**
- On golden cross with trend strength confirmed: emit LIMIT buy at `bar.close`
- Each subsequent bar: if unfilled and crossover still valid, CANCEL + submit new LIMIT at current close
- If crossover reverses before fill: CANCEL, no entry

**Engine SL-M stops:**
- On entry fill (detected via `snapshot.fills`): submit SL-M sell at trailing stop level
- Each bar: if trailing stop ratcheted up, CANCEL + submit new SL-M at updated level
- Engine handles gap fills automatically

**Dynamic CNC/MIS:**
- SMA spread `(fast - slow) / slow > 0.01` (strong trend): CNC — expect multi-day hold
- Spread 0.005-0.01 (moderate): MIS — take intraday, auto-squareoff protects downside

**CANCEL usage:**
- Cancel unfilled limit entries on crossover reversal
- Cancel pending SL-M on death cross exit (full exit overrides stop)
- Cancel all pending before short entry

**New state fields:**
- `pending_entry: bool` — limit entry outstanding
- `has_engine_stop: bool` — SL-M in engine

### Strategy 2: RSI Daily Trend

**Limit entries (mean-reversion advantage):**
- RSI < 35 entry: LIMIT buy at `bar.close * 0.999` (0.1% below current — captures dip)
- Cancel unfilled limits after 3 bars (45 min) or when RSI reverses above threshold
- Pyramid levels also use limits at progressively lower prices

**Engine SL-M stops:**
- On entry fill: submit SL-M at `avg_entry * (1 - max_loss_pct)`
- Trailing stop ratchets: CANCEL + new SL-M at updated level
- For shorts: SL-M buy at `avg_entry * (1 + max_loss_pct)`, ratchets down

**Dynamic CNC/MIS:**
- Deep oversold (RSI < 25) in strong uptrend (EMA rising steeply): CNC — high conviction
- Moderate oversold (RSI < 35): MIS — lower conviction
- Short entries: always MIS

**CANCEL usage:**
- Cancel unfilled limit entries when RSI reverses direction
- On partial exit at RSI > 60: CANCEL old SL-M, submit new SL-M at breakeven
- Cancel all pending on full exit

**New state fields:**
- `pending_entry_bar: int` — bar when limit was submitted (cancel after 3 bars)
- `has_engine_stop: bool`

### Strategy 3: Donchian Breakout

**Market entries (keep):**
- Breakouts need immediate execution — limit orders risk missing the move

**Limit orders for partial profit:**
- On entry fill: submit LIMIT sell at `avg_entry + profit_target_atr * ATR` for 1/3 qty
- Rests in engine — fills automatically at exact target price
- Better than waiting for next on_bar (up to 15 min delay)

**Engine SL-M stops:**
- On entry fill: submit SL-M at `entry - atr * multiplier`
- Trailing stop ratchets: CANCEL + new SL-M
- On partial profit fill detected: CANCEL old SL-M, submit new at breakeven

**Dynamic CNC/MIS:**
- Volume > 1.5x average + ATR expanding: CNC — strong breakout, multi-day trend
- Volume 1.0-1.5x, no ATR expansion: MIS — weaker, let auto-squareoff limit risk

**CANCEL usage:**
- Cancel profit target + SL-M on full exit (trailing stop, channel low, max loss, time stop)
- Cancel all pending before short entry

**New state fields:**
- `has_profit_target: bool` — limit sell resting
- `has_engine_stop: bool`

### Strategy 4: LLM Signal Generator

**Expanded system prompt:**
Add to the JSON schema instruction:
```
- "order_type": "MARKET" (immediate), "LIMIT" (at limit_price), "SL_M" (stop at stop_price)
- "limit_price": required for LIMIT orders
- "stop_price": required for SL_M orders
- "product_type": "CNC" (delivery, multi-day), "MIS" (intraday, auto-closed 3:20 PM)
```

Include guidance:
```
Use LIMIT for mean-reversion entries (buy below market).
Use SL_M to set automatic stop-losses.
Use CANCEL to remove pending orders.
Use MIS for intraday, CNC for multi-day positions.
```

**Enhanced snapshot:**
Add pending orders to `format_snapshot()`:
```
Pending orders:
  RELIANCE: SL_M SELL qty=50 stop=1180.00
  INFY: LIMIT BUY qty=30 limit=1450.00
```

**Validation:**
- LIMIT without limit_price → default to MARKET with warning
- SL_M without stop_price → skip signal with warning
- Already validates order_type and product_type against known values

**No structural changes** — Signal dataclass, parse_signals, and engine already handle all fields.

### Shared Pattern: Fill Detection & Order Lifecycle

All 3 deterministic strategies follow the same lifecycle:

```
1. Entry signal → LIMIT or MARKET
2. Fill detected (snapshot.fills) → submit SL-M stop
3. Each bar in position:
   a. Trailing stop moved → CANCEL + new SL-M
   b. Partial profit target (Donchian) → resting LIMIT in engine
4. Partial fill detected → CANCEL old SL-M + new SL-M at breakeven
5. Full exit condition → CANCEL all for symbol + SELL MARKET
6. Stop hit detected (SL-M fill) → reset state
```

**Fill detection:**
```python
for fill in snapshot.fills:
    if fill.symbol == symbol and fill.side == "BUY":
        # Entry filled — submit engine stop
    if fill.symbol == symbol and fill.side == "SELL" and state.has_engine_stop:
        # Stop triggered — position closed by engine, reset state
```

**Reconciliation update:**
Check `snapshot.pending_orders` to verify stops are still active. If a stop disappeared without a fill, re-submit.

## Files Impacted

| File | Changes |
|------|---------|
| `strategies/deterministic/sma_crossover.py` | Limit entries, engine SL-M, dynamic CNC/MIS, CANCEL |
| `strategies/deterministic/rsi_daily_trend.py` | Limit entries, engine SL-M, dynamic CNC/MIS, CANCEL |
| `strategies/deterministic/donchian_breakout.py` | Limit profit target, engine SL-M, dynamic CNC/MIS, CANCEL |
| `strategies/llm/llm_signal_generator.py` | Expanded system prompt |
| `strategies/llm_base.py` | format_snapshot adds pending orders |
| `tests/test_sma_crossover.py` | Tests for limit/SL-M/cancel/dynamic product |
| `tests/test_rsi_daily_trend.py` | Tests for limit/SL-M/cancel/dynamic product |
| `tests/test_donchian_breakout.py` | Tests for limit profit/SL-M/cancel/dynamic product |
| `tests/test_llm_base.py` | Test pending orders in snapshot |
| `tests/test_llm_signal_generator.py` | Test expanded prompt |

## What Doesn't Change

- Engine code (all features already implemented)
- Proto definitions (all fields already defined)
- `base.py` Signal/MarketSnapshot (already have all fields)
- `server.py` (already maps all order types and actions)

## Verification

```bash
cd strategies && pytest tests/ -v    # all tests pass
# Then run backtests on 10 stocks full year 2025:
backtest run --strategy sma_crossover --symbols RELIANCE,INFY,TCS,BAJFINANCE,HINDUNILVR,BHARTIARTL,SBIN,ICICIBANK,HDFCBANK,ITC --from 2025-01-01 --to 2025-12-31 --capital 1000000 --interval day --max-drawdown 0.15 --max-volume-pct 0.10 --max-exposure 0.80
# Repeat for rsi_daily_trend (--interval 15minute) and donchian_breakout (--interval 15minute)
# Compare trade counts, fill prices, costs, and returns vs previous versions
```
