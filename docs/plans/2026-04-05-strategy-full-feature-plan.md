# Strategy Full Feature Utilization — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade all 4 strategies to use limit orders, engine SL-M stops, order cancellation, and dynamic CNC/MIS product type selection.

**Architecture:** No engine changes needed — all features already exist. Each strategy is modified to emit richer Signal objects (with order_type, limit_price, stop_price, product_type) and manage order lifecycle via snapshot.fills and snapshot.pending_orders. The LLM strategy gets an expanded prompt.

**Tech Stack:** Python 3.11+, pytest

**Design doc:** `docs/plans/2026-04-05-strategy-full-feature-utilization-design.md`

---

### Task 1: LLM Strategy — Expanded Prompt + Pending Orders in Snapshot

**Files:**
- Modify: `strategies/strategies/llm_base.py` — add pending orders to `format_snapshot()`
- Modify: `strategies/strategies/llm/llm_signal_generator.py` — expand system prompt
- Modify: `strategies/tests/test_llm_base.py` — test pending orders in snapshot
- Modify: `strategies/tests/test_llm_signal_generator.py` — test prompt mentions order types

**Changes:**

1. In `llm_base.py` `format_snapshot()`, after the portfolio section, add pending orders:
   ```python
   # Pending orders
   if snapshot.pending_orders:
       lines.append("Pending orders:")
       for po in snapshot.pending_orders:
           lines.append(f"  {po.symbol}: {po.order_type} {po.side} qty={po.quantity} "
                        f"limit={po.limit_price} stop={po.stop_price}")
   ```

2. In `llm_base.py` `parse_signals()`, add validation:
   ```python
   # Require limit_price for LIMIT orders
   if order_type == "LIMIT" and float(item.get("limit_price", 0.0)) <= 0:
       logger.warning("LIMIT order without limit_price, defaulting to MARKET")
       order_type = "MARKET"
   # Require stop_price for SL/SL_M orders
   if order_type in ("SL", "SL_M") and float(item.get("stop_price", 0.0)) <= 0:
       logger.warning("SL/SL_M order without stop_price, skipping")
       continue
   ```

3. In `llm_signal_generator.py`, expand `DEFAULT_SYSTEM_PROMPT` to include all order types, CANCEL action, limit/stop prices, CNC/MIS guidance (per design doc Section 4).

4. Tests:
   - `test_format_snapshot_includes_pending_orders` — snapshot with pending orders, verify text output
   - `test_parse_signals_limit_without_price_defaults_market` — LIMIT with no limit_price → MARKET
   - `test_parse_signals_slm_without_price_skipped` — SL_M with no stop_price → skipped
   - `test_build_prompt_mentions_limit_slm` — system prompt contains "LIMIT" and "SL_M"

**Verify:** `cd strategies && source .venv/bin/activate && pytest tests/test_llm_base.py tests/test_llm_signal_generator.py -v`

**Commit:** `feat: LLM strategy with full order type support and pending orders in snapshot`

---

### Task 2: SMA Crossover — Limit Entries + Engine Stops + Dynamic CNC/MIS

**Files:**
- Modify: `strategies/strategies/deterministic/sma_crossover.py`
- Modify: `strategies/tests/test_sma_crossover.py`

**Changes to SymbolState dataclass:**
```python
# Add to SymbolState:
pending_entry: bool = False       # limit entry outstanding
has_engine_stop: bool = False     # SL-M in engine
```

**Changes to on_bar — Entry logic:**
Replace market BUY with limit:
```python
# Instead of:
signals.append(Signal(action="BUY", symbol=symbol, quantity=qty, product_type="CNC"))

# New:
product = "CNC" if spread > 0.01 else "MIS"
signals.append(Signal(
    action="BUY", symbol=symbol, quantity=qty,
    order_type="LIMIT", limit_price=bar.close,
    product_type=product,
))
state.pending_entry = True
```

On each bar with `pending_entry=True`:
- Check `snapshot.fills` for entry fill → if filled, set `pending_entry=False`, submit SL-M stop
- If crossover reversed → emit CANCEL, set `pending_entry=False`
- If still valid → CANCEL old limit + submit new at current close

**Changes to on_bar — Stop management:**
On entry fill detected:
```python
signals.append(Signal(
    action="SELL", symbol=symbol, quantity=state.position_qty,
    order_type="SL_M", stop_price=state.trailing_stop,
    product_type=product,
))
state.has_engine_stop = True
```

Each bar with `has_engine_stop=True` and trailing stop moved:
```python
signals.append(Signal(action="CANCEL", symbol=symbol, quantity=0))
signals.append(Signal(
    action="SELL", symbol=symbol, quantity=state.position_qty,
    order_type="SL_M", stop_price=state.trailing_stop,
    product_type=product,
))
```

**Changes to on_bar — Exit logic:**
On death cross or time stop:
```python
# Cancel any pending orders first
signals.append(Signal(action="CANCEL", symbol=symbol, quantity=0))
state.has_engine_stop = False
# Then market sell
signals.append(Signal(action="SELL", symbol=symbol, quantity=state.position_qty, product_type=product))
```

On stop-hit detection (fill from SL-M):
```python
for fill in snapshot.fills:
    if fill.symbol == symbol and fill.side == "SELL" and state.has_engine_stop:
        state.has_engine_stop = False
        # Reset state — engine closed position
        self._reset_state(state)
```

Remove all software trailing stop checking (the `if bar.close < state.trailing_stop` code) — engine handles it now.

Same pattern for short side (mirror all logic).

**Tests:**
- `test_limit_entry_on_golden_cross` — verify LIMIT order emitted with limit_price=bar.close
- `test_cancel_unfilled_limit_on_reversal` — crossover reverses, verify CANCEL emitted
- `test_engine_stop_submitted_on_fill` — simulate fill in snapshot, verify SL-M emitted
- `test_trailing_stop_ratchet_cancel_resubmit` — verify CANCEL + new SL-M on stop update
- `test_dynamic_cnc_strong_trend` — spread > 1%, verify product_type="CNC"
- `test_dynamic_mis_weak_trend` — spread < 1%, verify product_type="MIS"
- `test_death_cross_cancels_all_pending` — verify CANCEL before SELL on death cross
- `test_stop_hit_detection_resets_state` — fill from SL-M, verify state reset

**Verify:** `cd strategies && pytest tests/test_sma_crossover.py -v`

**Commit:** `feat: SMA crossover with limit entries, engine stops, dynamic CNC/MIS`

---

### Task 3: RSI Daily Trend — Limit Entries + Engine Stops + Dynamic CNC/MIS

**Files:**
- Modify: `strategies/strategies/deterministic/rsi_daily_trend.py`
- Modify: `strategies/tests/test_rsi_daily_trend.py`

**Changes to SymbolState:**
```python
pending_entry_bar: int = 0    # bar when limit was submitted (cancel after 3 bars)
has_engine_stop: bool = False
```

**Entry changes:**
Replace market BUY with limit at 0.1% below:
```python
product = "CNC" if rsi < 25 and trend_up else "MIS"
signals.append(Signal(
    action="BUY", symbol=symbol, quantity=level_qty,
    order_type="LIMIT", limit_price=bar_close * 0.999,
    product_type=product,
))
state.pending_entry_bar = state.bar_count
```

Cancel stale entries after 3 bars:
```python
if state.pending_entry_bar > 0 and state.bar_count - state.pending_entry_bar > 3:
    signals.append(Signal(action="CANCEL", symbol=symbol, quantity=0))
    state.pending_entry_bar = 0
```

**Stop management:**
Same pattern as SMA — on fill detected, submit SL-M at max-loss level. Ratchet trailing stop via CANCEL + new SL-M.

On partial exit at RSI > 60:
```python
# Sell half
signals.append(Signal(action="SELL", symbol=symbol, quantity=current_qty // 2, product_type=product))
# Cancel old stop, submit new at breakeven
signals.append(Signal(action="CANCEL", symbol=symbol, quantity=0))
signals.append(Signal(
    action="SELL", symbol=symbol, quantity=remaining_qty,
    order_type="SL_M", stop_price=state.avg_entry_price,
    product_type=product,
))
```

Short entries: same pattern with SELL + SL-M buy. Always MIS for shorts.

Remove software stop-loss checking code.

**Tests:**
- `test_limit_entry_below_market` — verify limit_price = close * 0.999
- `test_cancel_stale_limit_after_3_bars` — verify CANCEL after 3 unfilled bars
- `test_engine_stop_on_entry_fill` — SL-M submitted at max_loss level
- `test_partial_exit_replaces_stop_at_breakeven` — CANCEL + new SL-M at avg_entry
- `test_dynamic_cnc_deep_oversold` — RSI < 25 in uptrend → CNC
- `test_dynamic_mis_moderate_oversold` — RSI < 35 → MIS
- `test_short_always_mis` — short entries always MIS

**Verify:** `cd strategies && pytest tests/test_rsi_daily_trend.py -v`

**Commit:** `feat: RSI strategy with limit entries, engine stops, dynamic CNC/MIS`

---

### Task 4: Donchian Breakout — Limit Profit Target + Engine Stops + Dynamic CNC/MIS

**Files:**
- Modify: `strategies/strategies/deterministic/donchian_breakout.py`
- Modify: `strategies/tests/test_donchian_breakout.py`

**Changes to per-symbol state:**
```python
# Add:
has_profit_target: bool = False
has_engine_stop: bool = False
```

**Entry stays MARKET** — breakouts need speed.

**On entry fill detected, submit two engine orders:**
```python
# 1. SL-M stop at trailing stop level
signals.append(Signal(
    action="SELL", symbol=symbol, quantity=state.position_qty,
    order_type="SL_M", stop_price=state.trailing_stop,
    product_type=product,
))
state.has_engine_stop = True

# 2. Limit sell for partial profit at entry + 2*ATR
partial_qty = max(1, state.position_qty // 3)
signals.append(Signal(
    action="SELL", symbol=symbol, quantity=partial_qty,
    order_type="LIMIT", limit_price=state.avg_entry + profit_target_atr * atr,
    product_type=product,
))
state.has_profit_target = True
```

**On partial profit fill detected:**
```python
for fill in snapshot.fills:
    if fill.symbol == symbol and fill.side == "SELL" and state.has_profit_target:
        state.has_profit_target = False
        state.partial_taken = True
        # Move stop to breakeven
        signals.append(Signal(action="CANCEL", symbol=symbol, quantity=0))
        remaining = state.position_qty  # after partial
        signals.append(Signal(
            action="SELL", symbol=symbol, quantity=remaining,
            order_type="SL_M", stop_price=state.avg_entry,
            product_type=product,
        ))
```

**Dynamic CNC/MIS:**
```python
volumes = list(state.daily_volumes)
avg_volume = sum(volumes[-(channel_period+1):-1]) / channel_period
volume_strong = volumes[-1] > avg_volume * 1.5
product = "CNC" if volume_strong else "MIS"
```

**Full exit (channel low, max loss, time stop):**
```python
signals.append(Signal(action="CANCEL", symbol=symbol, quantity=0))
state.has_engine_stop = False
state.has_profit_target = False
signals.append(Signal(action="SELL", symbol=symbol, quantity=held_qty, product_type=product))
```

Remove software trailing stop checking and software partial profit checking.

Same pattern for short side.

**Tests:**
- `test_limit_profit_target_on_entry` — verify LIMIT sell at entry + 2*ATR submitted after fill
- `test_slm_stop_on_entry` — verify SL-M at trailing stop level after fill
- `test_partial_fill_moves_stop_to_breakeven` — simulate profit target fill, verify CANCEL + SL-M at avg_entry
- `test_stop_ratchet_cancel_resubmit` — trailing stop moved, verify CANCEL + new SL-M
- `test_dynamic_cnc_strong_volume` — volume > 1.5x avg → CNC
- `test_dynamic_mis_normal_volume` — volume < 1.5x → MIS
- `test_full_exit_cancels_all_pending` — channel low exit, verify CANCEL before SELL
- `test_short_limit_cover_profit_target` — short side: LIMIT buy at entry - 2*ATR

**Verify:** `cd strategies && pytest tests/test_donchian_breakout.py -v`

**Commit:** `feat: Donchian breakout with limit profit targets, engine stops, dynamic CNC/MIS`

---

### Task 5: Final Verification — All Tests + Backtests

**Step 1:** Run all tests
```bash
cd strategies && source .venv/bin/activate && pytest tests/ -v
```
Expected: All tests pass.

**Step 2:** Run backtests on 10 stocks full year 2025
```bash
SYM="RELIANCE,INFY,TCS,BAJFINANCE,HINDUNILVR,BHARTIARTL,SBIN,ICICIBANK,HDFCBANK,ITC"
CLI=./engine/target/release/backtest

$CLI run --strategy sma_crossover --symbols $SYM --from 2025-01-01 --to 2025-12-31 --capital 1000000 --interval day --max-drawdown 0.15 --max-volume-pct 0.10 --max-exposure 0.80

$CLI run --strategy rsi_daily_trend --symbols $SYM --from 2025-01-01 --to 2025-12-31 --capital 1000000 --interval 15minute --max-drawdown 0.15 --max-volume-pct 0.10 --max-exposure 0.80

$CLI run --strategy donchian_breakout --symbols $SYM --from 2025-01-01 --to 2025-12-31 --capital 1000000 --interval 15minute --max-drawdown 0.15 --max-volume-pct 0.10 --max-exposure 0.80
```

**Step 3:** Compare results with previous baselines:
- SMA: was -0.07%, 37 trades
- RSI: was +49.86%, 14 trades (after tuning)
- Donchian: was +0.43%, 36 trades

Look for: fewer trades (limits cancel unfilled), better avg fill prices (limits vs market), similar or better returns, lower costs.

---

## Execution Order

```
Task 1 (LLM)         ── Independent, small scope
Task 2 (SMA)          ── Independent
Task 3 (RSI)          ── Independent
Task 4 (Donchian)     ── Independent
Task 5 (Verification) ── After 1-4
```

Tasks 1-4 are fully independent (different files). Can run in parallel.
