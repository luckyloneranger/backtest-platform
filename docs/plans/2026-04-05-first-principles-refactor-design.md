# First-Principles Refactor — Design Document

Date: 2026-04-05

## Context

Over multiple sessions, the strategies accumulated layered patches: original logic → advanced features → short selling → LIMIT/SL-M engine stops → DAY expiry handling. An audit revealed strategies are 93% infrastructure and 7% trading logic, with ~1,200 lines of duplicated order management code across 3 strategies, 8 bugs, and dead state fields.

The engine has been separately fixed (10 realism issues from first-principles audit). This design covers only the strategy rewrite.

## Goal

Rewrite all 3 deterministic strategies from scratch using a shared `PositionManager` class that handles all order lifecycle infrastructure. Strategies become 120-150 lines of pure trading logic. LLM strategy gets minor cleanup only.

## Architecture

```
Strategy (on_bar)                    PositionManager (shared)
  │                                    │
  ├─ compute indicators               ├─ process_fills(snapshot)
  ├─ detect entry signal               │   ├─ detect entry fill → submit SL-M
  │   └─ pm.enter_long(...)            │   ├─ detect stop hit → reset state
  ├─ detect exit signal                │   ├─ detect profit target fill → breakeven
  │   └─ pm.exit_position(...)         │   └─ detect expired pending → reset
  ├─ update trailing stop              ├─ resubmit_expired(snapshot)
  │   └─ pm.update_trailing_stop(...)  │   ├─ re-submit missing SL-M
  └─ return signals                    │   └─ re-submit missing LIMIT target
                                       ├─ reconcile(snapshot)
                                       └─ enter_long/short, exit, pyramid, etc.
```

## New File: `strategies/strategies/position_manager.py`

### PositionState dataclass

```python
@dataclass
class PositionState:
    direction: str = "flat"       # "flat", "long", "short"
    qty: int = 0
    avg_entry: float = 0.0
    entry_bar: int = 0
    trailing_stop: float = 0.0
    product_type: str = "CNC"
    has_engine_stop: bool = False
    has_profit_target: bool = False
    partial_taken: bool = False
    pending_entry: bool = False
    pending_side: str = ""        # "BUY" or "SELL"
    pending_qty: int = 0
    pending_bar: int = 0
    pyramid_count: int = 0
    original_qty: int = 0
    bars_held: int = 0
```

### PositionManager class

```python
class PositionManager:
    def __init__(self):
        self.states: dict[str, PositionState] = {}
        self.bar_count: int = 0

    # === Entry methods (return list[Signal]) ===

    def enter_long(self, symbol, qty, limit_price, product_type, stop_price) -> list[Signal]:
        """Submit LIMIT BUY entry. CNC shorts blocked — forces MIS for shorts."""
        # Sets pending state, returns LIMIT BUY signal

    def enter_short(self, symbol, qty, limit_price, stop_price) -> list[Signal]:
        """Submit LIMIT SELL entry. Always MIS (CNC shorts not allowed)."""

    def add_pyramid(self, symbol, qty, limit_price) -> list[Signal]:
        """Add to existing position. Returns LIMIT signal."""

    def set_profit_target(self, symbol, qty, limit_price) -> list[Signal]:
        """Set a resting LIMIT order for partial profit taking."""

    # === Exit methods ===

    def exit_position(self, symbol, qty=None) -> list[Signal]:
        """Full or partial exit. Emits CANCEL (clear pending) + MARKET order."""

    def update_trailing_stop(self, symbol, new_stop) -> list[Signal]:
        """Ratchet trailing stop. If moved, emits CANCEL + new SL-M."""

    # === Lifecycle methods (call each bar) ===

    def process_fills(self, snapshot) -> list[Signal]:
        """Detect fills in snapshot.fills:
        - Entry fill → set position state, submit SL-M stop
        - Pyramid fill → update avg_entry, resubmit SL-M for new total
        - Stop hit → reset state (engine closed position)
        - Profit target fill → set partial_taken, move stop to breakeven
        """

    def resubmit_expired(self, snapshot) -> list[Signal]:
        """DAY order expiry handling:
        - If has_engine_stop but no SL-M in pending_orders → re-submit
        - If has_profit_target but no LIMIT in pending_orders → re-submit
        - If pending_entry but no LIMIT in pending_orders and no fill → reset pending
        """

    def reconcile(self, snapshot):
        """Sync internal state with portfolio positions.
        If direction != flat but portfolio shows 0 qty → reset state.
        If portfolio qty differs from internal qty → update.
        """

    # === Query methods ===

    def is_flat(self, symbol) -> bool
    def is_long(self, symbol) -> bool
    def is_short(self, symbol) -> bool
    def position_qty(self, symbol) -> int
    def avg_entry_price(self, symbol) -> float
    def has_pending_entry(self, symbol) -> bool
    def get_state(self, symbol) -> PositionState
```

### Key design decisions:

1. **All methods return `list[Signal]`** — strategies collect them and return from on_bar
2. **enter_short always uses MIS** — CNC short restriction enforced at strategy level
3. **Entries use LIMIT orders** — except Donchian which calls a separate `enter_market_long/short` method for breakout entries
4. **Stale entry cancellation** — `process_fills` cancels pending entries after `max_pending_bars` (default 3 for 15-min, 1 for daily)
5. **Fill disambiguation** — detects which order filled by checking side + state (pending_entry vs has_engine_stop vs has_profit_target)
6. **bar_count incremented by strategies** — PM doesn't know the timeframe, strategies call `pm.bar_count += 1` each bar

## Strategy Rewrites

### SMA Crossover (~120 lines)

```
required_data: day, lookback 200
indicators: fast_sma, slow_sma, ATR (from shared indicators module)
state: per-symbol prev_fast_above, highest_since_entry, lowest_since_entry + price deques

on_bar:
  signals = pm.process_fills(snapshot) + pm.resubmit_expired(snapshot)
  pm.reconcile(snapshot)

  for symbol, bar in day_bars:
    compute sma, atr
    detect crossover

    if flat:
      golden_cross + spread > min_spread → pm.enter_long(qty, close, product, stop)
      death_cross + spread > min_spread → pm.enter_short(qty, close, stop)

    if long:
      update highest → pm.update_trailing_stop(highest - atr * mult)
      death_cross → pm.exit_position()
      time_stop → pm.exit_position()
      pyramid_condition → pm.add_pyramid(qty, close)

    if short:
      mirror of long

    prev[symbol] = fast_above
  return signals
```

### RSI Daily Trend (~150 lines)

```
required_data: 15minute lookback 100, day lookback 50
indicators: RSI (15m), EMA (daily), ATR (daily)
state: per-symbol prices_15m, prices_daily, ema_history, prev_rsi + daily OHLC deques

on_bar:
  update daily data + trend when daily bar arrives

  signals = pm.process_fills(snapshot) + pm.resubmit_expired(snapshot)
  pm.reconcile(snapshot)

  for symbol, bar in 15m_bars:
    compute rsi, atr
    target_qty = risk_pct * cash / price, level_qty = target / 3
    product = "CNC" if rsi < 25 and trend_up else "MIS"

    if flat:
      rsi < 35 and trend_up → pm.enter_long(level_qty, close*0.999, product, stop)
      rsi > 65 and trend_down → pm.enter_short(level_qty, close*1.001, stop)

    if long:
      pyramid: rsi < 25 and level < max → pm.add_pyramid(level_qty, close*0.999)
      partial: rsi > 60 → pm.exit_position(half)
      full: rsi > 70 or trend_reversed or time_stop → pm.exit_position()
      trailing: pm.update_trailing_stop(new_stop)

    if short: mirror
  return signals
```

### Donchian Breakout (~130 lines)

```
required_data: day lookback 60, 15minute lookback 30
indicators: ATR (daily), channel high/low (daily)
state: per-symbol daily OHLCV deques, highest/lowest_since_entry

on_bar:
  update daily data when daily bar arrives

  signals = pm.process_fills(snapshot) + pm.resubmit_expired(snapshot)
  pm.reconcile(snapshot)

  for symbol, bar in 15m_bars:
    compute channel, atr, avg_volume
    product = "CNC" if volume > 1.5x avg else "MIS"

    if flat:
      price > channel_high + volume → pm.enter_long(qty, MARKET, product, stop)
      price < channel_low + volume → pm.enter_short(qty, MARKET, stop)
      on entry: pm.set_profit_target(qty/3, entry + 2*atr)

    if long:
      trailing: pm.update_trailing_stop(highest - atr * 1.5)
      channel_low_exit or max_loss or time_stop → pm.exit_position()
      pyramid: price > avg_entry + atr → pm.add_pyramid(qty/2, MARKET)

    if short: mirror
  return signals
```

Note: Donchian uses MARKET for entries (breakouts need speed) while SMA/RSI use LIMIT. PositionManager supports both — `enter_long` with `limit_price=0` means MARKET order.

### LLM Signal Generator (minimal changes)

- No PositionManager — LLM controls its own orders via the full signal schema
- Remove dead `prev_rsi` style fields if any
- Ensure `format_snapshot` includes pending orders (already done)
- Add symbol/quantity validation in `parse_signals` (fix from audit)
- No structural rewrite needed — the LLM layer is already clean (~70 lines)

## Tests

### position_manager tests (`tests/test_position_manager.py`, ~20 tests):
- `test_enter_long_emits_limit_buy`
- `test_enter_short_forces_mis`
- `test_exit_emits_cancel_plus_market`
- `test_process_fills_detects_entry_fill`
- `test_process_fills_submits_stop_on_entry`
- `test_process_fills_detects_stop_hit`
- `test_process_fills_detects_profit_target_fill`
- `test_resubmit_expired_stop`
- `test_resubmit_expired_profit_target`
- `test_resubmit_expired_pending_entry`
- `test_reconcile_resets_on_empty_portfolio`
- `test_update_trailing_stop_ratchets`
- `test_update_trailing_stop_no_change`
- `test_add_pyramid_updates_avg_entry`
- `test_state_queries` (is_flat, is_long, is_short, etc.)

### Strategy tests (~8-10 tests each, focused on TRADING LOGIC not infrastructure):
- SMA: crossover detection, trend strength filter, pyramid condition, short entry
- RSI: RSI threshold entries, trend filter, partial/full exits, dynamic CNC/MIS
- Donchian: channel breakout, volume filter, profit target placement, channel low exit
- LLM: prompt content, signal validation

## Files

| Action | File |
|--------|------|
| CREATE | `strategies/strategies/position_manager.py` |
| CREATE | `strategies/tests/test_position_manager.py` |
| REWRITE | `strategies/strategies/deterministic/sma_crossover.py` |
| REWRITE | `strategies/strategies/deterministic/rsi_daily_trend.py` |
| REWRITE | `strategies/strategies/deterministic/donchian_breakout.py` |
| REWRITE | `strategies/tests/test_sma_crossover.py` |
| REWRITE | `strategies/tests/test_rsi_daily_trend.py` |
| REWRITE | `strategies/tests/test_donchian_breakout.py` |
| MODIFY | `strategies/strategies/llm_base.py` (signal validation fix) |
| MODIFY | `strategies/tests/test_llm_base.py` (validation test) |

## Execution Order

```
Task 1: PositionManager + tests        ── foundation, do first
Task 2: SMA rewrite + tests            ─┐
Task 3: RSI rewrite + tests            ─┤── parallel after Task 1
Task 4: Donchian rewrite + tests       ─┘
Task 5: LLM validation fix             ── independent, parallel with 2-4
Task 6: Backtests + verification       ── after all
```

## Verification

```bash
cd strategies && pytest tests/ -v       # all tests pass
# Run backtests on 10 stocks 2025
# Compare trade counts, returns vs previous versions
```

## Expected Outcomes

| Metric | Before | After |
|--------|--------|-------|
| Lines per strategy | 586-758 | 120-150 |
| Signal:infrastructure ratio | 7:93 | ~60:40 |
| Duplicated infrastructure | ~1,200 lines | 0 (shared PM) |
| State fields per symbol | 15-21 | 16 (shared) |
| Dead code / bugs | 8 issues | 0 |
| New strategy effort | ~400 lines boilerplate | ~20 lines (extend PM) |
