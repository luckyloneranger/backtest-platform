# Intraday 5-Min Strategies — Design Document

Date: 2026-04-05

## Goal

Build 2 intraday strategies that specifically exploit 5-minute candle patterns unavailable on daily/15-min bars: VWAP Mean Reversion and Bollinger Squeeze Breakout. Both are pure intraday MIS strategies with daily state resets.

## Why These Strategies

Our existing strategies are daily-signal strategies that use 15-min/5-min for entry timing. These new strategies compute indicators ON the 5-min bars themselves and exploit intraday microstructure:

- VWAP exists only intraday (resets daily) — institutional benchmark for execution quality
- Bollinger squeeze on 5-min captures 30-90 minute volatility cycles invisible on daily charts
- Both enter and exit same day — MIS product, auto-squareoff protection

## New Indicators

### compute_vwap (in indicators.py)

```python
def compute_vwap(highs: list[float], lows: list[float], closes: list[float],
                 volumes: list[int]) -> float | None:
    """Cumulative VWAP = sum(typical_price * volume) / sum(volume).
    Typical price = (high + low + close) / 3.
    Caller provides today's intraday bars only (VWAP resets daily)."""
```

### compute_vwap_bands (in indicators.py)

```python
def compute_vwap_bands(highs: list[float], lows: list[float], closes: list[float],
                       volumes: list[int], std_mult: float = 1.0) -> tuple[float, float, float] | None:
    """Returns (vwap, upper_band, lower_band).
    Bands = VWAP ± std_mult * std_dev of (typical_price - vwap)."""
```

## Strategy 1: VWAP Mean Reversion

**File:** `strategies/strategies/deterministic/vwap_reversion.py`
**Register:** `@register("vwap_reversion")`
**Timeframe:** 5minute only, lookback 100

### Daily Cycle:
1. **Warmup (9:15-9:45):** First 6 bars accumulate data, compute VWAP. No trading.
2. **Trading (9:45-15:00):** Check entry/exit each bar.
3. **Close (15:00):** Exit all positions before MIS squareoff.

### Data Management:
- Track today's bars per symbol: `today_highs`, `today_lows`, `today_closes`, `today_volumes`
- Detect new trading day via timestamp (IST hour resets to 9)
- Reset all daily state on new day: clear bar deques, reset `warmup_complete`, `trades_today`

### Entry:
- **Long:** close < VWAP - std_mult * std_dev → LIMIT buy at close
- **Short:** close > VWAP + std_mult * std_dev → LIMIT sell at close
- Skip if `trades_today >= max_trades_per_day`
- All entries MIS product type

### Exit:
- **Profit:** price reaches VWAP → pm.exit_position()
- **Stop:** price moves to VWAP - 2 * std_dev (longs) or VWAP + 2 * std_dev (shorts)
- **Time:** IST hour >= 15 → pm.exit_position() for all open positions

### Config:
- std_mult: 1.0
- risk_pct: 0.15
- warmup_bars: 6
- exit_time_hour: 15
- max_trades_per_day: 3

### PositionManager usage:
- pm.enter_long/short with LIMIT at the band level
- pm.exit_position on VWAP touch (profit), 2-sigma breach (stop), or time
- pm.update_trailing_stop not used (fixed stop at 2-sigma)
- Engine SL-M submitted at stop level on entry fill

## Strategy 2: Bollinger Squeeze Breakout

**File:** `strategies/strategies/deterministic/bollinger_squeeze.py`
**Register:** `@register("bollinger_squeeze")`
**Timeframe:** 5minute only, lookback 100

### Squeeze Detection:
- Compute BBW on 5-min bars every bar
- Track rolling avg_bbw over last 20 bars (deque)
- Squeeze = BBW < squeeze_threshold * avg_BBW (default 0.5)
- Set `squeeze_active = True` when detected

### Entry (on squeeze release):
- squeeze_active AND BBW > avg_BBW (expanding):
  - close > upper BB + volume > 1.5x avg → LIMIT long at close
  - close < lower BB + volume > 1.5x avg → LIMIT short at close
- After entry: `squeeze_active = False`
- Skip if `trades_today >= max_trades_per_day`
- All entries MIS

### Exit:
- **Profit target:** pm.set_profit_target at entry + 1.5 * ATR
- **Stop:** price returns to BB mid (SMA 20) → pm.exit_position (breakout failed)
- **Trailing stop:** pm.update_trailing_stop at 1x ATR from highest/lowest
- **Time:** IST hour >= 15 → pm.exit_position()

### Daily Reset:
- Same as VWAP: detect new day via timestamp, reset squeeze state, bar deques, trade counter
- Skip first warmup_bars for BB to stabilize

### Config:
- bb_period: 20
- bb_std: 2.0
- squeeze_threshold: 0.5
- volume_confirm: 1.5
- risk_per_trade: 0.015
- atr_period: 14
- profit_target_atr: 1.5
- atr_stop_mult: 1.0
- warmup_bars: 6
- exit_time_hour: 15
- max_trades_per_day: 3

## Shared Intraday Pattern

Both strategies share:
1. **Daily state reset** — detect new day via IST timestamp, reset bar deques + counters
2. **Warmup period** — first N bars no trading
3. **Time exit** — close positions at 15:00 IST
4. **MIS only** — all entries use product_type="MIS"
5. **Trade limits** — max_trades_per_day prevents overtrading
6. **IST time parsing** — extract hour/minute from timestamp_ms for trading window checks

This pattern can be a helper function or method in each strategy. Not extracted to a base class since there are only 2 strategies.

## Files

| Action | File |
|--------|------|
| MODIFY | `strategies/strategies/indicators.py` — add compute_vwap, compute_vwap_bands |
| CREATE | `strategies/strategies/deterministic/vwap_reversion.py` (~150 lines) |
| CREATE | `strategies/strategies/deterministic/bollinger_squeeze.py` (~160 lines) |
| MODIFY | `strategies/server/server.py` — add imports |
| CREATE | `strategies/tests/test_vwap_reversion.py` |
| CREATE | `strategies/tests/test_bollinger_squeeze.py` |
| MODIFY | `strategies/tests/test_indicators.py` — add VWAP tests |

## Implementation Order

```
Task 1: Add compute_vwap + compute_vwap_bands to indicators.py + tests
Task 2: VWAP Mean Reversion strategy + tests    ─┐ parallel after Task 1
Task 3: Bollinger Squeeze Breakout strategy + tests ─┘
Task 4: Run backtests on 5-min data for 2024 + 2025
```

## Verification

```bash
cd strategies && pytest tests/ -v
# Run on 5-min data:
backtest run --strategy vwap_reversion --symbols RELIANCE,INFY,TCS,BAJFINANCE,HINDUNILVR,BHARTIARTL,SBIN,ICICIBANK,HDFCBANK,ITC --from 2024-01-01 --to 2024-12-31 --capital 1000000 --interval 5minute --max-drawdown 0.20 --max-volume-pct 0.10
backtest run --strategy bollinger_squeeze --symbols ... --from 2024-01-01 --to 2024-12-31 --capital 1000000 --interval 5minute ...
```
