# 6 Strategies for Consistent Profitability — Design Document

Date: 2026-04-06

## Goal

Build 6 new strategies covering untapped approaches: ORB, portfolio combination, intraday momentum, time-of-day patterns, relative strength, and multi-timeframe confirmation. These address the regime-dependency problem (strategies that win in 2024 lose in 2025 and vice versa).

## Strategy Designs

### 1. Opening Range Breakout (ORB)
**File:** `strategies/deterministic/orb_breakout.py`
**Register:** `@register("orb_breakout")`
**Interval:** 5minute, lookback 50

- First 6 bars (9:15-9:45) define opening range: `range_high = max(highs)`, `range_low = min(lows)`
- Break above range_high with volume > 1.5x avg → long at MARKET. Stop at range_low. Target at entry + 2*(range_high - range_low).
- Break below range_low with volume → short (MIS). Stop at range_high. Target at entry - 2*range_width.
- Max 1 trade/day/symbol. Exit at 15:00. All MIS.
- Uses PositionManager for entries/exits/stops.

### 2. Portfolio Combiner (Donchian + RSI Dynamic Allocation)
**File:** `strategies/deterministic/portfolio_combiner.py`
**Register:** `@register("portfolio_combiner")`
**Interval:** 15minute + day, lookback 100/50

- Computes ADX on daily bars per symbol.
- ADX > 25 (trending): Donchian-style channel breakout entry. ATR trailing stop. CNC.
- ADX < 20 (ranging): RSI mean-reversion entry (RSI < 35 in uptrend). Tighter stop. MIS.
- ADX 20-25 (neutral): half size from whichever signal fires.
- Combines best of both worlds — trend-following when trending, mean-reversion when ranging.
- Single strategy, not two running separately.

### 3. Intraday Momentum
**File:** `strategies/deterministic/intraday_momentum.py`
**Register:** `@register("intraday_momentum")`
**Interval:** 5minute, lookback 50

- Detects momentum burst: 3-bar price move > 1.5x ATR with volume > 2x avg.
- Enter in move direction at MARKET. Trail stop at 1x ATR.
- Pure trend-following on 5-min — no mean-reversion.
- Max 2 trades/day. Exit at 15:00. All MIS.

### 4. Time-of-Day Adaptive
**File:** `strategies/deterministic/time_adaptive.py`
**Register:** `@register("time_adaptive")`
**Interval:** 5minute, lookback 100

- 9:15-10:15 (opening): Momentum mode — if first-hour direction is up (close > VWAP), enter long on pullback to VWAP.
- 10:15-14:00 (midday): Mean-reversion mode — fade moves >1 std from VWAP, exit at VWAP.
- 14:00-15:00 (closing): Momentum mode — enter in direction of last hour's trend.
- VWAP as anchor for all modes. Computed from today's bars.
- Max 2 trades/day. Exit at 15:00. All MIS.

### 5. Relative Strength Rotation
**File:** `strategies/deterministic/relative_strength.py`
**Register:** `@register("relative_strength")`
**Interval:** 15minute + day, lookback 30

- At 9:45 daily (after warmup), rank all symbols by first-30-min return.
- Long top 3 (strongest momentum), short bottom 3 (weakest).
- Equal dollar allocation: (capital * risk_pct / 6) per leg.
- Exit all at 15:00. All MIS. Re-rank next day.
- Market-neutral: always 3 long + 3 short.

### 6. Multi-Timeframe Confirmation
**File:** `strategies/deterministic/multi_tf_confirm.py`
**Register:** `@register("multi_tf_confirm")`
**Interval:** 5minute + 15minute + day, lookback 50/30/20

- Daily EMA(20): determines direction. Only long if close > EMA. Only short if close < EMA.
- 15-min MACD: confirms entry. MACD histogram crosses positive (long) or negative (short).
- 5-min RSI(14): times entry. Enter when RSI < 35 (long oversold in uptrend) or RSI > 65 (short overbought in downtrend).
- All three must agree → entry. Any level disagrees → exit.
- CNC if daily trend strong, MIS otherwise.

## Files

| Action | File |
|--------|------|
| CREATE | `strategies/deterministic/orb_breakout.py` |
| CREATE | `strategies/deterministic/portfolio_combiner.py` |
| CREATE | `strategies/deterministic/intraday_momentum.py` |
| CREATE | `strategies/deterministic/time_adaptive.py` |
| CREATE | `strategies/deterministic/relative_strength.py` |
| CREATE | `strategies/deterministic/multi_tf_confirm.py` |
| CREATE | Tests for each |
| MODIFY | `strategies/server/server.py` — add imports |

## Implementation Order

All 6 are independent files. Can run in parallel after design approval.

## Verification

Run all on 10 stocks for 2024 + 2025 at both ₹1L and ₹10L capital.
