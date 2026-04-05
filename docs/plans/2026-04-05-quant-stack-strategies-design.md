# Quant Stack Strategies — Design Document

Date: 2026-04-05

## Goal

Add pandas + numpy + pandas-ta + scikit-learn + statsmodels to the platform. Upgrade `indicators.py` to use pandas-ta as the computation backend. Build 3 new sophisticated strategies: multi-indicator confluence, statistical pairs trading, and regime-adaptive. Keep existing strategies unchanged.

## Library Integration

### pyproject.toml additions:
```toml
dependencies = [
    # ... existing ...
    "pandas>=2.0",
    "numpy>=1.24",
    "pandas-ta>=0.3.14b1",
    "scikit-learn>=1.3",
    "statsmodels>=0.14",
]
```

### indicators.py overhaul

ALL indicator functions backed by pandas-ta internally. Existing function signatures unchanged (`list[float]` in, plain types out). Strategies never import pandas-ta directly.

**Existing functions (rewritten to use pandas-ta):**
- `compute_sma(prices, period)` → pandas rolling mean
- `compute_ema(prices, period)` → pandas-ta EMA
- `compute_rsi(prices, period)` → Wilder's RSI (fixes the Cutler's RSI variant issue from audit)
- `compute_atr(highs, lows, closes, period)` → pandas-ta ATR

**New functions:**
- `compute_macd(closes, fast=12, slow=26, signal=9) -> tuple[float, float, float] | None` — returns (macd_line, signal_line, histogram)
- `compute_bollinger(closes, period=20, std=2.0) -> tuple[float, float, float] | None` — returns (upper, mid, lower)
- `compute_adx(highs, lows, closes, period=14) -> float | None` — trend strength 0-100
- `compute_obv(closes, volumes) -> list[float] | None` — on-balance volume series
- `compute_obv_slope(closes, volumes, period=10) -> float | None` — OBV regression slope
- `compute_stochastic(highs, lows, closes, k=14, d=3) -> tuple[float, float] | None` — (%K, %D)
- `compute_bbw(closes, period=20, std=2.0) -> float | None` — Bollinger Band width (volatility)
- `compute_zscore(series, period=20) -> float | None` — rolling z-score
- `compute_correlation(series_a, series_b, period=30) -> float | None` — Pearson correlation
- `compute_cointegration(series_a, series_b) -> tuple[float, float] | None` — (p_value, hedge_ratio) via Engle-Granger
- `compute_halflife(series) -> int | None` — Ornstein-Uhlenbeck mean-reversion halflife

Pattern for each function:
```python
def compute_rsi(prices: list[float], period: int) -> float | None:
    if len(prices) < period + 1:
        return None
    s = pd.Series(prices)
    result = ta.rsi(s, length=period)
    if result is None or result.empty:
        return None
    val = result.iloc[-1]
    return None if pd.isna(val) else float(val)
```

## Strategy 1: Multi-Indicator Confluence

**File:** `strategies/strategies/deterministic/confluence.py`
**Register:** `@register("confluence")`
**Timeframe:** day, lookback 50

### Indicators (all from indicators.py):
- RSI (14): oversold < 35 = +1, overbought > 65 = -1
- MACD: histogram > 0 = +1, < 0 = -1
- Bollinger Bands: close < lower = +1, close > upper = -1
- ADX (14): > 25 confirms trend (amplifies trend signals)
- OBV slope (10): positive slope = +1, negative = -1

### Logic:
```
score = rsi_signal + macd_signal + bb_signal + obv_signal  # -4 to +4
ADX > 25 amplifies: if trending, double the macd_signal weight

if flat:
    score >= 3 → enter_long (3 of 4 indicators agree)
    score <= -3 → enter_short

if long:
    score <= 0 → exit (consensus lost)
    trailing stop via pm.update_trailing_stop

if short:
    score >= 0 → exit
```

### Config:
- `confluence_threshold`: 3 (min score to enter)
- `rsi_oversold/overbought`: 35/65
- `adx_threshold`: 25
- `risk_per_trade`: 0.02
- `atr_multiplier`: 2.0
- `max_hold_bars`: 30

### Expected characteristics:
- Fewer trades than single-indicator strategies (higher bar for entry)
- Higher win rate (multiple confirmations)
- Works in both trending and ranging (different indicators dominate)

## Strategy 2: Statistical Pairs Trading

**File:** `strategies/strategies/deterministic/pairs_trading.py`
**Register:** `@register("pairs_trading")`
**Timeframe:** day, lookback 60

### Pair Selection (in initialize or first bar):
- From all symbol pairs, compute pairwise cointegration (Engle-Granger test)
- Select pair with lowest p-value < 0.05
- Compute hedge ratio from cointegration regression
- If no cointegrated pair found, strategy stays flat

### Spread & Z-Score:
```
spread = price_A - hedge_ratio * price_B
zscore = compute_zscore(spread_history, period=20)
```

### Logic:
```
if flat:
    zscore > +2.0 → short A, long B (spread will converge)
    zscore < -2.0 → long A, short B

if in position:
    zscore crosses 0 → exit both legs (mean-reversion complete)
    |zscore| > 3.0 → exit both legs (stop-loss, spread diverging)
    bars_held > max_hold_bars → exit
```

### Position sizing:
- Equal dollar allocation per leg: `(capital * risk_pct / 2) / price` for each
- Always two signals per entry (one per leg)

### Config:
- `zscore_entry`: 2.0
- `zscore_exit`: 0.0
- `zscore_stop`: 3.0
- `lookback_period`: 60
- `zscore_period`: 20
- `risk_pct`: 0.3
- `max_hold_bars`: 30 (or computed from halflife)
- `min_cointegration_pvalue`: 0.05

### Expected characteristics:
- Market-neutral: long one stock, short another → hedged against market moves
- Profits in all market regimes (bull, bear, sideways)
- Requires at least 2 symbols; best with related stocks (INFY/TCS, ICICIBANK/HDFCBANK)

### PositionManager usage:
- Pairs trading manages TWO positions simultaneously
- Uses PositionManager for each leg independently
- Custom logic to coordinate entry/exit of both legs together

## Strategy 3: Regime-Adaptive

**File:** `strategies/strategies/deterministic/regime_adaptive.py`
**Register:** `@register("regime_adaptive")`
**Timeframe:** 15minute + day, lookback 50/100

### Regime Detection (daily bars):
```python
adx = compute_adx(highs, lows, closes, 14)
bbw = compute_bbw(closes, 20)
avg_bbw = compute_sma(bbw_history, 20)

if adx > 25:
    regime = "TRENDING"
elif bbw > avg_bbw * 1.5:
    regime = "VOLATILE"
else:
    regime = "RANGING"
```

### Sub-Strategies:

**TRENDING regime:**
- Entry: MACD cross + ADX > 25 + ADX rising
- Stop: 2x ATR trailing
- Pyramid: price > entry + ATR
- No time stop (let trends run)

**RANGING regime:**
- Entry: close < Bollinger lower + RSI < 35 (long) / close > upper + RSI > 65 (short)
- Exit: price reaches Bollinger mid (SMA 20) or opposite band
- Stop: 1x ATR (tight — small ranges)
- No pyramiding

**VOLATILE regime:**
- Default: stay flat
- Exception: enter at 50% normal size with 3x ATR stops IF MACD + RSI + OBV all agree
- Very selective — most bars produce no trades

### Regime Transition:
- When regime changes, existing positions stay open
- Stop parameters adjust to new regime's settings
- TRENDING → VOLATILE: tighten stop to 1.5x ATR
- RANGING → TRENDING: widen stop to 2x ATR

### Config:
- `adx_trend`: 25
- `adx_range`: 20
- `bbw_volatile_mult`: 1.5
- `trending_atr_mult`: 2.0
- `ranging_atr_mult`: 1.0
- `volatile_size_pct`: 0.5
- `risk_per_trade`: 0.02
- `pyramid_levels`: 2

### Expected characteristics:
- Adapts to market conditions automatically
- Should perform consistently across 2024 (trending) and 2025 (choppy)
- More complex but solves the regime-dependency problem seen in our backtests

## Files

| Action | File |
|--------|------|
| MODIFY | `strategies/pyproject.toml` — add pandas, numpy, pandas-ta, scikit-learn, statsmodels |
| REWRITE | `strategies/strategies/indicators.py` — pandas-ta backed, add ~10 new functions |
| CREATE | `strategies/strategies/deterministic/confluence.py` (~130 lines) |
| CREATE | `strategies/strategies/deterministic/pairs_trading.py` (~160 lines) |
| CREATE | `strategies/strategies/deterministic/regime_adaptive.py` (~180 lines) |
| MODIFY | `strategies/server/server.py` — add imports for new strategies |
| CREATE | `strategies/tests/test_confluence.py` |
| CREATE | `strategies/tests/test_pairs_trading.py` |
| CREATE | `strategies/tests/test_regime_adaptive.py` |
| MODIFY | `strategies/tests/test_indicators.py` or existing indicator tests — update for pandas-ta backend |

## Implementation Order

```
Task 1: Library integration + indicators.py overhaul + tests
Task 2: Confluence strategy + tests              ─┐
Task 3: Pairs trading strategy + tests            ─┤── parallel after Task 1
Task 4: Regime-adaptive strategy + tests          ─┘
Task 5: Integration test — run all 6 strategies on 2024+2025 data
```

Task 1 must complete first (indicators is a dependency). Tasks 2-4 are independent.

## Verification

```bash
pip install -e ".[dev]"
pytest tests/ -v                    # all tests pass
# Run all 6 strategies on 10 stocks 2024 + 2025
```

## Expected Outcome

| Strategy | Type | 2024 Expected | 2025 Expected |
|----------|------|---------------|---------------|
| SMA | Trend | Low activity | Low activity |
| RSI | Mean-reversion | Poor (no dips) | Good (Feb dip) |
| Donchian | Trend-following | Good (trends) | Poor (choppy) |
| **Confluence** | Multi-signal | Moderate (selective) | Moderate (selective) |
| **Pairs** | Market-neutral | Steady (hedged) | Steady (hedged) |
| **Regime** | Adaptive | Good (detects trends) | Good (detects chop) |
