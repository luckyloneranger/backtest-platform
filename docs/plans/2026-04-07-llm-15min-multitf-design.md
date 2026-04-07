# LLM Autonomous Trader — 15-Min Multi-Timeframe Redesign

Date: 2026-04-07

## Goal

Redesign the LLM autonomous trader from daily-only to 15-minute multi-timeframe execution, using ALL 17 indicator functions from `indicators.py`. The current version uses 11/17 and misses the 5 most sophisticated: VWAP, VWAP bands, cointegration, halflife, z-score.

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │         LLM Autonomous Trader        │
                    │   required_data: 15min + day         │
                    └──────────────┬──────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                     │
    ┌─────────▼──────────┐  ┌─────▼──────────┐  ┌──────▼───────────┐
    │ Daily Context       │  │ Intraday Context│  │ Cross-Stock      │
    │ - Trend (SMA/EMA)   │  │ - VWAP position │  │ - Correlation    │
    │ - Momentum (RSI/    │  │ - VWAP bands    │  │ - Cointegration  │
    │   MACD/Stoch)       │  │ - Intraday ATR  │  │ - Relative Z-    │
    │ - Regime (ADX/BBW)  │  │ - 15m RSI/MACD  │  │   scores         │
    │ - Volatility (BB)   │  │ - Volume profile │  │ - Halflife       │
    │ - OBV slope         │  │ - Session gaps   │  │                  │
    └────────────────────┘  └────────────────┘  └──────────────────┘
              │                    │                     │
              └────────────────────┼─────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │      Narrative Builder (Enhanced)     │
                    │  6 sections: Portfolio, Regime, Daily,│
                    │    Intraday, Cross-Stock, History     │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │         LLM (Azure OpenAI)           │
                    │  Thesis-driven decision framework    │
                    └─────────────────────────────────────┘
```

## All 17 Indicators — Where They Appear

| # | Function | Used Before? | Where in New Narrative | Example |
|---|----------|-------------|----------------------|---------|
| 1 | `compute_sma` | Yes | Daily symbol section | "20-day SMA: ₹1,238" |
| 2 | `compute_ema` | Yes | Daily symbol section | "+1.2% from 20-day EMA" |
| 3 | `compute_rsi` | Yes | Daily + 15-min | "Daily RSI 42. 15-min RSI 56." |
| 4 | `compute_atr` | Yes | Daily + 15-min | "Daily ATR ₹28.50, 15-min ATR ₹4.20" |
| 5 | `compute_macd` | Yes | Daily + 15-min | "Daily MACD hist -0.42 falling. 15-min hist +0.15 rising" |
| 6 | `compute_bollinger` | Yes | Daily symbol section | "BB position 68th percentile" |
| 7 | `compute_adx` | Yes | Regime section | "Average ADX: 24.5" |
| 8 | `compute_obv` | **NO** | Via obv_slope (raw OBV not useful as text) | N/A — slope is more informative |
| 9 | `compute_obv_slope` | Yes | Daily symbol section | "OBV rising" |
| 10 | `compute_stochastic` | Yes | Daily symbol section | "Stochastic %K=35 %D=38" |
| 11 | `compute_bbw` | Yes | Regime section | "BBW at average levels" |
| 12 | **`compute_zscore`** | **NO** | Daily symbol + cross-stock | "20-day z-score: +1.8σ" |
| 13 | `compute_correlation` | Yes (autocorr only) | **Cross-stock section** | "SBIN-ICICIBANK: ρ=0.85" |
| 14 | **`compute_cointegration`** | **NO** | Cross-stock section | "Cointegrated (p=0.02, hedge=0.73)" |
| 15 | **`compute_halflife`** | **NO** | Cross-stock section | "Spread halflife: 8 bars" |
| 16 | **`compute_vwap`** | **NO** | Intraday section | "VWAP: ₹1,238. Price 0.6% above" |
| 17 | **`compute_vwap_bands`** | **NO** | Intraday section | "Upper band ₹1,252, lower ₹1,224" |

Note: `compute_obv` (raw series) is used indirectly via `compute_obv_slope`. Raw OBV values are not meaningful as narrative text — the slope conveys the signal. That's 16 of 17 directly used, with #8 used indirectly.

## New Narrative Builder Functions

### `build_intraday_narrative(symbol, intraday_data, daily_close)`

Presents intraday context from today's 15-min bars:

```
RELIANCE INTRADAY (15-min):
  VWAP: ₹1,238.50. Price at ₹1,245 (+0.5% above VWAP).
  VWAP bands (1σ): Upper ₹1,252, Lower ₹1,224. Price at 68th percentile.
  Session gap: +0.8% from yesterday's close (₹1,235).
  15-min RSI: 56. 15-min MACD histogram: +0.15, rising.
  Intraday ATR: ₹4.20.
  Volume profile: 12 bars today, avg volume 45K (vs daily avg 380K — on pace for 90% of typical).
```

### `build_cross_stock_narrative(all_daily_closes, symbols)`

Cross-stock statistical analysis (computed once per day):

```
CROSS-STOCK ANALYSIS:
  Strongest correlations (30-day):
    SBIN-ICICIBANK: ρ=0.85
    TCS-INFY: ρ=0.82
    HDFCBANK-KOTAKBANK: ρ=0.78

  Cointegrated pairs (p < 0.05):
    SBIN-ICICIBANK: p=0.02, hedge ratio 0.73, spread z-score +2.1
      Spread halflife: 8 bars (mean-reversion in ~8 trading days)
    TCS-INFY: p=0.04, hedge ratio 1.12, spread z-score -0.4

  Relative z-scores (most extended):
    Most overbought: BAJFINANCE z=+2.3, RELIANCE z=+1.8
    Most oversold: SBIN z=-1.9, ITC z=-1.5
    Near mean: TCS z=-0.1, INFY z=+0.3
```

### Enhanced `build_symbol_narrative`

Adds z-score to existing factual narrative:

```
  Statistical position: 20-day z-score +1.8σ (1.8 standard deviations above rolling mean).
```

## LLM Call Throttling

At 15-min bars: ~25 bars/day × 250 days = 6,250 bars/year.

| Mode | Frequency | Calls/Year | Runtime (~1s/call) |
|------|-----------|-----------|-------------------|
| Conservative | Every 8 bars (2 hours) | ~750 | ~12 min |
| **Default** | **Every 4 bars (1 hour)** | **~1,500** | **~25 min** |
| Aggressive | Every 2 bars (30 min) | ~3,000 | ~50 min |
| Ultra | Every bar (15 min) | ~6,250 | ~100 min |

Default: `llm_interval_bars=4` (once per hour). PositionManager runs every bar for fill processing, stop management, and order lifecycle.

## Buffer Management

```python
# Daily buffers — append once per day when daily bar arrives (maxlen=300)
self.daily_closes[symbol]
self.daily_highs[symbol]
self.daily_lows[symbol]
self.daily_volumes[symbol]

# Intraday buffers — reset each trading day (for VWAP calculation)
self.intraday_closes[symbol]  # today's 15-min bars only
self.intraday_highs[symbol]
self.intraday_lows[symbol]
self.intraday_volumes[symbol]

# 15-min rolling buffers — don't reset (for RSI, MACD, ATR on 15-min)
self.m15_closes[symbol]  # (maxlen=200)
self.m15_highs[symbol]
self.m15_lows[symbol]
self.m15_volumes[symbol]
```

Day detection: compare current bar's date with previous bar's date. On new day, reset intraday buffers.

## Cross-Stock Analysis (Daily Only)

Computed once per day when the daily bar arrives (too expensive for intraday):

1. **Correlation matrix**: `compute_correlation(closes_a, closes_b, 30)` for all pairs with n≥30 bars
2. **Cointegration**: `compute_cointegration(closes_a, closes_b)` only for pairs with |ρ| > 0.7
3. **Halflife**: `compute_halflife(spread)` only for cointegrated pairs (p < 0.05)
4. **Relative z-scores**: `compute_zscore(closes, 20)` for all symbols, rank by extremity

With 12 stocks: 66 pairs for correlation, typically 5-10 pairs for cointegration test, 2-3 for halflife. Total: ~1 second per day.

## System Prompt Changes

The thesis-driven framework (THESIS → EVIDENCE → COUNTER-THESIS → CONVICTION → ACTION) stays identical. Only the data context description is added:

```
You receive a multi-timeframe dashboard updated every hour:
- PORTFOLIO: Current holdings, P&L, drawdown, costs
- MARKET REGIME: ADX trend strength, Bollinger Band Width volatility
- PER-STOCK DAILY: Trend, momentum, volatility, volume, returns, z-score
- PER-STOCK INTRADAY: VWAP position, intraday momentum, session context
- CROSS-STOCK: Correlations, cointegrated pairs, relative z-scores
- TRADE HISTORY: Recent outcomes, win rate, profit factor

Use daily context for DIRECTION (which side of the market).
Use intraday context for TIMING (when to enter/exit).
Use cross-stock analysis for SELECTION (which stocks, pair trades).
```

## Config Parameters

| Param | Default | Description |
|-------|---------|-------------|
| `temperature` | 0.3 | LLM temperature |
| `max_tokens` | 1024 | LLM max completion tokens |
| `max_positions` | 4 | Max simultaneous positions |
| `max_daily_trades` | 3 | Max new entries per day |
| `risk_pct` | 0.03 | Max capital risked per trade |
| `auto_stop_pct` | 0.03 | Fallback stop distance if LLM omits |
| `llm_interval_bars` | 4 | Call LLM every N 15-min bars (4=hourly) |

## Files

| Action | File | Lines (est) |
|--------|------|-------------|
| REWRITE | `strategies/strategies/narrative_builder.py` | ~400 |
| REWRITE | `strategies/strategies/llm/llm_autonomous_trader.py` | ~500 |
| UPDATE | `strategies/tests/test_narrative_builder.py` | +100 |
| UPDATE | `strategies/tests/test_llm_autonomous_trader.py` | +80 |

## Verification

```bash
cd strategies && pytest tests/ -v

# 15-min backtest (2024):
backtest run --strategy llm_autonomous_trader \
  --symbols RELIANCE,INFY,TCS,BAJFINANCE,HINDUNILVR,BHARTIARTL,SBIN,ICICIBANK,HDFCBANK,ITC,KOTAKBANK,LT \
  --from 2024-01-01 --to 2024-12-31 --capital 100000 --interval 15minute

# 15-min backtest (2025):
backtest run --strategy llm_autonomous_trader \
  --symbols RELIANCE,INFY,TCS,BAJFINANCE,HINDUNILVR,BHARTIARTL,SBIN,ICICIBANK,HDFCBANK,ITC,KOTAKBANK,LT \
  --from 2025-01-01 --to 2025-12-31 --capital 100000 --interval 15minute
```

Compare against daily-only version (2024: +33.35%, 2025: +0.59%).
