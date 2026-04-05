# Quant Stack Strategies — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add pandas/numpy/pandas-ta/scikit-learn/statsmodels, rewrite indicators.py with pandas-ta backend, build 3 new strategies (confluence, pairs trading, regime-adaptive).

**Architecture:** indicators.py is the single gateway to pandas-ta — strategies never import pandas-ta directly. All indicator functions accept `list[float]` and return plain Python types. New strategies use PositionManager for order lifecycle. Existing strategies unchanged.

**Tech Stack:** Python 3.11+, pandas, numpy, pandas-ta, statsmodels, scikit-learn

**Design doc:** `docs/plans/2026-04-05-quant-stack-strategies-design.md`

---

### Task 1: Library Integration + indicators.py Overhaul

**Files:**
- Modify: `strategies/pyproject.toml` — add dependencies
- Rewrite: `strategies/strategies/indicators.py` — pandas-ta backend for all functions
- Modify: `strategies/tests/test_rsi_daily_trend.py` — update indicator test assertions (Wilder's RSI values differ from Cutler's)
- Modify: `strategies/tests/test_donchian_breakout.py` — update ATR test if values change

**Changes:**

1. **pyproject.toml** — add to dependencies:
   ```toml
   "pandas>=2.0",
   "numpy>=1.24",
   "pandas-ta>=0.3.14b1",
   "scikit-learn>=1.3",
   "statsmodels>=0.14",
   ```

2. **indicators.py** — complete rewrite. All functions use pandas-ta internally:

   **Existing functions (same signatures, pandas-ta backend):**
   ```python
   import pandas as pd
   import pandas_ta as ta
   import numpy as np

   def compute_sma(prices: list[float], period: int) -> float | None:
       if len(prices) < period:
           return None
       s = pd.Series(prices)
       result = s.rolling(period).mean()
       val = result.iloc[-1]
       return None if pd.isna(val) else float(val)

   def compute_ema(prices: list[float], period: int) -> float | None:
       if len(prices) < period:
           return None
       s = pd.Series(prices)
       result = ta.ema(s, length=period)
       if result is None or result.empty:
           return None
       val = result.iloc[-1]
       return None if pd.isna(val) else float(val)

   def compute_rsi(prices: list[float], period: int) -> float | None:
       if len(prices) < period + 1:
           return None
       s = pd.Series(prices)
       result = ta.rsi(s, length=period)
       if result is None or result.empty:
           return None
       val = result.iloc[-1]
       return None if pd.isna(val) else float(val)

   def compute_atr(highs: list[float], lows: list[float], closes: list[float], period: int) -> float | None:
       if len(highs) < period + 1:
           return None
       h, l, c = pd.Series(highs), pd.Series(lows), pd.Series(closes)
       result = ta.atr(h, l, c, length=period)
       if result is None or result.empty:
           return None
       val = result.iloc[-1]
       return None if pd.isna(val) else float(val)
   ```

   **New functions:**
   ```python
   def compute_macd(closes: list[float], fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[float, float, float] | None:
       """Returns (macd_line, signal_line, histogram) or None."""

   def compute_bollinger(closes: list[float], period: int = 20, std: float = 2.0) -> tuple[float, float, float] | None:
       """Returns (upper, mid, lower) or None."""

   def compute_adx(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> float | None:
       """Returns ADX value (0-100) or None."""

   def compute_obv(closes: list[float], volumes: list[int]) -> list[float] | None:
       """Returns OBV series or None."""

   def compute_obv_slope(closes: list[float], volumes: list[int], period: int = 10) -> float | None:
       """Returns OBV regression slope over last N bars or None."""

   def compute_stochastic(highs: list[float], lows: list[float], closes: list[float], k: int = 14, d: int = 3) -> tuple[float, float] | None:
       """Returns (%K, %D) or None."""

   def compute_bbw(closes: list[float], period: int = 20, std: float = 2.0) -> float | None:
       """Returns Bollinger Band Width (upper-lower)/mid or None."""

   def compute_zscore(series: list[float], period: int = 20) -> float | None:
       """Returns rolling z-score of last value or None."""

   def compute_correlation(series_a: list[float], series_b: list[float], period: int = 30) -> float | None:
       """Returns Pearson correlation over last N values or None."""

   def compute_cointegration(series_a: list[float], series_b: list[float]) -> tuple[float, float] | None:
       """Returns (p_value, hedge_ratio) via Engle-Granger test or None. Uses statsmodels."""

   def compute_halflife(series: list[float]) -> int | None:
       """Returns mean-reversion halflife in bars via OLS on lagged spread. Uses statsmodels."""
   ```

3. **Install and verify:**
   ```bash
   cd strategies && pip install -e ".[dev]"
   ```

4. **Update existing tests:** RSI values will change (Wilder's vs Cutler's). The test `test_compute_rsi` currently checks relative properties (uptrend → high, downtrend → low, insufficient data → None). These should still pass. If any test checks exact RSI values, update the assertion.

**Tests to add/update:**
- `test_compute_macd` — known input, verify (macd, signal, hist) tuple
- `test_compute_bollinger` — verify (upper, mid, lower) with mid ≈ SMA
- `test_compute_adx` — trending series → high ADX, flat series → low ADX
- `test_compute_obv_slope` — rising prices + rising volume → positive slope
- `test_compute_zscore` — known series, verify z-score at last value
- `test_compute_cointegration` — two correlated series → low p-value
- `test_compute_halflife` — mean-reverting series → reasonable halflife
- `test_sma_matches_pandas` — verify our SMA matches pandas rolling
- `test_rsi_wilder` — verify RSI is Wilder's (not Cutler's SMA variant)
- All existing indicator tests still pass

**Verify:** `cd strategies && pytest tests/ -v`

**Commit:** `feat: pandas-ta backed indicators with 11 new functions`

---

### Task 2: Confluence Strategy

**Files:**
- Create: `strategies/strategies/deterministic/confluence.py` (~130 lines)
- Create: `strategies/tests/test_confluence.py`
- Modify: `strategies/server/server.py` — add import

**Strategy implementation:**

```python
"""Multi-Indicator Confluence — trade only when 3+ indicators agree."""

from collections import deque
from server.registry import register
from strategies.base import Strategy, MarketSnapshot, InstrumentInfo, Signal
from strategies.indicators import (
    compute_rsi, compute_macd, compute_bollinger,
    compute_adx, compute_obv_slope, compute_atr,
)
from strategies.position_manager import PositionManager


@register("confluence")
class Confluence(Strategy):

    def required_data(self):
        return [{"interval": "day", "lookback": 50}]

    def initialize(self, config, instruments):
        self.confluence_threshold = config.get("confluence_threshold", 3)
        self.rsi_oversold = config.get("rsi_oversold", 35)
        self.rsi_overbought = config.get("rsi_overbought", 65)
        self.adx_threshold = config.get("adx_threshold", 25)
        self.risk_per_trade = config.get("risk_per_trade", 0.02)
        self.atr_multiplier = config.get("atr_multiplier", 2.0)
        self.max_hold_bars = config.get("max_hold_bars", 30)
        self.instruments = instruments
        self.pm = PositionManager(max_pending_bars=1)

        # Per-symbol indicator data
        self.closes: dict[str, deque] = {}
        self.highs: dict[str, deque] = {}
        self.lows: dict[str, deque] = {}
        self.volumes: dict[str, deque] = {}
        self.highest: dict[str, float] = {}
        self.lowest: dict[str, float] = {}

    def on_bar(self, snapshot):
        self.pm.increment_bars()
        signals = self.pm.process_fills(snapshot)
        signals += self.pm.resubmit_expired(snapshot)
        self.pm.reconcile(snapshot)

        for interval, bars in snapshot.timeframes.items():
            for symbol, bar in bars.items():
                # Update deques, compute all indicators
                # Compute confluence score (-4 to +4)
                # Entry/exit based on score vs threshold
                # Use PositionManager for all orders
                ...

        return signals
```

**Tests (~10):**
- test_required_data
- test_high_confluence_long_entry (score >= 3)
- test_high_confluence_short_entry (score <= -3)
- test_low_confluence_no_entry (score between -2 and +2)
- test_score_flip_exits_position
- test_adx_amplifies_trend_signal
- test_trailing_stop_exit
- test_time_stop_exit
- test_position_sizing_atr_based

**server.py:** Add `import strategies.deterministic.confluence  # noqa: F401`

**Verify:** `cd strategies && pytest tests/test_confluence.py -v`

**Commit:** `feat: multi-indicator confluence strategy`

---

### Task 3: Pairs Trading Strategy

**Files:**
- Create: `strategies/strategies/deterministic/pairs_trading.py` (~160 lines)
- Create: `strategies/tests/test_pairs_trading.py`
- Modify: `strategies/server/server.py` — add import

**Strategy implementation:**

```python
"""Statistical Pairs Trading — trade mean-reversion of cointegrated spread."""

from collections import deque
from server.registry import register
from strategies.base import Strategy, MarketSnapshot, InstrumentInfo, Signal
from strategies.indicators import compute_cointegration, compute_zscore, compute_halflife
from strategies.position_manager import PositionManager


@register("pairs_trading")
class PairsTrading(Strategy):

    def required_data(self):
        return [{"interval": "day", "lookback": 60}]

    def initialize(self, config, instruments):
        self.zscore_entry = config.get("zscore_entry", 2.0)
        self.zscore_exit = config.get("zscore_exit", 0.0)
        self.zscore_stop = config.get("zscore_stop", 3.0)
        self.lookback = config.get("lookback_period", 60)
        self.zscore_period = config.get("zscore_period", 20)
        self.risk_pct = config.get("risk_pct", 0.3)
        self.max_hold_bars = config.get("max_hold_bars", 30)
        self.min_pvalue = config.get("min_cointegration_pvalue", 0.05)
        self.instruments = instruments

        self.pm_a = PositionManager(max_pending_bars=1)  # leg A
        self.pm_b = PositionManager(max_pending_bars=1)  # leg B

        # Pair state
        self.pair: tuple[str, str] | None = None
        self.hedge_ratio: float = 0.0
        self.spread_history: deque = deque(maxlen=100)
        self.prices: dict[str, deque] = {}
        self.pair_selected: bool = False
        self.in_trade: bool = False
        self.trade_bar: int = 0
        self.bar_count: int = 0

    def on_bar(self, snapshot):
        self.bar_count += 1
        self.pm_a.increment_bars()
        self.pm_b.increment_bars()
        signals = self.pm_a.process_fills(snapshot)
        signals += self.pm_b.process_fills(snapshot)
        signals += self.pm_a.resubmit_expired(snapshot)
        signals += self.pm_b.resubmit_expired(snapshot)
        self.pm_a.reconcile(snapshot)
        self.pm_b.reconcile(snapshot)

        # Update price history for all symbols
        # Select pair on first bar with enough data
        # Compute spread and z-score each bar
        # Entry/exit based on z-score thresholds
        # Two PositionManagers: one per leg
        ...

        return signals
```

**Key complexity:** Two PositionManagers for two legs. Entry/exit signals come in pairs. The strategy must coordinate both legs.

**Tests (~10):**
- test_required_data
- test_pair_selection_cointegrated
- test_no_pair_when_not_cointegrated
- test_long_spread_entry (zscore > +2)
- test_short_spread_entry (zscore < -2)
- test_exit_on_mean_reversion (zscore crosses 0)
- test_stop_on_divergence (|zscore| > 3)
- test_time_stop_exit
- test_equal_dollar_sizing
- test_both_legs_exit_together

**server.py:** Add `import strategies.deterministic.pairs_trading  # noqa: F401`

**Verify:** `cd strategies && pytest tests/test_pairs_trading.py -v`

**Commit:** `feat: statistical pairs trading strategy`

---

### Task 4: Regime-Adaptive Strategy

**Files:**
- Create: `strategies/strategies/deterministic/regime_adaptive.py` (~180 lines)
- Create: `strategies/tests/test_regime_adaptive.py`
- Modify: `strategies/server/server.py` — add import

**Strategy implementation:**

```python
"""Regime-Adaptive — switches between trend-following and mean-reversion."""

from collections import deque
from server.registry import register
from strategies.base import Strategy, MarketSnapshot, InstrumentInfo, Signal
from strategies.indicators import (
    compute_adx, compute_macd, compute_bollinger, compute_bbw,
    compute_rsi, compute_atr, compute_sma, compute_obv_slope,
)
from strategies.position_manager import PositionManager


@register("regime_adaptive")
class RegimeAdaptive(Strategy):

    def required_data(self):
        return [
            {"interval": "15minute", "lookback": 100},
            {"interval": "day", "lookback": 50},
        ]

    def initialize(self, config, instruments):
        self.adx_trend = config.get("adx_trend", 25)
        self.adx_range = config.get("adx_range", 20)
        self.bbw_volatile_mult = config.get("bbw_volatile_mult", 1.5)
        self.trending_atr_mult = config.get("trending_atr_mult", 2.0)
        self.ranging_atr_mult = config.get("ranging_atr_mult", 1.0)
        self.volatile_size_pct = config.get("volatile_size_pct", 0.5)
        self.risk_per_trade = config.get("risk_per_trade", 0.02)
        self.pyramid_levels = config.get("pyramid_levels", 2)
        self.rsi_oversold = config.get("rsi_oversold", 35)
        self.rsi_overbought = config.get("rsi_overbought", 65)
        self.max_hold_bars = config.get("max_hold_bars", 40)
        self.instruments = instruments

        self.pm = PositionManager(max_pending_bars=3)

        # Per-symbol state
        self.regime: dict[str, str] = {}  # "TRENDING", "RANGING", "VOLATILE"
        self.daily_closes: dict[str, deque] = {}
        self.daily_highs: dict[str, deque] = {}
        self.daily_lows: dict[str, deque] = {}
        self.daily_volumes: dict[str, deque] = {}
        self.prices_15m: dict[str, deque] = {}
        self.bbw_history: dict[str, deque] = {}
        self.highest: dict[str, float] = {}
        self.lowest: dict[str, float] = {}

    def on_bar(self, snapshot):
        self.pm.increment_bars()
        signals = self.pm.process_fills(snapshot)
        signals += self.pm.resubmit_expired(snapshot)
        self.pm.reconcile(snapshot)

        # Update daily data + detect regime when daily bar arrives
        # Process 15-minute bars with regime-specific logic
        # TRENDING: MACD cross + ADX confirmation
        # RANGING: Bollinger band + RSI mean-reversion
        # VOLATILE: stay flat or very selective entry
        ...

        return signals
```

**Tests (~12):**
- test_required_data
- test_regime_trending_detected (high ADX)
- test_regime_ranging_detected (low ADX, low BBW)
- test_regime_volatile_detected (high BBW)
- test_trending_macd_entry
- test_ranging_bollinger_entry
- test_volatile_stays_flat
- test_regime_transition_adjusts_stops
- test_trending_pyramid
- test_ranging_no_pyramid
- test_trailing_stop_trending_wider
- test_trailing_stop_ranging_tighter

**server.py:** Add `import strategies.deterministic.regime_adaptive  # noqa: F401`

**Verify:** `cd strategies && pytest tests/test_regime_adaptive.py -v`

**Commit:** `feat: regime-adaptive strategy with trend/range/volatile detection`

---

### Task 5: Integration Test — Run All Strategies

**Step 1:** Run all 6 deterministic strategies + LLM on 10 stocks for 2024 and 2025.

```bash
SYM="RELIANCE,INFY,TCS,BAJFINANCE,HINDUNILVR,BHARTIARTL,SBIN,ICICIBANK,HDFCBANK,ITC"
CLI=./engine/target/release/backtest

# 2024
for strategy in sma_crossover rsi_daily_trend donchian_breakout confluence pairs_trading regime_adaptive; do
  $CLI run --strategy $strategy --symbols $SYM \
    --from 2024-01-01 --to 2024-12-31 --capital 1000000 \
    --interval $([ "$strategy" = "sma_crossover" ] && echo "day" || echo "15minute") \
    --max-drawdown 0.15 --max-volume-pct 0.10 --max-exposure 0.80
done

# 2025: same
```

**Step 2:** Compare results across all strategies and both years.

**Commit:** `test: integration backtests for all 6 strategies on 2024+2025`

---

## Execution Order

```
Task 1: Libraries + indicators.py    ── foundation, must go first
Task 2: Confluence strategy           ─┐
Task 3: Pairs trading strategy        ─┤── all parallel after Task 1
Task 4: Regime-adaptive strategy      ─┘
Task 5: Integration backtests         ── after 2-4
```

## Expected Test Counts After Completion

| Component | Tests |
|-----------|-------|
| indicators.py | ~15 (existing + new) |
| position_manager.py | 20 |
| SMA crossover | 12 |
| RSI daily trend | 11 |
| Donchian breakout | 12 |
| Confluence | ~10 |
| Pairs trading | ~10 |
| Regime adaptive | ~12 |
| LLM (base + generator) | 20 |
| **Python Total** | **~122** |
| **Rust Total** | **176** |
| **Grand Total** | **~298** |
