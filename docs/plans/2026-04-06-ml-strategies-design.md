# ML-Powered Strategies — Design Document

Date: 2026-04-06

## Goal

Build 3 ML-powered strategies using scikit-learn and statsmodels that learn from data rather than using fixed thresholds. Online/rolling training prevents overfitting.

## Shared: Feature Engine

Add `compute_features()` to `indicators.py`:

```python
def compute_features(closes, highs, lows, volumes, period=20) -> dict[str, float] | None:
```

Features (~20):
- rsi_14, rsi_percentile (rank vs last 50 bars)
- macd_hist, macd_hist_slope (current - 3 bars ago)
- bb_pct_b (where price is in Bollinger bands: 0=lower, 1=upper)
- adx_14
- obv_slope_10
- atr_norm (ATR / close)
- volume_zscore (vs 20-bar rolling)
- ret_1, ret_5, ret_10, ret_20 (returns at various lookbacks)
- ret_autocorr (correlation of ret_1 with lag-1 ret_1 over 20 bars)
- bbw (Bollinger Band Width)
- stoch_k, stoch_d

All computed from existing indicator functions. Returns None if insufficient data.

## Strategy 1: ML Signal Classifier

**File:** `strategies/deterministic/ml_classifier.py`
**Register:** `@register("ml_classifier")`
**Interval:** day, lookback 200

- Each bar: compute features, store in rolling buffer
- Label: next `forward_bars` (5) return > `target_return` (1%) → +1, < -1% → -1, else 0
- After `min_train_bars` (120): train GradientBoostingClassifier(max_depth=3, n_estimators=50, min_samples_leaf=10)
- Retrain every `retrain_interval` (20) bars on last `max_train_bars` (250) samples
- Predict probability. prob > 0.65 → long. prob < 0.35 → short. Else hold.
- PositionManager for entry/exit. ATR trailing stop.
- Only 1 position per symbol. CNC if prob > 0.75, MIS otherwise.

Config: min_train_bars=120, retrain_interval=20, forward_bars=5, target_return=0.01, confidence_threshold=0.65, risk_pct=0.03, max_train_bars=250

## Strategy 2: OU Mean Reversion

**File:** `strategies/deterministic/ou_mean_reversion.py`
**Register:** `@register("ou_mean_reversion")`
**Interval:** day, lookback 200

- Each daily bar: accumulate close prices per symbol
- After 60 bars: fit OU process via statsmodels OLS:
  - regress diff(price) on lag(price) → get θ (mean-reversion speed)
  - μ = -intercept/θ (long-term mean)
  - halflife = -ln(2)/θ
  - Residual std = σ
- Stock filter: only trade if θ > 0 AND halflife < 30 AND p-value < 0.05
- Entry: z = (price - μ)/σ. z < -2.0 → long. z > +2.0 → short.
- Exit: z crosses 0 (mean-reverted). Stop: |z| > 3.0.
- Refit daily (new OLS each bar with full history).
- PositionManager. CNC product.

Config: min_history=60, max_halflife=30, zscore_entry=2.0, zscore_exit=0.0, zscore_stop=3.0, min_pvalue=0.05, risk_pct=0.03

## Strategy 3: Adaptive Ensemble Meta-Learner

**File:** `strategies/deterministic/ensemble_meta.py`
**Register:** `@register("ensemble_meta")`
**Interval:** day, lookback 200

- Each bar: compute 5 sub-signals:
  1. SMA(10) vs SMA(30): +1/-1/0
  2. RSI(14): <35=+1, >65=-1, else 0
  3. MACD histogram: >0=+1, <0=-1
  4. Bollinger %B: <0=+1 (below lower), >1=-1 (above upper), else 0
  5. ADX direction: ADX>25 AND trending up=+1, trending down=-1, else 0
- Feature vector: 5 sub-signals + ADX value + BBW value = 7 features
- Label: same as ML classifier (next 5-bar return > 1%)
- After 120 bars: train LogisticRegression on (features → labels)
- Retrain every 20 bars
- Predict probability. prob > 0.60 → long. prob < 0.40 → short.
- PositionManager. CNC/MIS based on confidence.

Config: min_train_bars=120, retrain_interval=20, forward_bars=5, target_return=0.01, confidence_threshold=0.60, risk_pct=0.03

## Implementation Order

```
Task 1: compute_features() in indicators.py + tests
Task 2: ML Classifier + tests        ─┐
Task 3: OU Mean Reversion + tests     ─┤ parallel after Task 1
Task 4: Ensemble Meta + tests         ─┘
Task 5: Run backtests at ₹1L for 2024 + 2025
```

## Verification

```bash
pytest tests/ -v
# Run at ₹1L:
backtest run --strategy ml_classifier --symbols ... --capital 100000 --interval day
backtest run --strategy ou_mean_reversion --symbols ... --capital 100000 --interval day
backtest run --strategy ensemble_meta --symbols ... --capital 100000 --interval day
```
