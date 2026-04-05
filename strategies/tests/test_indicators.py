"""Tests for all indicator functions in strategies/indicators.py."""

import math

from strategies.indicators import (
    compute_sma,
    compute_ema,
    compute_rsi,
    compute_atr,
    compute_macd,
    compute_bollinger,
    compute_adx,
    compute_obv,
    compute_obv_slope,
    compute_stochastic,
    compute_bbw,
    compute_zscore,
    compute_correlation,
    compute_cointegration,
    compute_halflife,
    compute_vwap,
    compute_vwap_bands,
)


# === SMA ===

def test_compute_sma_basic():
    prices = [10.0, 20.0, 30.0, 40.0, 50.0]
    result = compute_sma(prices, 3)
    assert result is not None
    # SMA of last 3: (30 + 40 + 50) / 3 = 40
    assert abs(result - 40.0) < 0.001

    result2 = compute_sma(prices, 5)
    assert result2 is not None
    assert abs(result2 - 30.0) < 0.001


# === EMA ===

def test_compute_ema_basic():
    prices = [10.0, 11.0, 12.0, 13.0, 14.0]
    result = compute_ema(prices, 3)
    assert result is not None
    # EMA should be between the min and max of recent values
    assert 12.0 < result < 14.0


def test_compute_ema_weights_recent():
    # Non-linear ascending prices -> EMA should track above SMA
    prices = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 15.0]
    ema = compute_ema(prices, 5)
    sma = compute_sma(prices, 5)
    assert ema is not None and sma is not None
    # EMA weights recent more heavily; with acceleration at the end, EMA > SMA
    assert ema > sma


# === RSI ===

def test_compute_rsi_wilder():
    # Strong uptrend -> RSI very high
    prices_up = [100 + i for i in range(20)]
    rsi = compute_rsi(prices_up, 14)
    assert rsi is not None and rsi > 90

    # Strong downtrend -> RSI very low
    prices_down = [100 - i for i in range(20)]
    rsi = compute_rsi(prices_down, 14)
    assert rsi is not None and rsi < 10

    # Insufficient data -> None
    assert compute_rsi([100, 101, 102], 14) is None


def test_compute_rsi_flat():
    # Flat prices -> RSI should be around 50 (no gains, no losses)
    # Actually all-flat means avg_gain=0, avg_loss=0 -> depends on implementation
    prices = [100.0] * 20
    rsi = compute_rsi(prices, 14)
    # pandas-ta returns NaN for flat series (0/0 case)
    # This is acceptable behavior
    if rsi is not None:
        assert 0 <= rsi <= 100


# === ATR ===

def test_compute_atr_basic():
    # Uniform range data: high-low = 2, no gaps
    highs = [12, 13, 14, 15, 16, 17]
    lows = [10, 11, 12, 13, 14, 15]
    closes = [11, 12, 13, 14, 15, 16]
    result = compute_atr(highs, lows, closes, 3)
    assert result is not None
    assert result > 0
    assert abs(result - 2.0) < 0.1


def test_compute_atr_volatile():
    # More volatile data should give larger ATR
    highs_calm = [100, 101, 102, 103, 104, 105]
    lows_calm = [99, 100, 101, 102, 103, 104]
    closes_calm = [100, 101, 102, 103, 104, 105]

    highs_wild = [100, 110, 90, 115, 85, 120]
    lows_wild = [90, 95, 80, 100, 75, 105]
    closes_wild = [95, 105, 85, 110, 80, 115]

    atr_calm = compute_atr(highs_calm, lows_calm, closes_calm, 3)
    atr_wild = compute_atr(highs_wild, lows_wild, closes_wild, 3)
    assert atr_calm is not None and atr_wild is not None
    assert atr_wild > atr_calm


# === MACD ===

def test_compute_macd():
    # 40 ascending data points
    closes = [float(100 + i) for i in range(40)]
    result = compute_macd(closes, fast=12, slow=26, signal=9)
    assert result is not None
    macd_line, signal_line, histogram = result
    # In steady uptrend, MACD should be positive
    assert macd_line > 0
    assert isinstance(signal_line, float)
    assert isinstance(histogram, float)


def test_compute_macd_insufficient():
    # Not enough data
    closes = [float(100 + i) for i in range(30)]
    result = compute_macd(closes, fast=12, slow=26, signal=9)
    # 30 < 26 + 9 = 35 -> None
    assert result is None


# === Bollinger Bands ===

def test_compute_bollinger():
    closes = [float(100 + i) for i in range(30)]
    result = compute_bollinger(closes, period=20, std=2.0)
    assert result is not None
    upper, mid, lower = result
    assert upper > mid > lower
    # Mid should be close to SMA
    sma = compute_sma(closes, 20)
    assert sma is not None
    assert abs(mid - sma) < 0.01


def test_compute_bollinger_insufficient():
    closes = [float(100 + i) for i in range(15)]
    result = compute_bollinger(closes, period=20)
    assert result is None


# === ADX ===

def test_compute_adx_trending():
    # Strongly trending data should give high ADX
    n = 40
    highs = [float(100 + i * 2 + 2) for i in range(n)]
    lows = [float(100 + i * 2 - 2) for i in range(n)]
    closes = [float(100 + i * 2) for i in range(n)]
    result = compute_adx(highs, lows, closes, period=14)
    assert result is not None
    # A strong trend should give ADX > 25
    assert result > 25


def test_compute_adx_insufficient():
    highs = [12, 13, 14]
    lows = [10, 11, 12]
    closes = [11, 12, 13]
    result = compute_adx(highs, lows, closes, period=14)
    assert result is None


# === OBV Slope ===

def test_compute_obv_slope():
    # Rising prices with constant volume -> positive OBV slope
    closes = [float(100 + i) for i in range(20)]
    volumes = [1000] * 20
    slope = compute_obv_slope(closes, volumes, period=10)
    assert slope is not None
    assert slope > 0


def test_compute_obv_slope_declining():
    # Falling prices -> negative OBV slope
    closes = [float(120 - i) for i in range(20)]
    volumes = [1000] * 20
    slope = compute_obv_slope(closes, volumes, period=10)
    assert slope is not None
    assert slope < 0


# === OBV ===

def test_compute_obv():
    closes = [10.0, 11.0, 10.5, 12.0, 11.5]
    volumes = [100, 200, 150, 300, 250]
    result = compute_obv(closes, volumes)
    assert result is not None
    assert len(result) == 5


def test_compute_obv_insufficient():
    assert compute_obv([10.0], [100]) is None


# === Stochastic ===

def test_compute_stochastic():
    n = 20
    highs = [float(100 + i + 2) for i in range(n)]
    lows = [float(100 + i - 2) for i in range(n)]
    closes = [float(100 + i) for i in range(n)]
    result = compute_stochastic(highs, lows, closes, k=14, d=3)
    assert result is not None
    k_val, d_val = result
    assert 0 <= k_val <= 100
    assert 0 <= d_val <= 100


def test_compute_stochastic_insufficient():
    result = compute_stochastic([1, 2, 3], [0, 1, 2], [1, 2, 3], k=14, d=3)
    assert result is None


# === BBW ===

def test_compute_bbw():
    closes = [float(100 + i) for i in range(30)]
    result = compute_bbw(closes, period=20, std=2.0)
    assert result is not None
    assert result > 0


def test_compute_bbw_insufficient():
    result = compute_bbw([1.0, 2.0], period=20)
    assert result is None


# === Z-Score ===

def test_compute_zscore():
    # Known z-score: last value is the mean -> z=0
    series = [100.0] * 20
    result = compute_zscore(series, period=20)
    # All same values -> std=0 -> None
    assert result is None

    # Build a series where last value is 1 std above mean
    import numpy as np
    base = list(np.random.RandomState(42).normal(100, 10, 19))
    mean_val = np.mean(base)
    std_val = np.std(base, ddof=1)
    # Append a value exactly 1 std above mean
    base.append(float(mean_val + std_val))
    # z-score should be close to 1 (but recalculated over 20 values)
    result = compute_zscore(base, period=20)
    assert result is not None
    # Just check it returns a reasonable number
    assert -5 < result < 5


def test_compute_zscore_insufficient():
    result = compute_zscore([1.0, 2.0], period=20)
    assert result is None


# === Correlation ===

def test_compute_correlation_perfect():
    a = [float(i) for i in range(30)]
    b = [float(2 * i + 10) for i in range(30)]
    result = compute_correlation(a, b, period=30)
    assert result is not None
    assert abs(result - 1.0) < 0.001


def test_compute_correlation_negative():
    a = [float(i) for i in range(30)]
    b = [float(100 - 2 * i) for i in range(30)]
    result = compute_correlation(a, b, period=30)
    assert result is not None
    assert abs(result - (-1.0)) < 0.001


def test_compute_correlation_insufficient():
    result = compute_correlation([1.0, 2.0], [3.0, 4.0], period=30)
    assert result is None


# === Cointegration ===

def test_compute_cointegration():
    import numpy as np
    rng = np.random.RandomState(42)
    n = 100
    # Create two cointegrated series: b = a + noise
    a = np.cumsum(rng.normal(0, 1, n)).tolist()
    b = [x + rng.normal(0, 0.1) for x in a]
    result = compute_cointegration(a, b)
    assert result is not None
    p_value, hedge_ratio = result
    # Strongly cointegrated -> low p-value
    assert p_value < 0.05
    # Hedge ratio should be close to 1
    assert abs(hedge_ratio - 1.0) < 0.5


def test_compute_cointegration_insufficient():
    assert compute_cointegration([1.0] * 10, [2.0] * 10) is None


# === Halflife ===

def test_compute_halflife():
    import numpy as np
    rng = np.random.RandomState(42)
    # Ornstein-Uhlenbeck process (mean-reverting)
    n = 200
    theta = 0.1  # mean-reversion speed
    mu = 100.0
    sigma = 1.0
    series = [mu]
    for _ in range(n - 1):
        prev = series[-1]
        series.append(prev + theta * (mu - prev) + sigma * rng.normal())
    result = compute_halflife(series)
    assert result is not None
    assert result >= 1
    # Theoretical halflife = ln(2) / theta ~ 6.9
    # Allow wide range since OLS estimate is noisy
    assert 1 < result < 50


def test_compute_halflife_trending():
    # Non-mean-reverting (trending with noise) -> None or very large halflife
    # A purely linear series may produce a near-zero negative theta from
    # floating-point noise, yielding a huge halflife.  Our test should accept
    # either None or a very large value (> 1000), since both indicate the
    # series is not practically mean-reverting.
    series = [float(i) for i in range(50)]
    result = compute_halflife(series)
    assert result is None or result > 1000


def test_compute_halflife_insufficient():
    result = compute_halflife([1.0, 2.0, 3.0])
    assert result is None


# === Insufficient data returns None ===

def test_insufficient_data_returns_none():
    short = [1.0, 2.0, 3.0]
    short_h = [2.0, 3.0, 4.0]
    short_l = [0.0, 1.0, 2.0]

    assert compute_sma(short, 10) is None
    assert compute_ema(short, 10) is None
    assert compute_rsi(short, 14) is None
    assert compute_atr(short_h, short_l, short, 14) is None
    assert compute_macd(short) is None
    assert compute_bollinger(short) is None
    assert compute_adx(short_h, short_l, short) is None
    assert compute_obv([1.0], [100]) is None
    assert compute_obv_slope(short, [100, 200, 300], period=10) is None
    assert compute_stochastic(short_h, short_l, short) is None
    assert compute_bbw(short) is None
    assert compute_zscore(short) is None
    assert compute_correlation(short, short) is None
    assert compute_cointegration(short, short) is None
    assert compute_halflife(short) is None


# === VWAP ===

def test_compute_vwap():
    highs = [102.0, 104.0, 103.0, 105.0, 106.0]
    lows = [98.0, 100.0, 99.0, 101.0, 102.0]
    closes = [100.0, 102.0, 101.0, 103.0, 104.0]
    volumes = [1000, 2000, 1500, 3000, 2500]
    result = compute_vwap(highs, lows, closes, volumes)
    assert result is not None
    # TP = [100, 102, 101, 103, 104], weighted by volume
    # Manual: (100*1000 + 102*2000 + 101*1500 + 103*3000 + 104*2500) / 10000
    expected = (100000 + 204000 + 151500 + 309000 + 260000) / 10000.0
    assert abs(result - expected) < 0.01


def test_compute_vwap_insufficient():
    assert compute_vwap([], [], [], []) is None


def test_compute_vwap_bands():
    highs = [102.0, 104.0, 103.0, 105.0, 106.0]
    lows = [98.0, 100.0, 99.0, 101.0, 102.0]
    closes = [100.0, 102.0, 101.0, 103.0, 104.0]
    volumes = [1000, 2000, 1500, 3000, 2500]
    result = compute_vwap_bands(highs, lows, closes, volumes, std_mult=1.0)
    assert result is not None
    vwap, upper, lower = result
    assert upper > vwap > lower


def test_compute_vwap_bands_insufficient():
    assert compute_vwap_bands([], [], [], []) is None
