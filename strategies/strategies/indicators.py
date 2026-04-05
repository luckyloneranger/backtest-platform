"""Technical indicators backed by pandas-ta.

All functions accept plain Python lists and return plain Python types.
Strategies import from here -- never import pandas-ta directly.
"""

# pandas-ta requires numba at import time, but numba does not support
# Python 3.14+.  We shim it with a no-op njit decorator so that
# pandas-ta can load without the compiled backend.
import sys
import types

if "numba" not in sys.modules:
    _numba = types.ModuleType("numba")

    def _njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]

        def wrapper(fn):
            return fn

        return wrapper

    _numba.njit = _njit
    sys.modules["numba"] = _numba

import pandas as pd
import numpy as np
import pandas_ta as ta


def _to_series(values: list[float]) -> pd.Series:
    """Convert list to pandas Series."""
    return pd.Series(values, dtype=float)


def _last_valid(result: pd.Series | None) -> float | None:
    """Extract last non-NaN value from a pandas Series."""
    if result is None or result.empty:
        return None
    val = result.iloc[-1]
    return None if pd.isna(val) else float(val)


# === Existing functions (pandas-ta backend) ===


def compute_sma(prices: list[float], period: int) -> float | None:
    """Simple Moving Average."""
    if len(prices) < period:
        return None
    return _last_valid(_to_series(prices).rolling(period).mean())


def compute_ema(prices: list[float], period: int) -> float | None:
    """Exponential Moving Average."""
    if len(prices) < period:
        return None
    return _last_valid(ta.ema(_to_series(prices), length=period))


def compute_rsi(prices: list[float], period: int) -> float | None:
    """RSI using Wilder's smoothing method (industry standard)."""
    if len(prices) < period + 1:
        return None
    return _last_valid(ta.rsi(_to_series(prices), length=period))


def compute_atr(
    highs: list[float], lows: list[float], closes: list[float], period: int,
) -> float | None:
    """Average True Range."""
    if len(highs) < period + 1:
        return None
    return _last_valid(
        ta.atr(_to_series(highs), _to_series(lows), _to_series(closes), length=period)
    )


# === New functions ===


def compute_macd(
    closes: list[float], fast: int = 12, slow: int = 26, signal: int = 9,
) -> tuple[float, float, float] | None:
    """MACD. Returns (macd_line, signal_line, histogram) or None."""
    if len(closes) < slow + signal:
        return None
    result = ta.macd(_to_series(closes), fast=fast, slow=slow, signal=signal)
    if result is None or result.empty:
        return None
    # pandas-ta returns DataFrame with columns: MACD_f_s_sig, MACDh_f_s_sig, MACDs_f_s_sig
    cols = result.columns.tolist()
    macd_val = result[cols[0]].iloc[-1]
    hist_val = result[cols[1]].iloc[-1]
    signal_val = result[cols[2]].iloc[-1]
    if any(pd.isna(v) for v in [macd_val, signal_val, hist_val]):
        return None
    return (float(macd_val), float(signal_val), float(hist_val))


def compute_bollinger(
    closes: list[float], period: int = 20, std: float = 2.0,
) -> tuple[float, float, float] | None:
    """Bollinger Bands. Returns (upper, mid, lower) or None."""
    if len(closes) < period:
        return None
    result = ta.bbands(_to_series(closes), length=period, std=std)
    if result is None or result.empty:
        return None
    cols = result.columns.tolist()
    # bbands returns: BBL, BBM, BBU, BBB, BBP
    lower = result[cols[0]].iloc[-1]
    mid = result[cols[1]].iloc[-1]
    upper = result[cols[2]].iloc[-1]
    if any(pd.isna(v) for v in [lower, mid, upper]):
        return None
    return (float(upper), float(mid), float(lower))


def compute_adx(
    highs: list[float], lows: list[float], closes: list[float], period: int = 14,
) -> float | None:
    """Average Directional Index (trend strength 0-100)."""
    if len(highs) < period * 2:
        return None
    result = ta.adx(
        _to_series(highs), _to_series(lows), _to_series(closes), length=period,
    )
    if result is None or result.empty:
        return None
    # adx returns DataFrame with ADX_14, ADXR_14_2, DMP_14, DMN_14
    adx_col = [c for c in result.columns if c.startswith("ADX_")]
    if not adx_col:
        return None
    val = result[adx_col[0]].iloc[-1]
    return None if pd.isna(val) else float(val)


def compute_obv(closes: list[float], volumes: list[int]) -> list[float] | None:
    """On-Balance Volume series."""
    if len(closes) < 2 or len(volumes) < 2:
        return None
    result = ta.obv(_to_series(closes), _to_series([float(v) for v in volumes]))
    if result is None or result.empty:
        return None
    return [float(v) if not pd.isna(v) else 0.0 for v in result.tolist()]


def compute_obv_slope(
    closes: list[float], volumes: list[int], period: int = 10,
) -> float | None:
    """OBV regression slope over last N bars."""
    obv = compute_obv(closes, volumes)
    if obv is None or len(obv) < period:
        return None
    recent = obv[-period:]
    x = np.arange(period, dtype=float)
    y = np.array(recent, dtype=float)
    # Simple linear regression slope
    slope = (np.sum(x * y) - period * np.mean(x) * np.mean(y)) / (
        np.sum(x**2) - period * np.mean(x) ** 2
    )
    return float(slope)


def compute_stochastic(
    highs: list[float], lows: list[float], closes: list[float],
    k: int = 14, d: int = 3,
) -> tuple[float, float] | None:
    """Stochastic Oscillator. Returns (%K, %D) or None."""
    if len(highs) < k + d:
        return None
    result = ta.stoch(
        _to_series(highs), _to_series(lows), _to_series(closes), k=k, d=d,
    )
    if result is None or result.empty:
        return None
    cols = result.columns.tolist()
    k_val = result[cols[0]].iloc[-1]
    d_val = result[cols[1]].iloc[-1]
    if any(pd.isna(v) for v in [k_val, d_val]):
        return None
    return (float(k_val), float(d_val))


def compute_bbw(
    closes: list[float], period: int = 20, std: float = 2.0,
) -> float | None:
    """Bollinger Band Width = (upper - lower) / mid."""
    bb = compute_bollinger(closes, period, std)
    if bb is None:
        return None
    upper, mid, lower = bb
    if mid == 0:
        return None
    return (upper - lower) / mid


def compute_zscore(series: list[float], period: int = 20) -> float | None:
    """Rolling z-score of the last value."""
    if len(series) < period:
        return None
    s = _to_series(series)
    mean = s.rolling(period).mean().iloc[-1]
    std = s.rolling(period).std().iloc[-1]
    if pd.isna(mean) or pd.isna(std) or std == 0:
        return None
    return float((s.iloc[-1] - mean) / std)


def compute_correlation(
    series_a: list[float], series_b: list[float], period: int = 30,
) -> float | None:
    """Pearson correlation over last N values."""
    if len(series_a) < period or len(series_b) < period:
        return None
    a = pd.Series(series_a[-period:])
    b = pd.Series(series_b[-period:])
    corr = a.corr(b)
    return None if pd.isna(corr) else float(corr)


def compute_cointegration(
    series_a: list[float], series_b: list[float],
) -> tuple[float, float] | None:
    """Engle-Granger cointegration test. Returns (p_value, hedge_ratio) or None."""
    if len(series_a) < 30 or len(series_b) < 30:
        return None
    try:
        from statsmodels.tsa.stattools import coint
        import statsmodels.api as sm

        a = np.array(series_a, dtype=float)
        b = np.array(series_b, dtype=float)
        _, p_value, _ = coint(a, b)
        # Hedge ratio via OLS: a = hedge_ratio * b + intercept
        b_with_const = sm.add_constant(b)
        model = sm.OLS(a, b_with_const).fit()
        hedge_ratio = model.params[1]
        return (float(p_value), float(hedge_ratio))
    except Exception:
        return None


def compute_vwap(highs: list[float], lows: list[float], closes: list[float],
                 volumes: list[int]) -> float | None:
    """Cumulative VWAP = sum(typical_price * volume) / sum(volume).
    Typical price = (high + low + close) / 3.
    Caller provides today's intraday bars only (VWAP resets daily)."""
    if len(highs) < 1 or len(volumes) < 1:
        return None
    tp = np.array([(h + l + c) / 3.0 for h, l, c in zip(highs, lows, closes)])
    vol = np.array(volumes, dtype=float)
    total_vol = vol.sum()
    if total_vol == 0:
        return None
    return float((tp * vol).sum() / total_vol)


def compute_vwap_bands(highs: list[float], lows: list[float], closes: list[float],
                       volumes: list[int], std_mult: float = 1.0) -> tuple[float, float, float] | None:
    """Returns (vwap, upper_band, lower_band).
    Bands = VWAP +/- std_mult * std_dev of (typical_price - vwap)."""
    vwap = compute_vwap(highs, lows, closes, volumes)
    if vwap is None:
        return None
    tp = np.array([(h + l + c) / 3.0 for h, l, c in zip(highs, lows, closes)])
    std = float(np.std(tp - vwap))
    if std == 0:
        return (vwap, vwap, vwap)
    return (vwap, vwap + std_mult * std, vwap - std_mult * std)


def compute_halflife(series: list[float]) -> int | None:
    """Mean-reversion halflife via OLS on lagged spread."""
    if len(series) < 20:
        return None
    try:
        import statsmodels.api as sm

        s = np.array(series, dtype=float)
        lag = s[:-1]
        diff = np.diff(s)
        lag_with_const = sm.add_constant(lag)
        model = sm.OLS(diff, lag_with_const).fit()
        theta = model.params[1]
        if theta >= 0:
            return None  # not mean-reverting
        halflife = int(-np.log(2) / theta)
        return max(1, halflife)
    except Exception:
        return None
