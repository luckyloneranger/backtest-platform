"""Shared technical indicator functions for deterministic strategies.

All indicators operate on plain lists of floats and return None when
there is insufficient data. No external dependencies.
"""


def compute_sma(prices: list[float], period: int) -> float | None:
    """Simple Moving Average over the last `period` prices."""
    if len(prices) < period:
        return None
    return sum(prices[-period:]) / period


def compute_ema(prices: list[float], period: int) -> float | None:
    """Exponential Moving Average. Seeds with SMA of first `period` values."""
    if len(prices) < period:
        return None

    multiplier = 2.0 / (period + 1)
    ema = sum(prices[:period]) / period  # seed with SMA

    for price in prices[period:]:
        ema = (price - ema) * multiplier + ema

    return ema


def compute_rsi(prices: list[float], period: int) -> float | None:
    """RSI using simple moving average of gains/losses (Cutler's RSI).

    Note: This is NOT Wilder's smoothed RSI used by most charting platforms.
    Values will differ from TradingView/Bloomberg RSI, especially for short periods.
    """
    if len(prices) < period + 1:
        return None

    gains = []
    losses = []
    for i in range(-period, 0):
        change = prices[i] - prices[i - 1]
        if change > 0:
            gains.append(change)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(abs(change))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def compute_atr(
    highs: list[float], lows: list[float], closes: list[float], period: int,
) -> float | None:
    """Average True Range — measures volatility."""
    if len(highs) < period + 1:
        return None

    true_ranges = []
    for i in range(-period, 0):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        true_ranges.append(tr)

    return sum(true_ranges) / period
