"""RSI + Daily EMA Trend strategy.

Multi-timeframe strategy that uses:
- 15-minute RSI for entry/exit signals
- Daily EMA for trend direction (regime filter)

Rules:
- Only BUY when daily EMA trend is UP and 15-min RSI drops below oversold level
- Only SELL when 15-min RSI rises above overbought level, or daily trend reverses
- Position sizing: allocate a configurable % of available capital per trade

Config params:
- rsi_period: RSI lookback period (default 14)
- rsi_oversold: RSI level to trigger buy (default 30)
- rsi_overbought: RSI level to trigger sell (default 70)
- ema_period: Daily EMA period for trend (default 20)
- risk_pct: fraction of available capital to allocate per trade (default 0.2 = 20%)
"""

from collections import deque

from server.registry import register
from strategies.base import Strategy, MarketSnapshot, InstrumentInfo, Signal


def compute_rsi(prices: list[float], period: int) -> float | None:
    """Compute RSI from a list of prices. Returns None if not enough data."""
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


def compute_ema(prices: list[float], period: int) -> float | None:
    """Compute EMA from a list of prices. Returns None if not enough data."""
    if len(prices) < period:
        return None

    multiplier = 2.0 / (period + 1)
    ema = sum(prices[:period]) / period  # seed with SMA

    for price in prices[period:]:
        ema = (price - ema) * multiplier + ema

    return ema


@register("rsi_daily_trend")
class RsiDailyTrend(Strategy):
    """Multi-timeframe RSI + Daily EMA Trend strategy.

    Uses 15-minute RSI for timing entries/exits and daily EMA for trend
    direction. Only takes long positions when the daily trend is bullish.
    """

    def required_data(self) -> list[dict]:
        return [
            {"interval": "15minute", "lookback": 100},
            {"interval": "day", "lookback": 50},
        ]

    def initialize(self, config: dict, instruments: dict[str, InstrumentInfo]) -> None:
        self.rsi_period = config.get("rsi_period", 14)
        self.rsi_oversold = config.get("rsi_oversold", 30)
        self.rsi_overbought = config.get("rsi_overbought", 70)
        self.ema_period = config.get("ema_period", 20)
        self.risk_pct = config.get("risk_pct", 0.2)  # 20% of capital per trade
        self.instruments = instruments

        # Per-symbol state
        self.prices_15m: dict[str, deque[float]] = {}
        self.prices_daily: dict[str, deque[float]] = {}
        self.in_position: dict[str, bool] = {}
        self.prev_rsi: dict[str, float | None] = {}

    def on_bar(self, snapshot: MarketSnapshot) -> list[Signal]:
        signals = []

        # Update daily prices when a new daily bar arrives
        if "day" in snapshot.timeframes:
            for symbol, bar in snapshot.timeframes["day"].items():
                if symbol not in self.prices_daily:
                    self.prices_daily[symbol] = deque(maxlen=self.ema_period + 10)
                self.prices_daily[symbol].append(bar.close)

        # Process 15-minute bars for RSI signals
        if "15minute" not in snapshot.timeframes:
            return signals

        for symbol, bar in snapshot.timeframes["15minute"].items():
            if symbol not in self.prices_15m:
                self.prices_15m[symbol] = deque(maxlen=self.rsi_period + 10)
                self.in_position[symbol] = False
                self.prev_rsi[symbol] = None

            self.prices_15m[symbol].append(bar.close)

            # Compute RSI on 15-minute data
            rsi = compute_rsi(list(self.prices_15m[symbol]), self.rsi_period)
            if rsi is None:
                continue

            # Compute daily EMA for trend direction
            daily_prices = list(self.prices_daily.get(symbol, []))
            ema = compute_ema(daily_prices, self.ema_period)
            daily_trend_up = (
                ema is not None
                and len(daily_prices) > 0
                and daily_prices[-1] > ema
            )

            # --- Entry: RSI crosses below oversold AND daily trend is up ---
            prev = self.prev_rsi[symbol]
            if (
                not self.in_position[symbol]
                and daily_trend_up
                and prev is not None
                and prev >= self.rsi_oversold
                and rsi < self.rsi_oversold
            ):
                # Dynamic position sizing: allocate risk_pct of available cash
                allocation = snapshot.portfolio.cash * self.risk_pct
                qty = int(allocation / bar.close)

                # Round to lot size if instrument metadata available
                inst = self.instruments.get(symbol)
                if inst and inst.lot_size > 1:
                    qty = (qty // inst.lot_size) * inst.lot_size

                if qty > 0:
                    signals.append(Signal(
                        action="BUY", symbol=symbol, quantity=qty,
                    ))
                    self.in_position[symbol] = True

            # --- Exit: RSI crosses above overbought OR daily trend reverses ---
            elif self.in_position[symbol]:
                should_exit = False

                # RSI overbought exit
                if prev is not None and prev <= self.rsi_overbought and rsi > self.rsi_overbought:
                    should_exit = True

                # Daily trend reversal exit
                if not daily_trend_up and ema is not None:
                    should_exit = True

                if should_exit:
                    # Sell entire position
                    held_qty = 0
                    for pos in snapshot.portfolio.positions:
                        if pos.symbol == symbol:
                            held_qty = pos.quantity
                            break

                    if held_qty > 0:
                        signals.append(Signal(
                            action="SELL", symbol=symbol, quantity=held_qty,
                        ))
                    self.in_position[symbol] = False

            self.prev_rsi[symbol] = rsi

        return signals

    def on_complete(self) -> dict:
        return {
            "strategy_type": "rsi_daily_trend",
            "rsi_period": self.rsi_period,
            "ema_period": self.ema_period,
            "risk_pct": self.risk_pct,
        }
