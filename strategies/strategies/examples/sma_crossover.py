from collections import deque

from server.registry import register
from strategies.base import Strategy, Bar, Portfolio, Signal


@register("sma_crossover")
class SmaCrossover(Strategy):
    """Simple Moving Average crossover strategy.

    Generates a BUY signal when the fast SMA crosses above the slow SMA
    (golden cross) and a SELL signal when the fast SMA crosses below the
    slow SMA (death cross).
    """

    def initialize(self, config: dict) -> None:
        self.fast_period = config.get("fast_period", 10)
        self.slow_period = config.get("slow_period", 30)
        self.prices: deque[float] = deque(maxlen=self.slow_period)
        self.prev_fast_above: bool | None = None

    def on_bar(self, bar: Bar, portfolio: Portfolio) -> list[Signal]:
        self.prices.append(bar.close)
        if len(self.prices) < self.slow_period:
            return []

        fast_sma = sum(list(self.prices)[-self.fast_period:]) / self.fast_period
        slow_sma = sum(self.prices) / len(self.prices)
        fast_above = fast_sma > slow_sma

        signals: list[Signal] = []
        if self.prev_fast_above is not None:
            if fast_above and not self.prev_fast_above:
                # Golden cross: fast crosses above slow -> BUY
                signals.append(Signal(
                    action="BUY", symbol=bar.symbol, quantity=1,
                ))
            elif not fast_above and self.prev_fast_above:
                # Death cross: fast crosses below slow -> SELL
                signals.append(Signal(
                    action="SELL", symbol=bar.symbol, quantity=1,
                ))

        self.prev_fast_above = fast_above
        return signals
