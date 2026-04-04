from collections import deque

from server.registry import register
from strategies.base import Strategy, MarketSnapshot, InstrumentInfo, Signal


@register("sma_crossover")
class SmaCrossover(Strategy):
    """Simple Moving Average crossover strategy.

    Generates a BUY signal when the fast SMA crosses above the slow SMA
    (golden cross) and a SELL signal when the fast SMA crosses below the
    slow SMA (death cross).
    """

    def required_data(self) -> list[dict]:
        # Default requirements - called before initialize, so use conservative defaults
        return [{"interval": "day", "lookback": 200}]

    def initialize(self, config: dict, instruments: dict[str, InstrumentInfo]) -> None:
        self.fast_period = config.get("fast_period", 10)
        self.slow_period = config.get("slow_period", 30)
        self.prices: dict[str, deque[float]] = {}
        self.prev_fast_above: dict[str, bool | None] = {}

    def on_bar(self, snapshot: MarketSnapshot) -> list[Signal]:
        signals = []

        # Get bars from the first available timeframe
        for interval, bars in snapshot.timeframes.items():
            for symbol, bar in bars.items():
                if symbol not in self.prices:
                    self.prices[symbol] = deque(maxlen=self.slow_period)
                    self.prev_fast_above[symbol] = None

                self.prices[symbol].append(bar.close)

                if len(self.prices[symbol]) < self.slow_period:
                    continue

                fast_sma = sum(list(self.prices[symbol])[-self.fast_period:]) / self.fast_period
                slow_sma = sum(self.prices[symbol]) / len(self.prices[symbol])
                fast_above = fast_sma > slow_sma

                if self.prev_fast_above[symbol] is not None:
                    if fast_above and not self.prev_fast_above[symbol]:
                        signals.append(Signal(action="BUY", symbol=symbol, quantity=1))
                    elif not fast_above and self.prev_fast_above[symbol]:
                        signals.append(Signal(action="SELL", symbol=symbol, quantity=1))

                self.prev_fast_above[symbol] = fast_above

        return signals
