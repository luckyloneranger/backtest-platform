"""SMA Crossover — trend following with ATR sizing, trailing stops, pyramiding.

Golden cross (fast > slow) → enter long
Death cross (fast < slow) → enter short (MIS)
Trailing stop: highest/lowest since entry - ATR * multiplier
Pyramiding: add on continued trend (price > entry + ATR)
"""

from collections import deque
from server.registry import register
from strategies.base import Strategy, MarketSnapshot, InstrumentInfo, Signal
from strategies.indicators import compute_sma, compute_atr
from strategies.position_manager import PositionManager


@register("sma_crossover")
class SmaCrossover(Strategy):

    def required_data(self):
        return [{"interval": "day", "lookback": 200}]

    def initialize(self, config, instruments):
        self.fast_period = config.get("fast_period", 10)
        self.slow_period = config.get("slow_period", 30)
        if self.fast_period >= self.slow_period:
            raise ValueError(f"fast_period ({self.fast_period}) >= slow_period ({self.slow_period})")
        self.risk_per_trade = config.get("risk_per_trade", 0.02)
        self.atr_period = config.get("atr_period", 14)
        self.atr_multiplier = config.get("atr_multiplier", 2.0)
        self.min_spread = config.get("min_spread", 0.005)
        self.max_hold_bars = config.get("max_hold_bars", 30)
        self.pyramid_levels = config.get("pyramid_levels", 2)

        self.pm = PositionManager(max_pending_bars=1)  # daily bars, cancel next day

        # Per-symbol indicator state
        self.prices: dict[str, deque] = {}
        self.highs: dict[str, deque] = {}
        self.lows: dict[str, deque] = {}
        self.closes: dict[str, deque] = {}
        self.prev_fast_above: dict[str, bool | None] = {}
        self.highest: dict[str, float] = {}
        self.lowest: dict[str, float] = {}

    def _ensure(self, symbol):
        if symbol not in self.prices:
            maxlen = self.slow_period + self.atr_period + 10
            self.prices[symbol] = deque(maxlen=maxlen)
            self.highs[symbol] = deque(maxlen=maxlen)
            self.lows[symbol] = deque(maxlen=maxlen)
            self.closes[symbol] = deque(maxlen=maxlen)
            self.prev_fast_above[symbol] = None
            self.highest[symbol] = 0.0
            self.lowest[symbol] = float('inf')

    def on_bar(self, snapshot):
        self.pm.increment_bars()
        signals = self.pm.process_fills(snapshot)
        signals += self.pm.resubmit_expired(snapshot)
        self.pm.reconcile(snapshot)

        for interval, bars in snapshot.timeframes.items():
            for symbol, bar in bars.items():
                self._ensure(symbol)
                self.prices[symbol].append(bar.close)
                self.highs[symbol].append(bar.high)
                self.lows[symbol].append(bar.low)
                self.closes[symbol].append(bar.close)

                # Compute indicators
                fast = compute_sma(list(self.prices[symbol]), self.fast_period)
                slow = compute_sma(list(self.prices[symbol]), self.slow_period)
                atr = compute_atr(list(self.highs[symbol]), list(self.lows[symbol]),
                                  list(self.closes[symbol]), self.atr_period)
                if fast is None or slow is None or atr is None or atr <= 0:
                    self.prev_fast_above[symbol] = None
                    continue

                fast_above = fast > slow
                spread = abs(fast - slow) / slow
                prev = self.prev_fast_above[symbol]
                qty = int(snapshot.context.initial_capital * self.risk_per_trade / (atr * self.atr_multiplier))

                # --- Flat: check entries ---
                if self.pm.is_flat(symbol) and not self.pm.has_pending_entry(symbol):
                    if prev is not None and fast_above and not prev and spread > self.min_spread and qty > 0:
                        # Golden cross
                        product = "CNC" if spread > 0.01 else "MIS"
                        stop = bar.close - atr * self.atr_multiplier
                        signals += self.pm.enter_long(symbol, qty, bar.close, product, stop)
                        self.highest[symbol] = bar.close
                    elif prev is not None and not fast_above and prev and spread > self.min_spread and qty > 0:
                        # Death cross
                        stop = bar.close + atr * self.atr_multiplier
                        signals += self.pm.enter_short(symbol, qty, bar.close, stop)
                        self.lowest[symbol] = bar.close

                # --- Long: trailing stop + exits ---
                elif self.pm.is_long(symbol):
                    state = self.pm.get_state(symbol)
                    if bar.close > self.highest[symbol]:
                        self.highest[symbol] = bar.close
                    new_stop = self.highest[symbol] - atr * self.atr_multiplier
                    signals += self.pm.update_trailing_stop(symbol, new_stop)

                    if prev is not None and not fast_above and prev:
                        signals += self.pm.exit_position(symbol)
                    elif state.bars_held > self.max_hold_bars:
                        gain = (bar.close - state.avg_entry) / state.avg_entry if state.avg_entry > 0 else 0
                        if gain < 0.005:
                            signals += self.pm.exit_position(symbol)
                    elif (state.pyramid_count < self.pyramid_levels and
                          bar.close > state.avg_entry + atr and qty > 0):
                        add_qty = max(1, state.original_qty // 2)
                        signals += self.pm.add_pyramid(symbol, add_qty, bar.close)

                # --- Short: mirror ---
                elif self.pm.is_short(symbol):
                    state = self.pm.get_state(symbol)
                    if bar.close < self.lowest[symbol]:
                        self.lowest[symbol] = bar.close
                    new_stop = self.lowest[symbol] + atr * self.atr_multiplier
                    signals += self.pm.update_trailing_stop(symbol, new_stop)

                    if prev is not None and fast_above and not prev:
                        signals += self.pm.exit_position(symbol)
                    elif state.bars_held > self.max_hold_bars:
                        gain = (state.avg_entry - bar.close) / state.avg_entry if state.avg_entry > 0 else 0
                        if gain < 0.005:
                            signals += self.pm.exit_position(symbol)
                    elif (state.pyramid_count < self.pyramid_levels and
                          bar.close < state.avg_entry - atr and qty > 0):
                        add_qty = max(1, state.original_qty // 2)
                        signals += self.pm.add_pyramid(symbol, add_qty, bar.close)

                self.prev_fast_above[symbol] = fast_above
        return signals

    def on_complete(self):
        return {"strategy_type": "sma_crossover"}
