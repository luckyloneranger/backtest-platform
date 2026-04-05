"""RSI Mean Reversion — pyramid entries with trend filter.

Long: RSI < 35/25/15 in daily uptrend -> 3-level pyramid entry
Short: RSI > 65/75/85 in daily downtrend -> 3-level short pyramid (MIS)
Exits: partial at RSI > 60, full at RSI > 70/trend reversal/time stop
"""

from collections import deque
from server.registry import register
from strategies.base import Strategy, MarketSnapshot, InstrumentInfo, Signal
from strategies.indicators import compute_rsi, compute_ema, compute_atr
from strategies.position_manager import PositionManager


@register("rsi_daily_trend")
class RsiDailyTrend(Strategy):

    def required_data(self):
        return [
            {"interval": "15minute", "lookback": 100},
            {"interval": "day", "lookback": 50},
        ]

    def initialize(self, config, instruments):
        self.rsi_period = config.get("rsi_period", 14)
        self.rsi_entry_1 = config.get("rsi_entry_1", 35)
        self.rsi_entry_2 = config.get("rsi_entry_2", 25)
        self.rsi_entry_3 = config.get("rsi_entry_3", 15)
        self.rsi_partial_exit = config.get("rsi_partial_exit", 60)
        self.rsi_full_exit = config.get("rsi_full_exit", 70)
        self.ema_period = config.get("ema_period", 20)
        self.risk_pct = config.get("risk_pct", 0.2)
        self.max_pyramid_levels = config.get("max_pyramid_levels", 2)
        self.atr_period = config.get("atr_period", 14)
        self.atr_multiplier = config.get("atr_multiplier", 2.0)
        self.max_loss_pct = config.get("max_loss_pct", 0.03)
        self.max_hold_bars = config.get("max_hold_bars", 40)
        self.cooldown_bars = config.get("cooldown_bars", 50)
        self.instruments = instruments

        self.pm = PositionManager(max_pending_bars=3)

        # Per-symbol indicator state (position state lives in PM)
        self.prices_15m: dict[str, deque] = {}
        self.prices_daily: dict[str, deque] = {}
        self.daily_highs: dict[str, deque] = {}
        self.daily_lows: dict[str, deque] = {}
        self.daily_closes: dict[str, deque] = {}
        self.ema_history: dict[str, deque] = {}
        self.trend_up: dict[str, bool] = {}
        self.trend_down: dict[str, bool] = {}
        self.last_exit_bar: dict[str, int] = {}

    def _ensure(self, symbol):
        if symbol not in self.prices_15m:
            self.prices_15m[symbol] = deque(maxlen=self.rsi_period + 20)
            self.prices_daily[symbol] = deque(maxlen=60)
            self.daily_highs[symbol] = deque(maxlen=60)
            self.daily_lows[symbol] = deque(maxlen=60)
            self.daily_closes[symbol] = deque(maxlen=60)
            self.ema_history[symbol] = deque(maxlen=10)
            self.trend_up[symbol] = False
            self.trend_down[symbol] = False
            self.last_exit_bar[symbol] = 0

    def on_bar(self, snapshot):
        self.pm.increment_bars()
        signals = self.pm.process_fills(snapshot)
        signals += self.pm.resubmit_expired(snapshot)
        self.pm.reconcile(snapshot)

        # Update daily data + trend
        if "day" in snapshot.timeframes:
            for symbol, bar in snapshot.timeframes["day"].items():
                self._ensure(symbol)
                self.prices_daily[symbol].append(bar.close)
                self.daily_highs[symbol].append(bar.high)
                self.daily_lows[symbol].append(bar.low)
                self.daily_closes[symbol].append(bar.close)
                ema = compute_ema(list(self.prices_daily[symbol]), self.ema_period)
                if ema is not None:
                    self.ema_history[symbol].append(ema)
                    self.trend_up[symbol] = (bar.close > ema and
                        (len(self.ema_history[symbol]) < 6 or
                         self.ema_history[symbol][-1] > self.ema_history[symbol][-6]))
                    self.trend_down[symbol] = (bar.close < ema and
                        (len(self.ema_history[symbol]) < 6 or
                         self.ema_history[symbol][-1] < self.ema_history[symbol][-6]))

        # Process 15-minute bars
        if "15minute" not in snapshot.timeframes:
            return signals

        for symbol, bar in snapshot.timeframes["15minute"].items():
            self._ensure(symbol)
            self.prices_15m[symbol].append(bar.close)

            rsi = compute_rsi(list(self.prices_15m[symbol]), self.rsi_period)
            if rsi is None:
                continue

            atr = compute_atr(list(self.daily_highs.get(symbol, [])),
                              list(self.daily_lows.get(symbol, [])),
                              list(self.daily_closes.get(symbol, [])), self.atr_period)

            target_qty = int(self.risk_pct * snapshot.portfolio.cash / bar.close) if bar.close > 0 else 0
            inst = self.instruments.get(symbol)
            if inst and inst.lot_size > 1:
                target_qty = (target_qty // inst.lot_size) * inst.lot_size
            level_qty = max(target_qty // 3, 1) if target_qty > 0 else 0

            state = self.pm.get_state(symbol)

            # --- Flat: check entries (with cooldown) ---
            if self.pm.is_flat(symbol) and not self.pm.has_pending_entry(symbol):
                in_cooldown = (self.last_exit_bar.get(symbol, 0) > 0 and
                               self.pm.bar_count - self.last_exit_bar[symbol] < self.cooldown_bars)
                if not in_cooldown and level_qty > 0:
                    # Long entry
                    if rsi < self.rsi_entry_1 and self.trend_up.get(symbol, False):
                        product = "CNC" if rsi < self.rsi_entry_2 and self.trend_up[symbol] else "MIS"
                        stop = bar.close * (1 - self.max_loss_pct) if atr is None else bar.close - self.atr_multiplier * atr
                        signals += self.pm.enter_long(symbol, level_qty, bar.close * 0.999, product, stop)
                    # Short entry
                    elif rsi > (100 - self.rsi_entry_1) and self.trend_down.get(symbol, False):
                        stop = bar.close * (1 + self.max_loss_pct) if atr is None else bar.close + self.atr_multiplier * atr
                        signals += self.pm.enter_short(symbol, level_qty, bar.close * 1.001, stop)

            # --- Long: pyramids + exits ---
            elif self.pm.is_long(symbol):
                # Pyramid entries
                if (state.pyramid_count < self.max_pyramid_levels and level_qty > 0 and
                        not self.pm.has_pending_entry(symbol)):
                    if state.pyramid_count == 0 and rsi < self.rsi_entry_2:
                        signals += self.pm.add_pyramid(symbol, level_qty, bar.close * 0.999)
                    elif state.pyramid_count == 1 and rsi < self.rsi_entry_3:
                        signals += self.pm.add_pyramid(symbol, level_qty, bar.close * 0.999)

                # Trailing stop update
                if atr is not None:
                    new_stop = bar.close - self.atr_multiplier * atr
                    signals += self.pm.update_trailing_stop(symbol, new_stop)

                # Exits
                if rsi > self.rsi_full_exit or self.trend_down.get(symbol, False):
                    self.last_exit_bar[symbol] = self.pm.bar_count
                    signals += self.pm.exit_position(symbol)
                elif rsi > self.rsi_partial_exit and not state.partial_taken and state.qty > 1:
                    signals += self.pm.exit_position(symbol, qty=state.qty // 2)
                elif state.bars_held > self.max_hold_bars:
                    gain = (bar.close - state.avg_entry) / state.avg_entry if state.avg_entry > 0 else 0
                    if gain < 0.005:
                        self.last_exit_bar[symbol] = self.pm.bar_count
                        signals += self.pm.exit_position(symbol)

            # --- Short: mirror ---
            elif self.pm.is_short(symbol):
                if (state.pyramid_count < self.max_pyramid_levels and level_qty > 0 and
                        not self.pm.has_pending_entry(symbol)):
                    if state.pyramid_count == 0 and rsi > (100 - self.rsi_entry_2):
                        signals += self.pm.add_pyramid(symbol, level_qty, bar.close * 1.001)
                    elif state.pyramid_count == 1 and rsi > (100 - self.rsi_entry_3):
                        signals += self.pm.add_pyramid(symbol, level_qty, bar.close * 1.001)

                if atr is not None:
                    new_stop = bar.close + self.atr_multiplier * atr
                    signals += self.pm.update_trailing_stop(symbol, new_stop)

                if rsi < (100 - self.rsi_full_exit) or self.trend_up.get(symbol, False):
                    self.last_exit_bar[symbol] = self.pm.bar_count
                    signals += self.pm.exit_position(symbol)
                elif rsi < (100 - self.rsi_partial_exit) and not state.partial_taken and state.qty > 1:
                    signals += self.pm.exit_position(symbol, qty=state.qty // 2)
                elif state.bars_held > self.max_hold_bars:
                    gain = (state.avg_entry - bar.close) / state.avg_entry if state.avg_entry > 0 else 0
                    if gain < 0.005:
                        self.last_exit_bar[symbol] = self.pm.bar_count
                        signals += self.pm.exit_position(symbol)

        return signals

    def on_complete(self):
        return {"strategy_type": "rsi_daily_trend", "rsi_period": self.rsi_period}
