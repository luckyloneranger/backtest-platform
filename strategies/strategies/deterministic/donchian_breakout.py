"""Donchian Breakout — channel breakout with risk sizing and trailing stops.

Long: price > N-day channel high with volume → MARKET entry
Short: price < N-day channel low with volume → MARKET short (MIS)
Profit target: LIMIT at entry + 2*ATR (partial)
Trailing stop: 1.5*ATR from highest/lowest since entry
"""

from collections import deque
from server.registry import register
from strategies.base import Strategy, MarketSnapshot, InstrumentInfo, Signal
from strategies.indicators import compute_atr, compute_adx
from strategies.position_manager import PositionManager


@register("donchian_breakout")
class DonchianBreakout(Strategy):

    def required_data(self):
        return [
            {"interval": "day", "lookback": 60},
            {"interval": "15minute", "lookback": 30},
        ]

    def initialize(self, config, instruments):
        self.channel_period = config.get("channel_period", 20)
        self.atr_period = config.get("atr_period", 14)
        self.atr_multiplier = config.get("atr_multiplier", 2.0)
        self.volume_factor = config.get("volume_factor", 1.0)
        self.risk_per_trade = config.get("risk_per_trade", 0.01)
        self.profit_target_atr = config.get("profit_target_atr", 2.0)
        self.max_loss_pct = config.get("max_loss_pct", 0.02)
        self.max_hold_bars = config.get("max_hold_bars", 30)
        self.pyramid_levels = config.get("pyramid_levels", 2)
        self.min_adx = config.get("min_adx", 0)
        self.instruments = instruments

        self.pm = PositionManager(max_pending_bars=1)

        # Per-symbol indicator state
        self.daily_highs: dict[str, deque] = {}
        self.daily_lows: dict[str, deque] = {}
        self.daily_closes: dict[str, deque] = {}
        self.daily_volumes: dict[str, deque] = {}
        self.current_atr: dict[str, float] = {}
        self.highest: dict[str, float] = {}
        self.lowest: dict[str, float] = {}
        self.adx_values: dict[str, float] = {}

    def _ensure(self, symbol):
        if symbol not in self.daily_highs:
            maxlen = self.channel_period + self.atr_period + 10
            self.daily_highs[symbol] = deque(maxlen=maxlen)
            self.daily_lows[symbol] = deque(maxlen=maxlen)
            self.daily_closes[symbol] = deque(maxlen=maxlen)
            self.daily_volumes[symbol] = deque(maxlen=maxlen)
            self.current_atr[symbol] = 0.0
            self.highest[symbol] = 0.0
            self.lowest[symbol] = float('inf')

    def on_bar(self, snapshot):
        self.pm.increment_bars()
        signals = self.pm.process_fills(snapshot)
        signals += self.pm.resubmit_expired(snapshot)
        self.pm.reconcile(snapshot)

        # Update daily data
        if "day" in snapshot.timeframes:
            for symbol, bar in snapshot.timeframes["day"].items():
                self._ensure(symbol)
                self.daily_highs[symbol].append(bar.high)
                self.daily_lows[symbol].append(bar.low)
                self.daily_closes[symbol].append(bar.close)
                self.daily_volumes[symbol].append(bar.volume)
                atr = compute_atr(list(self.daily_highs[symbol]), list(self.daily_lows[symbol]),
                                  list(self.daily_closes[symbol]), self.atr_period)
                if atr is not None:
                    self.current_atr[symbol] = atr
                adx = compute_adx(list(self.daily_highs[symbol]), list(self.daily_lows[symbol]),
                                  list(self.daily_closes[symbol]), 14)
                if adx is not None:
                    self.adx_values[symbol] = adx

        # Process 15-minute bars
        if "15minute" not in snapshot.timeframes:
            return signals

        for symbol, bar in snapshot.timeframes["15minute"].items():
            self._ensure(symbol)
            highs = list(self.daily_highs.get(symbol, []))
            lows = list(self.daily_lows.get(symbol, []))
            volumes = list(self.daily_volumes.get(symbol, []))

            if len(highs) < self.channel_period + 1:
                continue

            # Donchian channel (exclude current day)
            ch_high = max(highs[-(self.channel_period + 1):-1])
            ch_low = min(lows[-(self.channel_period + 1):-1])
            avg_vol = sum(volumes[-(self.channel_period + 1):-1]) / self.channel_period
            atr = self.current_atr.get(symbol, 0.0)

            state = self.pm.get_state(symbol)

            # --- Flat: check breakout entries ---
            if self.pm.is_flat(symbol) and not self.pm.has_pending_entry(symbol) and atr > 0:
                vol_ok = volumes and volumes[-1] >= avg_vol * self.volume_factor
                qty = int(snapshot.portfolio.cash * self.risk_per_trade / (atr * self.atr_multiplier))
                inst = self.instruments.get(symbol)
                if inst and inst.lot_size > 1:
                    qty = (qty // inst.lot_size) * inst.lot_size

                if qty > 0 and vol_ok and self.adx_values.get(symbol, 0) > self.min_adx:
                    # Long breakout
                    if bar.close > ch_high:
                        vol_strong = volumes[-1] > avg_vol * 1.5 if volumes else False
                        product = "CNC" if vol_strong else "MIS"
                        stop = bar.close - atr * self.atr_multiplier
                        signals += self.pm.enter_long(symbol, qty, 0, product, stop)
                        self.highest[symbol] = bar.close
                        target = bar.close + self.profit_target_atr * atr
                        signals += self.pm.set_profit_target(symbol, max(1, qty // 3), target)

                    # Short breakout
                    elif bar.close < ch_low:
                        stop = bar.close + atr * self.atr_multiplier
                        signals += self.pm.enter_short(symbol, qty, 0, stop)
                        self.lowest[symbol] = bar.close
                        target = bar.close - self.profit_target_atr * atr
                        signals += self.pm.set_profit_target(symbol, max(1, qty // 3), target)

            # --- Long: trailing + exits + pyramid ---
            elif self.pm.is_long(symbol):
                if bar.close > self.highest[symbol]:
                    self.highest[symbol] = bar.close
                new_stop = self.highest[symbol] - atr * self.atr_multiplier if atr > 0 else state.trailing_stop
                signals += self.pm.update_trailing_stop(symbol, new_stop)

                if bar.close < ch_low:
                    signals += self.pm.exit_position(symbol)
                elif state.avg_entry > 0 and bar.close < state.avg_entry * (1 - self.max_loss_pct):
                    signals += self.pm.exit_position(symbol)
                elif state.bars_held > self.max_hold_bars:
                    gain = (bar.close - state.avg_entry) / state.avg_entry if state.avg_entry > 0 else 0
                    if gain < 0.005:
                        signals += self.pm.exit_position(symbol)
                elif (state.pyramid_count < self.pyramid_levels and atr > 0 and
                      bar.close > state.avg_entry + atr and not self.pm.has_pending_entry(symbol)):
                    add_qty = max(1, state.original_qty // 2)
                    signals += self.pm.add_pyramid(symbol, add_qty, 0)

            # --- Short: mirror ---
            elif self.pm.is_short(symbol):
                if bar.close < self.lowest[symbol]:
                    self.lowest[symbol] = bar.close
                new_stop = self.lowest[symbol] + atr * self.atr_multiplier if atr > 0 else state.trailing_stop
                signals += self.pm.update_trailing_stop(symbol, new_stop)

                if bar.close > ch_high:
                    signals += self.pm.exit_position(symbol)
                elif state.avg_entry > 0 and bar.close > state.avg_entry * (1 + self.max_loss_pct):
                    signals += self.pm.exit_position(symbol)
                elif state.bars_held > self.max_hold_bars:
                    gain = (state.avg_entry - bar.close) / state.avg_entry if state.avg_entry > 0 else 0
                    if gain < 0.005:
                        signals += self.pm.exit_position(symbol)
                elif (state.pyramid_count < self.pyramid_levels and atr > 0 and
                      bar.close < state.avg_entry - atr and not self.pm.has_pending_entry(symbol)):
                    add_qty = max(1, state.original_qty // 2)
                    signals += self.pm.add_pyramid(symbol, add_qty, 0)

        return signals

    def on_complete(self):
        return {"strategy_type": "donchian_breakout"}
