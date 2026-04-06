"""Multi-Timeframe Confirmation -- 3-level signal agreement required.

Daily EMA(20) determines direction (only long above EMA, short below).
15-min MACD histogram confirms momentum (positive for longs, negative for shorts).
5-min RSI(14) times entry (RSI < 35 for long entry, RSI > 65 for short entry).

All three levels must agree before entry. Any disagreement -> exit.
"""

from collections import deque
from server.registry import register
from strategies.base import Strategy, MarketSnapshot, InstrumentInfo, Signal
from strategies.indicators import compute_ema, compute_macd, compute_rsi, compute_atr
from strategies.position_manager import PositionManager


@register("multi_tf_confirm")
class MultiTfConfirm(Strategy):

    def required_data(self):
        return [
            {"interval": "5minute", "lookback": 50},
            {"interval": "15minute", "lookback": 30},
            {"interval": "day", "lookback": 20},
        ]

    def initialize(self, config, instruments):
        self.ema_period = config.get("ema_period", 20)
        self.rsi_period = config.get("rsi_period", 14)
        self.rsi_entry_long = config.get("rsi_entry_long", 35)
        self.rsi_entry_short = config.get("rsi_entry_short", 65)
        self.risk_pct = config.get("risk_pct", 0.02)
        self.atr_period = config.get("atr_period", 14)
        self.atr_mult = config.get("atr_mult", 2.0)
        self.max_hold_bars = config.get("max_hold_bars", 40)
        self.ema_strong_pct = config.get("ema_strong_pct", 0.02)
        self.instruments = instruments

        self.pm = PositionManager(max_pending_bars=3)

        # Per-symbol indicator state
        self.daily_closes: dict[str, deque] = {}
        self.prices_15m: dict[str, deque] = {}
        self.prices_5m: dict[str, deque] = {}
        self.highs_5m: dict[str, deque] = {}
        self.lows_5m: dict[str, deque] = {}
        self.closes_5m: dict[str, deque] = {}

        # Per-symbol signal state
        self.daily_trend: dict[str, str] = {}      # "UP", "DOWN", "NEUTRAL"
        self.daily_ema: dict[str, float] = {}       # last EMA value
        self.daily_close: dict[str, float] = {}     # last daily close
        self.macd_bullish: dict[str, bool] = {}

    def _ensure(self, symbol):
        if symbol not in self.daily_closes:
            self.daily_closes[symbol] = deque(maxlen=self.ema_period + 10)
            self.prices_15m[symbol] = deque(maxlen=50)
            self.prices_5m[symbol] = deque(maxlen=self.rsi_period + 20)
            self.highs_5m[symbol] = deque(maxlen=self.atr_period + 10)
            self.lows_5m[symbol] = deque(maxlen=self.atr_period + 10)
            self.closes_5m[symbol] = deque(maxlen=self.atr_period + 10)
            self.daily_trend[symbol] = "NEUTRAL"
            self.daily_ema[symbol] = 0.0
            self.daily_close[symbol] = 0.0
            self.macd_bullish[symbol] = False

    def on_bar(self, snapshot):
        self.pm.increment_bars()
        signals = self.pm.process_fills(snapshot)
        signals += self.pm.resubmit_expired(snapshot)
        self.pm.reconcile(snapshot)

        # --- Daily bar: determine trend via EMA(20) ---
        if "day" in snapshot.timeframes:
            for symbol, bar in snapshot.timeframes["day"].items():
                self._ensure(symbol)
                self.daily_closes[symbol].append(bar.close)
                ema = compute_ema(list(self.daily_closes[symbol]), self.ema_period)
                if ema is not None:
                    self.daily_ema[symbol] = ema
                    self.daily_close[symbol] = bar.close
                    if bar.close > ema:
                        self.daily_trend[symbol] = "UP"
                    elif bar.close < ema:
                        self.daily_trend[symbol] = "DOWN"
                    else:
                        self.daily_trend[symbol] = "NEUTRAL"

        # --- 15-min bar: track MACD histogram ---
        if "15minute" in snapshot.timeframes:
            for symbol, bar in snapshot.timeframes["15minute"].items():
                self._ensure(symbol)
                self.prices_15m[symbol].append(bar.close)
                macd = compute_macd(list(self.prices_15m[symbol]))
                if macd is not None:
                    _, _, histogram = macd
                    self.macd_bullish[symbol] = histogram > 0

        # --- 5-min bar: RSI timing for entries/exits ---
        if "5minute" not in snapshot.timeframes:
            return signals

        for symbol, bar in snapshot.timeframes["5minute"].items():
            self._ensure(symbol)
            self.prices_5m[symbol].append(bar.close)
            self.highs_5m[symbol].append(bar.high)
            self.lows_5m[symbol].append(bar.low)
            self.closes_5m[symbol].append(bar.close)

            rsi = compute_rsi(list(self.prices_5m[symbol]), self.rsi_period)
            if rsi is None:
                continue

            atr = compute_atr(
                list(self.highs_5m[symbol]),
                list(self.lows_5m[symbol]),
                list(self.closes_5m[symbol]),
                self.atr_period,
            )

            trend = self.daily_trend.get(symbol, "NEUTRAL")
            macd_bull = self.macd_bullish.get(symbol, False)
            state = self.pm.get_state(symbol)

            # --- Flat: check for triple-agreement entries ---
            if self.pm.is_flat(symbol) and not self.pm.has_pending_entry(symbol):
                # Position sizing: risk_pct * cash / price, capped to available cash
                target_qty = int(self.risk_pct * snapshot.portfolio.cash / bar.close) if bar.close > 0 else 0
                target_qty = min(target_qty, int(snapshot.portfolio.cash / bar.close)) if bar.close > 0 else 0
                inst = self.instruments.get(symbol)
                if inst and inst.lot_size > 1:
                    target_qty = (target_qty // inst.lot_size) * inst.lot_size

                if target_qty > 0:
                    # Long entry: daily UP + MACD bullish + RSI < threshold
                    if trend == "UP" and macd_bull and rsi < self.rsi_entry_long:
                        stop = bar.close * 0.97 if atr is None else bar.close - self.atr_mult * atr
                        # CNC if daily trend is strong (close > EMA by > ema_strong_pct), else MIS
                        ema_val = self.daily_ema.get(symbol, 0.0)
                        daily_c = self.daily_close.get(symbol, 0.0)
                        strong = ema_val > 0 and daily_c > ema_val * (1 + self.ema_strong_pct)
                        product = "CNC" if strong else "MIS"
                        signals += self.pm.enter_long(symbol, target_qty, 0, product, stop)

                    # Short entry: daily DOWN + MACD bearish + RSI > threshold
                    elif trend == "DOWN" and not macd_bull and rsi > self.rsi_entry_short:
                        stop = bar.close * 1.03 if atr is None else bar.close + self.atr_mult * atr
                        signals += self.pm.enter_short(symbol, target_qty, 0, stop)

            # --- Long position: exit on disagreement ---
            elif self.pm.is_long(symbol):
                # Trailing stop update
                if atr is not None:
                    new_stop = bar.close - self.atr_mult * atr
                    signals += self.pm.update_trailing_stop(symbol, new_stop)

                # Exit if any timeframe disagrees
                if trend != "UP" or not macd_bull:
                    signals += self.pm.exit_position(symbol)
                # Time stop
                elif state.bars_held > self.max_hold_bars:
                    gain = (bar.close - state.avg_entry) / state.avg_entry if state.avg_entry > 0 else 0
                    if gain < 0.005:
                        signals += self.pm.exit_position(symbol)

            # --- Short position: exit on disagreement ---
            elif self.pm.is_short(symbol):
                # Trailing stop update
                if atr is not None:
                    new_stop = bar.close + self.atr_mult * atr
                    signals += self.pm.update_trailing_stop(symbol, new_stop)

                # Exit if any timeframe disagrees
                if trend != "DOWN" or macd_bull:
                    signals += self.pm.exit_position(symbol)
                # Time stop
                elif state.bars_held > self.max_hold_bars:
                    gain = (state.avg_entry - bar.close) / state.avg_entry if state.avg_entry > 0 else 0
                    if gain < 0.005:
                        signals += self.pm.exit_position(symbol)

        return signals

    def on_complete(self):
        return {
            "strategy_type": "multi_tf_confirm",
            "ema_period": self.ema_period,
            "rsi_period": self.rsi_period,
            "atr_mult": self.atr_mult,
        }
