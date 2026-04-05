"""Multi-Indicator Confluence -- trade only when 3+ indicators agree on direction.

Indicators: RSI, MACD, Bollinger Bands, ADX (trend strength), OBV slope (volume)
Each contributes +1 (bullish), -1 (bearish), or 0 (neutral)
Entry: score >= threshold (default 2) for long, <= -threshold for short
Exit: score drops below entry threshold (< 1 long, > -1 short), trailing stop, time stop
"""

from collections import deque
from server.registry import register
from strategies.base import Strategy, MarketSnapshot, InstrumentInfo, Signal
from strategies.indicators import (
    compute_rsi, compute_macd, compute_bollinger,
    compute_adx, compute_obv_slope, compute_atr,
)
from strategies.position_manager import PositionManager


@register("confluence")
class Confluence(Strategy):

    def required_data(self):
        return [{"interval": "day", "lookback": 50}]

    def initialize(self, config, instruments):
        self.rsi_period = config.get("rsi_period", 14)
        self.rsi_oversold = config.get("rsi_oversold", 35)
        self.rsi_overbought = config.get("rsi_overbought", 65)
        self.macd_fast = config.get("macd_fast", 12)
        self.macd_slow = config.get("macd_slow", 26)
        self.macd_signal = config.get("macd_signal", 9)
        self.bb_period = config.get("bb_period", 20)
        self.bb_std = config.get("bb_std", 2.0)
        self.adx_period = config.get("adx_period", 14)
        self.adx_trend_threshold = config.get("adx_trend_threshold", 25)
        self.obv_period = config.get("obv_period", 10)
        self.atr_period = config.get("atr_period", 14)
        self.atr_multiplier = config.get("atr_multiplier", 1.5)
        self.risk_per_trade = config.get("risk_per_trade", 0.02)
        self.threshold = config.get("threshold", 2)
        self.max_hold_bars = config.get("max_hold_bars", 20)
        self.min_gain_for_hold = config.get("min_gain_for_hold", 0.005)

        self.pm = PositionManager(max_pending_bars=1)

        # Per-symbol price/volume deques
        self.closes: dict[str, deque] = {}
        self.highs: dict[str, deque] = {}
        self.lows: dict[str, deque] = {}
        self.volumes: dict[str, deque] = {}
        self.highest: dict[str, float] = {}
        self.lowest: dict[str, float] = {}

    def _ensure(self, symbol):
        if symbol not in self.closes:
            maxlen = max(self.macd_slow + self.macd_signal,
                         self.bb_period, self.adx_period * 2,
                         self.rsi_period + 1, self.atr_period + 1) + 10
            self.closes[symbol] = deque(maxlen=maxlen)
            self.highs[symbol] = deque(maxlen=maxlen)
            self.lows[symbol] = deque(maxlen=maxlen)
            self.volumes[symbol] = deque(maxlen=maxlen)
            self.highest[symbol] = 0.0
            self.lowest[symbol] = float("inf")

    def _compute_score(self, symbol):
        """Compute confluence score from all 5 indicators. Returns (score, adx_value)."""
        c = list(self.closes[symbol])
        h = list(self.highs[symbol])
        lo = list(self.lows[symbol])
        v = list(self.volumes[symbol])

        rsi = compute_rsi(c, self.rsi_period)
        macd = compute_macd(c, self.macd_fast, self.macd_slow, self.macd_signal)
        bb = compute_bollinger(c, self.bb_period, self.bb_std)
        adx = compute_adx(h, lo, c, self.adx_period)
        obv_slope = compute_obv_slope(c, v, self.obv_period)

        if any(x is None for x in [rsi, macd, bb, adx, obv_slope]):
            return None, None

        score = 0
        _, _, hist = macd
        upper, _, lower = bb
        close = c[-1]

        # RSI
        if rsi < self.rsi_oversold:
            score += 1
        elif rsi > self.rsi_overbought:
            score -= 1

        # MACD histogram
        macd_vote = 1 if hist > 0 else -1 if hist < 0 else 0
        # ADX amplification: double MACD weight in strong trends
        if adx > self.adx_trend_threshold:
            macd_vote *= 2
        score += macd_vote

        # Bollinger Bands
        if close < lower:
            score += 1
        elif close > upper:
            score -= 1

        # OBV slope
        if obv_slope > 0:
            score += 1
        elif obv_slope < 0:
            score -= 1

        return score, adx

    def on_bar(self, snapshot):
        self.pm.increment_bars()
        signals = self.pm.process_fills(snapshot)
        signals += self.pm.resubmit_expired(snapshot)
        self.pm.reconcile(snapshot)

        for interval, bars in snapshot.timeframes.items():
            for symbol, bar in bars.items():
                self._ensure(symbol)
                self.closes[symbol].append(bar.close)
                self.highs[symbol].append(bar.high)
                self.lows[symbol].append(bar.low)
                self.volumes[symbol].append(bar.volume)

                score, adx = self._compute_score(symbol)
                if score is None:
                    continue

                c = list(self.closes[symbol])
                h = list(self.highs[symbol])
                lo = list(self.lows[symbol])
                atr = compute_atr(h, lo, c, self.atr_period)
                if atr is None or atr <= 0:
                    continue

                qty = int(snapshot.context.initial_capital
                          * self.risk_per_trade / (atr * self.atr_multiplier))
                if qty <= 0:
                    continue

                # --- Flat: check entries ---
                if self.pm.is_flat(symbol) and not self.pm.has_pending_entry(symbol):
                    if score >= self.threshold:
                        product = "CNC" if adx > 30 else "MIS"
                        stop = bar.close - atr * self.atr_multiplier
                        signals += self.pm.enter_long(symbol, qty, 0, product, stop)
                        self.highest[symbol] = bar.close
                    elif score <= -self.threshold:
                        stop = bar.close + atr * self.atr_multiplier
                        signals += self.pm.enter_short(symbol, qty, 0, stop)
                        self.lowest[symbol] = bar.close

                # --- Long: trailing stop + exits ---
                elif self.pm.is_long(symbol):
                    state = self.pm.get_state(symbol)
                    if bar.close > self.highest[symbol]:
                        self.highest[symbol] = bar.close
                    new_stop = self.highest[symbol] - atr * self.atr_multiplier
                    signals += self.pm.update_trailing_stop(symbol, new_stop)

                    if score < 1:
                        signals += self.pm.exit_position(symbol)
                    elif state.bars_held > self.max_hold_bars:
                        gain = ((bar.close - state.avg_entry) / state.avg_entry
                                if state.avg_entry > 0 else 0)
                        if gain < self.min_gain_for_hold:
                            signals += self.pm.exit_position(symbol)

                # --- Short: trailing stop + exits ---
                elif self.pm.is_short(symbol):
                    state = self.pm.get_state(symbol)
                    if bar.close < self.lowest[symbol]:
                        self.lowest[symbol] = bar.close
                    new_stop = self.lowest[symbol] + atr * self.atr_multiplier
                    signals += self.pm.update_trailing_stop(symbol, new_stop)

                    if score > -1:
                        signals += self.pm.exit_position(symbol)
                    elif state.bars_held > self.max_hold_bars:
                        gain = ((state.avg_entry - bar.close) / state.avg_entry
                                if state.avg_entry > 0 else 0)
                        if gain < self.min_gain_for_hold:
                            signals += self.pm.exit_position(symbol)

        return signals

    def on_complete(self):
        return {"strategy_type": "confluence"}
