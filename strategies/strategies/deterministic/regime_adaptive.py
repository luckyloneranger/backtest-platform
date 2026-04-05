"""Regime-Adaptive -- detects market regime and switches sub-strategies.

Regime detection (from daily bars):
  ADX > 25 -> TRENDING: use MACD crossover for entries
  ADX < 20, low BBW -> RANGING: use Bollinger band mean-reversion
  High BBW -> VOLATILE: stay flat or very selective entries

Trending: MACD cross + ADX rising -> entry, 2x ATR trailing stop, pyramiding
Ranging: close < BB lower + RSI < 35 -> long, close > BB upper + RSI > 65 -> short, 1x ATR stop
Volatile: mostly flat, enter only with 3+ signals confirming, 50% size, 3x ATR stop
"""

from collections import deque
from server.registry import register
from strategies.base import Strategy, MarketSnapshot, InstrumentInfo, Signal
from strategies.indicators import (
    compute_adx, compute_macd, compute_bollinger, compute_bbw,
    compute_rsi, compute_atr, compute_sma, compute_obv_slope,
)
from strategies.position_manager import PositionManager


@register("regime_adaptive")
class RegimeAdaptive(Strategy):

    def required_data(self):
        return [
            {"interval": "15minute", "lookback": 100},
            {"interval": "day", "lookback": 60},
        ]

    def initialize(self, config, instruments):
        # Regime thresholds
        self.adx_trend = config.get("adx_trend", 25)
        self.adx_range = config.get("adx_range", 20)
        self.bbw_volatile_mult = config.get("bbw_volatile_mult", 1.5)

        # Stop multipliers per regime
        self.trending_atr_mult = config.get("trending_atr_mult", 2.0)
        self.ranging_atr_mult = config.get("ranging_atr_mult", 1.5)
        self.volatile_atr_mult = config.get("volatile_atr_mult", 3.0)

        # Sizing / risk
        self.volatile_size_pct = config.get("volatile_size_pct", 0.5)
        self.risk_per_trade = config.get("risk_per_trade", 0.015)
        self.pyramid_levels = config.get("pyramid_levels", 2)

        # Indicator params
        self.rsi_oversold = config.get("rsi_oversold", 35)
        self.rsi_overbought = config.get("rsi_overbought", 65)
        self.max_hold_bars = config.get("max_hold_bars", 20)

        # Regime smoothing
        self.regime_confirm_bars = config.get("regime_confirm_bars", 3)

        self.instruments = instruments
        self.pm = PositionManager(max_pending_bars=3)

        # Per-symbol state
        self.daily_highs: dict[str, deque] = {}
        self.daily_lows: dict[str, deque] = {}
        self.daily_closes: dict[str, deque] = {}
        self.bbw_history: dict[str, deque] = {}
        self.prices_15m: dict[str, deque] = {}
        self.volumes_15m: dict[str, deque] = {}
        self.highs_15m: dict[str, deque] = {}
        self.lows_15m: dict[str, deque] = {}
        self.regime: dict[str, str] = {}            # "TRENDING", "RANGING", "VOLATILE"
        self.pending_regime: dict[str, str] = {}    # candidate regime for smoothing
        self.regime_counter: dict[str, int] = {}    # consecutive bars in candidate
        self.prev_macd_hist: dict[str, float | None] = {}
        self.current_atr: dict[str, float] = {}

    def _ensure(self, symbol):
        if symbol not in self.daily_highs:
            self.daily_highs[symbol] = deque(maxlen=60)
            self.daily_lows[symbol] = deque(maxlen=60)
            self.daily_closes[symbol] = deque(maxlen=60)
            self.bbw_history[symbol] = deque(maxlen=30)
            self.prices_15m[symbol] = deque(maxlen=200)
            self.volumes_15m[symbol] = deque(maxlen=200)
            self.highs_15m[symbol] = deque(maxlen=200)
            self.lows_15m[symbol] = deque(maxlen=200)
            self.regime[symbol] = "RANGING"
            self.prev_macd_hist[symbol] = None
            self.current_atr[symbol] = 0.0

    def _detect_regime(self, symbol):
        """Classify regime from daily data."""
        highs = list(self.daily_highs[symbol])
        lows = list(self.daily_lows[symbol])
        closes = list(self.daily_closes[symbol])

        adx = compute_adx(highs, lows, closes, 14)
        bbw = compute_bbw(closes, 20)

        if bbw is not None:
            self.bbw_history[symbol].append(bbw)

        avg_bbw = compute_sma(list(self.bbw_history[symbol]), 20)

        atr = compute_atr(highs, lows, closes, 14)
        if atr is not None:
            self.current_atr[symbol] = atr

        # Determine detected regime
        if bbw is not None and avg_bbw is not None and bbw > avg_bbw * self.bbw_volatile_mult:
            detected = "VOLATILE"
        elif adx is not None and adx > self.adx_trend:
            detected = "TRENDING"
        else:
            detected = "RANGING"

        # Regime smoothing: require regime_confirm_bars consecutive detections
        current = self.regime.get(symbol, "RANGING")
        if detected != current:
            if detected == self.pending_regime.get(symbol):
                self.regime_counter[symbol] = self.regime_counter.get(symbol, 0) + 1
                if self.regime_counter[symbol] >= self.regime_confirm_bars:
                    self.regime[symbol] = detected
                    self.regime_counter[symbol] = 0
            else:
                self.pending_regime[symbol] = detected
                self.regime_counter[symbol] = 1
        else:
            # Current regime re-confirmed, reset any pending switch
            self.pending_regime.pop(symbol, None)
            self.regime_counter.pop(symbol, None)

    def _atr_mult_for_regime(self, regime: str) -> float:
        if regime == "TRENDING":
            return self.trending_atr_mult
        elif regime == "VOLATILE":
            return self.volatile_atr_mult
        return self.ranging_atr_mult

    def _calc_qty(self, price, cash, regime):
        qty = int(self.risk_per_trade * cash / price) if price > 0 else 0
        if regime == "VOLATILE":
            qty = int(qty * self.volatile_size_pct)
        return max(qty, 0)

    def on_bar(self, snapshot):
        self.pm.increment_bars()
        signals = self.pm.process_fills(snapshot)
        signals += self.pm.resubmit_expired(snapshot)
        self.pm.reconcile(snapshot)

        # --- Daily bar processing: update regime ---
        if "day" in snapshot.timeframes:
            for symbol, bar in snapshot.timeframes["day"].items():
                self._ensure(symbol)
                self.daily_highs[symbol].append(bar.high)
                self.daily_lows[symbol].append(bar.low)
                self.daily_closes[symbol].append(bar.close)
                self._detect_regime(symbol)

        # --- Adjust stops on regime change for open positions ---
        for symbol in list(self.pm.states.keys()):
            state = self.pm.get_state(symbol)
            if state.direction == "flat":
                continue
            regime = self.regime.get(symbol, "RANGING")
            atr = self.current_atr.get(symbol, 0.0)
            if atr <= 0 or not state.has_engine_stop:
                continue
            mult = self._atr_mult_for_regime(regime)
            if regime == "VOLATILE" and mult > self.trending_atr_mult:
                # Tighten on transition to volatile: use 1.5x
                mult = 1.5
            if state.direction == "long":
                new_stop = state.avg_entry - mult * atr
                if new_stop > state.trailing_stop:
                    signals += self.pm.update_trailing_stop(symbol, new_stop)
            elif state.direction == "short":
                new_stop = state.avg_entry + mult * atr
                if new_stop < state.trailing_stop:
                    signals += self.pm.update_trailing_stop(symbol, new_stop)

        # --- 15-minute bar processing: trade based on regime ---
        if "15minute" not in snapshot.timeframes:
            return signals

        for symbol, bar in snapshot.timeframes["15minute"].items():
            self._ensure(symbol)
            self.prices_15m[symbol].append(bar.close)
            self.volumes_15m[symbol].append(bar.volume)
            self.highs_15m[symbol].append(bar.high)
            self.lows_15m[symbol].append(bar.low)

            regime = self.regime.get(symbol, "RANGING")
            atr = self.current_atr.get(symbol, 0.0)
            state = self.pm.get_state(symbol)
            prices = list(self.prices_15m[symbol])

            # --- Time stop for any regime ---
            if state.direction != "flat" and state.bars_held > self.max_hold_bars:
                signals += self.pm.exit_position(symbol)
                continue

            # --- Trailing stop update for open positions ---
            if state.direction != "flat" and atr > 0:
                mult = self._atr_mult_for_regime(regime)
                if state.direction == "long":
                    new_stop = bar.close - mult * atr
                    signals += self.pm.update_trailing_stop(symbol, new_stop)
                elif state.direction == "short":
                    new_stop = bar.close + mult * atr
                    signals += self.pm.update_trailing_stop(symbol, new_stop)

            # --- Pyramid for TRENDING only ---
            if (regime == "TRENDING" and state.direction != "flat"
                    and state.pyramid_count < self.pyramid_levels
                    and not self.pm.has_pending_entry(symbol) and atr > 0):
                if state.direction == "long" and bar.close > state.avg_entry + atr:
                    add_qty = max(1, state.original_qty // 2)
                    signals += self.pm.add_pyramid(symbol, add_qty, 0.0)
                elif state.direction == "short" and bar.close < state.avg_entry - atr:
                    add_qty = max(1, state.original_qty // 2)
                    signals += self.pm.add_pyramid(symbol, add_qty, 0.0)

            # --- New entries (only if flat) ---
            if not self.pm.is_flat(symbol) or self.pm.has_pending_entry(symbol):
                continue

            qty = self._calc_qty(bar.close, snapshot.portfolio.cash, regime)
            if qty <= 0 or atr <= 0:
                continue

            if regime == "TRENDING":
                signals += self._trending_entry(symbol, bar, prices, qty, atr)
            elif regime == "RANGING":
                signals += self._ranging_entry(symbol, bar, prices, qty, atr)
            elif regime == "VOLATILE":
                signals += self._volatile_entry(symbol, bar, prices, qty, atr)

        return signals

    def _trending_entry(self, symbol, bar, prices, qty, atr):
        """MACD crossover entry in trending regime."""
        macd = compute_macd(prices, 12, 26, 9)
        if macd is None:
            return []
        _, _, hist = macd
        prev = self.prev_macd_hist.get(symbol)
        self.prev_macd_hist[symbol] = hist

        if prev is None:
            return []

        stop_mult = self.trending_atr_mult
        if prev <= 0 < hist:  # bullish cross
            stop = bar.close - stop_mult * atr
            return self.pm.enter_long(symbol, qty, 0.0, "CNC", stop)
        elif prev >= 0 > hist:  # bearish cross
            stop = bar.close + stop_mult * atr
            return self.pm.enter_short(symbol, qty, 0.0, stop)
        return []

    def _ranging_entry(self, symbol, bar, prices, qty, atr):
        """Bollinger + RSI mean-reversion entry."""
        bb = compute_bollinger(prices, 20, 2.0)
        rsi = compute_rsi(prices, 14)
        if bb is None or rsi is None:
            return []

        upper, mid, lower = bb
        stop_mult = self.ranging_atr_mult

        if bar.close < lower and rsi < self.rsi_oversold:
            stop = bar.close - stop_mult * atr
            return self.pm.enter_long(symbol, qty, 0.0, "MIS", stop)
        elif bar.close > upper and rsi > self.rsi_overbought:
            stop = bar.close + stop_mult * atr
            return self.pm.enter_short(symbol, qty, 0.0, stop)
        return []

    def _volatile_entry(self, symbol, bar, prices, qty, atr):
        """Only enter when 3+ signals agree in volatile regime."""
        confirms = 0

        macd = compute_macd(prices, 12, 26, 9)
        rsi = compute_rsi(prices, 14)
        obv_slope = compute_obv_slope(
            prices, list(self.volumes_15m.get(symbol, [])), 10,
        )

        # Track MACD hist for volatile regime too
        if macd is not None:
            _, _, hist = macd
            self.prev_macd_hist[symbol] = hist

        # Bullish check
        bull = 0
        if macd is not None and macd[2] > 0:
            bull += 1
        if rsi is not None and rsi < self.rsi_oversold:
            bull += 1
        if obv_slope is not None and obv_slope > 0:
            bull += 1
        if bull >= 3:
            stop = bar.close - self.volatile_atr_mult * atr
            return self.pm.enter_long(symbol, qty, 0.0, "MIS", stop)

        # Bearish check
        bear = 0
        if macd is not None and macd[2] < 0:
            bear += 1
        if rsi is not None and rsi > self.rsi_overbought:
            bear += 1
        if obv_slope is not None and obv_slope < 0:
            bear += 1
        if bear >= 3:
            stop = bar.close + self.volatile_atr_mult * atr
            return self.pm.enter_short(symbol, qty, 0.0, stop)

        return []

    def on_complete(self):
        return {"strategy_type": "regime_adaptive"}
