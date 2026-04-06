"""Adaptive Ensemble Meta-Learner — learns which sub-signal combinations predict returns.

Runs 5 sub-strategies internally, each producing +1/0/-1 signals.
A LogisticRegression meta-model learns which combinations work.
Rolling retraining adapts weights to changing market conditions.

Why LogisticRegression: 7 features is too few for tree-based models to shine.
LogisticRegression learns linear combinations of sub-signals, which is exactly
what we want — "strong SMA + strong MACD + high ADX = strong signal."
Less prone to overfitting on 7 features.
"""

from collections import deque
import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
except ImportError:
    LogisticRegression = None

from server.registry import register
from strategies.base import Strategy, MarketSnapshot, InstrumentInfo, Signal
from strategies.indicators import (
    compute_sma, compute_rsi, compute_macd, compute_bollinger,
    compute_adx, compute_bbw, compute_atr,
)
from strategies.position_manager import PositionManager


@register("ensemble_meta")
class EnsembleMeta(Strategy):

    def required_data(self):
        return [{"interval": "day", "lookback": 200}]

    def initialize(self, config, instruments):
        self.min_train_bars = config.get("min_train_bars", 120)
        self.retrain_interval = config.get("retrain_interval", 20)
        self.forward_bars = config.get("forward_bars", 5)
        self.target_return = config.get("target_return", 0.01)
        self.confidence_threshold = config.get("confidence_threshold", 0.60)
        self.risk_pct = config.get("risk_pct", 0.03)
        self.atr_period = config.get("atr_period", 14)
        self.atr_mult = config.get("atr_mult", 2.0)
        self.max_hold_bars = config.get("max_hold_bars", 40)
        self.instruments = instruments

        self.pm = PositionManager(max_pending_bars=3)

        # Per-symbol state
        self.close_buffer: dict[str, deque] = {}
        self.high_buffer: dict[str, deque] = {}
        self.low_buffer: dict[str, deque] = {}
        self.volume_buffer: dict[str, deque] = {}
        self.feature_buffer: dict[str, deque] = {}   # feature vectors (lists)
        self.feature_closes: dict[str, deque] = {}   # close at each feature bar
        self.models: dict[str, object] = {}
        self.bars_since_retrain: dict[str, int] = {}
        self.current_atr: dict[str, float] = {}
        self.highest: dict[str, float] = {}
        self.lowest: dict[str, float] = {}

    def _ensure(self, symbol):
        if symbol not in self.close_buffer:
            maxlen = 300
            self.close_buffer[symbol] = deque(maxlen=maxlen)
            self.high_buffer[symbol] = deque(maxlen=maxlen)
            self.low_buffer[symbol] = deque(maxlen=maxlen)
            self.volume_buffer[symbol] = deque(maxlen=maxlen)
            self.feature_buffer[symbol] = deque(maxlen=maxlen)
            self.feature_closes[symbol] = deque(maxlen=maxlen)
            self.models[symbol] = None
            self.bars_since_retrain[symbol] = 0
            self.current_atr[symbol] = 0.0
            self.highest[symbol] = 0.0
            self.lowest[symbol] = float("inf")

    # === Sub-signal computation ===

    def _compute_sub_signals(self, closes, highs, lows):
        """Compute 5 sub-signals + 2 context features = 7-dim feature vector.

        Returns list of 7 floats or None if insufficient data.
        """
        if len(closes) < 35:
            return None

        signals = {}

        # 1. SMA: fast(10) vs slow(30)
        fast = compute_sma(closes, 10)
        slow = compute_sma(closes, 30)
        if fast is not None and slow is not None:
            signals["sma"] = 1 if fast > slow else -1
        else:
            signals["sma"] = 0

        # 2. RSI: < 35 = +1 (oversold buy), > 65 = -1 (overbought sell)
        rsi = compute_rsi(closes, 14)
        if rsi is not None:
            signals["rsi"] = 1 if rsi < 35 else (-1 if rsi > 65 else 0)
        else:
            signals["rsi"] = 0

        # 3. MACD: histogram > 0 = +1, < 0 = -1
        macd = compute_macd(closes)
        if macd is not None:
            signals["macd"] = 1 if macd[2] > 0 else -1
        else:
            signals["macd"] = 0

        # 4. Bollinger %B: < 0 = +1 (below lower), > 1 = -1 (above upper)
        bb = compute_bollinger(closes, 20)
        if bb is not None:
            upper, mid, lower = bb
            bb_range = upper - lower
            pct_b = (closes[-1] - lower) / bb_range if bb_range > 0 else 0.5
            signals["bb"] = 1 if pct_b < 0 else (-1 if pct_b > 1 else 0)
        else:
            signals["bb"] = 0

        # 5. ADX direction: ADX > 25 = trending, combine with price vs SMA20
        adx = compute_adx(highs, lows, closes, 14)
        if adx is not None and adx > 25:
            sma20 = compute_sma(closes, 20)
            signals["adx_dir"] = 1 if sma20 and closes[-1] > sma20 else -1
        else:
            signals["adx_dir"] = 0

        # Context features
        adx_val = adx / 100.0 if adx is not None else 0.5
        bbw_val = compute_bbw(closes, 20) or 0.0

        return [
            signals["sma"], signals["rsi"], signals["macd"],
            signals["bb"], signals["adx_dir"],
            adx_val, bbw_val,
        ]

    # === Label generation ===

    def _build_labels(self, closes):
        """Generate labels from close prices. Returns array of {-1, 0, 1}."""
        n = len(closes)
        labels = []
        for i in range(n - self.forward_bars):
            future_ret = (closes[i + self.forward_bars] - closes[i]) / closes[i]
            if future_ret > self.target_return:
                labels.append(1)
            elif future_ret < -self.target_return:
                labels.append(-1)
            else:
                labels.append(0)
        return np.array(labels)

    # === Rolling training ===

    def _try_train(self, symbol):
        """Train or retrain the LogisticRegression model if conditions are met."""
        if LogisticRegression is None:
            return

        buf = self.feature_buffer[symbol]
        closes = list(self.feature_closes[symbol])

        if len(buf) < self.min_train_bars:
            return
        if (self.bars_since_retrain[symbol] < self.retrain_interval
                and self.models[symbol] is not None):
            return

        feature_list = list(buf)
        labels = self._build_labels(closes)

        n_usable = len(labels)
        if n_usable < 20:
            return

        X = np.array(feature_list[:n_usable], dtype=float)

        # Filter out HOLD labels -- only learn BUY vs SELL
        mask = labels != 0
        if mask.sum() < 20:
            return

        unique_classes = np.unique(labels[mask])
        if len(unique_classes) < 2:
            return

        try:
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X[mask], labels[mask])
            self.models[symbol] = model
            self.bars_since_retrain[symbol] = 0
        except Exception:
            pass

    # === Prediction ===

    def _predict(self, symbol, features):
        """Predict class probabilities. Returns (buy_prob, sell_prob) or (0, 0)."""
        model = self.models.get(symbol)
        if model is None:
            return 0.0, 0.0

        try:
            X_pred = np.array([features], dtype=float)
            proba = model.predict_proba(X_pred)
            classes = model.classes_

            buy_prob = 0.0
            sell_prob = 0.0
            for i, c in enumerate(classes):
                if c == 1:
                    buy_prob = proba[0][i]
                elif c == -1:
                    sell_prob = proba[0][i]

            return buy_prob, sell_prob
        except Exception:
            return 0.0, 0.0

    # === Main event handler ===

    def on_bar(self, snapshot):
        self.pm.increment_bars()
        signals = self.pm.process_fills(snapshot)
        signals += self.pm.resubmit_expired(snapshot)
        self.pm.reconcile(snapshot)

        if "day" not in snapshot.timeframes:
            return signals

        for symbol, bar in snapshot.timeframes["day"].items():
            self._ensure(symbol)

            # Accumulate price data
            self.close_buffer[symbol].append(bar.close)
            self.high_buffer[symbol].append(bar.high)
            self.low_buffer[symbol].append(bar.low)
            self.volume_buffer[symbol].append(bar.volume)

            closes = list(self.close_buffer[symbol])
            highs = list(self.high_buffer[symbol])
            lows = list(self.low_buffer[symbol])

            # Compute ATR
            atr = compute_atr(highs, lows, closes, self.atr_period)
            if atr is not None:
                self.current_atr[symbol] = atr

            # Compute sub-signals feature vector
            feats = self._compute_sub_signals(closes, highs, lows)
            if feats is not None:
                self.feature_buffer[symbol].append(feats)
                self.feature_closes[symbol].append(bar.close)

            self.bars_since_retrain[symbol] += 1

            # Try to train / retrain
            self._try_train(symbol)

            # Prediction and trading logic
            atr_val = self.current_atr.get(symbol, 0.0)
            state = self.pm.get_state(symbol)

            if feats is not None and self.models.get(symbol) is not None:
                buy_prob, sell_prob = self._predict(symbol, feats)

                # === Flat: check for entries ===
                if (self.pm.is_flat(symbol)
                        and not self.pm.has_pending_entry(symbol)
                        and atr_val > 0):
                    qty = int(
                        snapshot.portfolio.cash * self.risk_pct
                        / (atr_val * self.atr_mult)
                    )
                    # Cap to available cash
                    if bar.close > 0:
                        qty = min(qty, int(snapshot.portfolio.cash / bar.close))
                    inst = self.instruments.get(symbol)
                    if inst and inst.lot_size > 1:
                        qty = (qty // inst.lot_size) * inst.lot_size

                    if qty > 0:
                        if buy_prob > self.confidence_threshold:
                            stop = bar.close - atr_val * self.atr_mult
                            signals += self.pm.enter_long(
                                symbol, qty, 0, "CNC", stop,
                            )
                            self.highest[symbol] = bar.close
                        elif sell_prob > self.confidence_threshold:
                            stop = bar.close + atr_val * self.atr_mult
                            signals += self.pm.enter_short(
                                symbol, qty, 0, stop,
                            )
                            self.lowest[symbol] = bar.close

                # === Long: trailing stop + model-based exit ===
                elif self.pm.is_long(symbol):
                    if bar.close > self.highest.get(symbol, 0.0):
                        self.highest[symbol] = bar.close
                    if atr_val > 0:
                        new_stop = self.highest[symbol] - atr_val * self.atr_mult
                        signals += self.pm.update_trailing_stop(
                            symbol, new_stop,
                        )

                    # Exit when model confidence drops below 0.5
                    if buy_prob < 0.5:
                        signals += self.pm.exit_position(symbol)
                    elif state.bars_held > self.max_hold_bars:
                        gain = (
                            (bar.close - state.avg_entry) / state.avg_entry
                            if state.avg_entry > 0
                            else 0
                        )
                        if gain < 0.005:
                            signals += self.pm.exit_position(symbol)

                # === Short: trailing stop + model-based exit ===
                elif self.pm.is_short(symbol):
                    if bar.close < self.lowest.get(symbol, float("inf")):
                        self.lowest[symbol] = bar.close
                    if atr_val > 0:
                        new_stop = (
                            self.lowest[symbol] + atr_val * self.atr_mult
                        )
                        signals += self.pm.update_trailing_stop(
                            symbol, new_stop,
                        )

                    # Exit when model confidence drops below 0.5
                    if sell_prob < 0.5:
                        signals += self.pm.exit_position(symbol)
                    elif state.bars_held > self.max_hold_bars:
                        gain = (
                            (state.avg_entry - bar.close) / state.avg_entry
                            if state.avg_entry > 0
                            else 0
                        )
                        if gain < 0.005:
                            signals += self.pm.exit_position(symbol)

        return signals

    def on_complete(self):
        return {"strategy_type": "ensemble_meta"}
