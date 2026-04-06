"""ML Signal Classifier — GradientBoosting predicts next-bar direction.

Rolling training: trains on last 250 bars, retrains every 20 bars.
Only trades when model confidence > 65%. Uses 20+ features from
compute_features().
"""

from collections import deque
import numpy as np

try:
    from sklearn.ensemble import GradientBoostingClassifier
except ImportError:
    GradientBoostingClassifier = None

from server.registry import register
from strategies.base import Strategy, MarketSnapshot, InstrumentInfo, Signal
from strategies.indicators import compute_features, compute_atr
from strategies.position_manager import PositionManager


@register("ml_classifier")
class MLClassifier(Strategy):

    def required_data(self):
        return [{"interval": "day", "lookback": 200}]

    def initialize(self, config, instruments):
        self.min_train_bars = config.get("min_train_bars", 120)
        self.max_train_bars = config.get("max_train_bars", 250)
        self.retrain_interval = config.get("retrain_interval", 20)
        self.forward_bars = config.get("forward_bars", 5)
        self.target_return = config.get("target_return", 0.01)
        self.confidence_threshold = config.get("confidence_threshold", 0.65)
        self.risk_pct = config.get("risk_pct", 0.03)
        self.atr_period = config.get("atr_period", 14)
        self.atr_mult = config.get("atr_mult", 2.0)
        self.max_hold_bars = config.get("max_hold_bars", 30)
        self.instruments = instruments

        self.pm = PositionManager(max_pending_bars=3)

        # Per-symbol state
        self.feature_buffer: dict[str, deque] = {}   # feature dicts
        self.feature_closes: dict[str, deque] = {}   # close at each feature bar (aligned)
        self.close_buffer: dict[str, deque] = {}      # raw close prices for ATR/features
        self.high_buffer: dict[str, deque] = {}
        self.low_buffer: dict[str, deque] = {}
        self.volume_buffer: dict[str, deque] = {}
        self.models: dict[str, object] = {}
        self.bars_since_retrain: dict[str, int] = {}
        self.feature_keys: dict[str, list[str]] = {}
        self.current_atr: dict[str, float] = {}
        self.highest: dict[str, float] = {}
        self.lowest: dict[str, float] = {}

    def _ensure(self, symbol):
        if symbol not in self.feature_buffer:
            maxlen = self.max_train_bars + 10
            self.feature_buffer[symbol] = deque(maxlen=maxlen)
            self.feature_closes[symbol] = deque(maxlen=maxlen)
            self.close_buffer[symbol] = deque(maxlen=maxlen)
            self.high_buffer[symbol] = deque(maxlen=maxlen)
            self.low_buffer[symbol] = deque(maxlen=maxlen)
            self.volume_buffer[symbol] = deque(maxlen=maxlen)
            self.models[symbol] = None
            self.bars_since_retrain[symbol] = 0
            self.feature_keys[symbol] = []
            self.current_atr[symbol] = 0.0
            self.highest[symbol] = 0.0
            self.lowest[symbol] = float("inf")

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

    def _features_to_array(self, feature_dicts, keys):
        """Convert list of feature dicts to numpy array with consistent column order."""
        rows = []
        for fd in feature_dicts:
            rows.append([fd.get(k, 0.0) for k in keys])
        return np.array(rows, dtype=float)

    def _try_train(self, symbol):
        """Train or retrain the model if conditions are met."""
        if GradientBoostingClassifier is None:
            return

        buf = self.feature_buffer[symbol]
        closes = list(self.feature_closes[symbol])

        if len(buf) < self.min_train_bars:
            return
        if self.bars_since_retrain[symbol] < self.retrain_interval and self.models[symbol] is not None:
            return

        # Build feature matrix and labels from aligned feature/close data
        feature_list = list(buf)
        labels = self._build_labels(closes)

        # Align: features[:-forward_bars] matches labels
        n_usable = len(labels)
        if n_usable < 20:
            return

        X_features = feature_list[:n_usable]

        # Establish sorted feature keys from the first feature dict
        keys = sorted(X_features[0].keys())
        if not keys:
            return
        self.feature_keys[symbol] = keys
        X = self._features_to_array(X_features, keys)

        # Filter out HOLD labels — only learn BUY vs SELL
        mask = labels != 0
        if mask.sum() < 20:
            return

        # Need at least two classes
        unique_classes = np.unique(labels[mask])
        if len(unique_classes) < 2:
            return

        try:
            model = GradientBoostingClassifier(
                max_depth=3,
                n_estimators=50,
                min_samples_leaf=10,
                random_state=42,
            )
            model.fit(X[mask], labels[mask])
            self.models[symbol] = model
            self.bars_since_retrain[symbol] = 0
        except Exception:
            pass

    def _predict(self, symbol, features):
        """Predict class probabilities. Returns (buy_prob, sell_prob) or (0, 0)."""
        model = self.models.get(symbol)
        if model is None or not self.feature_keys.get(symbol):
            return 0.0, 0.0

        keys = self.feature_keys[symbol]
        try:
            X_pred = np.array([[features.get(k, 0.0) for k in keys]], dtype=float)
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

            # Compute ATR
            atr = compute_atr(
                list(self.high_buffer[symbol]),
                list(self.low_buffer[symbol]),
                list(self.close_buffer[symbol]),
                self.atr_period,
            )
            if atr is not None:
                self.current_atr[symbol] = atr

            # Compute features and store
            feats = compute_features(
                list(self.close_buffer[symbol]),
                list(self.high_buffer[symbol]),
                list(self.low_buffer[symbol]),
                list(self.volume_buffer[symbol]),
            )
            if feats is not None:
                self.feature_buffer[symbol].append(feats)
                self.feature_closes[symbol].append(bar.close)

            self.bars_since_retrain[symbol] += 1

            # Try to train/retrain
            self._try_train(symbol)

            # Prediction and trading logic
            atr_val = self.current_atr.get(symbol, 0.0)
            state = self.pm.get_state(symbol)

            if feats is not None and self.models.get(symbol) is not None:
                buy_prob, sell_prob = self._predict(symbol, feats)

                # === Flat: check for entries ===
                if self.pm.is_flat(symbol) and not self.pm.has_pending_entry(symbol) and atr_val > 0:
                    qty = int(
                        snapshot.portfolio.cash * self.risk_pct / (atr_val * self.atr_mult)
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
                            signals += self.pm.enter_long(symbol, qty, 0, "CNC", stop)
                            self.highest[symbol] = bar.close
                        elif sell_prob > self.confidence_threshold:
                            stop = bar.close + atr_val * self.atr_mult
                            signals += self.pm.enter_short(symbol, qty, 0, stop)
                            self.lowest[symbol] = bar.close

                # === Long: trailing stop + model-based exit ===
                elif self.pm.is_long(symbol):
                    if bar.close > self.highest.get(symbol, 0.0):
                        self.highest[symbol] = bar.close
                    if atr_val > 0:
                        new_stop = self.highest[symbol] - atr_val * self.atr_mult
                        signals += self.pm.update_trailing_stop(symbol, new_stop)

                    # Exit when model flips: buy confidence drops below 0.5
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
                        new_stop = self.lowest[symbol] + atr_val * self.atr_mult
                        signals += self.pm.update_trailing_stop(symbol, new_stop)

                    # Exit when model flips: sell confidence drops below 0.5
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
        return {"strategy_type": "ml_classifier"}
