"""Portfolio Combiner — Donchian trend-following + RSI mean-reversion with dynamic ADX allocation.

When ADX > 25 (trending): use Donchian-style channel breakout entries
When ADX < 20 (ranging): use RSI mean-reversion entries
When ADX 20-25 (neutral): half position size from whichever signal fires

Product selection: CNC for trending entries, MIS for ranging entries.
Shorts always MIS regardless of regime.
"""

from collections import deque
from server.registry import register
from strategies.base import Strategy, MarketSnapshot, InstrumentInfo, Signal
from strategies.indicators import compute_atr, compute_adx, compute_rsi, compute_ema
from strategies.position_manager import PositionManager


@register("portfolio_combiner")
class PortfolioCombiner(Strategy):

    def required_data(self):
        return [
            {"interval": "day", "lookback": 60},
            {"interval": "15minute", "lookback": 100},
        ]

    def initialize(self, config, instruments):
        # Donchian params
        self.channel_period = config.get("channel_period", 20)
        # RSI params
        self.rsi_period = config.get("rsi_period", 14)
        self.rsi_oversold = config.get("rsi_oversold", 35)
        self.rsi_overbought = config.get("rsi_overbought", 65)
        # ADX regime thresholds
        self.adx_trend = config.get("adx_trend", 25)
        self.adx_range = config.get("adx_range", 20)
        # Sizing and risk
        self.risk_per_trade = config.get("risk_per_trade", 0.015)
        self.atr_period = config.get("atr_period", 14)
        self.atr_mult = config.get("atr_mult", 2.0)
        self.max_hold_bars = config.get("max_hold_bars", 30)
        self.ema_period = config.get("ema_period", 20)
        self.instruments = instruments

        self.pm = PositionManager(max_pending_bars=3)

        # Per-symbol indicator state
        self.daily_highs: dict[str, deque] = {}
        self.daily_lows: dict[str, deque] = {}
        self.daily_closes: dict[str, deque] = {}
        self.prices_15m: dict[str, deque] = {}
        self.ema_history: dict[str, deque] = {}
        self.current_atr: dict[str, float] = {}
        self.current_adx: dict[str, float] = {}
        self.trend_up: dict[str, bool] = {}
        self.trend_down: dict[str, bool] = {}
        self.highest: dict[str, float] = {}
        self.lowest: dict[str, float] = {}
        # Track which regime was used for entry
        self.entry_regime: dict[str, str] = {}

    def _ensure(self, symbol):
        if symbol not in self.daily_highs:
            maxlen = self.channel_period + self.atr_period + 20
            self.daily_highs[symbol] = deque(maxlen=maxlen)
            self.daily_lows[symbol] = deque(maxlen=maxlen)
            self.daily_closes[symbol] = deque(maxlen=maxlen)
            self.prices_15m[symbol] = deque(maxlen=self.rsi_period + 20)
            self.ema_history[symbol] = deque(maxlen=10)
            self.current_atr[symbol] = 0.0
            self.current_adx[symbol] = 0.0
            self.trend_up[symbol] = False
            self.trend_down[symbol] = False
            self.highest[symbol] = 0.0
            self.lowest[symbol] = float('inf')

    def _get_regime(self, symbol):
        """Return 'trending', 'ranging', or 'neutral' based on ADX."""
        adx = self.current_adx.get(symbol, 0.0)
        if adx > self.adx_trend:
            return "trending"
        elif adx < self.adx_range:
            return "ranging"
        return "neutral"

    def _compute_qty(self, symbol, cash, price, atr, regime):
        """ATR-based position sizing, capped to available cash. Half size in neutral."""
        if atr <= 0 or price <= 0:
            return 0
        qty = int(cash * self.risk_per_trade / (atr * self.atr_mult))
        # Cap to available cash (prevent leverage blow-ups)
        qty = min(qty, int(cash / price))
        # Neutral regime: half size
        if regime == "neutral":
            qty = max(1, qty // 2)
        inst = self.instruments.get(symbol)
        if inst and inst.lot_size > 1:
            qty = (qty // inst.lot_size) * inst.lot_size
        return max(0, qty)

    def on_bar(self, snapshot):
        self.pm.increment_bars()
        signals = self.pm.process_fills(snapshot)
        signals += self.pm.resubmit_expired(snapshot)
        self.pm.reconcile(snapshot)

        # --- Update daily indicators ---
        if "day" in snapshot.timeframes:
            for symbol, bar in snapshot.timeframes["day"].items():
                self._ensure(symbol)
                self.daily_highs[symbol].append(bar.high)
                self.daily_lows[symbol].append(bar.low)
                self.daily_closes[symbol].append(bar.close)

                atr = compute_atr(
                    list(self.daily_highs[symbol]), list(self.daily_lows[symbol]),
                    list(self.daily_closes[symbol]), self.atr_period,
                )
                if atr is not None:
                    self.current_atr[symbol] = atr

                adx = compute_adx(
                    list(self.daily_highs[symbol]), list(self.daily_lows[symbol]),
                    list(self.daily_closes[symbol]), 14,
                )
                if adx is not None:
                    self.current_adx[symbol] = adx

                ema = compute_ema(list(self.daily_closes[symbol]), self.ema_period)
                if ema is not None:
                    self.ema_history[symbol].append(ema)
                    self.trend_up[symbol] = (
                        bar.close > ema and
                        (len(self.ema_history[symbol]) < 6 or
                         self.ema_history[symbol][-1] > self.ema_history[symbol][-6])
                    )
                    self.trend_down[symbol] = (
                        bar.close < ema and
                        (len(self.ema_history[symbol]) < 6 or
                         self.ema_history[symbol][-1] < self.ema_history[symbol][-6])
                    )

        # --- Process 15-minute bars ---
        if "15minute" not in snapshot.timeframes:
            return signals

        for symbol, bar in snapshot.timeframes["15minute"].items():
            self._ensure(symbol)
            self.prices_15m[symbol].append(bar.close)

            atr = self.current_atr.get(symbol, 0.0)
            regime = self._get_regime(symbol)
            state = self.pm.get_state(symbol)

            # === Flat: check regime-based entries ===
            if self.pm.is_flat(symbol) and not self.pm.has_pending_entry(symbol) and atr > 0:
                qty = self._compute_qty(symbol, snapshot.portfolio.cash, bar.close, atr, regime)
                if qty <= 0:
                    continue

                entered = False

                # --- TRENDING or NEUTRAL: Donchian channel breakout ---
                if regime in ("trending", "neutral"):
                    highs = list(self.daily_highs.get(symbol, []))
                    lows = list(self.daily_lows.get(symbol, []))
                    if len(highs) >= self.channel_period + 1:
                        ch_high = max(highs[-(self.channel_period + 1):-1])
                        ch_low = min(lows[-(self.channel_period + 1):-1])

                        if bar.close > ch_high:
                            product = "CNC" if regime == "trending" else "MIS"
                            stop = bar.close - atr * self.atr_mult
                            signals += self.pm.enter_long(symbol, qty, 0, product, stop)
                            self.highest[symbol] = bar.close
                            self.entry_regime[symbol] = regime
                            entered = True

                        elif bar.close < ch_low:
                            stop = bar.close + atr * self.atr_mult
                            signals += self.pm.enter_short(symbol, qty, 0, stop)
                            self.lowest[symbol] = bar.close
                            self.entry_regime[symbol] = regime
                            entered = True

                # --- RANGING or NEUTRAL (if Donchian didn't fire): RSI mean-reversion ---
                if not entered and regime in ("ranging", "neutral"):
                    rsi = compute_rsi(list(self.prices_15m[symbol]), self.rsi_period)
                    if rsi is not None:
                        # Long: RSI oversold in uptrend
                        if rsi < self.rsi_oversold and self.trend_up.get(symbol, False):
                            product = "MIS" if regime == "ranging" else "MIS"
                            stop = bar.close - atr * self.atr_mult
                            signals += self.pm.enter_long(symbol, qty, bar.close * 0.999, product, stop)
                            self.highest[symbol] = bar.close
                            self.entry_regime[symbol] = regime
                        # Short: RSI overbought in downtrend
                        elif rsi > self.rsi_overbought and self.trend_down.get(symbol, False):
                            stop = bar.close + atr * self.atr_mult
                            signals += self.pm.enter_short(symbol, qty, bar.close * 1.001, stop)
                            self.lowest[symbol] = bar.close
                            self.entry_regime[symbol] = regime

            # === Long: trailing stop + time exit ===
            elif self.pm.is_long(symbol):
                if bar.close > self.highest.get(symbol, 0.0):
                    self.highest[symbol] = bar.close
                if atr > 0:
                    new_stop = self.highest[symbol] - atr * self.atr_mult
                    signals += self.pm.update_trailing_stop(symbol, new_stop)

                # Time stop: exit if held too long with no meaningful gain
                if state.bars_held > self.max_hold_bars:
                    gain = (bar.close - state.avg_entry) / state.avg_entry if state.avg_entry > 0 else 0
                    if gain < 0.005:
                        signals += self.pm.exit_position(symbol)

            # === Short: trailing stop + time exit ===
            elif self.pm.is_short(symbol):
                if bar.close < self.lowest.get(symbol, float('inf')):
                    self.lowest[symbol] = bar.close
                if atr > 0:
                    new_stop = self.lowest[symbol] + atr * self.atr_mult
                    signals += self.pm.update_trailing_stop(symbol, new_stop)

                if state.bars_held > self.max_hold_bars:
                    gain = (state.avg_entry - bar.close) / state.avg_entry if state.avg_entry > 0 else 0
                    if gain < 0.005:
                        signals += self.pm.exit_position(symbol)

        return signals

    def on_complete(self):
        return {"strategy_type": "portfolio_combiner"}
