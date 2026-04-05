"""Bollinger Squeeze Breakout — enter on volatility expansion after compression.

Pure intraday 5-minute strategy. Bollinger Bands compress (squeeze) then expand.
Enter in the expansion direction with volume confirmation.

Daily cycle:
  9:15-9:45: Warmup (accumulate bars for BB, no trading)
  9:45-15:00: Detect squeeze, enter on expansion
  15:00: Close all positions before MIS squareoff
"""

from collections import deque
from datetime import datetime, timezone, timedelta

from server.registry import register
from strategies.base import Strategy, MarketSnapshot, InstrumentInfo, Signal
from strategies.indicators import (
    compute_bollinger, compute_bbw, compute_atr, compute_sma,
)
from strategies.position_manager import PositionManager


IST = timezone(timedelta(hours=5, minutes=30))


def _ist_time(timestamp_ms: int) -> tuple[int, int]:
    """Convert epoch ms to IST (hour, minute)."""
    dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=IST)
    return dt.hour, dt.minute


@register("bollinger_squeeze")
class BollingerSqueeze(Strategy):

    def required_data(self):
        return [{"interval": "5minute", "lookback": 100}]

    def initialize(self, config, instruments):
        self.bb_period = config.get("bb_period", 20)
        self.bb_std = config.get("bb_std", 2.0)
        self.squeeze_threshold = config.get("squeeze_threshold", 0.7)
        self.volume_confirm = config.get("volume_confirm", 1.5)
        self.risk_per_trade = config.get("risk_per_trade", 0.005)
        self.atr_period = config.get("atr_period", 14)
        self.profit_target_atr = config.get("profit_target_atr", 1.5)
        self.atr_stop_mult = config.get("atr_stop_mult", 1.0)
        self.warmup_bars = config.get("warmup_bars", 6)
        self.exit_time_hour = config.get("exit_time_hour", 15)
        self.max_trades_per_day = config.get("max_trades_per_day", 3)
        self.instruments = instruments

        self.pm = PositionManager(max_pending_bars=2)

        # Per-symbol state
        self.closes_5m: dict[str, deque] = {}
        self.highs_5m: dict[str, deque] = {}
        self.lows_5m: dict[str, deque] = {}
        self.volumes_5m: dict[str, deque] = {}
        self.bbw_history: dict[str, deque] = {}
        self.squeeze_active: dict[str, bool] = {}
        self.bars_today: dict[str, int] = {}
        self.trades_today: dict[str, int] = {}
        self.last_hour: dict[str, int] = {}

    def _ensure(self, symbol: str):
        if symbol not in self.closes_5m:
            self.closes_5m[symbol] = deque(maxlen=100)
            self.highs_5m[symbol] = deque(maxlen=100)
            self.lows_5m[symbol] = deque(maxlen=100)
            self.volumes_5m[symbol] = deque(maxlen=100)
            self.bbw_history[symbol] = deque(maxlen=50)
            self.squeeze_active[symbol] = False
            self.bars_today[symbol] = 0
            self.trades_today[symbol] = 0
            self.last_hour[symbol] = 0

    def _is_new_day(self, symbol: str, timestamp_ms: int) -> bool:
        """Detect day transition from timestamp."""
        hour, _ = _ist_time(timestamp_ms)
        if symbol not in self.last_hour:
            return True
        if hour == 9 and self.last_hour[symbol] >= 15:
            return True
        return False

    def _reset_day(self, symbol: str):
        """Clear daily state. Keep closes_5m for BB continuity."""
        self.highs_5m[symbol].clear()
        self.lows_5m[symbol].clear()
        self.volumes_5m[symbol].clear()
        self.closes_5m[symbol].clear()
        self.bbw_history[symbol].clear()
        self.squeeze_active[symbol] = False
        self.bars_today[symbol] = 0
        self.trades_today[symbol] = 0

    def on_bar(self, snapshot: MarketSnapshot) -> list[Signal]:
        self.pm.increment_bars()
        signals = self.pm.process_fills(snapshot)
        signals += self.pm.resubmit_expired(snapshot)
        self.pm.reconcile(snapshot)

        if "5minute" not in snapshot.timeframes:
            return signals

        for symbol, bar in snapshot.timeframes["5minute"].items():
            self._ensure(symbol)
            hour, minute = _ist_time(snapshot.timestamp_ms)

            # New day detection — reset daily state
            if self._is_new_day(symbol, snapshot.timestamp_ms):
                self._reset_day(symbol)

            self.last_hour[symbol] = hour

            # Accumulate bar data
            self.closes_5m[symbol].append(bar.close)
            self.highs_5m[symbol].append(bar.high)
            self.lows_5m[symbol].append(bar.low)
            self.volumes_5m[symbol].append(bar.volume)
            self.bars_today[symbol] += 1

            # Skip warmup period (first 30 min = 6 bars of 5min)
            if self.bars_today[symbol] < self.warmup_bars:
                continue

            # Compute BBW for squeeze detection
            closes_list = list(self.closes_5m[symbol])
            bbw = compute_bbw(closes_list, self.bb_period, self.bb_std)
            if bbw is not None:
                self.bbw_history[symbol].append(bbw)

            # Need enough BBW history for average
            avg_bbw = compute_sma(list(self.bbw_history[symbol]), 20)

            # Squeeze detection: BBW < threshold * avg_BBW
            if avg_bbw is not None and bbw is not None:
                if bbw < self.squeeze_threshold * avg_bbw:
                    self.squeeze_active[symbol] = True

            # Time exit: close all positions at 15:00
            if hour >= self.exit_time_hour:
                if not self.pm.is_flat(symbol):
                    signals += self.pm.exit_position(symbol)
                continue  # no new entries after exit time

            # --- Trailing stop update for existing positions ---
            if not self.pm.is_flat(symbol):
                highs_list = list(self.highs_5m[symbol])
                lows_list = list(self.lows_5m[symbol])
                atr = compute_atr(highs_list, lows_list, closes_list,
                                  self.atr_period)
                bb = compute_bollinger(closes_list, self.bb_period, self.bb_std)
                if atr and bb:
                    _, mid, _ = bb
                    if self.pm.is_long(symbol):
                        new_stop = max(mid, bar.close - self.atr_stop_mult * atr)
                        signals += self.pm.update_trailing_stop(symbol, new_stop)
                    elif self.pm.is_short(symbol):
                        new_stop = min(mid, bar.close + self.atr_stop_mult * atr)
                        signals += self.pm.update_trailing_stop(symbol, new_stop)
                continue  # already in a position, skip entry logic

            # --- Entry logic: expansion after squeeze ---
            if not self.pm.is_flat(symbol) or self.pm.has_pending_entry(symbol):
                continue

            if self.trades_today.get(symbol, 0) >= self.max_trades_per_day:
                continue

            if not self.squeeze_active.get(symbol, False):
                continue

            if bbw is None or avg_bbw is None:
                continue

            # Bands must be expanding (BBW > avg_bbw)
            if bbw <= avg_bbw:
                continue

            bb = compute_bollinger(closes_list, self.bb_period, self.bb_std)
            if bb is None:
                continue
            upper, mid, lower = bb

            # Volume confirmation
            vol_list = list(self.volumes_5m[symbol])
            avg_vol = (sum(vol_list[-20:]) / len(vol_list[-20:])
                       if len(vol_list) >= 20 else bar.volume)
            volume_ok = bar.volume > self.volume_confirm * avg_vol

            if not volume_ok:
                continue

            # ATR for position sizing and stops
            highs_list = list(self.highs_5m[symbol])
            lows_list = list(self.lows_5m[symbol])
            atr = compute_atr(highs_list, lows_list, closes_list,
                              self.atr_period)

            capital = snapshot.portfolio.cash

            if bar.close > upper:
                # Upward breakout
                stop = mid  # stop at BB mid — breakout failed if returns
                qty = (int(capital * self.risk_per_trade
                           / (atr * self.atr_stop_mult))
                       if atr and atr > 0 else 0)
                if qty > 0:
                    signals += self.pm.enter_long(
                        symbol, qty, bar.close, "MIS", stop)
                    if atr:
                        target_price = bar.close + self.profit_target_atr * atr
                        signals += self.pm.set_profit_target(
                            symbol, max(1, qty // 3), target_price)
                    self.squeeze_active[symbol] = False
                    self.trades_today[symbol] = (
                        self.trades_today.get(symbol, 0) + 1)

            elif bar.close < lower:
                # Downward breakout — short (MIS)
                stop = mid
                qty = (int(capital * self.risk_per_trade
                           / (atr * self.atr_stop_mult))
                       if atr and atr > 0 else 0)
                if qty > 0:
                    signals += self.pm.enter_short(
                        symbol, qty, bar.close, stop)
                    if atr:
                        target_price = bar.close - self.profit_target_atr * atr
                        signals += self.pm.set_profit_target(
                            symbol, max(1, qty // 3), target_price)
                    self.squeeze_active[symbol] = False
                    self.trades_today[symbol] = (
                        self.trades_today.get(symbol, 0) + 1)

        return signals

    def on_complete(self):
        return {"strategy_type": "bollinger_squeeze"}
