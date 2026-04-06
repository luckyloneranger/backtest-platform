"""Intraday Momentum -- ride strong directional moves on 5-min bars.

Detects momentum bursts: price moves > 1.5x ATR over 3 bars with volume > 2x avg.
Enters in the move direction. Trails stop at 1x ATR.
Pure trend-following on 5-min -- no mean-reversion.

Daily cycle:
  9:15-9:45: Warmup (accumulate bars for ATR, no trading)
  9:45-15:00: Detect momentum bursts, enter and trail
  15:00: Close all positions before MIS squareoff
"""

from collections import deque
from datetime import datetime, timezone, timedelta

from server.registry import register
from strategies.base import Strategy, MarketSnapshot, Signal
from strategies.indicators import compute_atr
from strategies.position_manager import PositionManager


IST = timezone(timedelta(hours=5, minutes=30))


def _ist_time(timestamp_ms: int) -> tuple[int, int]:
    """Convert epoch ms to IST (hour, minute)."""
    dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=IST)
    return dt.hour, dt.minute


@register("intraday_momentum")
class IntradayMomentum(Strategy):

    def required_data(self):
        return [{"interval": "5minute", "lookback": 100}]

    def initialize(self, config, instruments):
        self.momentum_atr_mult = config.get("momentum_atr_mult", 1.5)
        self.volume_mult = config.get("volume_mult", 2.0)
        self.atr_period = config.get("atr_period", 14)
        self.atr_stop_mult = config.get("atr_stop_mult", 1.0)
        self.risk_pct = config.get("risk_pct", 0.03)
        self.warmup_bars = config.get("warmup_bars", 6)
        self.exit_time_hour = config.get("exit_time_hour", 15)
        self.max_trades_per_day = config.get("max_trades_per_day", 2)
        self.lookback_bars = config.get("lookback_bars", 3)
        self.vol_avg_period = config.get("vol_avg_period", 20)
        self.instruments = instruments

        self.pm = PositionManager(max_pending_bars=2)

        # Per-symbol daily state
        self.closes_5m: dict[str, deque] = {}
        self.highs_5m: dict[str, deque] = {}
        self.lows_5m: dict[str, deque] = {}
        self.volumes_5m: dict[str, deque] = {}
        self.bars_today: dict[str, int] = {}
        self.trades_today: dict[str, int] = {}
        self.last_hour: dict[str, int] = {}

    def _ensure(self, symbol: str):
        if symbol not in self.closes_5m:
            self.closes_5m[symbol] = deque(maxlen=100)
            self.highs_5m[symbol] = deque(maxlen=100)
            self.lows_5m[symbol] = deque(maxlen=100)
            self.volumes_5m[symbol] = deque(maxlen=100)
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
        """Clear daily state for a symbol."""
        self.closes_5m[symbol].clear()
        self.highs_5m[symbol].clear()
        self.lows_5m[symbol].clear()
        self.volumes_5m[symbol].clear()
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

            # New day detection -- reset daily state
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
            if self.bars_today[symbol] <= self.warmup_bars:
                continue

            # Time exit: close all positions at 15:00
            if hour >= self.exit_time_hour:
                if not self.pm.is_flat(symbol):
                    signals += self.pm.exit_position(symbol)
                continue  # no new entries after exit time

            # Need enough bars for ATR and lookback
            closes_list = list(self.closes_5m[symbol])
            highs_list = list(self.highs_5m[symbol])
            lows_list = list(self.lows_5m[symbol])
            volumes_list = list(self.volumes_5m[symbol])

            if len(closes_list) < self.atr_period + 1:
                continue

            atr = compute_atr(highs_list, lows_list, closes_list,
                              self.atr_period)
            if atr is None or atr <= 0:
                continue

            # --- Trailing stop update for existing positions ---
            if not self.pm.is_flat(symbol):
                if self.pm.is_long(symbol):
                    new_stop = bar.close - self.atr_stop_mult * atr
                    signals += self.pm.update_trailing_stop(symbol, new_stop)
                elif self.pm.is_short(symbol):
                    new_stop = bar.close + self.atr_stop_mult * atr
                    signals += self.pm.update_trailing_stop(symbol, new_stop)
                continue  # already in position, skip entry logic

            # --- Entry logic: momentum burst detection ---
            if self.pm.has_pending_entry(symbol):
                continue

            if self.trades_today.get(symbol, 0) >= self.max_trades_per_day:
                continue

            # Need enough bars for lookback move
            if len(closes_list) < self.lookback_bars + 1:
                continue

            # 3-bar price move
            move = closes_list[-1] - closes_list[-(self.lookback_bars + 1)]

            # Momentum burst: |move| > momentum_atr_mult * ATR
            if abs(move) <= self.momentum_atr_mult * atr:
                continue

            # Volume confirmation: current volume > volume_mult * avg volume
            vol_window = min(self.vol_avg_period, len(volumes_list) - 1)
            if vol_window <= 0:
                continue
            avg_volume = sum(volumes_list[-vol_window - 1:-1]) / vol_window
            if avg_volume <= 0 or bar.volume <= self.volume_mult * avg_volume:
                continue

            # Position sizing: cap to available cash
            capital = snapshot.portfolio.cash
            qty = int(self.risk_pct * capital / bar.close) if bar.close > 0 else 0
            max_qty = int(capital / bar.close) if bar.close > 0 else 0
            qty = min(qty, max_qty)
            if qty <= 0:
                continue

            if move > 0:
                # Upward burst -> long
                stop = bar.close - self.atr_stop_mult * atr
                signals += self.pm.enter_long(symbol, qty, 0, "MIS", stop)
                self.trades_today[symbol] = (
                    self.trades_today.get(symbol, 0) + 1)
            else:
                # Downward burst -> short
                stop = bar.close + self.atr_stop_mult * atr
                signals += self.pm.enter_short(symbol, qty, 0, stop)
                self.trades_today[symbol] = (
                    self.trades_today.get(symbol, 0) + 1)

        return signals

    def on_complete(self):
        return {"strategy_type": "intraday_momentum"}
