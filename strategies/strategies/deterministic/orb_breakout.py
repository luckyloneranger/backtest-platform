"""Opening Range Breakout — trade breakout of first 30-min range.

First 6 five-min bars (9:15-9:45) define the opening range.
Breakout above range high -> long. Below range low -> short.
Target at 2x range width. Stop at opposite end of range.

Daily cycle:
  9:15-9:45: Build opening range (track high/low of 6 bars)
  9:45-15:00: Trade breakouts of range
  15:00: Close all positions before MIS squareoff
"""

from collections import deque
from datetime import datetime, timezone, timedelta

from server.registry import register
from strategies.base import Strategy, MarketSnapshot, Signal
from strategies.position_manager import PositionManager


IST = timezone(timedelta(hours=5, minutes=30))


def _ist_time(timestamp_ms: int) -> tuple[int, int]:
    """Convert epoch ms to IST (hour, minute)."""
    dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=IST)
    return dt.hour, dt.minute


@register("orb_breakout")
class OrbBreakout(Strategy):

    def required_data(self):
        return [{"interval": "5minute", "lookback": 50}]

    def initialize(self, config, instruments):
        self.warmup_bars = config.get("warmup_bars", 6)
        self.volume_confirm = config.get("volume_confirm", 1.5)
        self.risk_pct = config.get("risk_pct", 0.03)
        self.max_trades_per_day = config.get("max_trades_per_day", 1)
        self.exit_time_hour = config.get("exit_time_hour", 15)
        self.instruments = instruments

        self.pm = PositionManager(max_pending_bars=2)

        # Per-symbol daily state
        self.range_high: dict[str, float] = {}
        self.range_low: dict[str, float] = {}
        self.range_set: dict[str, bool] = {}
        self.bars_today: dict[str, int] = {}
        self.trades_today: dict[str, int] = {}
        self.last_hour: dict[str, int] = {}
        self.today_highs: dict[str, deque] = {}
        self.today_volumes: dict[str, deque] = {}
        # Deferred profit target: set after entry fill
        self.pending_target: dict[str, tuple[int, float]] = {}  # symbol -> (qty, price)

    def _ensure(self, symbol: str):
        if symbol not in self.range_high:
            self.range_high[symbol] = 0.0
            self.range_low[symbol] = float("inf")
            self.range_set[symbol] = False
            self.bars_today[symbol] = 0
            self.trades_today[symbol] = 0
            self.last_hour[symbol] = 0
            self.today_highs[symbol] = deque()
            self.today_volumes[symbol] = deque()

    def _is_new_day(self, symbol: str, timestamp_ms: int) -> bool:
        """Detect day transition from timestamp."""
        hour, _ = _ist_time(timestamp_ms)
        if symbol not in self.last_hour:
            return True
        if hour == 9 and self.last_hour[symbol] >= 15:
            return True
        return False

    def _reset_day(self, symbol: str):
        """Clear all daily state for a symbol."""
        self.range_high[symbol] = 0.0
        self.range_low[symbol] = float("inf")
        self.range_set[symbol] = False
        self.bars_today[symbol] = 0
        self.trades_today[symbol] = 0
        self.today_highs[symbol] = deque()
        self.today_volumes[symbol] = deque()
        self.pending_target.pop(symbol, None)

    def on_bar(self, snapshot: MarketSnapshot) -> list[Signal]:
        self.pm.increment_bars()
        signals = self.pm.process_fills(snapshot)
        signals += self.pm.resubmit_expired(snapshot)
        self.pm.reconcile(snapshot)

        # Submit deferred profit targets after entry fill
        for symbol in list(self.pending_target):
            if not self.pm.is_flat(symbol) and not self.pm.has_pending_entry(symbol):
                qty, price = self.pending_target.pop(symbol)
                signals += self.pm.set_profit_target(symbol, qty, price)

        if "5minute" not in snapshot.timeframes:
            return signals

        for symbol, bar in snapshot.timeframes["5minute"].items():
            self._ensure(symbol)
            hour, minute = _ist_time(snapshot.timestamp_ms)

            # New day detection -- reset daily state
            if self._is_new_day(symbol, snapshot.timestamp_ms):
                self._reset_day(symbol)

            self.last_hour[symbol] = hour
            self.bars_today[symbol] += 1
            self.today_volumes[symbol].append(bar.volume)

            # Phase 1: Build opening range (first 6 bars = 9:15-9:45)
            if self.bars_today[symbol] <= self.warmup_bars:
                self.range_high[symbol] = max(self.range_high[symbol], bar.high)
                self.range_low[symbol] = min(self.range_low[symbol], bar.low)
                if self.bars_today[symbol] == self.warmup_bars:
                    self.range_set[symbol] = True
                continue

            # Range must be set before any trading
            if not self.range_set[symbol]:
                continue

            # Time exit: close all positions at 15:00
            if hour >= self.exit_time_hour:
                if not self.pm.is_flat(symbol):
                    signals += self.pm.exit_position(symbol)
                continue

            # Skip if already in position or have pending entry
            if not self.pm.is_flat(symbol) or self.pm.has_pending_entry(symbol):
                continue

            # Max trades per day check
            if self.trades_today.get(symbol, 0) >= self.max_trades_per_day:
                continue

            # Volume confirmation
            vol_list = list(self.today_volumes[symbol])
            avg_vol = (sum(vol_list) / len(vol_list)) if vol_list else bar.volume
            if bar.volume < self.volume_confirm * avg_vol:
                continue

            range_width = self.range_high[symbol] - self.range_low[symbol]
            if range_width <= 0:
                continue

            # Position sizing: cap to available cash
            qty = int(self.risk_pct * snapshot.portfolio.cash / bar.close) if bar.close > 0 else 0
            max_qty = int(snapshot.portfolio.cash / bar.close) if bar.close > 0 else 0
            qty = min(qty, max_qty)
            if qty <= 0:
                continue

            # Long breakout: close above range high
            if bar.close > self.range_high[symbol]:
                stop = self.range_low[symbol]
                signals += self.pm.enter_long(symbol, qty, 0, "MIS", stop)
                target = bar.close + 2 * range_width
                self.pending_target[symbol] = (qty, target)
                self.trades_today[symbol] = self.trades_today.get(symbol, 0) + 1

            # Short breakout: close below range low
            elif bar.close < self.range_low[symbol]:
                stop = self.range_high[symbol]
                signals += self.pm.enter_short(symbol, qty, 0, stop)
                target = bar.close - 2 * range_width
                self.pending_target[symbol] = (qty, target)
                self.trades_today[symbol] = self.trades_today.get(symbol, 0) + 1

        return signals

    def on_complete(self):
        return {"strategy_type": "orb_breakout"}
