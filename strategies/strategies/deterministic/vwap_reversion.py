"""VWAP Mean Reversion — buy below VWAP, sell at VWAP.

Pure intraday 5-minute strategy. Stocks revert to VWAP intraday.
Buy when price drops below VWAP - 1 std dev, exit when price returns to VWAP.

Daily cycle:
  9:15-9:45: Warmup (accumulate bars, compute VWAP, no trading)
  9:45-15:00: Trade entries and exits
  15:00: Close all positions before MIS squareoff
"""

from collections import deque
from datetime import datetime, timezone, timedelta

from server.registry import register
from strategies.base import Strategy, MarketSnapshot, InstrumentInfo, Signal
from strategies.indicators import compute_vwap_bands
from strategies.position_manager import PositionManager


IST = timezone(timedelta(hours=5, minutes=30))


def _ist_time(timestamp_ms: int) -> tuple[int, int]:
    """Convert epoch ms to IST (hour, minute)."""
    dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=IST)
    return dt.hour, dt.minute


@register("vwap_reversion")
class VwapReversion(Strategy):

    def required_data(self):
        return [{"interval": "5minute", "lookback": 100}]

    def initialize(self, config, instruments):
        self.std_mult = config.get("std_mult", 1.0)
        self.risk_pct = config.get("risk_pct", 0.02)
        self.warmup_bars = config.get("warmup_bars", 6)
        self.exit_time_hour = config.get("exit_time_hour", 15)
        self.max_trades_per_day = config.get("max_trades_per_day", 3)
        self.instruments = instruments

        self.pm = PositionManager(max_pending_bars=2)

        # Per-symbol daily state
        self.today_highs: dict[str, deque] = {}
        self.today_lows: dict[str, deque] = {}
        self.today_closes: dict[str, deque] = {}
        self.today_volumes: dict[str, deque] = {}
        self.bars_today: dict[str, int] = {}
        self.trades_today: dict[str, int] = {}
        self.last_hour: dict[str, int] = {}
        self.entry_vwap: dict[str, float] = {}

    def _ensure(self, symbol: str):
        if symbol not in self.today_highs:
            self.today_highs[symbol] = deque()
            self.today_lows[symbol] = deque()
            self.today_closes[symbol] = deque()
            self.today_volumes[symbol] = deque()
            self.bars_today[symbol] = 0
            self.trades_today[symbol] = 0
            self.last_hour[symbol] = 0
            self.entry_vwap[symbol] = 0.0

    def _is_new_day(self, symbol: str, timestamp_ms: int) -> bool:
        """Detect day transition from timestamp."""
        hour, _ = _ist_time(timestamp_ms)
        if symbol not in self.last_hour:
            return True
        if hour == 9 and self.last_hour[symbol] >= 15:
            return True
        return False

    def _reset_day(self, symbol: str):
        """Clear all daily accumulators for a symbol."""
        self.today_highs[symbol] = deque()
        self.today_lows[symbol] = deque()
        self.today_closes[symbol] = deque()
        self.today_volumes[symbol] = deque()
        self.bars_today[symbol] = 0
        self.trades_today[symbol] = 0
        self.entry_vwap[symbol] = 0.0

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

            # Accumulate today's bars
            self.today_highs[symbol].append(bar.high)
            self.today_lows[symbol].append(bar.low)
            self.today_closes[symbol].append(bar.close)
            self.today_volumes[symbol].append(bar.volume)
            self.bars_today[symbol] += 1

            # Skip warmup period (first 30 min = 6 bars of 5min)
            if self.bars_today[symbol] < self.warmup_bars:
                continue

            # Compute VWAP + bands from today's bars
            vwap_result = compute_vwap_bands(
                list(self.today_highs[symbol]),
                list(self.today_lows[symbol]),
                list(self.today_closes[symbol]),
                list(self.today_volumes[symbol]),
                self.std_mult,
            )
            if vwap_result is None:
                continue
            vwap, upper, lower = vwap_result

            # Time exit: close all positions at 15:00
            if hour >= self.exit_time_hour:
                if not self.pm.is_flat(symbol):
                    signals += self.pm.exit_position(symbol)
                continue  # no new entries after exit time

            # --- Entry logic ---
            if self.pm.is_flat(symbol) and not self.pm.has_pending_entry(symbol):
                if self.trades_today.get(symbol, 0) >= self.max_trades_per_day:
                    continue

                qty = int(self.risk_pct * snapshot.portfolio.cash / bar.close) if bar.close > 0 else 0
                if qty <= 0:
                    continue

                # Long: price below lower band
                if bar.close < lower:
                    stop = vwap - 2 * (vwap - lower)  # 2x std dev below VWAP
                    signals += self.pm.enter_long(symbol, qty, lower, "MIS", stop)
                    self.entry_vwap[symbol] = vwap
                    self.trades_today[symbol] = self.trades_today.get(symbol, 0) + 1

                # Short: price above upper band
                elif bar.close > upper:
                    stop = vwap + 2 * (upper - vwap)  # 2x std dev above VWAP
                    signals += self.pm.enter_short(symbol, qty, upper, stop)
                    self.entry_vwap[symbol] = vwap
                    self.trades_today[symbol] = self.trades_today.get(symbol, 0) + 1

            # --- Exit logic: profit at VWAP ---
            elif self.pm.is_long(symbol):
                if bar.close >= vwap:
                    signals += self.pm.exit_position(symbol)

            elif self.pm.is_short(symbol):
                if bar.close <= vwap:
                    signals += self.pm.exit_position(symbol)

        return signals

    def on_complete(self):
        return {"strategy_type": "vwap_reversion"}
