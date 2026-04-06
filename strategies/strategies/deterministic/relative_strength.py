"""Relative Strength Rotation -- long strongest, short weakest stocks.

Market-neutral intraday strategy. Each day at 9:45 (after 30-min warmup),
ranks all symbols by their opening momentum (first 30-min return).
Goes long the top N stocks, short the bottom N stocks. Equal dollar allocation.
Exits all at 15:00.

Daily cycle:
  9:15-9:45: Warmup (track open price and accumulate closes per symbol)
  9:45: Rank symbols by 30-min return, enter long top N / short bottom N
  9:45-15:00: Trail stops at 1.5x ATR
  15:00: Close all positions before MIS squareoff

Requires at least n_long + n_short symbols to trade.
"""

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


@register("relative_strength")
class RelativeStrength(Strategy):

    def required_data(self):
        return [{"interval": "15minute", "lookback": 30}]

    def initialize(self, config, instruments):
        self.n_long = config.get("n_long", 3)
        self.n_short = config.get("n_short", 3)
        self.risk_pct = config.get("risk_pct", 0.20)
        self.warmup_bars = config.get("warmup_bars", 2)  # 2 x 15min = 30min
        self.exit_time_hour = config.get("exit_time_hour", 15)
        self.atr_period = config.get("atr_period", 14)
        self.atr_stop_mult = config.get("atr_stop_mult", 1.5)
        self.instruments = instruments

        self.pm = PositionManager(max_pending_bars=2)

        # Per-symbol daily state
        self.open_prices: dict[str, float] = {}
        self.highs: dict[str, list[float]] = {}
        self.lows: dict[str, list[float]] = {}
        self.closes: dict[str, list[float]] = {}
        self.bars_today: int = 0
        self.rankings_done: bool = False
        self.entry_done: bool = False
        self.last_hour: int = 0

    def _is_new_day(self, timestamp_ms: int) -> bool:
        """Detect day transition from timestamp."""
        hour, _ = _ist_time(timestamp_ms)
        if self.last_hour == 0 and self.bars_today == 0:
            return True
        if hour == 9 and self.last_hour >= 15:
            return True
        return False

    def _reset_day(self):
        """Clear all daily state."""
        self.open_prices.clear()
        self.highs.clear()
        self.lows.clear()
        self.closes.clear()
        self.bars_today = 0
        self.rankings_done = False
        self.entry_done = False

    def on_bar(self, snapshot: MarketSnapshot) -> list[Signal]:
        self.pm.increment_bars()
        signals = self.pm.process_fills(snapshot)
        signals += self.pm.resubmit_expired(snapshot)
        self.pm.reconcile(snapshot)

        if "15minute" not in snapshot.timeframes:
            return signals

        bars = snapshot.timeframes["15minute"]
        hour, minute = _ist_time(snapshot.timestamp_ms)

        # New day detection -- reset daily state
        if self._is_new_day(snapshot.timestamp_ms):
            self._reset_day()

        self.last_hour = hour
        self.bars_today += 1

        # Accumulate bar data for all symbols
        for symbol, bar in bars.items():
            # Track first bar's open as the opening price
            if symbol not in self.open_prices:
                self.open_prices[symbol] = bar.open
            if symbol not in self.highs:
                self.highs[symbol] = []
                self.lows[symbol] = []
                self.closes[symbol] = []
            self.highs[symbol].append(bar.high)
            self.lows[symbol].append(bar.low)
            self.closes[symbol].append(bar.close)

        # Phase 1: Warmup -- no trading during first N bars
        if self.bars_today <= self.warmup_bars:
            return signals

        # Time exit: close all positions at 15:00
        if hour >= self.exit_time_hour:
            for symbol in list(bars.keys()):
                if not self.pm.is_flat(symbol):
                    signals += self.pm.exit_position(symbol)
            return signals

        # --- Ranking and entry at bar warmup_bars + 1 (9:45) ---
        if not self.rankings_done:
            self.rankings_done = True

            # Compute 30-min returns for each symbol
            returns: dict[str, float] = {}
            for symbol in bars:
                if symbol in self.open_prices and symbol in self.closes:
                    open_p = self.open_prices[symbol]
                    close_p = self.closes[symbol][-1]
                    if open_p > 0:
                        returns[symbol] = (close_p - open_p) / open_p

            # Need enough symbols to rank
            min_symbols = self.n_long + self.n_short
            if len(returns) < min_symbols:
                return signals

            # Rank by return: highest first
            ranked = sorted(returns.items(), key=lambda x: x[1], reverse=True)
            long_symbols = [sym for sym, _ in ranked[:self.n_long]]
            short_symbols = [sym for sym, _ in ranked[-self.n_short:]]

            # Equal dollar allocation
            n_positions = self.n_long + self.n_short
            capital = snapshot.portfolio.cash
            alloc_per_pos = capital * self.risk_pct / n_positions

            for symbol in long_symbols:
                bar = bars[symbol]
                if bar.close <= 0:
                    continue
                qty = int(alloc_per_pos / bar.close)
                max_qty = int(capital / bar.close) if bar.close > 0 else 0
                qty = min(qty, max_qty)
                if qty <= 0:
                    continue

                # Compute ATR for stop
                h = self.highs.get(symbol, [])
                l = self.lows.get(symbol, [])
                c = self.closes.get(symbol, [])
                atr = compute_atr(h, l, c, min(self.atr_period, max(len(h) - 1, 1)))
                stop = bar.close - (atr or bar.close * 0.02) * self.atr_stop_mult

                signals += self.pm.enter_long(symbol, qty, 0, "MIS", stop)

            for symbol in short_symbols:
                bar = bars[symbol]
                if bar.close <= 0:
                    continue
                qty = int(alloc_per_pos / bar.close)
                max_qty = int(capital / bar.close) if bar.close > 0 else 0
                qty = min(qty, max_qty)
                if qty <= 0:
                    continue

                # Compute ATR for stop
                h = self.highs.get(symbol, [])
                l = self.lows.get(symbol, [])
                c = self.closes.get(symbol, [])
                atr = compute_atr(h, l, c, min(self.atr_period, max(len(h) - 1, 1)))
                stop = bar.close + (atr or bar.close * 0.02) * self.atr_stop_mult

                signals += self.pm.enter_short(symbol, qty, 0, stop)

            self.entry_done = True
            return signals

        # --- Trailing stop updates for existing positions ---
        for symbol, bar in bars.items():
            if self.pm.is_flat(symbol):
                continue

            h = self.highs.get(symbol, [])
            l = self.lows.get(symbol, [])
            c = self.closes.get(symbol, [])
            atr = compute_atr(h, l, c, min(self.atr_period, max(len(h) - 1, 1)))
            if atr is None or atr <= 0:
                continue

            if self.pm.is_long(symbol):
                new_stop = bar.close - self.atr_stop_mult * atr
                signals += self.pm.update_trailing_stop(symbol, new_stop)
            elif self.pm.is_short(symbol):
                new_stop = bar.close + self.atr_stop_mult * atr
                signals += self.pm.update_trailing_stop(symbol, new_stop)

        return signals

    def on_complete(self):
        return {"strategy_type": "relative_strength"}
