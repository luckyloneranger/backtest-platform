"""Time-of-Day Adaptive -- switches mode based on Indian market time patterns.

9:15-10:15 (opening): Momentum mode -- trade in first-hour direction
10:15-14:00 (midday): Mean-reversion mode -- fade moves away from VWAP
14:00-15:00 (closing): Momentum mode -- trade in closing push direction
15:00: Exit all positions

Uses VWAP as the anchor for all modes.
"""

from collections import deque
from datetime import datetime, timezone, timedelta

from server.registry import register
from strategies.base import Strategy, MarketSnapshot, InstrumentInfo, Signal
from strategies.indicators import compute_vwap_bands, compute_atr
from strategies.position_manager import PositionManager


IST = timezone(timedelta(hours=5, minutes=30))

# Time mode constants
OPENING = "opening"
MIDDAY = "midday"
CLOSING = "closing"
EXIT = "exit"


def _ist_time(timestamp_ms: int) -> tuple[int, int]:
    """Convert epoch ms to IST (hour, minute)."""
    dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=IST)
    return dt.hour, dt.minute


def _time_mode(hour: int, minute: int) -> str:
    """Determine trading mode from IST time."""
    if hour == 9 or (hour == 10 and minute < 15):
        return OPENING
    if (hour == 10 and minute >= 15) or (11 <= hour < 14):
        return MIDDAY
    if 14 <= hour < 15:
        return CLOSING
    return EXIT


@register("time_adaptive")
class TimeAdaptive(Strategy):

    def required_data(self):
        return [{"interval": "5minute", "lookback": 100}]

    def initialize(self, config, instruments):
        self.risk_pct = config.get("risk_pct", 0.03)
        self.std_mult = config.get("std_mult", 1.5)
        self.warmup_bars = config.get("warmup_bars", 6)
        self.max_trades_per_day = config.get("max_trades_per_day", 2)
        self.exit_time_hour = config.get("exit_time_hour", 15)
        self.atr_period = config.get("atr_period", 14)
        self.atr_stop_mult = config.get("atr_stop_mult", 1.0)
        self.closing_lookback = config.get("closing_lookback", 10)
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
        self.first_hour_direction: dict[str, str] = {}  # "UP", "DOWN", ""

    def _ensure(self, symbol: str):
        if symbol not in self.today_highs:
            self.today_highs[symbol] = deque()
            self.today_lows[symbol] = deque()
            self.today_closes[symbol] = deque()
            self.today_volumes[symbol] = deque()
            self.bars_today[symbol] = 0
            self.trades_today[symbol] = 0
            self.last_hour[symbol] = 0
            self.first_hour_direction[symbol] = ""

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
        self.first_hour_direction[symbol] = ""

    def _compute_qty(self, cash: float, price: float) -> int:
        """Position sizing: risk_pct * cash / price, capped to available cash."""
        if price <= 0:
            return 0
        qty = int(self.risk_pct * cash / price)
        max_qty = int(cash / price)
        return min(qty, max_qty)

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

            # Accumulate today's bars
            self.today_highs[symbol].append(bar.high)
            self.today_lows[symbol].append(bar.low)
            self.today_closes[symbol].append(bar.close)
            self.today_volumes[symbol].append(bar.volume)
            self.bars_today[symbol] += 1

            # Skip warmup period (first N bars)
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

            mode = _time_mode(hour, minute)

            # EXIT mode: close all positions at 15:00+
            if mode == EXIT:
                if not self.pm.is_flat(symbol):
                    signals += self.pm.exit_position(symbol)
                continue  # no new entries after exit time

            # Determine first-hour direction after warmup
            if mode == OPENING and not self.first_hour_direction.get(symbol):
                if bar.close > vwap:
                    self.first_hour_direction[symbol] = "UP"
                elif bar.close < vwap:
                    self.first_hour_direction[symbol] = "DOWN"

            # --- Trailing stop update for existing positions ---
            if not self.pm.is_flat(symbol):
                highs_list = list(self.today_highs[symbol])
                lows_list = list(self.today_lows[symbol])
                closes_list = list(self.today_closes[symbol])
                atr = compute_atr(highs_list, lows_list, closes_list,
                                  self.atr_period)
                if atr and atr > 0:
                    if self.pm.is_long(symbol):
                        new_stop = bar.close - self.atr_stop_mult * atr
                        signals += self.pm.update_trailing_stop(symbol, new_stop)
                    elif self.pm.is_short(symbol):
                        new_stop = bar.close + self.atr_stop_mult * atr
                        signals += self.pm.update_trailing_stop(symbol, new_stop)

                # Midday mean-reversion exit at VWAP
                if mode == MIDDAY:
                    if self.pm.is_long(symbol) and bar.close >= vwap:
                        signals += self.pm.exit_position(symbol)
                    elif self.pm.is_short(symbol) and bar.close <= vwap:
                        signals += self.pm.exit_position(symbol)

                continue  # already in a position, skip entry logic

            # --- Skip if already have pending entry or hit trade limit ---
            if self.pm.has_pending_entry(symbol):
                continue

            if self.trades_today.get(symbol, 0) >= self.max_trades_per_day:
                continue

            qty = self._compute_qty(snapshot.portfolio.cash, bar.close)
            if qty <= 0:
                continue

            # === OPENING mode (9:15 - 10:15) ===
            if mode == OPENING:
                direction = self.first_hour_direction.get(symbol, "")
                if not direction:
                    continue

                # Momentum: trade in first-hour direction on pullback to VWAP
                std_dev = upper - vwap  # one std dev
                if direction == "UP" and bar.close <= vwap * 1.005 and bar.close > vwap - std_dev:
                    # Pullback to VWAP but direction is UP -> long
                    stop = vwap - 2 * std_dev
                    signals += self.pm.enter_long(symbol, qty, 0, "MIS", stop)
                    self.trades_today[symbol] += 1

                elif direction == "DOWN" and bar.close >= vwap * 0.995 and bar.close < vwap + std_dev:
                    # Bounce to VWAP but direction is DOWN -> short
                    stop = vwap + 2 * std_dev
                    signals += self.pm.enter_short(symbol, qty, 0, stop)
                    self.trades_today[symbol] += 1

            # === MIDDAY mode (10:15 - 14:00) ===
            elif mode == MIDDAY:
                std_dev = upper - vwap
                if bar.close < lower:
                    # Below lower band -> fade the dip, long
                    stop = vwap - 2 * std_dev
                    signals += self.pm.enter_long(symbol, qty, 0, "MIS", stop)
                    self.trades_today[symbol] += 1

                elif bar.close > upper:
                    # Above upper band -> fade the rally, short
                    stop = vwap + 2 * std_dev
                    signals += self.pm.enter_short(symbol, qty, 0, stop)
                    self.trades_today[symbol] += 1

            # === CLOSING mode (14:00 - 15:00) ===
            elif mode == CLOSING:
                closes_list = list(self.today_closes[symbol])
                lookback = self.closing_lookback
                if len(closes_list) > lookback:
                    close_ago = closes_list[-lookback]
                    trending_up = bar.close > close_ago and bar.close > vwap
                    trending_down = bar.close < close_ago and bar.close < vwap

                    highs_list = list(self.today_highs[symbol])
                    lows_list = list(self.today_lows[symbol])
                    atr = compute_atr(highs_list, lows_list, closes_list,
                                      self.atr_period)

                    if trending_up:
                        stop = bar.close - self.atr_stop_mult * atr if atr else vwap
                        signals += self.pm.enter_long(symbol, qty, 0, "MIS", stop)
                        self.trades_today[symbol] += 1

                    elif trending_down:
                        stop = bar.close + self.atr_stop_mult * atr if atr else vwap
                        signals += self.pm.enter_short(symbol, qty, 0, stop)
                        self.trades_today[symbol] += 1

        return signals

    def on_complete(self):
        return {"strategy_type": "time_adaptive"}
