"""Tests for Time-of-Day Adaptive strategy -- trading logic only.

PositionManager infrastructure is tested in test_position_manager.py.
These tests verify the time-mode switching (opening/midday/closing/exit),
first-hour direction detection, mean-reversion entries, closing momentum,
exit timing, MIS enforcement, and daily trade limits.
"""

import pytest
from datetime import datetime, timezone, timedelta
from strategies.base import (
    BarData, FillInfo, MarketSnapshot, PendingOrder, Portfolio,
    Signal, SessionContext, Position,
)
from strategies.deterministic.time_adaptive import TimeAdaptive, _time_mode

CAPITAL = 1_000_000.0
SYMBOL = "TEST"
IST = timezone(timedelta(hours=5, minutes=30))

# Base datetime for the trading day
_BASE = datetime(2025, 3, 10, 9, 15, 0, tzinfo=IST)


def _ts(hour: int, minute: int = 0) -> int:
    """Build an IST timestamp_ms for a given hour:minute on a fixed date.

    Handles minute values >= 60 by rolling over into subsequent hours.
    """
    extra_hours, minute = divmod(minute, 60)
    dt = datetime(2025, 3, 10, hour + extra_hours, minute, 0, tzinfo=IST)
    return int(dt.timestamp() * 1000)


def _ts_offset(bar_index: int) -> int:
    """Return timestamp_ms for the bar_index-th 5-minute bar starting at 9:15 IST."""
    dt = _BASE + timedelta(minutes=bar_index * 5)
    return int(dt.timestamp() * 1000)


def _snap(close, timestamp_ms, symbol=SYMBOL, high=None, low=None, volume=10000,
          capital=CAPITAL, fills=None, pending_orders=None, positions=None):
    """Build a minimal MarketSnapshot for one 5-minute bar."""
    h = high if high is not None else close + 1
    lo = low if low is not None else close - 1
    bar = BarData(symbol, close, h, lo, close, volume, 0, timestamp_ms)
    return MarketSnapshot(
        timestamp_ms=timestamp_ms,
        timeframes={"5minute": {symbol: bar}},
        history={},
        portfolio=Portfolio(capital, capital, positions or []),
        instruments={},
        fills=fills or [],
        rejections=[],
        closed_trades=[],
        context=SessionContext(capital, 0, 200, "2025-03-10", "2025-03-10",
                               ["5minute"], 100),
        pending_orders=pending_orders or [],
    )


def _init(**overrides):
    """Create and initialize a TimeAdaptive."""
    s = TimeAdaptive()
    config = {
        "risk_pct": 0.03,
        "std_mult": 1.5,
        "warmup_bars": 6,
        "max_trades_per_day": 2,
        "exit_time_hour": 15,
        "atr_period": 14,
        "atr_stop_mult": 1.0,
        "closing_lookback": 10,
    }
    config.update(overrides)
    s.initialize(config, {})
    return s


def _feed_bars(s, n, start_bar_index, price=100.0, high_d=0.5, low_d=0.5,
               volume=10000):
    """Feed n bars at a fixed price starting at the given bar index offset.

    Returns all signals produced.
    """
    all_sigs = []
    for i in range(n):
        ts = _ts_offset(start_bar_index + i)
        snap = _snap(price, ts, high=price + high_d, low=price - low_d,
                     volume=volume)
        all_sigs.extend(s.on_bar(snap))
    return all_sigs


# === Tests ===


class TestRequiredData:
    def test_required_data(self):
        s = TimeAdaptive()
        reqs = s.required_data()
        assert reqs == [{"interval": "5minute", "lookback": 100}]


class TestTimeMode:
    def test_opening_mode_9_15(self):
        assert _time_mode(9, 15) == "opening"

    def test_opening_mode_10_10(self):
        assert _time_mode(10, 10) == "opening"

    def test_midday_mode_10_15(self):
        assert _time_mode(10, 15) == "midday"

    def test_midday_mode_12_00(self):
        assert _time_mode(12, 0) == "midday"

    def test_midday_mode_13_55(self):
        assert _time_mode(13, 55) == "midday"

    def test_closing_mode_14_00(self):
        assert _time_mode(14, 0) == "closing"

    def test_closing_mode_14_55(self):
        assert _time_mode(14, 55) == "closing"

    def test_exit_mode_15_00(self):
        assert _time_mode(15, 0) == "exit"

    def test_exit_mode_15_30(self):
        assert _time_mode(15, 30) == "exit"


class TestNoTradeDuringWarmup:
    def test_no_trade_during_warmup(self):
        """First 6 bars (warmup) should produce no entry signals."""
        s = _init()
        for i in range(6):
            ts = _ts_offset(i)  # 9:15, 9:20, 9:25, 9:30, 9:35, 9:40
            snap = _snap(100.0, ts)
            sigs = s.on_bar(snap)
            entries = [sig for sig in sigs
                       if sig.action in ("BUY", "SELL") and sig.order_type != "SL_M"]
            assert len(entries) == 0, f"Entry signal on warmup bar {i}"


class TestOpeningMomentumLong:
    def test_opening_momentum_long(self):
        """First hour direction UP, then pullback to VWAP -> long entry."""
        s = _init()

        # Feed 6 warmup bars at stable price (9:15-9:40)
        _feed_bars(s, 6, start_bar_index=0, price=100.0)

        # 7th bar at 9:45: close well above flat VWAP -> set direction UP
        ts = _ts_offset(6)  # 9:45
        snap = _snap(103.0, ts, high=103.5, low=102.5, volume=10000)
        s.on_bar(snap)

        # 8th bar at 9:50: pullback to VWAP (~100)
        ts = _ts_offset(7)  # 9:50
        snap = _snap(100.1, ts, high=100.5, low=99.5, volume=10000)
        sigs = s.on_bar(snap)

        buys = [sig for sig in sigs if sig.action == "BUY"]
        assert len(buys) == 1
        assert buys[0].product_type == "MIS"
        assert buys[0].order_type == "MARKET"  # limit_price=0 -> MARKET


class TestOpeningMomentumShort:
    def test_opening_momentum_short(self):
        """First hour direction DOWN, then bounce to VWAP -> short entry."""
        s = _init()

        # 6 warmup bars at stable price (9:15-9:40)
        _feed_bars(s, 6, start_bar_index=0, price=100.0)

        # 7th bar at 9:45: close well below flat VWAP -> set direction DOWN
        ts = _ts_offset(6)  # 9:45
        snap = _snap(97.0, ts, high=97.5, low=96.5, volume=10000)
        s.on_bar(snap)

        # 8th bar at 9:50: bounce to VWAP (~100)
        ts = _ts_offset(7)  # 9:50
        snap = _snap(99.9, ts, high=100.5, low=99.5, volume=10000)
        sigs = s.on_bar(snap)

        sells = [sig for sig in sigs if sig.action == "SELL"]
        assert len(sells) == 1
        assert sells[0].product_type == "MIS"


class TestMiddayMeanReversionLong:
    def test_midday_mean_reversion_long(self):
        """Close < VWAP lower band at midday -> long entry."""
        s = _init()

        # Feed 6 warmup bars at stable price (9:15-9:40, bar indices 0-5)
        _feed_bars(s, 6, start_bar_index=0, price=100.0)

        # 6 more opening bars (9:45-10:10, bar indices 6-11) at stable price
        _feed_bars(s, 6, start_bar_index=6, price=100.0)

        # Bar at 10:15 (bar index 12) -- midday, price well below VWAP lower band
        ts = _ts_offset(12)  # 10:15
        snap = _snap(95.0, ts, high=95.5, low=94.5, volume=10000)
        sigs = s.on_bar(snap)

        buys = [sig for sig in sigs if sig.action == "BUY"]
        assert len(buys) == 1
        assert buys[0].product_type == "MIS"


class TestMiddayExitAtVwap:
    def test_midday_exit_at_vwap(self):
        """Once long, price reaching VWAP at midday triggers exit."""
        s = _init()

        # Feed 12 bars (9:15-10:10, bar indices 0-11) at stable VWAP ~100
        _feed_bars(s, 12, start_bar_index=0, price=100.0)

        # Bar at 10:15 (index 12): midday, price below lower band -> enter long
        ts = _ts_offset(12)  # 10:15
        snap = _snap(95.0, ts, high=95.5, low=94.5, volume=10000)
        sigs = s.on_bar(snap)
        buys = [sig for sig in sigs if sig.action == "BUY"]
        assert len(buys) >= 1
        entry_qty = buys[0].quantity

        # Simulate fill
        fill = FillInfo(SYMBOL, "BUY", entry_qty, 95.0, 0.0, _ts_offset(12))
        position = Position(SYMBOL, entry_qty, 95.0, 0.0)
        pending_stop = PendingOrder(SYMBOL, "SELL", entry_qty, "SL_M", 0.0, 90.0)

        # Next bar at 10:20 (index 13): price back at VWAP (~100)
        ts = _ts_offset(13)  # 10:20
        snap = _snap(100.0, ts, high=100.5, low=99.5, volume=10000,
                     fills=[fill], positions=[position],
                     pending_orders=[pending_stop])
        sigs = s.on_bar(snap)

        # Should see exit signals (CANCEL + SELL MARKET)
        sells = [sig for sig in sigs
                 if sig.action == "SELL" and sig.order_type == "MARKET"]
        assert len(sells) >= 1


class TestClosingMomentumLong:
    def test_closing_momentum_long(self):
        """14:00+ with price above VWAP and trending up -> long entry."""
        s = _init(closing_lookback=3)

        # Feed bars from 9:15 to 10:10 (12 bars, indices 0-11) at stable price
        _feed_bars(s, 12, start_bar_index=0, price=100.0)

        # Feed midday bars from 10:15 to ~13:55 using _ts_offset
        # Bar index 12 = 10:15, need to get to 14:00 = bar index 57
        # (14:00 - 9:15) = 285 min / 5 = 57 bars from start
        _feed_bars(s, 45, start_bar_index=12, price=100.0)

        # Closing mode bars at 14:00, 14:05, 14:10 with rising prices
        # bar index 57 = 14:00, 58 = 14:05, 59 = 14:10
        for i in range(3):
            price = 103.0 + i * 1.0
            ts = _ts_offset(57 + i)
            snap = _snap(price, ts, high=price + 0.5, low=price - 0.5,
                         volume=10000)
            s.on_bar(snap)

        # Bar index 60 = 14:15, close > close_3_bars_ago and close > VWAP
        ts = _ts_offset(60)  # 14:15
        snap = _snap(107.0, ts, high=107.5, low=106.5, volume=10000)
        sigs = s.on_bar(snap)

        buys = [sig for sig in sigs if sig.action == "BUY"]
        assert len(buys) == 1
        assert buys[0].product_type == "MIS"


class TestExitAt1500:
    def test_exit_at_1500(self):
        """All positions closed at 15:00+."""
        s = _init()

        # Feed 12 bars (9:15-10:10, indices 0-11) at stable price
        _feed_bars(s, 12, start_bar_index=0, price=100.0)

        # Enter at midday 10:15 (index 12)
        ts = _ts_offset(12)  # 10:15
        snap = _snap(95.0, ts, high=95.5, low=94.5, volume=10000)
        sigs = s.on_bar(snap)
        buys = [sig for sig in sigs if sig.action == "BUY"]
        assert len(buys) >= 1, "Should enter long below VWAP lower band"

        entry_qty = buys[0].quantity
        fill = FillInfo(SYMBOL, "BUY", entry_qty, 95.0, 0.0, _ts_offset(12))
        position = Position(SYMBOL, entry_qty, 95.0, 0.0)
        pending_stop = PendingOrder(SYMBOL, "SELL", entry_qty, "SL_M",
                                    0.0, 90.0)

        # Feed bar at 10:20 (index 13) to register the fill -- keep price
        # below VWAP so midday exit does NOT trigger
        ts = _ts_offset(13)  # 10:20
        snap = _snap(96.0, ts, high=96.5, low=95.5, volume=10000,
                     fills=[fill], positions=[position],
                     pending_orders=[pending_stop])
        s.on_bar(snap)

        assert not s.pm.is_flat(SYMBOL)

        # EXIT at 15:00 (index 69, since (15:00-9:15)/5 = 69)
        ts = _ts(15, 0)
        snap = _snap(96.0, ts, high=96.5, low=95.5, volume=10000,
                     positions=[position],
                     pending_orders=[pending_stop])
        sigs = s.on_bar(snap)

        sells = [sig for sig in sigs
                 if sig.action == "SELL" and sig.order_type == "MARKET"]
        assert len(sells) >= 1


class TestAllEntriesMIS:
    def test_all_entries_mis(self):
        """Every entry signal should use MIS product type."""
        s = _init()

        # Feed 6 warmup bars (indices 0-5)
        _feed_bars(s, 6, start_bar_index=0, price=100.0)

        all_sigs = []

        # Opening direction UP at 9:45 (index 6)
        ts = _ts_offset(6)
        snap = _snap(103.0, ts, high=103.5, low=102.5, volume=10000)
        all_sigs.extend(s.on_bar(snap))

        # Opening pullback entry at 9:50 (index 7)
        ts = _ts_offset(7)
        snap = _snap(100.1, ts, high=100.5, low=99.5, volume=10000)
        all_sigs.extend(s.on_bar(snap))

        entries = [sig for sig in all_sigs
                   if sig.action in ("BUY", "SELL") and sig.order_type != "SL_M"]
        for sig in entries:
            assert sig.product_type == "MIS", (
                f"Non-MIS entry: {sig.action} {sig.product_type}")


class TestMaxTradesPerDay:
    def test_max_trades_per_day(self):
        """No new entries after max_trades_per_day reached."""
        s = _init(max_trades_per_day=1)

        # Feed 12 bars (9:15-10:10, indices 0-11) at stable price
        _feed_bars(s, 12, start_bar_index=0, price=100.0)

        # 1st midday trade at 10:15 (index 12): below lower band
        ts = _ts_offset(12)
        snap = _snap(95.0, ts, high=95.5, low=94.5, volume=10000)
        sigs = s.on_bar(snap)
        buys1 = [sig for sig in sigs if sig.action == "BUY"]
        assert len(buys1) >= 1, "First trade should be allowed"

        # Simulate fill + exit at VWAP to become flat again
        entry_qty = buys1[0].quantity
        fill = FillInfo(SYMBOL, "BUY", entry_qty, 95.0, 0.0, _ts_offset(12))
        position = Position(SYMBOL, entry_qty, 95.0, 0.0)
        pending_stop = PendingOrder(SYMBOL, "SELL", entry_qty, "SL_M", 0.0, 90.0)

        # Bar at 10:20 (index 13): price at VWAP -> exit
        ts = _ts_offset(13)
        snap = _snap(100.0, ts, high=100.5, low=99.5, volume=10000,
                     fills=[fill], positions=[position],
                     pending_orders=[pending_stop])
        s.on_bar(snap)

        # Should have exited at VWAP
        assert s.pm.is_flat(SYMBOL), "Should have exited at VWAP"

        # 2nd attempt at 10:25 (index 14): should be blocked
        ts = _ts_offset(14)
        snap = _snap(95.0, ts, high=95.5, low=94.5, volume=10000)
        sigs = s.on_bar(snap)
        buys2 = [sig for sig in sigs if sig.action == "BUY"]
        assert len(buys2) == 0, "Second trade should be blocked by max_trades_per_day"


class TestOnComplete:
    def test_on_complete_returns_strategy_type(self):
        s = _init()
        result = s.on_complete()
        assert result == {"strategy_type": "time_adaptive"}
