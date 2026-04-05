"""Tests for VWAP Mean Reversion strategy.

Covers: required_data, entry logic (long below lower band, short above upper band),
warmup gating, VWAP exit, time exit, max trades per day, new-day reset,
and MIS-only product type.
"""

import pytest
from datetime import datetime, timezone, timedelta

from strategies.base import (
    BarData, FillInfo, MarketSnapshot, PendingOrder, Portfolio,
    Signal, SessionContext, Position,
)
from strategies.deterministic.vwap_reversion import VwapReversion, _ist_time


IST = timezone(timedelta(hours=5, minutes=30))
CAPITAL = 1_000_000.0
SYMBOL = "TEST"


def _ts(hour: int, minute: int = 0, day: int = 2) -> int:
    """Build epoch ms for a given IST hour:minute on 2024-01-{day}."""
    dt = datetime(2024, 1, day, hour, minute, tzinfo=IST)
    return int(dt.timestamp() * 1000)


def _snap(close: float, timestamp_ms: int, symbol: str = SYMBOL,
          high=None, low=None, volume: int = 10000,
          capital: float = CAPITAL, fills=None,
          pending_orders=None, positions=None) -> MarketSnapshot:
    """Build a minimal 5-minute MarketSnapshot."""
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
        context=SessionContext(capital, 0, 200, "2024-01-01", "2024-12-31",
                               ["5minute"], 100),
        pending_orders=pending_orders or [],
    )


def _init(**overrides) -> VwapReversion:
    """Create and initialize a VwapReversion with defaults."""
    s = VwapReversion()
    config = {
        "std_mult": 1.0,
        "risk_pct": 0.02,
        "warmup_bars": 6,
        "exit_time_hour": 15,
        "max_trades_per_day": 3,
    }
    config.update(overrides)
    s.initialize(config, {})
    return s


def _feed_warmup(s: VwapReversion, n: int = 6, base_price: float = 100.0,
                 start_minute: int = 15):
    """Feed n warmup bars (9:15..9:40 by default) with stable prices.

    Returns list of all signals emitted during warmup.
    """
    all_signals = []
    for i in range(n):
        minute = start_minute + i * 5
        ts = _ts(9, minute)
        snap = _snap(base_price, ts, volume=50000)
        sigs = s.on_bar(snap)
        all_signals.extend(sigs)
    return all_signals


class TestRequiredData:
    def test_required_data(self):
        s = VwapReversion()
        reqs = s.required_data()
        assert reqs == [{"interval": "5minute", "lookback": 100}]


class TestIstTimeHelper:
    def test_ist_conversion(self):
        ts = _ts(9, 15)
        hour, minute = _ist_time(ts)
        assert hour == 9
        assert minute == 15

    def test_afternoon(self):
        ts = _ts(15, 0)
        hour, minute = _ist_time(ts)
        assert hour == 15
        assert minute == 0


class TestNoEntryDuringWarmup:
    def test_first_6_bars_no_signals(self):
        """No entry signals during the warmup period (first 6 bars)."""
        s = _init()
        signals = _feed_warmup(s, n=6, base_price=100.0)

        buys = [sig for sig in signals if sig.action == "BUY"]
        sells = [sig for sig in signals if sig.action == "SELL"]
        assert len(buys) == 0
        assert len(sells) == 0


class TestLongEntryBelowLowerBand:
    def test_long_entry_below_lower_band(self):
        """Price below VWAP - std_dev triggers a BUY signal."""
        s = _init()

        # Warmup: 6 bars at 100 with some variation to create std dev
        prices = [101, 102, 100, 99, 101, 100]
        for i, p in enumerate(prices):
            ts = _ts(9, 15 + i * 5)
            snap = _snap(p, ts, high=p + 2, low=p - 2, volume=50000)
            s.on_bar(snap)

        # Bar 7: price drops well below VWAP - 1 std dev
        ts7 = _ts(9, 45)
        snap7 = _snap(93.0, ts7, high=94, low=92, volume=50000)
        signals = s.on_bar(snap7)

        buys = [sig for sig in signals if sig.action == "BUY"]
        assert len(buys) == 1
        assert buys[0].symbol == SYMBOL
        assert buys[0].order_type == "LIMIT"
        assert buys[0].quantity > 0


class TestShortEntryAboveUpperBand:
    def test_short_entry_above_upper_band(self):
        """Price above VWAP + std_dev triggers a SELL signal."""
        s = _init()

        # Warmup: 6 bars around 100
        prices = [99, 100, 101, 100, 99, 100]
        for i, p in enumerate(prices):
            ts = _ts(9, 15 + i * 5)
            snap = _snap(p, ts, high=p + 2, low=p - 2, volume=50000)
            s.on_bar(snap)

        # Bar 7: price jumps well above VWAP + 1 std dev
        ts7 = _ts(9, 45)
        snap7 = _snap(107.0, ts7, high=108, low=106, volume=50000)
        signals = s.on_bar(snap7)

        sells = [sig for sig in signals if sig.action == "SELL"]
        assert len(sells) == 1
        assert sells[0].symbol == SYMBOL
        assert sells[0].product_type == "MIS"  # shorts always MIS


class TestExitAtVwap:
    def test_long_exit_when_price_reaches_vwap(self):
        """After a long entry, price reaching VWAP triggers exit."""
        s = _init()

        # Warmup bars
        prices = [101, 102, 100, 99, 101, 100]
        for i, p in enumerate(prices):
            ts = _ts(9, 15 + i * 5)
            snap = _snap(p, ts, high=p + 2, low=p - 2, volume=50000)
            s.on_bar(snap)

        # Entry bar: price below lower band
        ts_entry = _ts(9, 45)
        snap_entry = _snap(93.0, ts_entry, high=94, low=92, volume=50000)
        entry_sigs = s.on_bar(snap_entry)
        entry_buys = [sig for sig in entry_sigs if sig.action == "BUY"]
        assert len(entry_buys) == 1
        entry_qty = entry_buys[0].quantity

        # Simulate fill
        fill = FillInfo(SYMBOL, "BUY", entry_qty, 93.0, 0.0, ts_entry)
        position = Position(SYMBOL, entry_qty, 93.0, 0.0)
        pending_stop = PendingOrder(SYMBOL, "SELL", entry_qty, "SL_M", 0.0, 85.0)

        ts_fill = _ts(9, 50)
        snap_fill = _snap(94.0, ts_fill, high=95, low=93, volume=50000,
                          fills=[fill], positions=[position],
                          pending_orders=[pending_stop])
        s.on_bar(snap_fill)

        # Price reaches VWAP (~100) -> exit
        ts_exit = _ts(9, 55)
        snap_exit = _snap(100.5, ts_exit, high=101, low=99, volume=50000,
                          positions=[position],
                          pending_orders=[PendingOrder(SYMBOL, "SELL", entry_qty,
                                                      "SL_M", 0.0, 85.0)])
        exit_sigs = s.on_bar(snap_exit)

        # Should see CANCEL + SELL (exit)
        sell_exits = [sig for sig in exit_sigs
                      if sig.action == "SELL" and sig.order_type == "MARKET"]
        assert len(sell_exits) >= 1


class TestTimeExitAt1500:
    def test_exit_at_1500(self):
        """All positions closed when hour >= 15."""
        s = _init()

        # Warmup bars
        prices = [101, 102, 100, 99, 101, 100]
        for i, p in enumerate(prices):
            ts = _ts(9, 15 + i * 5)
            snap = _snap(p, ts, high=p + 2, low=p - 2, volume=50000)
            s.on_bar(snap)

        # Entry: price below lower band
        ts_entry = _ts(9, 45)
        snap_entry = _snap(93.0, ts_entry, high=94, low=92, volume=50000)
        entry_sigs = s.on_bar(snap_entry)
        entry_buys = [sig for sig in entry_sigs if sig.action == "BUY"]
        assert len(entry_buys) == 1
        entry_qty = entry_buys[0].quantity

        # Simulate fill
        fill = FillInfo(SYMBOL, "BUY", entry_qty, 93.0, 0.0, ts_entry)
        position = Position(SYMBOL, entry_qty, 93.0, 0.0)
        pending_stop = PendingOrder(SYMBOL, "SELL", entry_qty, "SL_M", 0.0, 85.0)

        ts_fill = _ts(10, 0)
        snap_fill = _snap(94.0, ts_fill, high=95, low=93, volume=50000,
                          fills=[fill], positions=[position],
                          pending_orders=[pending_stop])
        s.on_bar(snap_fill)

        # Feed bars until 15:00 — price still below VWAP so no profit exit
        # Jump directly to 15:00
        ts_1500 = _ts(15, 0)
        snap_1500 = _snap(95.0, ts_1500, high=96, low=94, volume=50000,
                          positions=[position],
                          pending_orders=[PendingOrder(SYMBOL, "SELL", entry_qty,
                                                      "SL_M", 0.0, 85.0)])
        sigs_1500 = s.on_bar(snap_1500)

        # Should have exit signals: CANCEL + SELL
        sell_exits = [sig for sig in sigs_1500
                      if sig.action == "SELL" and sig.order_type == "MARKET"]
        assert len(sell_exits) >= 1


class TestMaxTradesPerDay:
    def test_no_entry_after_max_trades(self):
        """After max_trades_per_day reached, no more entries."""
        s = _init(max_trades_per_day=1)

        # Warmup
        prices = [101, 102, 100, 99, 101, 100]
        for i, p in enumerate(prices):
            ts = _ts(9, 15 + i * 5)
            snap = _snap(p, ts, high=p + 2, low=p - 2, volume=50000)
            s.on_bar(snap)

        # First entry: price below lower band
        ts1 = _ts(9, 45)
        snap1 = _snap(93.0, ts1, high=94, low=92, volume=50000)
        sigs1 = s.on_bar(snap1)
        buys1 = [sig for sig in sigs1 if sig.action == "BUY"]
        assert len(buys1) == 1

        # Simulate fill + full exit so position is flat
        entry_qty = buys1[0].quantity
        fill = FillInfo(SYMBOL, "BUY", entry_qty, 93.0, 0.0, ts1)
        position = Position(SYMBOL, entry_qty, 93.0, 0.0)
        pending_stop = PendingOrder(SYMBOL, "SELL", entry_qty, "SL_M", 0.0, 85.0)
        ts_fill = _ts(9, 50)
        snap_fill = _snap(94.0, ts_fill, high=95, low=93, volume=50000,
                          fills=[fill], positions=[position],
                          pending_orders=[pending_stop])
        s.on_bar(snap_fill)

        # Exit at VWAP
        ts_exit = _ts(9, 55)
        snap_exit = _snap(100.5, ts_exit, high=101, low=99, volume=50000,
                          positions=[position],
                          pending_orders=[PendingOrder(SYMBOL, "SELL", entry_qty,
                                                      "SL_M", 0.0, 85.0)])
        s.on_bar(snap_exit)

        # Now flat; try another entry — should be blocked by max_trades
        ts_try = _ts(10, 0)
        snap_try = _snap(93.0, ts_try, high=94, low=92, volume=50000)
        sigs_try = s.on_bar(snap_try)

        buys_try = [sig for sig in sigs_try if sig.action == "BUY"]
        assert len(buys_try) == 0


class TestNewDayResetsState:
    def test_new_day_resets_bars_today(self):
        """A timestamp on a new day resets daily counters."""
        s = _init()

        # Day 1: feed warmup bars
        for i in range(6):
            ts = _ts(9, 15 + i * 5, day=2)
            snap = _snap(100.0, ts, volume=50000)
            s.on_bar(snap)

        assert s.bars_today[SYMBOL] == 6

        # Feed a 15:00 bar on day 1 to set last_hour
        ts_close = _ts(15, 0, day=2)
        snap_close = _snap(100.0, ts_close, volume=50000)
        s.on_bar(snap_close)
        assert s.last_hour[SYMBOL] == 15

        # Day 2: 9:15 bar triggers new day reset
        ts_new = _ts(9, 15, day=3)
        snap_new = _snap(100.0, ts_new, volume=50000)
        s.on_bar(snap_new)

        assert s.bars_today[SYMBOL] == 1  # reset + 1 new bar


class TestAllEntriesMis:
    def test_long_entry_is_mis(self):
        """Long entries use MIS product type."""
        s = _init()

        prices = [101, 102, 100, 99, 101, 100]
        for i, p in enumerate(prices):
            ts = _ts(9, 15 + i * 5)
            snap = _snap(p, ts, high=p + 2, low=p - 2, volume=50000)
            s.on_bar(snap)

        ts_entry = _ts(9, 45)
        snap_entry = _snap(93.0, ts_entry, high=94, low=92, volume=50000)
        sigs = s.on_bar(snap_entry)

        buys = [sig for sig in sigs if sig.action == "BUY"]
        assert len(buys) == 1
        assert buys[0].product_type == "MIS"

    def test_short_entry_is_mis(self):
        """Short entries use MIS product type."""
        s = _init()

        prices = [99, 100, 101, 100, 99, 100]
        for i, p in enumerate(prices):
            ts = _ts(9, 15 + i * 5)
            snap = _snap(p, ts, high=p + 2, low=p - 2, volume=50000)
            s.on_bar(snap)

        ts_entry = _ts(9, 45)
        snap_entry = _snap(107.0, ts_entry, high=108, low=106, volume=50000)
        sigs = s.on_bar(snap_entry)

        sells = [sig for sig in sigs if sig.action == "SELL"]
        assert len(sells) == 1
        assert sells[0].product_type == "MIS"
