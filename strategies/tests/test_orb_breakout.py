"""Tests for Opening Range Breakout (ORB) strategy.

Covers: required_data, warmup gating, long entry above range,
short entry below range, no entry without volume, profit target at 2x range,
time exit at 15:00, max 1 trade per day.
"""

import pytest
from datetime import datetime, timezone, timedelta

from strategies.base import (
    BarData, FillInfo, MarketSnapshot, PendingOrder, Portfolio,
    Signal, SessionContext, Position,
)
from strategies.deterministic.orb_breakout import OrbBreakout, _ist_time


IST = timezone(timedelta(hours=5, minutes=30))
CAPITAL = 1_000_000.0
SYMBOL = "TEST"


def _ts(hour: int, minute: int = 0, day: int = 2) -> int:
    """Build epoch ms for a given IST hour:minute on 2024-01-{day}."""
    dt = datetime(2024, 1, day, hour, minute, tzinfo=IST)
    return int(dt.timestamp() * 1000)


def _snap(close: float, timestamp_ms: int, symbol: str = SYMBOL,
          high=None, low=None, volume: int = 100000,
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
                               ["5minute"], 50),
        pending_orders=pending_orders or [],
    )


def _init(**overrides) -> OrbBreakout:
    """Create and initialize an OrbBreakout with defaults."""
    s = OrbBreakout()
    config = {
        "warmup_bars": 6,
        "volume_confirm": 1.5,
        "risk_pct": 0.03,
        "max_trades_per_day": 1,
        "exit_time_hour": 15,
    }
    config.update(overrides)
    s.initialize(config, {})
    return s


def _feed_warmup(s: OrbBreakout, n: int = 6,
                 range_high: float = 105.0, range_low: float = 95.0,
                 volume: int = 50000):
    """Feed n warmup bars (9:15..9:40) with prices between range_low and range_high.

    Returns all signals generated during warmup.
    """
    all_signals = []
    for i in range(n):
        minute = 15 + i * 5
        ts = _ts(9, minute)
        # Oscillate between high and low to establish range
        if i == 0:
            price = range_high  # first bar sets high
        elif i == 1:
            price = range_low   # second bar sets low
        else:
            price = (range_high + range_low) / 2.0
        snap = _snap(price, ts, high=range_high, low=range_low, volume=volume)
        sigs = s.on_bar(snap)
        all_signals.extend(sigs)
    return all_signals


class TestRequiredData:
    def test_required_data(self):
        s = OrbBreakout()
        reqs = s.required_data()
        assert reqs == [{"interval": "5minute", "lookback": 50}]


class TestNoTradeDuringWarmup:
    def test_first_6_bars_no_trades(self):
        """No entry signals during the warmup/range-building period."""
        s = _init()
        signals = _feed_warmup(s, n=6, range_high=105.0, range_low=95.0)

        buys = [sig for sig in signals if sig.action == "BUY"]
        sells = [sig for sig in signals if sig.action == "SELL"]
        assert len(buys) == 0
        assert len(sells) == 0

    def test_range_set_after_warmup(self):
        """After 6 bars, range_high and range_low are set."""
        s = _init()
        _feed_warmup(s, n=6, range_high=105.0, range_low=95.0)

        assert s.range_set[SYMBOL] is True
        assert s.range_high[SYMBOL] == 105.0
        assert s.range_low[SYMBOL] == 95.0


class TestLongEntryAboveRange:
    def test_long_entry_above_range(self):
        """Close > range_high with volume -> BUY MARKET."""
        s = _init(volume_confirm=1.0)  # relax volume filter
        _feed_warmup(s, n=6, range_high=105.0, range_low=95.0)

        # Bar at 9:45 with close above range_high
        ts = _ts(9, 45)
        snap = _snap(106.0, ts, high=107.0, low=104.0, volume=100000)
        signals = s.on_bar(snap)

        buys = [sig for sig in signals if sig.action == "BUY"]
        assert len(buys) >= 1
        assert buys[0].symbol == SYMBOL
        assert buys[0].quantity > 0
        assert buys[0].order_type == "MARKET"
        assert buys[0].product_type == "MIS"


class TestShortEntryBelowRange:
    def test_short_entry_below_range(self):
        """Close < range_low with volume -> SELL MIS."""
        s = _init(volume_confirm=1.0)
        _feed_warmup(s, n=6, range_high=105.0, range_low=95.0)

        # Bar at 9:45 with close below range_low
        ts = _ts(9, 45)
        snap = _snap(94.0, ts, high=96.0, low=93.0, volume=100000)
        signals = s.on_bar(snap)

        sells = [sig for sig in signals if sig.action == "SELL"]
        assert len(sells) >= 1
        assert sells[0].symbol == SYMBOL
        assert sells[0].product_type == "MIS"
        assert sells[0].order_type == "MARKET"


class TestNoEntryWithoutVolume:
    def test_no_entry_low_volume(self):
        """Low volume bar -> no entry even at breakout."""
        s = _init(volume_confirm=2.0)  # require 2x avg volume
        _feed_warmup(s, n=6, range_high=105.0, range_low=95.0, volume=50000)

        # Breakout bar with very low volume (below 2x * avg_vol of ~50k)
        ts = _ts(9, 45)
        snap = _snap(106.0, ts, high=107.0, low=104.0, volume=1000)
        signals = s.on_bar(snap)

        buys = [sig for sig in signals if sig.action == "BUY"]
        assert len(buys) == 0


class TestProfitTarget2xRange:
    def test_profit_target_2x_range(self):
        """Profit target set at entry + 2 * range_width for longs."""
        s = _init(volume_confirm=1.0)
        _feed_warmup(s, n=6, range_high=105.0, range_low=95.0)

        # Range width = 105 - 95 = 10, target = 106 + 2*10 = 126
        ts_entry = _ts(9, 45)
        snap = _snap(106.0, ts_entry, high=107.0, low=104.0, volume=100000)
        entry_sigs = s.on_bar(snap)

        buys = [sig for sig in entry_sigs if sig.action == "BUY"]
        assert len(buys) >= 1
        entry_qty = buys[0].quantity

        # Target is deferred until fill -- verify it is stored
        assert SYMBOL in s.pending_target
        assert s.pending_target[SYMBOL] == (entry_qty, pytest.approx(126.0, abs=0.01))

        # Simulate fill on next bar -- profit target should be submitted
        fill = FillInfo(SYMBOL, "BUY", entry_qty, 106.0, 0.0, ts_entry)
        position = Position(SYMBOL, entry_qty, 106.0, 0.0)
        pending_stop = PendingOrder(SYMBOL, "SELL", entry_qty, "SL_M", 0.0, 95.0)

        ts_fill = _ts(9, 50)
        snap_fill = _snap(106.5, ts_fill, volume=50000,
                          fills=[fill], positions=[position],
                          pending_orders=[pending_stop])
        fill_sigs = s.on_bar(snap_fill)

        # Now the SELL LIMIT profit target should appear
        sells = [sig for sig in fill_sigs
                 if sig.action == "SELL" and sig.order_type == "LIMIT"]
        assert len(sells) >= 1
        assert sells[0].limit_price == pytest.approx(126.0, abs=0.01)


class TestExitAt1500:
    def test_exit_at_1500(self):
        """All positions closed when hour >= 15."""
        s = _init(volume_confirm=1.0)
        _feed_warmup(s, n=6, range_high=105.0, range_low=95.0)

        # Trigger entry
        ts_entry = _ts(9, 45)
        snap = _snap(106.0, ts_entry, high=107.0, low=104.0, volume=100000)
        entry_sigs = s.on_bar(snap)

        buys = [sig for sig in entry_sigs if sig.action == "BUY"]
        assert len(buys) >= 1
        entry_qty = buys[0].quantity

        # Simulate fill
        fill = FillInfo(SYMBOL, "BUY", entry_qty, 106.0, 0.0, ts_entry)
        position = Position(SYMBOL, entry_qty, 106.0, 0.0)
        pending_stop = PendingOrder(SYMBOL, "SELL", entry_qty, "SL_M", 0.0, 95.0)

        ts_fill = _ts(10, 0)
        snap_fill = _snap(106.5, ts_fill, volume=50000,
                          fills=[fill], positions=[position],
                          pending_orders=[pending_stop])
        s.on_bar(snap_fill)

        # Jump to 15:00 -- should exit
        ts_1500 = _ts(15, 0)
        snap_exit = _snap(108.0, ts_1500, volume=50000,
                          positions=[position],
                          pending_orders=[PendingOrder(SYMBOL, "SELL", entry_qty,
                                                      "SL_M", 0.0, 95.0)])
        sigs_exit = s.on_bar(snap_exit)

        sell_exits = [sig for sig in sigs_exit
                      if sig.action == "SELL" and sig.order_type == "MARKET"]
        assert len(sell_exits) >= 1


class TestMax1TradePerDay:
    def test_no_entry_after_max_trades(self):
        """After max_trades_per_day reached, no more entries."""
        s = _init(volume_confirm=1.0, max_trades_per_day=1)
        _feed_warmup(s, n=6, range_high=105.0, range_low=95.0)

        # First entry
        ts_entry = _ts(9, 45)
        snap = _snap(106.0, ts_entry, high=107.0, low=104.0, volume=100000)
        entry_sigs = s.on_bar(snap)

        buys = [sig for sig in entry_sigs if sig.action == "BUY"]
        assert len(buys) >= 1
        entry_qty = buys[0].quantity

        # Simulate fill
        fill = FillInfo(SYMBOL, "BUY", entry_qty, 106.0, 0.0, ts_entry)
        position = Position(SYMBOL, entry_qty, 106.0, 0.0)
        pending_stop = PendingOrder(SYMBOL, "SELL", entry_qty, "SL_M", 0.0, 95.0)

        ts_fill = _ts(10, 0)
        snap_fill = _snap(106.5, ts_fill, volume=50000,
                          fills=[fill], positions=[position],
                          pending_orders=[pending_stop])
        s.on_bar(snap_fill)

        # Simulate exit (stop hit)
        exit_fill = FillInfo(SYMBOL, "SELL", entry_qty, 95.0, 0.0, ts_fill)
        ts_exit = _ts(10, 5)
        snap_exit = _snap(95.0, ts_exit, volume=50000,
                          fills=[exit_fill])
        s.on_bar(snap_exit)

        # Try another entry -- should be blocked
        ts_try = _ts(11, 0)
        snap_try = _snap(106.0, ts_try, high=107.0, low=104.0, volume=100000)
        sigs_try = s.on_bar(snap_try)

        buys_try = [sig for sig in sigs_try if sig.action == "BUY"]
        assert len(buys_try) == 0
