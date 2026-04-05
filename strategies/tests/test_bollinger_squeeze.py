"""Tests for Bollinger Squeeze Breakout strategy.

Covers: required_data, squeeze detection, long/short entry on expansion,
no entry without squeeze, no entry without volume, warmup gating,
time exit at 15:00, max trades per day, and MIS-only product type.
"""

import pytest
from datetime import datetime, timezone, timedelta

from strategies.base import (
    BarData, FillInfo, MarketSnapshot, PendingOrder, Portfolio,
    Signal, SessionContext, Position,
)
from strategies.deterministic.bollinger_squeeze import BollingerSqueeze, _ist_time


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


def _init(**overrides) -> BollingerSqueeze:
    """Create and initialize a BollingerSqueeze with defaults."""
    s = BollingerSqueeze()
    config = {
        "bb_period": 20,
        "bb_std": 2.0,
        "squeeze_threshold": 0.5,
        "volume_confirm": 1.5,
        "risk_per_trade": 0.015,
        "atr_period": 14,
        "profit_target_atr": 1.5,
        "atr_stop_mult": 1.0,
        "warmup_bars": 6,
        "exit_time_hour": 15,
        "max_trades_per_day": 3,
    }
    config.update(overrides)
    s.initialize(config, {})
    return s


def _feed_warmup(s: BollingerSqueeze, n: int = 6, base_price: float = 100.0,
                 start_minute: int = 15):
    """Feed n warmup bars (9:15..9:40 by default) with stable prices."""
    all_signals = []
    for i in range(n):
        minute = start_minute + i * 5
        ts = _ts(9, minute)
        snap = _snap(base_price, ts, volume=50000)
        sigs = s.on_bar(snap)
        all_signals.extend(sigs)
    return all_signals


def _build_squeeze_scenario(s: BollingerSqueeze, base: float = 100.0,
                            narrow_range: float = 0.05):
    """Feed enough bars to create a Bollinger squeeze.

    Strategy:
        1. Feed 6 warmup bars with wide variation (± 5) to establish BB width.
        2. Feed 20 more bars with very tight range so BBW compresses.
        3. Feed 20 more bars of tight range so avg_bbw is computed from
           the compressed BBW values — making current BBW < 0.5 * avg_bbw.

    Actually, since squeeze_threshold = 0.5, we need BBW < 0.5 * avg_bbw.
    The simplest approach: first build 20 wide-range bars (large BBW), then
    20 tight-range bars. The avg_bbw will be a mix of wide+tight. The current
    (tight) BBW will be < 0.5 * avg.
    """
    # Phase 1: 6 warmup + 14 established-width bars = 20 bars with variation
    prices_wide = []
    for i in range(20):
        # Oscillate: 95, 105, 95, 105, ...
        p = base - 5.0 if i % 2 == 0 else base + 5.0
        prices_wide.append(p)

    for i, p in enumerate(prices_wide):
        minute = 15 + i * 5
        hour = 9 + minute // 60
        minute = minute % 60
        ts = _ts(hour, minute)
        snap = _snap(p, ts, high=p + 3, low=p - 3, volume=50000)
        s.on_bar(snap)

    # Phase 2: 20 bars with very tight range — BB compresses
    tight_prices = []
    for i in range(20):
        p = base + narrow_range * ((-1) ** i)
        tight_prices.append(p)

    for i, p in enumerate(tight_prices):
        bar_idx = 20 + i
        minute = 15 + bar_idx * 5
        hour = 9 + minute // 60
        minute = minute % 60
        ts = _ts(hour, minute)
        snap = _snap(p, ts, high=p + 0.1, low=p - 0.1, volume=50000)
        s.on_bar(snap)


class TestRequiredData:
    def test_required_data(self):
        s = BollingerSqueeze()
        reqs = s.required_data()
        assert reqs == [{"interval": "5minute", "lookback": 100}]


class TestSqueezeDetected:
    def test_squeeze_active_on_compression(self):
        """BBW < 0.5 * avg_bbw sets squeeze_active = True."""
        s = _init(warmup_bars=1)  # minimal warmup to speed things up
        _build_squeeze_scenario(s)
        assert s.squeeze_active.get(SYMBOL, False) is True


class TestLongEntryOnExpansion:
    def test_long_entry_on_squeeze_breakout_up(self):
        """Squeeze active + BBW expanding + close > upper + volume -> BUY."""
        s = _init(warmup_bars=1, volume_confirm=1.0)  # relax volume filter
        _build_squeeze_scenario(s)

        assert s.squeeze_active.get(SYMBOL, False) is True

        # Breakout bar: big up move with high volume
        bar_idx = 40
        minute = 15 + bar_idx * 5
        hour = 9 + minute // 60
        minute = minute % 60
        ts = _ts(hour, minute)
        # Price jumps well above upper band
        snap = _snap(115.0, ts, high=116.0, low=109.0, volume=100000)
        signals = s.on_bar(snap)

        buys = [sig for sig in signals if sig.action == "BUY"]
        assert len(buys) >= 1
        assert buys[0].symbol == SYMBOL
        assert buys[0].quantity > 0


class TestShortEntryOnExpansion:
    def test_short_entry_on_squeeze_breakout_down(self):
        """Squeeze active + BBW expanding + close < lower + volume -> SELL MIS."""
        s = _init(warmup_bars=1, volume_confirm=1.0)
        _build_squeeze_scenario(s)

        assert s.squeeze_active.get(SYMBOL, False) is True

        # Breakout bar: big down move
        bar_idx = 40
        minute = 15 + bar_idx * 5
        hour = 9 + minute // 60
        minute = minute % 60
        ts = _ts(hour, minute)
        snap = _snap(85.0, ts, high=91.0, low=84.0, volume=100000)
        signals = s.on_bar(snap)

        sells = [sig for sig in signals if sig.action == "SELL"]
        assert len(sells) >= 1
        assert sells[0].symbol == SYMBOL
        assert sells[0].product_type == "MIS"


class TestNoEntryWithoutSqueeze:
    def test_no_entry_without_squeeze(self):
        """BBW normal -> no entry even if close > upper."""
        s = _init(warmup_bars=1)

        # Feed 25 bars with wide range — no squeeze ever triggers
        for i in range(25):
            p = 100.0 + 5.0 * ((-1) ** i)
            minute = 15 + i * 5
            hour = 9 + minute // 60
            minute = minute % 60
            ts = _ts(hour, minute)
            snap = _snap(p, ts, high=p + 3, low=p - 3, volume=100000)
            s.on_bar(snap)

        assert s.squeeze_active.get(SYMBOL, False) is False

        # Try a bar with close above what would be upper band — no squeeze
        bar_idx = 25
        minute = 15 + bar_idx * 5
        hour = 9 + minute // 60
        minute = minute % 60
        ts = _ts(hour, minute)
        snap = _snap(120.0, ts, high=121.0, low=119.0, volume=100000)
        signals = s.on_bar(snap)

        buys = [sig for sig in signals if sig.action == "BUY"]
        assert len(buys) == 0


class TestNoEntryWithoutVolume:
    def test_no_entry_low_volume(self):
        """Low volume bar -> no entry even with squeeze + breakout."""
        s = _init(warmup_bars=1, volume_confirm=2.0)  # require 2x avg volume
        _build_squeeze_scenario(s)

        assert s.squeeze_active.get(SYMBOL, False) is True

        # Breakout bar but with low volume (well below 2x average)
        bar_idx = 40
        minute = 15 + bar_idx * 5
        hour = 9 + minute // 60
        minute = minute % 60
        ts = _ts(hour, minute)
        # Volume = 1000, avg is ~50000 so 1000 << 2.0 * 50000
        snap = _snap(115.0, ts, high=116.0, low=109.0, volume=1000)
        signals = s.on_bar(snap)

        buys = [sig for sig in signals if sig.action == "BUY"]
        assert len(buys) == 0


class TestNoEntryDuringWarmup:
    def test_first_6_bars_no_trades(self):
        """No entry signals during the warmup period (first 6 bars)."""
        s = _init(warmup_bars=6)
        signals = _feed_warmup(s, n=6, base_price=100.0)

        buys = [sig for sig in signals if sig.action == "BUY"]
        sells = [sig for sig in signals if sig.action == "SELL"]
        assert len(buys) == 0
        assert len(sells) == 0


class TestTimeExitAt1500:
    def test_exit_at_1500(self):
        """All positions closed when hour >= 15."""
        s = _init(warmup_bars=1, volume_confirm=1.0)
        _build_squeeze_scenario(s)

        # Trigger entry
        bar_idx = 40
        minute = 15 + bar_idx * 5
        hour = 9 + minute // 60
        minute = minute % 60
        ts = _ts(hour, minute)
        snap = _snap(115.0, ts, high=116.0, low=109.0, volume=100000)
        entry_sigs = s.on_bar(snap)

        buys = [sig for sig in entry_sigs if sig.action == "BUY"]
        assert len(buys) >= 1
        entry_qty = buys[0].quantity

        # Simulate fill
        fill = FillInfo(SYMBOL, "BUY", entry_qty, 115.0, 0.0, ts)
        position = Position(SYMBOL, entry_qty, 115.0, 0.0)
        pending_stop = PendingOrder(SYMBOL, "SELL", entry_qty, "SL_M", 0.0, 100.0)

        ts_fill = _ts(12, 0)
        snap_fill = _snap(115.5, ts_fill, high=116, low=115, volume=50000,
                          fills=[fill], positions=[position],
                          pending_orders=[pending_stop])
        s.on_bar(snap_fill)

        # Jump to 15:00 — should exit
        ts_1500 = _ts(15, 0)
        snap_1500 = _snap(116.0, ts_1500, high=117, low=115, volume=50000,
                          positions=[position],
                          pending_orders=[PendingOrder(SYMBOL, "SELL", entry_qty,
                                                      "SL_M", 0.0, 100.0)])
        sigs_1500 = s.on_bar(snap_1500)

        sell_exits = [sig for sig in sigs_1500
                      if sig.action == "SELL" and sig.order_type == "MARKET"]
        assert len(sell_exits) >= 1


class TestMaxTradesPerDay:
    def test_no_entry_after_max_trades(self):
        """After max_trades_per_day reached, no more entries."""
        s = _init(warmup_bars=1, volume_confirm=1.0, max_trades_per_day=1)
        _build_squeeze_scenario(s)

        # First entry
        bar_idx = 40
        minute = 15 + bar_idx * 5
        hour = 9 + minute // 60
        minute = minute % 60
        ts = _ts(hour, minute)
        snap = _snap(115.0, ts, high=116.0, low=109.0, volume=100000)
        entry_sigs = s.on_bar(snap)
        buys = [sig for sig in entry_sigs if sig.action == "BUY"]
        assert len(buys) >= 1
        entry_qty = buys[0].quantity

        # Simulate fill + position exit (so we are flat again)
        fill = FillInfo(SYMBOL, "BUY", entry_qty, 115.0, 0.0, ts)
        position = Position(SYMBOL, entry_qty, 115.0, 0.0)
        pending_stop = PendingOrder(SYMBOL, "SELL", entry_qty, "SL_M", 0.0, 100.0)

        bar_idx += 1
        minute2 = 15 + bar_idx * 5
        hour2 = 9 + minute2 // 60
        minute2 = minute2 % 60
        ts_fill = _ts(hour2, minute2)
        snap_fill = _snap(115.5, ts_fill, high=116, low=115, volume=50000,
                          fills=[fill], positions=[position],
                          pending_orders=[pending_stop])
        s.on_bar(snap_fill)

        # Exit position
        exit_fill = FillInfo(SYMBOL, "SELL", entry_qty, 116.0, 0.0, ts_fill)
        bar_idx += 1
        minute3 = 15 + bar_idx * 5
        hour3 = 9 + minute3 // 60
        minute3 = minute3 % 60
        ts_exit = _ts(hour3, minute3)
        snap_exit = _snap(116.0, ts_exit, volume=50000,
                          fills=[exit_fill])
        s.on_bar(snap_exit)

        # Re-establish squeeze for second entry attempt
        s.squeeze_active[SYMBOL] = True

        # Try another entry — should be blocked
        bar_idx += 1
        minute4 = 15 + bar_idx * 5
        hour4 = 9 + minute4 // 60
        minute4 = minute4 % 60
        ts_try = _ts(hour4, minute4)
        snap_try = _snap(120.0, ts_try, high=121, low=109, volume=100000)
        sigs_try = s.on_bar(snap_try)

        buys_try = [sig for sig in sigs_try if sig.action == "BUY"]
        assert len(buys_try) == 0


class TestAllEntriesMis:
    def test_long_entry_is_mis(self):
        """Long entries use MIS product type."""
        s = _init(warmup_bars=1, volume_confirm=1.0)
        _build_squeeze_scenario(s)

        bar_idx = 40
        minute = 15 + bar_idx * 5
        hour = 9 + minute // 60
        minute = minute % 60
        ts = _ts(hour, minute)
        snap = _snap(115.0, ts, high=116.0, low=109.0, volume=100000)
        signals = s.on_bar(snap)

        buys = [sig for sig in signals if sig.action == "BUY"]
        assert len(buys) >= 1
        assert buys[0].product_type == "MIS"

    def test_short_entry_is_mis(self):
        """Short entries use MIS product type."""
        s = _init(warmup_bars=1, volume_confirm=1.0)
        _build_squeeze_scenario(s)

        bar_idx = 40
        minute = 15 + bar_idx * 5
        hour = 9 + minute // 60
        minute = minute % 60
        ts = _ts(hour, minute)
        snap = _snap(85.0, ts, high=91.0, low=84.0, volume=100000)
        signals = s.on_bar(snap)

        sells = [sig for sig in signals if sig.action == "SELL"]
        assert len(sells) >= 1
        assert sells[0].product_type == "MIS"
