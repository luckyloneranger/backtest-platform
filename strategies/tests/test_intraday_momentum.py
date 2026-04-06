"""Tests for Intraday Momentum strategy.

Covers: required_data, warmup gating, long entry on upward burst,
short entry on downward burst, no entry without volume, no entry on small move,
trailing stop updates, time exit at 15:00.
"""

import pytest
from datetime import datetime, timezone, timedelta

from strategies.base import (
    BarData, FillInfo, MarketSnapshot, PendingOrder, Portfolio,
    Signal, SessionContext, Position,
)
from strategies.deterministic.intraday_momentum import IntradayMomentum, _ist_time


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
                               ["5minute"], 100),
        pending_orders=pending_orders or [],
    )


def _init(**overrides) -> IntradayMomentum:
    """Create and initialize an IntradayMomentum with defaults."""
    s = IntradayMomentum()
    config = {
        "momentum_atr_mult": 1.5,
        "volume_mult": 2.0,
        "atr_period": 14,
        "atr_stop_mult": 1.0,
        "risk_pct": 0.03,
        "warmup_bars": 6,
        "exit_time_hour": 15,
        "max_trades_per_day": 2,
        "lookback_bars": 3,
        "vol_avg_period": 20,
    }
    config.update(overrides)
    s.initialize(config, {})
    return s


def _feed_bars(s: IntradayMomentum, prices: list[float],
               start_minute: int = 15, volume: int = 50000,
               high_offset: float = 2.0, low_offset: float = 2.0):
    """Feed a sequence of bars starting from 9:{start_minute}.

    Returns all signals generated.
    """
    all_signals = []
    for i, price in enumerate(prices):
        minute = start_minute + i * 5
        hour = 9 + minute // 60
        minute = minute % 60
        ts = _ts(hour, minute)
        snap = _snap(price, ts, high=price + high_offset,
                     low=price - low_offset, volume=volume)
        sigs = s.on_bar(snap)
        all_signals.extend(sigs)
    return all_signals


def _build_momentum_scenario(s: IntradayMomentum, direction: str = "up",
                             burst_volume: int = 200000):
    """Feed enough bars to have ATR established, then inject a momentum burst.

    1. Feed 6 warmup bars with gentle oscillation (9:15-9:40).
    2. Feed ~15 more stable bars (9:45-10:55) to establish ATR.
    3. Return the bar index + timestamp for the burst bar.

    ATR will be approximately 4.0 (high-low range of 4, gentle movement).
    Momentum threshold = 1.5 * 4.0 = 6.0, so a 3-bar move > 6 triggers.
    """
    # Phase 1: warmup bars (gentle oscillation, range ~4 each bar)
    base = 100.0
    prices = []
    for i in range(6):
        p = base + ((-1) ** i) * 0.5
        prices.append(p)

    # Phase 2: stable bars to build ATR (each bar has H-L range ~4)
    for i in range(15):
        p = base + ((-1) ** i) * 0.3
        prices.append(p)

    _feed_bars(s, prices, volume=50000, high_offset=2.0, low_offset=2.0)

    # Phase 3: momentum burst (3 bars with strong directional move)
    # ATR ~ 4.0, threshold = 1.5 * 4.0 = 6.0, need 3-bar move > 6
    bar_start = len(prices)
    if direction == "up":
        burst_prices = [base + 1.0, base + 4.0, base + 9.0]
    else:
        burst_prices = [base - 1.0, base - 4.0, base - 9.0]

    burst_signals = []
    for i, p in enumerate(burst_prices):
        bar_idx = bar_start + i
        minute = 15 + bar_idx * 5
        hour = 9 + minute // 60
        minute = minute % 60
        ts = _ts(hour, minute)
        vol = burst_volume if i == len(burst_prices) - 1 else 50000
        snap = _snap(p, ts, high=p + 2.0, low=p - 2.0, volume=vol)
        sigs = s.on_bar(snap)
        burst_signals.extend(sigs)

    return burst_signals


class TestRequiredData:
    def test_required_data(self):
        s = IntradayMomentum()
        reqs = s.required_data()
        assert reqs == [{"interval": "5minute", "lookback": 100}]


class TestNoTradeDuringWarmup:
    def test_first_6_bars_no_trades(self):
        """No entry signals during the warmup period (first 6 bars)."""
        s = _init()
        prices = [100.0 + i * 3 for i in range(6)]  # even with big moves
        all_signals = _feed_bars(s, prices, volume=200000)

        buys = [sig for sig in all_signals if sig.action == "BUY"]
        sells = [sig for sig in all_signals if sig.action == "SELL"]
        assert len(buys) == 0
        assert len(sells) == 0


class TestLongOnUpwardBurst:
    def test_long_on_upward_burst(self):
        """Big upward move + high volume -> BUY MARKET MIS."""
        s = _init(volume_mult=1.0)  # relax volume filter for easier trigger
        signals = _build_momentum_scenario(s, direction="up", burst_volume=200000)

        buys = [sig for sig in signals if sig.action == "BUY"]
        assert len(buys) >= 1
        assert buys[0].symbol == SYMBOL
        assert buys[0].quantity > 0
        assert buys[0].order_type == "MARKET"
        assert buys[0].product_type == "MIS"


class TestShortOnDownwardBurst:
    def test_short_on_downward_burst(self):
        """Big downward move + high volume -> SELL MARKET MIS."""
        s = _init(volume_mult=1.0)
        signals = _build_momentum_scenario(s, direction="down", burst_volume=200000)

        sells = [sig for sig in signals if sig.action == "SELL"]
        assert len(sells) >= 1
        assert sells[0].symbol == SYMBOL
        assert sells[0].product_type == "MIS"
        assert sells[0].order_type == "MARKET"


class TestNoEntryWithoutVolume:
    def test_no_entry_low_volume(self):
        """Move OK but volume below threshold -> skip."""
        s = _init(volume_mult=5.0)  # require 5x avg volume -- very strict
        # burst_volume only slightly above avg (~50k), so 50k < 5.0 * 50k
        signals = _build_momentum_scenario(s, direction="up", burst_volume=60000)

        buys = [sig for sig in signals if sig.action == "BUY"]
        assert len(buys) == 0


class TestNoEntrySmallMove:
    def test_no_entry_small_move(self):
        """Volume OK but move < threshold -> skip."""
        s = _init(momentum_atr_mult=5.0)  # very high threshold
        # Build bars with small 3-bar moves that never exceed 5.0 * ATR
        base = 100.0
        prices = []
        for i in range(6):
            prices.append(base + ((-1) ** i) * 0.5)
        for i in range(15):
            prices.append(base + ((-1) ** i) * 0.3)
        _feed_bars(s, prices, volume=50000, high_offset=2.0, low_offset=2.0)

        # Small move even with high volume
        bar_idx = len(prices)
        for i in range(3):
            minute = 15 + (bar_idx + i) * 5
            hour = 9 + minute // 60
            minute = minute % 60
            ts = _ts(hour, minute)
            p = base + i * 1.0  # total 3-bar move = 2.0, ATR ~ 4, threshold = 5*4=20
            snap = _snap(p, ts, high=p + 2.0, low=p - 2.0, volume=200000)
            sigs = s.on_bar(snap)
            buys = [sig for sig in sigs if sig.action == "BUY"]
            assert len(buys) == 0


class TestTrailingStopUpdates:
    def test_trailing_stop_ratchets_up_for_long(self):
        """After long entry + fill, trailing stop moves up as price rises."""
        s = _init(volume_mult=1.0)
        _build_momentum_scenario(s, direction="up", burst_volume=200000)

        # Verify we triggered a long entry
        assert not s.pm.is_flat(SYMBOL) or s.pm.has_pending_entry(SYMBOL)

        # If pending entry, simulate fill
        if s.pm.has_pending_entry(SYMBOL):
            state = s.pm.get_state(SYMBOL)
            qty = state.pending_qty
            ts_fill = _ts(11, 0)
            fill = FillInfo(SYMBOL, "BUY", qty, 109.0, 0.0, ts_fill)
            position = Position(SYMBOL, qty, 109.0, 0.0)
            snap_fill = _snap(110.0, ts_fill, high=111.0, low=109.0,
                              volume=50000, fills=[fill], positions=[position])
            sigs_fill = s.on_bar(snap_fill)

        # Now in long position -- feed rising bars, stop should ratchet up
        old_stop = s.pm.get_state(SYMBOL).trailing_stop

        ts_next = _ts(11, 5)
        position = Position(SYMBOL, s.pm.position_qty(SYMBOL), 109.0, 0.0)
        pending_stop = PendingOrder(SYMBOL, "SELL", s.pm.position_qty(SYMBOL),
                                    "SL_M", 0.0, old_stop)
        snap_up = _snap(115.0, ts_next, high=116.0, low=114.0,
                        volume=50000, positions=[position],
                        pending_orders=[pending_stop])
        sigs_up = s.on_bar(snap_up)

        new_stop = s.pm.get_state(SYMBOL).trailing_stop
        assert new_stop > old_stop, f"Stop should ratchet up: {new_stop} > {old_stop}"


class TestExitAt1500:
    def test_exit_at_1500(self):
        """All positions closed when hour >= 15."""
        s = _init(volume_mult=1.0)
        _build_momentum_scenario(s, direction="up", burst_volume=200000)

        # Simulate fill if pending
        if s.pm.has_pending_entry(SYMBOL):
            state = s.pm.get_state(SYMBOL)
            qty = state.pending_qty
            ts_fill = _ts(11, 0)
            fill = FillInfo(SYMBOL, "BUY", qty, 109.0, 0.0, ts_fill)
            position = Position(SYMBOL, qty, 109.0, 0.0)
            pending_stop = PendingOrder(SYMBOL, "SELL", qty, "SL_M", 0.0, 105.0)
            snap_fill = _snap(110.0, ts_fill, high=111.0, low=109.0,
                              volume=50000, fills=[fill], positions=[position],
                              pending_orders=[pending_stop])
            s.on_bar(snap_fill)

        qty = s.pm.position_qty(SYMBOL)
        assert qty > 0, "Should be in a position before exit test"

        # Jump to 15:00 -- should exit
        position = Position(SYMBOL, qty, 109.0, 0.0)
        pending_stop = PendingOrder(SYMBOL, "SELL", qty, "SL_M", 0.0, 105.0)
        ts_1500 = _ts(15, 0)
        snap_1500 = _snap(112.0, ts_1500, high=113.0, low=111.0,
                          volume=50000, positions=[position],
                          pending_orders=[pending_stop])
        sigs_exit = s.on_bar(snap_1500)

        sell_exits = [sig for sig in sigs_exit
                      if sig.action == "SELL" and sig.order_type == "MARKET"]
        assert len(sell_exits) >= 1
