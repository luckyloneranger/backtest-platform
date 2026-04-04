"""Tests for Donchian Breakout strategy."""

from strategies.base import (
    BarData, MarketSnapshot, Portfolio, Position, SessionContext, InstrumentInfo,
)
from strategies.examples.donchian_breakout import DonchianBreakout, compute_atr


def make_snapshot(
    ts: int,
    close_15m: float | None = None,
    close_day: float | None = None,
    high_day: float | None = None,
    low_day: float | None = None,
    volume_day: int = 100_000,
    symbol: str = "TEST",
    cash: float = 100_000.0,
    positions: list[Position] | None = None,
) -> MarketSnapshot:
    timeframes = {}
    if close_15m is not None:
        bar = BarData(symbol, close_15m, close_15m + 1, close_15m - 1, close_15m, 1000, 0)
        timeframes["15minute"] = {symbol: bar}
    if close_day is not None:
        h = high_day if high_day is not None else close_day + 2
        l = low_day if low_day is not None else close_day - 2
        bar = BarData(symbol, close_day, h, l, close_day, volume_day, 0)
        timeframes["day"] = {symbol: bar}
    pos_list = positions if positions is not None else []
    equity = cash + sum(p.quantity * p.avg_price for p in pos_list)
    return MarketSnapshot(
        timestamp_ms=ts,
        timeframes=timeframes,
        history={},
        portfolio=Portfolio(cash=cash, equity=equity, positions=pos_list),
        instruments={},
        fills=[],
        rejections=[],
        closed_trades=[],
        context=SessionContext(100_000.0, ts, 1000, "2024-01-01", "2024-12-31", ["15minute", "day"], 200),
    )


def test_compute_atr():
    highs = [102, 104, 103, 105, 106, 104, 108, 107, 109, 110, 111, 112, 113, 114, 115, 116]
    lows = [98, 99, 97, 100, 101, 99, 103, 102, 104, 105, 106, 107, 108, 109, 110, 111]
    closes = [100, 101, 100, 103, 104, 102, 106, 105, 107, 108, 109, 110, 111, 112, 113, 114]
    atr = compute_atr(highs, lows, closes, 14)
    assert atr is not None
    assert atr > 0


def test_compute_atr_not_enough_data():
    atr = compute_atr([100, 101], [98, 99], [99, 100], 14)
    assert atr is None


def test_no_signal_during_warmup():
    s = DonchianBreakout()
    s.initialize({"channel_period": 5, "atr_period": 3}, {})
    # Feed only 3 daily bars — not enough for channel
    for i in range(3):
        s.on_bar(make_snapshot(i, close_day=100.0 + i))
    signals = s.on_bar(make_snapshot(10, close_15m=100.0))
    assert signals == []


def test_buy_on_breakout_with_volume():
    s = DonchianBreakout()
    s.initialize({"channel_period": 5, "atr_period": 3, "volume_factor": 1.0}, {})

    # Feed enough daily bars to build channel (range 95-105)
    for i in range(8):
        price = 95.0 + i
        s.on_bar(make_snapshot(i, close_day=price, high_day=price + 2, low_day=price - 2, volume_day=100_000))

    # Breakout day: send daily bar with high volume AND 15-min bar above channel
    # Channel high = max(highs of last 5 days) = max(99,100,101,102,103,104) = 104
    s.on_bar(make_snapshot(8, close_day=108.0, high_day=110.0, low_day=106.0, volume_day=200_000))
    all_signals = []
    for i in range(5):
        signals = s.on_bar(make_snapshot(100 + i, close_15m=108.0))
        all_signals.extend(signals)

    assert any(sig.action == "BUY" for sig in all_signals)


def test_no_buy_without_volume():
    s = DonchianBreakout()
    s.initialize({"channel_period": 5, "atr_period": 3, "volume_factor": 2.0}, {})

    # Feed daily bars with LOW volume
    for i in range(8):
        price = 95.0 + i
        s.on_bar(make_snapshot(i, close_day=price, high_day=price + 2, low_day=price - 2, volume_day=1000))

    signals = s.on_bar(make_snapshot(100, close_15m=108.0))
    assert not any(sig.action == "BUY" for sig in signals)


def test_trailing_stop_exit():
    s = DonchianBreakout()
    s.initialize({"channel_period": 5, "atr_period": 3, "atr_multiplier": 1.0, "volume_factor": 1.0}, {})

    # Build channel
    for i in range(8):
        price = 95.0 + i
        s.on_bar(make_snapshot(i, close_day=price, high_day=price + 2, low_day=price - 2, volume_day=100_000))

    # Breakout day with volume, then 15-min entry
    s.on_bar(make_snapshot(8, close_day=108.0, high_day=110.0, low_day=106.0, volume_day=200_000))
    s.on_bar(make_snapshot(100, close_15m=108.0))

    # Price rises — trailing stop moves up
    s.on_bar(make_snapshot(101, close_15m=112.0))
    s.on_bar(make_snapshot(102, close_15m=115.0))

    # Price drops through trailing stop — should exit
    held = Position(symbol="TEST", quantity=100, avg_price=108.0, unrealized_pnl=0.0)
    signals = s.on_bar(make_snapshot(103, close_15m=105.0, positions=[held]))
    assert any(sig.action == "SELL" for sig in signals)


def test_required_data():
    s = DonchianBreakout()
    reqs = s.required_data()
    intervals = [r["interval"] for r in reqs]
    assert "day" in intervals
    assert "15minute" in intervals
