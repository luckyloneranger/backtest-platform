from strategies.base import BarData, MarketSnapshot, Portfolio, Signal, SessionContext, InstrumentInfo
from strategies.deterministic.sma_crossover import SmaCrossover

import pytest


def make_snapshot(ts: int, close: float, symbol: str = "TEST") -> MarketSnapshot:
    bar = BarData(symbol, close, close, close, close, 1000, 0)
    return MarketSnapshot(
        timestamp_ms=ts,
        timeframes={"day": {symbol: bar}},
        history={},
        portfolio=Portfolio(cash=100_000.0, equity=100_000.0, positions=[]),
        instruments={},
        fills=[],
        rejections=[],
        closed_trades=[],
        context=SessionContext(100_000.0, ts, 100, "2024-01-01", "2024-12-31", ["day"], 200),
    )


def test_no_signal_before_enough_bars():
    s = SmaCrossover()
    s.initialize({"fast_period": 3, "slow_period": 5}, {})
    for i in range(4):
        signals = s.on_bar(make_snapshot(i, 100.0))
        assert signals == [] or all(sig.action == "HOLD" for sig in signals)


def test_buy_signal_on_golden_cross():
    s = SmaCrossover()
    s.initialize({"fast_period": 2, "slow_period": 3}, {})
    prices = [100, 95, 90, 92, 98, 105]
    all_signals = []
    for i, p in enumerate(prices):
        signals = s.on_bar(make_snapshot(i, float(p)))
        all_signals.extend(signals)
    assert any(sig.action == "BUY" for sig in all_signals)


def test_sell_signal_on_death_cross():
    s = SmaCrossover()
    s.initialize({"fast_period": 2, "slow_period": 3}, {})
    prices = [90, 95, 100, 105, 102, 96, 88]
    all_signals = []
    for i, p in enumerate(prices):
        signals = s.on_bar(make_snapshot(i, float(p)))
        all_signals.extend(signals)
    assert any(sig.action == "SELL" for sig in all_signals)


def test_fast_period_ge_slow_period_raises():
    """initialize raises ValueError when fast_period >= slow_period."""
    s = SmaCrossover()
    with pytest.raises(ValueError, match="fast_period"):
        s.initialize({"fast_period": 30, "slow_period": 10}, {})


def test_fast_period_equal_slow_period_raises():
    """initialize raises ValueError when fast_period == slow_period."""
    s = SmaCrossover()
    with pytest.raises(ValueError, match="fast_period"):
        s.initialize({"fast_period": 10, "slow_period": 10}, {})
