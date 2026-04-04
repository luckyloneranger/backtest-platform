from strategies.base import Bar, Portfolio, Signal
from strategies.examples.sma_crossover import SmaCrossover


def make_bar(ts: int, close: float, symbol: str = "TEST") -> Bar:
    return Bar(ts, symbol, close, close, close, close, 1000, 0)


def empty_portfolio() -> Portfolio:
    return Portfolio(cash=100_000.0, equity=100_000.0, positions=[])


def test_no_signal_before_enough_bars():
    s = SmaCrossover()
    s.initialize({"fast_period": 3, "slow_period": 5})
    for i in range(4):
        signals = s.on_bar(make_bar(i, 100.0), empty_portfolio())
        assert signals == [] or all(sig.action == "HOLD" for sig in signals)


def test_buy_signal_on_golden_cross():
    s = SmaCrossover()
    s.initialize({"fast_period": 2, "slow_period": 3})
    # Create a price sequence where fast SMA crosses above slow SMA
    prices = [100, 95, 90, 92, 98, 105]
    all_signals = []
    for i, p in enumerate(prices):
        all_signals.extend(s.on_bar(make_bar(i, float(p)), empty_portfolio()))
    assert any(sig.action == "BUY" for sig in all_signals)


def test_sell_signal_on_death_cross():
    s = SmaCrossover()
    s.initialize({"fast_period": 2, "slow_period": 3})
    prices = [90, 95, 100, 105, 102, 96, 88]
    all_signals = []
    for i, p in enumerate(prices):
        all_signals.extend(s.on_bar(make_bar(i, float(p)), empty_portfolio()))
    assert any(sig.action == "SELL" for sig in all_signals)
