"""Tests for OU Mean Reversion strategy."""

import numpy as np

from strategies.base import (
    BarData, MarketSnapshot, Portfolio, Position, SessionContext, InstrumentInfo,
    FillInfo, PendingOrder,
)
from strategies.deterministic.ou_mean_reversion import OUMeanReversion


# --- Helpers ---

DEFAULT_INSTRUMENTS = {
    "TEST": InstrumentInfo(
        symbol="TEST", exchange="NSE", instrument_type="EQ", lot_size=1,
        tick_size=0.05, expiry="", strike=0.0, option_type="",
        circuit_limit_upper=0.0, circuit_limit_lower=0.0,
    ),
}

DEFAULT_CONTEXT = SessionContext(
    initial_capital=100_000.0, bar_number=0, total_bars=300,
    start_date="2024-01-01", end_date="2024-12-31",
    intervals=["day"], lookback_window=200,
)


def make_snapshot(
    ts=0,
    close_day=None, high_day=None, low_day=None, volume_day=100_000,
    symbol="TEST", cash=100_000.0, positions=None, fills=None, pending_orders=None,
):
    timeframes = {}
    if close_day is not None:
        h = high_day if high_day is not None else close_day + 2
        l = low_day if low_day is not None else close_day - 2
        timeframes["day"] = {
            symbol: BarData(symbol, close_day, h, l, close_day, volume_day, 0)
        }
    pos_list = positions or []
    equity = cash + sum(p.quantity * p.avg_price for p in pos_list)
    return MarketSnapshot(
        timestamp_ms=ts,
        timeframes=timeframes,
        history={},
        portfolio=Portfolio(cash=cash, equity=equity, positions=pos_list),
        instruments=DEFAULT_INSTRUMENTS,
        fills=fills or [],
        rejections=[],
        closed_trades=[],
        context=DEFAULT_CONTEXT,
        pending_orders=pending_orders or [],
    )


def create_strategy(config=None):
    s = OUMeanReversion()
    s.initialize(config or {}, DEFAULT_INSTRUMENTS)
    return s


def _generate_ou_prices(mu=100.0, theta=0.1, sigma=1.0, n=200, seed=42):
    """Simulate an OU process: P(t+1) = P(t) + theta*(mu - P(t)) + noise."""
    rng = np.random.RandomState(seed)
    prices = [mu]
    for _ in range(n):
        p = prices[-1] + theta * (mu - prices[-1]) + rng.normal(0, sigma)
        prices.append(p)
    return prices


def _generate_trending_prices(n=200, base=100.0, slope=0.5):
    """Generate a straight trending series (no mean-reversion)."""
    return [base + i * slope for i in range(n)]


def _feed_prices(strategy, prices, symbol="TEST", cash=100_000.0):
    """Feed a list of prices as daily bars into the strategy. Returns all signals."""
    all_signals = []
    for i, p in enumerate(prices):
        snap = make_snapshot(ts=i, close_day=p, symbol=symbol, cash=cash)
        signals = strategy.on_bar(snap)
        all_signals.extend(signals)
    return all_signals


# --- Tests ---


def test_required_data():
    s = create_strategy()
    reqs = s.required_data()
    assert len(reqs) == 1
    assert reqs[0]["interval"] == "day"
    assert reqs[0]["lookback"] == 200


def test_no_trade_before_min_history():
    """First 59 bars should produce no entry signals."""
    s = create_strategy({"min_history": 60})
    prices = _generate_ou_prices(n=59)  # 60 values total (index 0..59)
    signals = _feed_prices(s, prices[:59])  # feed only 59 bars
    buys = [sig for sig in signals if sig.action == "BUY"]
    sells = [sig for sig in signals if sig.action == "SELL" and sig.order_type == "MARKET"]
    assert len(buys) == 0
    assert len(sells) == 0


def test_mean_reverting_stock_is_tradeable():
    """An OU-generated series should be detected as mean-reverting."""
    s = create_strategy({"min_history": 60, "max_halflife": 30, "min_pvalue": 0.10})
    prices = _generate_ou_prices(mu=100.0, theta=0.1, sigma=1.0, n=100)
    _feed_prices(s, prices)
    assert s.is_tradeable["TEST"]
    assert s.ou_theta["TEST"] > 0
    assert s.ou_halflife["TEST"] < 30


def test_trending_stock_not_tradeable():
    """A pure trending series should NOT be detected as mean-reverting."""
    s = create_strategy({"min_history": 60})
    prices = _generate_trending_prices(n=100)
    _feed_prices(s, prices)
    assert not s.is_tradeable["TEST"]


def test_long_entry_below_ou_mean():
    """Price deviating below OU mean by > zscore_entry should trigger BUY."""
    s = create_strategy({
        "min_history": 60, "zscore_entry": 2.0, "min_pvalue": 0.10,
    })
    # Build history with mean-reverting prices around mu=100
    prices = _generate_ou_prices(mu=100.0, theta=0.1, sigma=1.0, n=80, seed=42)
    _feed_prices(s, prices)

    assert s.is_tradeable["TEST"]
    mu = s.ou_mu["TEST"]
    sigma = s.ou_sigma["TEST"]

    # Now feed a price far below mu (z < -2)
    extreme_low = mu - 3.0 * sigma
    snap = make_snapshot(ts=100, close_day=extreme_low)
    signals = s.on_bar(snap)
    buys = [sig for sig in signals if sig.action == "BUY"]
    assert len(buys) == 1
    assert buys[0].product_type == "CNC"
    assert buys[0].quantity > 0


def test_short_entry_above_ou_mean():
    """Price deviating above OU mean by > zscore_entry should trigger SELL (short)."""
    s = create_strategy({
        "min_history": 60, "zscore_entry": 2.0, "min_pvalue": 0.10,
    })
    prices = _generate_ou_prices(mu=100.0, theta=0.1, sigma=1.0, n=80, seed=42)
    _feed_prices(s, prices)

    assert s.is_tradeable["TEST"]
    mu = s.ou_mu["TEST"]
    sigma = s.ou_sigma["TEST"]

    # Now feed a price far above mu (z > 2)
    extreme_high = mu + 3.0 * sigma
    snap = make_snapshot(ts=100, close_day=extreme_high)
    signals = s.on_bar(snap)
    sells = [sig for sig in signals if sig.action == "SELL" and sig.order_type == "MARKET"]
    assert len(sells) == 1
    assert sells[0].product_type == "MIS"  # shorts always MIS


def test_exit_at_ou_mean():
    """When price reverts to OU mean (z >= 0), a long position should exit."""
    s = create_strategy({
        "min_history": 60, "zscore_entry": 2.0, "zscore_exit": 0.0,
        "min_pvalue": 0.10,
    })
    prices = _generate_ou_prices(mu=100.0, theta=0.1, sigma=1.0, n=80, seed=42)
    _feed_prices(s, prices)

    mu = s.ou_mu["TEST"]
    sigma = s.ou_sigma["TEST"]

    # Trigger long entry at extreme low
    extreme_low = mu - 3.0 * sigma
    entry_snap = make_snapshot(ts=100, close_day=extreme_low)
    entry_signals = s.on_bar(entry_snap)
    buys = [sig for sig in entry_signals if sig.action == "BUY"]
    assert len(buys) == 1
    fill_qty = buys[0].quantity

    # Simulate fill
    fill_snap = make_snapshot(
        ts=101, close_day=extreme_low,
        positions=[Position("TEST", fill_qty, extreme_low, 0.0)],
        fills=[FillInfo("TEST", "BUY", fill_qty, extreme_low, 0.0, 0)],
    )
    s.on_bar(fill_snap)
    assert s.pm.is_long("TEST")

    # Price reverts to OU mean (z >= 0) -> should exit
    at_mean = mu + 0.1 * sigma  # slightly above mean, z > 0
    exit_snap = make_snapshot(
        ts=102, close_day=at_mean,
        positions=[Position("TEST", fill_qty, extreme_low, 0.0)],
        pending_orders=[PendingOrder("TEST", "SELL", fill_qty, "SL_M", 0.0,
                                     s.pm.get_state("TEST").trailing_stop)],
    )
    exit_signals = s.on_bar(exit_snap)
    sells = [sig for sig in exit_signals if sig.action == "SELL" and sig.order_type == "MARKET"]
    assert len(sells) >= 1


def test_short_always_mis():
    """Short entries must always use MIS product type (CNC shorts not allowed)."""
    s = create_strategy({
        "min_history": 60, "zscore_entry": 2.0, "min_pvalue": 0.10,
    })
    prices = _generate_ou_prices(mu=100.0, theta=0.1, sigma=1.0, n=80, seed=42)
    _feed_prices(s, prices)

    mu = s.ou_mu["TEST"]
    sigma = s.ou_sigma["TEST"]

    extreme_high = mu + 3.0 * sigma
    snap = make_snapshot(ts=100, close_day=extreme_high)
    signals = s.on_bar(snap)
    sells = [sig for sig in signals if sig.action == "SELL"]
    for sig in sells:
        if sig.order_type == "MARKET":
            assert sig.product_type == "MIS"
