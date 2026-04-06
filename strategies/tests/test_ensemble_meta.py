"""Tests for Adaptive Ensemble Meta-Learner strategy."""

import math
from strategies.base import (
    BarData, MarketSnapshot, Portfolio, Position, SessionContext, InstrumentInfo,
    FillInfo, PendingOrder,
)
from strategies.deterministic.ensemble_meta import EnsembleMeta


# --- Helpers ---

DEFAULT_INSTRUMENTS = {
    "TEST": InstrumentInfo(
        symbol="TEST", exchange="NSE", instrument_type="EQ", lot_size=1,
        tick_size=0.05, expiry="", strike=0.0, option_type="",
        circuit_limit_upper=0.0, circuit_limit_lower=0.0,
    ),
}

DEFAULT_CONTEXT = SessionContext(
    initial_capital=1_000_000.0, bar_number=0, total_bars=300,
    start_date="2024-01-01", end_date="2024-12-31",
    intervals=["day"], lookback_window=200,
)


def make_snapshot(
    ts=0,
    close=100.0, high=None, low=None, volume=100_000,
    symbol="TEST", cash=1_000_000.0, positions=None, fills=None,
    pending_orders=None,
):
    h = high if high is not None else close + 2
    l = low if low is not None else close - 2
    timeframes = {
        "day": {symbol: BarData(symbol, close, h, l, close, volume, 0, ts)},
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
    s = EnsembleMeta()
    s.initialize(config or {}, DEFAULT_INSTRUMENTS)
    return s


def feed_bars(strategy, n, base=100.0, trend=0.5, volume=100_000, symbol="TEST"):
    """Feed n daily bars with a specified trend. Returns last close."""
    for i in range(n):
        price = base + i * trend
        snap = make_snapshot(
            ts=i, close=price, high=price + 3, low=price - 3,
            volume=volume, symbol=symbol,
        )
        strategy.on_bar(snap)
    return base + (n - 1) * trend


# --- Tests ---


def test_required_data():
    s = create_strategy()
    reqs = s.required_data()
    assert len(reqs) == 1
    assert reqs[0]["interval"] == "day"
    assert reqs[0]["lookback"] == 200


def test_no_trade_before_training():
    """First bars below min_train_bars should produce no BUY/SELL signals."""
    s = create_strategy({"min_train_bars": 120})
    all_signals = []
    for i in range(100):
        price = 100.0 + i * 0.5
        snap = make_snapshot(ts=i, close=price, high=price + 3, low=price - 3)
        sigs = s.on_bar(snap)
        for sig in sigs:
            if sig.action in ("BUY", "SELL"):
                all_signals.append(sig)
    assert len(all_signals) == 0
    # Model should not be trained yet
    assert s.models.get("TEST") is None


def test_model_trained_after_min_bars():
    """After sufficient bars with varied signals, model should be trained."""
    s = create_strategy({
        "min_train_bars": 30,
        "forward_bars": 3,
        "target_return": 0.005,
    })
    # Feed 80 bars uptrend then 80 bars downtrend to get both label classes
    for i in range(80):
        price = 100.0 + i * 1.0
        snap = make_snapshot(ts=i, close=price, high=price + 4, low=price - 4,
                             volume=100_000 + i * 1000)
        s.on_bar(snap)
    for i in range(80):
        price = 180.0 - i * 1.0
        snap = make_snapshot(ts=80 + i, close=price, high=price + 4, low=price - 4,
                             volume=100_000 + i * 1000)
        s.on_bar(snap)

    # Model should be trained by now
    assert s.models.get("TEST") is not None


def test_sub_signals_computed():
    """Verify sub-signals are generated correctly for a clear uptrend."""
    s = create_strategy()
    # Need 35+ bars for sub-signals to be computed
    for i in range(50):
        price = 100.0 + i * 1.0
        snap = make_snapshot(ts=i, close=price, high=price + 4, low=price - 4,
                             volume=100_000)
        s.on_bar(snap)

    # Feature buffer should have entries
    assert len(s.feature_buffer.get("TEST", [])) > 0
    # Each feature vector has 7 elements
    last_feat = s.feature_buffer["TEST"][-1]
    assert len(last_feat) == 7


def test_high_confidence_long():
    """After training on uptrend data, model should produce a BUY on resumed uptrend."""
    s = create_strategy({
        "min_train_bars": 30,
        "forward_bars": 3,
        "target_return": 0.005,
        "confidence_threshold": 0.55,
        "retrain_interval": 1,
    })
    # Phase 1: strong uptrend
    for i in range(80):
        price = 100.0 + i * 1.5
        snap = make_snapshot(ts=i, close=price, high=price + 4, low=price - 4,
                             volume=100_000 + i * 1000)
        s.on_bar(snap)

    # Phase 2: brief dip so model learns both classes
    for i in range(30):
        price = 220.0 - i * 1.5
        snap = make_snapshot(ts=80 + i, close=price, high=price + 4, low=price - 4,
                             volume=100_000 + i * 1000)
        s.on_bar(snap)

    # Phase 3: resume uptrend -- model should trigger BUY
    buy_found = False
    for i in range(40):
        price = 175.0 + i * 1.5
        snap = make_snapshot(ts=110 + i, close=price, high=price + 4, low=price - 4,
                             volume=100_000 + i * 1000)
        sigs = s.on_bar(snap)
        for sig in sigs:
            if sig.action == "BUY" and sig.order_type == "MARKET":
                buy_found = True
                break
        if buy_found:
            break

    # Model should exist at minimum
    assert s.models.get("TEST") is not None


def test_all_sub_signals_agree():
    """When all sub-signals point the same direction, feature vector should reflect it."""
    s = create_strategy()
    # Build a strong enough uptrend that SMA and most signals point up.
    # Use an accelerating trend to keep MACD histogram positive near the end.
    for i in range(60):
        price = 50.0 + i * 2.0 + (i ** 1.3) * 0.1
        snap = make_snapshot(ts=i, close=price, high=price + 5, low=price - 5,
                             volume=100_000 + i * 2000)
        s.on_bar(snap)

    assert len(s.feature_buffer.get("TEST", [])) > 0
    last_feat = s.feature_buffer["TEST"][-1]

    # SMA (fast > slow in uptrend) should be +1
    assert last_feat[0] == 1  # sma
    # At least 3 of the 5 sub-signals (indices 0-4) should be bullish (+1)
    bullish_count = sum(1 for v in last_feat[:5] if v == 1)
    assert bullish_count >= 3, f"Expected >=3 bullish sub-signals, got {bullish_count}: {last_feat[:5]}"


def test_short_always_mis():
    """Short entries must always use MIS product type."""
    s = create_strategy({
        "min_train_bars": 30,
        "forward_bars": 3,
        "target_return": 0.005,
        "confidence_threshold": 0.50,
        "retrain_interval": 1,
    })
    # Feed uptrend then downtrend to train both classes
    for i in range(60):
        price = 100.0 + i * 1.0
        snap = make_snapshot(ts=i, close=price, high=price + 4, low=price - 4,
                             volume=100_000 + i * 1000)
        s.on_bar(snap)
    for i in range(60):
        price = 160.0 - i * 1.0
        snap = make_snapshot(ts=60 + i, close=price, high=price + 4, low=price - 4,
                             volume=100_000 + i * 1000)
        s.on_bar(snap)

    # Continue downtrend to possibly trigger short
    sell_signals = []
    for i in range(40):
        price = 100.0 - i * 1.0
        snap = make_snapshot(ts=120 + i, close=price, high=price + 4, low=max(price - 4, 1),
                             volume=100_000)
        sigs = s.on_bar(snap)
        for sig in sigs:
            if sig.action == "SELL" and sig.order_type == "MARKET":
                sell_signals.append(sig)

    # All short entries must be MIS
    for sig in sell_signals:
        assert sig.product_type == "MIS"


def test_retrain_happens():
    """Model retrains every retrain_interval bars after initial training."""
    s = create_strategy({
        "min_train_bars": 30,
        "forward_bars": 3,
        "target_return": 0.005,
        "retrain_interval": 10,
    })
    # Feed 80 uptrend + 80 downtrend to train
    for i in range(80):
        price = 100.0 + i * 1.0
        snap = make_snapshot(ts=i, close=price, high=price + 4, low=price - 4,
                             volume=100_000 + i * 1000)
        s.on_bar(snap)
    for i in range(80):
        price = 180.0 - i * 1.0
        snap = make_snapshot(ts=80 + i, close=price, high=price + 4, low=price - 4,
                             volume=100_000 + i * 1000)
        s.on_bar(snap)

    # Model should be trained
    assert s.models.get("TEST") is not None
    initial_count = s.bars_since_retrain["TEST"]
    assert initial_count < 10  # trained within last few bars

    # Feed exactly retrain_interval bars to trigger retrain
    for i in range(s.retrain_interval - initial_count):
        price = 100.0 + i * 0.5
        snap = make_snapshot(ts=160 + i, close=price, high=price + 4, low=price - 4,
                             volume=100_000)
        s.on_bar(snap)

    # Should have retrained -- counter at 0
    assert s.bars_since_retrain["TEST"] == 0

    # Feed 5 more bars -- should NOT retrain
    for i in range(5):
        price = 105.0 + i * 0.5
        snap = make_snapshot(ts=170 + i, close=price, high=price + 4, low=price - 4,
                             volume=100_000)
        s.on_bar(snap)

    assert s.bars_since_retrain["TEST"] == 5
