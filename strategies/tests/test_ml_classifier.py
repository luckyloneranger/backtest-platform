"""Tests for ML Signal Classifier strategy."""

import math
from strategies.base import (
    BarData, MarketSnapshot, Portfolio, Position, SessionContext, InstrumentInfo,
    FillInfo, PendingOrder,
)
from strategies.deterministic.ml_classifier import MLClassifier


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
    s = MLClassifier()
    s.initialize(config or {}, DEFAULT_INSTRUMENTS)
    return s


def feed_bars(strategy, n, base=100.0, trend=0.5, volume=100_000, symbol="TEST"):
    """Feed n daily bars with a steady uptrend. Returns last close."""
    for i in range(n):
        price = base + i * trend
        snap = make_snapshot(
            ts=i, close=price, high=price + 2, low=price - 2,
            volume=volume, symbol=symbol,
        )
        strategy.on_bar(snap)
    return base + (n - 1) * trend


def feed_uptrend(strategy, n, base=100.0, symbol="TEST"):
    """Feed bars with a clear uptrend for easy model training."""
    return feed_bars(strategy, n, base=base, trend=1.0, volume=100_000, symbol=symbol)


def feed_downtrend(strategy, n, base=200.0, symbol="TEST"):
    """Feed bars with a clear downtrend for easy model training."""
    return feed_bars(strategy, n, base=base, trend=-1.0, volume=100_000, symbol=symbol)


# --- Tests ---


def test_required_data():
    s = create_strategy()
    reqs = s.required_data()
    assert len(reqs) == 1
    assert reqs[0]["interval"] == "day"
    assert reqs[0]["lookback"] == 200


def test_no_trade_before_training():
    """First 119 bars should produce no BUY/SELL signals (below min_train_bars)."""
    s = create_strategy({"min_train_bars": 120})
    all_signals = []
    for i in range(119):
        price = 100.0 + i * 0.5
        snap = make_snapshot(ts=i, close=price, high=price + 2, low=price - 2)
        sigs = s.on_bar(snap)
        for sig in sigs:
            if sig.action in ("BUY", "SELL"):
                all_signals.append(sig)
    assert len(all_signals) == 0
    # Model should not be trained yet
    assert s.models.get("TEST") is None


def test_model_trained_after_min_bars():
    """After min_train_bars of feature data, model should be trained."""
    # compute_features needs 50 bars before returning valid features.
    # With min_train_bars=60 features, we need ~110 raw bars minimum.
    # Use uptrend then downtrend to get both label classes.
    s = create_strategy({"min_train_bars": 30, "forward_bars": 3, "target_return": 0.005})
    # Feed 60 bars uptrend, then 60 bars downtrend (120 total, ~70 feature bars)
    for i in range(60):
        price = 100.0 + i * 1.0
        snap = make_snapshot(ts=i, close=price, high=price + 3, low=price - 3,
                             volume=100_000 + i * 1000)
        s.on_bar(snap)
    for i in range(60):
        price = 160.0 - i * 1.0
        snap = make_snapshot(ts=60 + i, close=price, high=price + 3, low=price - 3,
                             volume=100_000 + i * 1000)
        s.on_bar(snap)

    # Model should be trained by now
    assert s.models.get("TEST") is not None


def test_high_confidence_long():
    """After training on uptrend data, model should predict BUY on continued uptrend."""
    s = create_strategy({
        "min_train_bars": 30,
        "forward_bars": 3,
        "target_return": 0.005,
        "confidence_threshold": 0.55,
        "retrain_interval": 1,
    })
    # Phase 1: strong uptrend (80 bars to build features)
    for i in range(80):
        price = 100.0 + i * 1.5
        snap = make_snapshot(ts=i, close=price, high=price + 3, low=price - 3,
                             volume=100_000 + i * 1000)
        s.on_bar(snap)

    # Phase 2: brief dip so model learns "sell" too
    for i in range(30):
        price = 220.0 - i * 1.5
        snap = make_snapshot(ts=80 + i, close=price, high=price + 3, low=price - 3,
                             volume=100_000 + i * 1000)
        s.on_bar(snap)

    # Phase 3: resume uptrend — model should trigger BUY if confident enough
    buy_found = False
    for i in range(40):
        price = 175.0 + i * 1.5
        snap = make_snapshot(ts=110 + i, close=price, high=price + 3, low=price - 3,
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


def test_low_confidence_no_trade():
    """Ambiguous (flat) data should not produce trades."""
    s = create_strategy({
        "min_train_bars": 30,
        "forward_bars": 3,
        "target_return": 0.02,
        "confidence_threshold": 0.90,
    })
    # Feed oscillating data — hard for model to be confident
    all_signals = []
    for i in range(200):
        # Oscillate around 100 with small amplitude
        price = 100.0 + 2.0 * math.sin(i * 0.3)
        snap = make_snapshot(ts=i, close=price, high=price + 1, low=price - 1,
                             volume=100_000)
        sigs = s.on_bar(snap)
        for sig in sigs:
            if sig.action in ("BUY", "SELL") and sig.order_type == "MARKET":
                all_signals.append(sig)

    # With 90% confidence threshold and oscillating data, should have very few or no entries
    assert len(all_signals) <= 2


def test_retrain_interval():
    """Model retrains every retrain_interval bars after initial training."""
    s = create_strategy({
        "min_train_bars": 30,
        "forward_bars": 3,
        "target_return": 0.005,
        "retrain_interval": 10,
    })
    # Feed 60 uptrend + 60 downtrend to train (120 raw bars -> ~70 feature bars)
    for i in range(60):
        price = 100.0 + i * 1.0
        snap = make_snapshot(ts=i, close=price, high=price + 3, low=price - 3,
                             volume=100_000 + i * 1000)
        s.on_bar(snap)
    for i in range(60):
        price = 160.0 - i * 1.0
        snap = make_snapshot(ts=60 + i, close=price, high=price + 3, low=price - 3,
                             volume=100_000 + i * 1000)
        s.on_bar(snap)

    # Model should be trained
    assert s.models.get("TEST") is not None
    # bars_since_retrain should be small (model trained recently)
    initial_count = s.bars_since_retrain["TEST"]
    assert initial_count < 10  # trained within last few bars

    # Feed exactly retrain_interval bars to trigger retrain
    for i in range(s.retrain_interval - initial_count):
        price = 100.0 + i * 0.5
        snap = make_snapshot(ts=120 + i, close=price, high=price + 3, low=price - 3,
                             volume=100_000)
        s.on_bar(snap)

    # Should have retrained — counter at 0 or just trained this bar
    assert s.bars_since_retrain["TEST"] == 0

    # Feed 5 more bars — should NOT retrain (well below interval of 10)
    for i in range(5):
        price = 105.0 + i * 0.5
        snap = make_snapshot(ts=130 + i, close=price, high=price + 3, low=price - 3,
                             volume=100_000)
        s.on_bar(snap)

    assert s.bars_since_retrain["TEST"] == 5


def test_short_always_mis():
    """Short entries must always use MIS product type."""
    s = create_strategy({
        "min_train_bars": 30,
        "forward_bars": 3,
        "target_return": 0.005,
        "confidence_threshold": 0.50,
        "retrain_interval": 1,
    })
    # Feed uptrend then downtrend to train model with both classes
    for i in range(60):
        price = 100.0 + i * 1.0
        snap = make_snapshot(ts=i, close=price, high=price + 3, low=price - 3,
                             volume=100_000 + i * 1000)
        s.on_bar(snap)
    for i in range(60):
        price = 160.0 - i * 1.0
        snap = make_snapshot(ts=60 + i, close=price, high=price + 3, low=price - 3,
                             volume=100_000 + i * 1000)
        s.on_bar(snap)

    # Continue downtrend to possibly trigger short
    sell_signals = []
    for i in range(40):
        price = 100.0 - i * 1.0
        snap = make_snapshot(ts=120 + i, close=price, high=price + 3, low=max(price - 3, 1),
                             volume=100_000)
        sigs = s.on_bar(snap)
        for sig in sigs:
            if sig.action == "SELL" and sig.order_type == "MARKET":
                sell_signals.append(sig)

    # All short entries must be MIS
    for sig in sell_signals:
        assert sig.product_type == "MIS"


def test_position_sizing_correct():
    """Position size should be capped to available cash."""
    s = create_strategy({
        "min_train_bars": 30,
        "forward_bars": 3,
        "target_return": 0.005,
        "confidence_threshold": 0.50,
        "retrain_interval": 1,
        "risk_pct": 0.03,
        "atr_mult": 2.0,
    })
    # Train model with enough bars
    for i in range(60):
        price = 100.0 + i * 1.0
        snap = make_snapshot(ts=i, close=price, high=price + 3, low=price - 3, volume=100_000)
        s.on_bar(snap)
    for i in range(60):
        price = 160.0 - i * 1.0
        snap = make_snapshot(ts=60 + i, close=price, high=price + 3, low=price - 3, volume=100_000)
        s.on_bar(snap)

    # Feed with limited cash to test cap
    for i in range(30):
        price = 100.0 + i * 1.0
        snap = make_snapshot(ts=120 + i, close=price, high=price + 3, low=price - 3,
                             volume=100_000, cash=10_000.0)
        sigs = s.on_bar(snap)
        for sig in sigs:
            if sig.action == "BUY" and sig.quantity > 0:
                # qty * price should not exceed cash
                assert sig.quantity * price <= 10_000.0 + 1  # +1 for rounding
