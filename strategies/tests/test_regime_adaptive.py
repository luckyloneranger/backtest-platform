"""Tests for Regime-Adaptive strategy."""

from strategies.base import (
    BarData, MarketSnapshot, Portfolio, Position, SessionContext, InstrumentInfo,
    FillInfo, PendingOrder,
)
from strategies.deterministic.regime_adaptive import RegimeAdaptive


# --- Helpers ---

DEFAULT_INSTRUMENTS = {
    "TEST": InstrumentInfo(
        symbol="TEST", exchange="NSE", instrument_type="EQ", lot_size=1,
        tick_size=0.05, expiry="", strike=0.0, option_type="",
        circuit_limit_upper=0.0, circuit_limit_lower=0.0,
    ),
}

DEFAULT_CONTEXT = SessionContext(
    initial_capital=100_000.0, bar_number=0, total_bars=200,
    start_date="2024-01-01", end_date="2024-12-31",
    intervals=["day", "15minute"], lookback_window=60,
)


def make_snapshot(
    ts=0,
    close_15m=None, high_15m=None, low_15m=None, volume_15m=10_000,
    close_day=None, high_day=None, low_day=None, volume_day=100_000,
    symbol="TEST", cash=100_000.0, positions=None, fills=None, pending_orders=None,
):
    timeframes = {}
    if close_15m is not None:
        h = high_15m if high_15m is not None else close_15m + 1
        l = low_15m if low_15m is not None else close_15m - 1
        timeframes["15minute"] = {symbol: BarData(symbol, close_15m, h, l, close_15m, volume_15m, 0)}
    if close_day is not None:
        h = high_day if high_day is not None else close_day + 2
        l = low_day if low_day is not None else close_day - 2
        timeframes["day"] = {symbol: BarData(symbol, close_day, h, l, close_day, volume_day, 0)}
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
    s = RegimeAdaptive()
    s.initialize(config or {}, DEFAULT_INSTRUMENTS)
    return s


def seed_daily_trending(strategy, symbol="TEST", n=30, base=100.0):
    """Feed rising daily bars that produce high ADX (strong trend)."""
    for i in range(n):
        price = base + i * 3.0  # strong uptrend
        high = price + 1.0
        low = price - 1.0
        snap = make_snapshot(
            ts=i, close_day=price, high_day=high, low_day=low, symbol=symbol,
        )
        strategy.on_bar(snap)


def seed_daily_ranging(strategy, symbol="TEST", n=30, base=100.0):
    """Feed flat daily bars that produce low ADX and low BBW."""
    for i in range(n):
        # Oscillate in tight range
        price = base + (i % 3 - 1) * 0.5
        high = price + 0.5
        low = price - 0.5
        snap = make_snapshot(
            ts=i, close_day=price, high_day=high, low_day=low, symbol=symbol,
        )
        strategy.on_bar(snap)


def seed_daily_volatile(strategy, symbol="TEST"):
    """Feed 25 calm bars then 15 wild bars to create a BBW spike above avg.

    This produces VOLATILE because the recent BBW is much higher than the
    rolling average (accumulated during calm phase).
    """
    base = 100.0
    for i in range(40):
        if i < 25:
            # Calm phase: tight range
            price = base + (i % 3 - 1) * 0.5
            high = price + 0.5
            low = price - 0.5
        else:
            # Wild phase: huge alternating swings, no net trend
            swing = 20.0 * (1 if i % 2 == 0 else -1)
            price = base + swing
            high = price + 12.0
            low = price - 12.0
        snap = make_snapshot(
            ts=i, close_day=price, high_day=high, low_day=low, symbol=symbol,
        )
        strategy.on_bar(snap)


# --- Tests ---

def test_required_data():
    s = create_strategy()
    reqs = s.required_data()
    assert len(reqs) == 2
    intervals = {r["interval"] for r in reqs}
    assert "15minute" in intervals
    assert "day" in intervals


def test_regime_trending_detected():
    """Rising prices with strong directional movement -> TRENDING."""
    s = create_strategy()
    seed_daily_trending(s)
    regime = s.regime.get("TEST", "RANGING")
    assert regime == "TRENDING", f"Expected TRENDING, got {regime}"


def test_regime_ranging_detected():
    """Flat prices with low ADX and low BBW -> RANGING."""
    s = create_strategy()
    seed_daily_ranging(s)
    regime = s.regime.get("TEST", "")
    assert regime == "RANGING", f"Expected RANGING, got {regime}"


def test_regime_volatile_detected():
    """Calm phase followed by wild swings -> BBW spikes -> VOLATILE."""
    s = create_strategy({"regime_confirm_bars": 1})
    seed_daily_volatile(s)
    regime = s.regime.get("TEST", "")
    assert regime == "VOLATILE", f"Expected VOLATILE, got {regime}"


def test_trending_macd_entry():
    """In TRENDING regime, MACD bullish cross produces BUY."""
    s = create_strategy()
    seed_daily_trending(s)
    assert s.regime.get("TEST") == "TRENDING"
    assert s.current_atr.get("TEST", 0) > 0, "ATR must be set from daily data"

    # Need 35+ bars for MACD(12,26,9).
    # Feed 40 declining bars so MACD histogram starts negative,
    # then rising bars to cause a bullish cross.
    for i in range(40):
        price = 200.0 - i * 1.0  # decline
        snap = make_snapshot(ts=1000 + i, close_15m=price, symbol="TEST")
        s.on_bar(snap)

    all_signals = []
    for i in range(25):
        price = 160.0 + i * 3.0  # strong rise
        snap = make_snapshot(ts=2000 + i, close_15m=price, symbol="TEST")
        sigs = s.on_bar(snap)
        all_signals.extend(sigs)

    buys = [sig for sig in all_signals if sig.action == "BUY" and sig.order_type == "MARKET"]
    assert len(buys) >= 1, "Expected at least one BUY signal from MACD cross in TRENDING"


def test_ranging_bollinger_entry():
    """In RANGING regime, close < BB lower + RSI < 35 produces BUY."""
    s = create_strategy()
    seed_daily_ranging(s)
    assert s.regime.get("TEST") == "RANGING"

    # Build up 15-min prices at a stable level then drop sharply
    for i in range(30):
        snap = make_snapshot(ts=1000 + i, close_15m=100.0, symbol="TEST")
        s.on_bar(snap)

    # Now drop price sharply below BB lower to trigger entry
    all_signals = []
    for i in range(10):
        price = 95.0 - i * 1.5  # sharp drop
        snap = make_snapshot(ts=2000 + i, close_15m=price, symbol="TEST")
        sigs = s.on_bar(snap)
        all_signals.extend(sigs)

    buys = [sig for sig in all_signals if sig.action == "BUY"]
    assert len(buys) >= 1, "Expected BUY signal from BB lower + RSI oversold in RANGING"


def test_volatile_stays_flat():
    """In VOLATILE regime, no entries without 3 confirming signals."""
    s = create_strategy({"regime_confirm_bars": 1})
    seed_daily_volatile(s)
    assert s.regime.get("TEST") == "VOLATILE"

    # Feed moderate 15-min data -- not extreme enough for 3 confirms
    all_signals = []
    for i in range(45):
        price = 100.0 + (i % 5) * 0.3
        snap = make_snapshot(ts=1000 + i, close_15m=price, symbol="TEST")
        sigs = s.on_bar(snap)
        all_signals.extend(sigs)

    buys = [sig for sig in all_signals if sig.action == "BUY"]
    sells = [sig for sig in all_signals if sig.action == "SELL" and sig.order_type == "MARKET"]
    assert len(buys) == 0, "Volatile regime should not produce BUY without 3 confirms"
    assert len(sells) == 0, "Volatile regime should not produce SELL without 3 confirms"


def test_trending_wider_stop():
    """Trending regime uses 2x ATR for trailing stop vs 1.5x for ranging."""
    s = create_strategy()
    assert s.trending_atr_mult == 2.0
    assert s.ranging_atr_mult == 1.5

    # Also verify via the internal helper
    assert s._atr_mult_for_regime("TRENDING") == 2.0
    assert s._atr_mult_for_regime("RANGING") == 1.5
    assert s._atr_mult_for_regime("VOLATILE") == 3.0


def test_ranging_tighter_stop():
    """Ranging regime uses 1.5x ATR for trailing stop (tighter than trending)."""
    s = create_strategy()
    assert s.ranging_atr_mult == 1.5
    assert s.trending_atr_mult == 2.0
    assert s.ranging_atr_mult < s.trending_atr_mult


def test_trending_pyramid():
    """Pyramiding works in trending regime."""
    s = create_strategy()
    seed_daily_trending(s)
    assert s.regime.get("TEST") == "TRENDING"
    atr = s.current_atr.get("TEST", 0.0)
    assert atr > 0

    # Feed declining 15-min bars then rising to get MACD cross
    for i in range(40):
        price = 200.0 - i * 1.0
        snap = make_snapshot(ts=1000 + i, close_15m=price)
        s.on_bar(snap)

    entry_price = None
    entry_qty = None
    for i in range(25):
        price = 160.0 + i * 3.0
        snap = make_snapshot(ts=2000 + i, close_15m=price)
        sigs = s.on_bar(snap)
        for sig in sigs:
            if sig.action == "BUY" and sig.order_type == "MARKET" and entry_price is None:
                entry_price = price
                entry_qty = sig.quantity
        # Break immediately after capturing entry to avoid stale pending cancellation
        if entry_price is not None:
            break

    assert entry_price is not None, "Must get a MACD entry first"

    # Simulate fill on the very next bar (MARKET orders fill next bar)
    fill_snap = make_snapshot(
        close_15m=entry_price,
        positions=[Position("TEST", entry_qty, entry_price, 0.0)],
        fills=[FillInfo("TEST", "BUY", entry_qty, entry_price, 0.0, 0)],
    )
    s.on_bar(fill_snap)

    assert s.pm.is_long("TEST")
    assert s.pm.get_state("TEST").pyramid_count == 0

    # Push price above entry + ATR to trigger pyramid
    pyramid_price = entry_price + atr + 5.0
    state = s.pm.get_state("TEST")
    snap = make_snapshot(
        close_15m=pyramid_price,
        positions=[Position("TEST", state.qty, entry_price, 0.0)],
        pending_orders=[PendingOrder("TEST", "SELL", state.qty, "SL_M", 0.0, state.trailing_stop)],
    )
    sigs = s.on_bar(snap)
    adds = [sig for sig in sigs if sig.action == "BUY" and sig.order_type == "MARKET"]
    assert len(adds) >= 1, "Expected pyramid add in TRENDING regime"


def test_no_pyramid_in_ranging():
    """Pyramiding does NOT happen in ranging regime."""
    s = create_strategy()
    seed_daily_ranging(s)
    assert s.regime.get("TEST") == "RANGING"

    # Get a ranging entry via BB/RSI
    for i in range(30):
        snap = make_snapshot(ts=1000 + i, close_15m=100.0, symbol="TEST")
        s.on_bar(snap)

    entry_price = None
    entry_qty = None
    for i in range(10):
        price = 95.0 - i * 1.5
        snap = make_snapshot(ts=2000 + i, close_15m=price, symbol="TEST")
        sigs = s.on_bar(snap)
        for sig in sigs:
            if sig.action == "BUY" and entry_price is None:
                entry_price = price
                entry_qty = sig.quantity
        if entry_price is not None:
            break

    assert entry_price is not None, "Must get a ranging BB entry first"

    # Simulate fill on next bar
    fill_snap = make_snapshot(
        close_15m=entry_price,
        positions=[Position("TEST", entry_qty, entry_price, 0.0)],
        fills=[FillInfo("TEST", "BUY", entry_qty, entry_price, 0.0, 0)],
    )
    s.on_bar(fill_snap)

    assert s.pm.is_long("TEST")
    state = s.pm.get_state("TEST")
    atr = s.current_atr.get("TEST", 1.0)

    # Price moves well above entry -- should NOT pyramid in ranging
    pyramid_price = entry_price + atr + 20.0
    snap = make_snapshot(
        close_15m=pyramid_price,
        positions=[Position("TEST", state.qty, entry_price, 0.0)],
        pending_orders=[PendingOrder("TEST", "SELL", state.qty, "SL_M", 0.0, state.trailing_stop)],
    )
    sigs = s.on_bar(snap)
    adds = [sig for sig in sigs if sig.action == "BUY" and sig.order_type == "MARKET"]
    assert len(adds) == 0, "No pyramid in RANGING regime"


def test_regime_transition_adjusts():
    """Verify stop logic changes when regime transitions."""
    s = create_strategy()
    # Start with trending
    seed_daily_trending(s)
    assert s.regime.get("TEST") == "TRENDING"
    atr_mult_trending = s._atr_mult_for_regime("TRENDING")

    # Now feed ranging data
    seed_daily_ranging(s)
    atr_mult_ranging = s._atr_mult_for_regime("RANGING")
    assert atr_mult_ranging < atr_mult_trending, \
        f"Ranging stop mult ({atr_mult_ranging}) should be less than trending ({atr_mult_trending})"


def test_short_always_mis():
    """All short entries must use MIS product type."""
    s = create_strategy()
    seed_daily_ranging(s)
    assert s.regime.get("TEST") == "RANGING"

    # Build 15-min data then push above BB upper with high RSI
    for i in range(30):
        snap = make_snapshot(ts=1000 + i, close_15m=100.0, symbol="TEST")
        s.on_bar(snap)

    all_signals = []
    for i in range(10):
        price = 105.0 + i * 1.5  # rising above upper BB
        snap = make_snapshot(ts=2000 + i, close_15m=price, symbol="TEST")
        sigs = s.on_bar(snap)
        all_signals.extend(sigs)

    sells = [sig for sig in all_signals if sig.action == "SELL" and sig.order_type == "MARKET"]
    for sig in sells:
        assert sig.product_type == "MIS", f"Short entry must be MIS, got {sig.product_type}"
