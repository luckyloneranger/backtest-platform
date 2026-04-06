"""Tests for Portfolio Combiner strategy (Donchian + RSI dynamic ADX allocation)."""

from strategies.base import (
    BarData, MarketSnapshot, Portfolio, Position, SessionContext,
    InstrumentInfo, FillInfo, PendingOrder,
)
from strategies.deterministic.portfolio_combiner import PortfolioCombiner


# --- Helpers ---

DEFAULT_INSTRUMENTS = {
    "TEST": InstrumentInfo(
        symbol="TEST", exchange="NSE", instrument_type="EQ", lot_size=1,
        tick_size=0.05, expiry="", strike=0.0, option_type="",
        circuit_limit_upper=0.0, circuit_limit_lower=0.0,
    ),
}

DEFAULT_CONTEXT = SessionContext(
    initial_capital=100_000.0, bar_number=0, total_bars=100,
    start_date="2024-01-01", end_date="2024-12-31",
    intervals=["day", "15minute"], lookback_window=60,
)


def make_snapshot(
    ts=0,
    close_15m=None, high_15m=None, low_15m=None, volume_15m=1000,
    close_day=None, high_day=None, low_day=None, volume_day=100_000,
    symbol="TEST", cash=100_000.0, positions=None, fills=None,
    pending_orders=None,
):
    timeframes = {}
    if close_15m is not None:
        h = high_15m if high_15m is not None else close_15m + 1
        l = low_15m if low_15m is not None else close_15m - 1
        timeframes["15minute"] = {
            symbol: BarData(symbol, close_15m, h, l, close_15m, volume_15m, 0),
        }
    if close_day is not None:
        h = high_day if high_day is not None else close_day + 2
        l = low_day if low_day is not None else close_day - 2
        timeframes["day"] = {
            symbol: BarData(symbol, close_day, h, l, close_day, volume_day, 0),
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
    s = PortfolioCombiner()
    s.initialize(config or {}, DEFAULT_INSTRUMENTS)
    return s


def seed_daily(strategy, symbol="TEST", n=30, base=100.0, volume=100_000):
    """Feed n daily bars to build channel + ATR + ADX. Returns last price."""
    for i in range(n):
        price = base + i * 0.5
        snap = make_snapshot(
            ts=i,
            close_day=price, high_day=price + 2, low_day=price - 2,
            volume_day=volume, symbol=symbol,
        )
        strategy.on_bar(snap)
    return base + (n - 1) * 0.5


def seed_daily_uptrend(strategy, symbol="TEST", n=30, base=100.0):
    """Feed daily bars with strong uptrend for EMA trend detection."""
    for i in range(n):
        price = base + i * 3.0
        snap = make_snapshot(
            ts=i,
            close_day=price, high_day=price + 4, low_day=price - 2,
            symbol=symbol,
        )
        strategy.on_bar(snap)
    return base + (n - 1) * 3.0


def seed_daily_downtrend(strategy, symbol="TEST", n=30, base=200.0):
    """Feed daily bars with strong downtrend for EMA trend detection."""
    for i in range(n):
        price = base - i * 3.0
        snap = make_snapshot(
            ts=i,
            close_day=price, high_day=price + 2, low_day=price - 4,
            symbol=symbol,
        )
        strategy.on_bar(snap)
    return base - (n - 1) * 3.0


def force_adx(strategy, symbol, adx_value):
    """Override ADX value for testing regime detection."""
    strategy._ensure(symbol)
    strategy.current_adx[symbol] = adx_value


def feed_15m_warmup(strategy, symbol="TEST", n=20, base=100.0):
    """Feed 15m bars to build RSI history."""
    for i in range(n):
        price = base + i * 0.5
        snap = make_snapshot(ts=100 + i, close_15m=price, symbol=symbol)
        strategy.on_bar(snap)


# --- Tests ---

def test_required_data():
    """Strategy should require both 15minute and day intervals."""
    s = create_strategy()
    reqs = s.required_data()
    intervals = {r["interval"] for r in reqs}
    assert "day" in intervals
    assert "15minute" in intervals
    assert len(reqs) == 2
    assert all("lookback" in r for r in reqs)


def test_trending_uses_donchian():
    """ADX > 25 (trending) should trigger Donchian channel breakout entry."""
    s = create_strategy()
    seed_daily(s)
    force_adx(s, "TEST", 30.0)  # Force trending regime

    # Channel high from seeded data: max of highs = (100 + 29*0.5) + 2 = 116.5
    # Break above channel high
    snap = make_snapshot(close_15m=120.0, volume_15m=1000)
    signals = s.on_bar(snap)

    buys = [sig for sig in signals if sig.action == "BUY" and sig.order_type == "MARKET"]
    assert len(buys) == 1, f"Expected 1 MARKET BUY for Donchian breakout, got {len(buys)}"
    assert buys[0].quantity > 0


def test_ranging_uses_rsi():
    """ADX < 20 (ranging) should trigger RSI mean-reversion entry."""
    s = create_strategy({"rsi_period": 5, "rsi_oversold": 40, "ema_period": 3})
    seed_daily_uptrend(s)
    force_adx(s, "TEST", 15.0)  # Force ranging regime

    # Feed ascending 15m bars to build RSI
    feed_15m_warmup(s, base=100.0)

    # Drop sharply to push RSI below oversold threshold
    ts = 200
    entry_signals = []
    for p in [108, 104, 100, 96, 92, 88]:
        snap = make_snapshot(ts=ts, close_15m=float(p))
        sigs = s.on_bar(snap)
        entry_signals += [sig for sig in sigs if sig.action == "BUY" and sig.order_type == "LIMIT"]
        if entry_signals:
            break
        ts += 1

    assert len(entry_signals) >= 1, "Expected LIMIT BUY from RSI mean-reversion in ranging regime"


def test_neutral_half_size():
    """ADX 20-25 (neutral) should produce half the position size of trending."""
    s_trending = create_strategy()
    seed_daily(s_trending)
    force_adx(s_trending, "TEST", 30.0)  # Trending

    s_neutral = create_strategy()
    seed_daily(s_neutral)
    force_adx(s_neutral, "TEST", 22.0)  # Neutral

    # Both break above channel high
    snap_t = make_snapshot(close_15m=120.0, volume_15m=1000)
    signals_t = s_trending.on_bar(snap_t)
    buys_t = [sig for sig in signals_t if sig.action == "BUY"]

    snap_n = make_snapshot(close_15m=120.0, volume_15m=1000)
    signals_n = s_neutral.on_bar(snap_n)
    buys_n = [sig for sig in signals_n if sig.action == "BUY"]

    assert len(buys_t) >= 1 and len(buys_n) >= 1, \
        f"Expected entries from both: trending={len(buys_t)}, neutral={len(buys_n)}"
    # Neutral should have roughly half the qty (within rounding)
    assert buys_n[0].quantity <= buys_t[0].quantity, \
        f"Neutral qty ({buys_n[0].quantity}) should be <= trending qty ({buys_t[0].quantity})"
    if buys_t[0].quantity >= 4:
        # Only check ratio when qty is large enough to halve meaningfully
        assert buys_n[0].quantity <= (buys_t[0].quantity // 2) + 1, \
            f"Neutral qty ({buys_n[0].quantity}) should be ~half of trending ({buys_t[0].quantity})"


def test_trailing_stop():
    """Long position should ratchet trailing stop upward when price rises."""
    s = create_strategy()
    seed_daily(s)
    force_adx(s, "TEST", 30.0)

    # Trigger entry
    snap = make_snapshot(close_15m=120.0, volume_15m=1000)
    entry_sigs = s.on_bar(snap)
    buys = [sig for sig in entry_sigs if sig.action == "BUY"]
    assert len(buys) >= 1

    fill_qty = buys[0].quantity
    fill_price = 120.0

    # Simulate fill
    fill_snap = make_snapshot(
        close_15m=120.0,
        positions=[Position("TEST", fill_qty, fill_price, 0.0)],
        fills=[FillInfo("TEST", "BUY", fill_qty, fill_price, 0.0, 0)],
    )
    s.on_bar(fill_snap)

    state = s.pm.get_state("TEST")
    assert state.has_engine_stop
    old_stop = state.trailing_stop

    # Price rises -> trailing stop should ratchet
    snap2 = make_snapshot(
        close_15m=130.0,
        positions=[Position("TEST", fill_qty, fill_price, 0.0)],
        pending_orders=[PendingOrder("TEST", "SELL", fill_qty, "SL_M", 0.0, old_stop)],
    )
    sigs2 = s.on_bar(snap2)
    new_stop = s.pm.get_state("TEST").trailing_stop
    assert new_stop > old_stop, f"Stop should ratchet up: {new_stop} > {old_stop}"

    cancels = [sig for sig in sigs2 if sig.action == "CANCEL"]
    sl_ms = [sig for sig in sigs2 if sig.order_type == "SL_M"]
    assert len(cancels) >= 1
    assert len(sl_ms) >= 1


def test_trending_cnc_ranging_mis():
    """Trending longs should use CNC, ranging longs should use MIS."""
    # Trending: CNC
    s_t = create_strategy()
    seed_daily(s_t)
    force_adx(s_t, "TEST", 30.0)

    snap_t = make_snapshot(close_15m=120.0)
    sigs_t = s_t.on_bar(snap_t)
    buys_t = [sig for sig in sigs_t if sig.action == "BUY"]
    assert len(buys_t) >= 1
    assert buys_t[0].product_type == "CNC", \
        f"Trending long should use CNC, got {buys_t[0].product_type}"

    # Ranging: MIS
    s_r = create_strategy({"rsi_period": 5, "rsi_oversold": 40, "ema_period": 3})
    seed_daily_uptrend(s_r)
    force_adx(s_r, "TEST", 15.0)
    feed_15m_warmup(s_r, base=100.0)

    ts = 200
    entry_signals = []
    for p in [108, 104, 100, 96, 92, 88]:
        snap = make_snapshot(ts=ts, close_15m=float(p))
        sigs = s_r.on_bar(snap)
        entry_signals += [sig for sig in sigs if sig.action == "BUY"]
        if entry_signals:
            break
        ts += 1

    assert len(entry_signals) >= 1
    assert entry_signals[0].product_type == "MIS", \
        f"Ranging long should use MIS, got {entry_signals[0].product_type}"


def test_short_always_mis():
    """Short entries should always use MIS regardless of regime."""
    s = create_strategy()
    seed_daily(s)
    force_adx(s, "TEST", 30.0)  # Trending

    # Channel low from seeded data: min of lows = 98.0
    # Break below channel low
    snap = make_snapshot(close_15m=90.0)
    signals = s.on_bar(snap)

    sells = [sig for sig in signals if sig.action == "SELL" and sig.order_type == "MARKET"]
    assert len(sells) >= 1, "Expected short entry on channel low breakout"
    assert sells[0].product_type == "MIS", \
        f"Short should always be MIS, got {sells[0].product_type}"


def test_time_stop_exit():
    """Position held beyond max_hold_bars should be exited (if no gain)."""
    s = create_strategy({"max_hold_bars": 5})
    seed_daily(s)
    force_adx(s, "TEST", 30.0)

    # Set up a long position directly in PM
    pm_state = s.pm.get_state("TEST")
    pm_state.direction = "long"
    pm_state.qty = 10
    pm_state.avg_entry = 120.0
    pm_state.has_engine_stop = True
    pm_state.product_type = "CNC"
    pm_state.trailing_stop = 110.0
    pm_state.bars_held = 6  # Already over max_hold_bars

    # Price near entry (no meaningful gain)
    snap = make_snapshot(
        close_15m=120.2,
        positions=[Position("TEST", 10, 120.0, 0.0)],
        pending_orders=[PendingOrder("TEST", "SELL", 10, "SL_M", 0.0, 110.0)],
    )
    signals = s.on_bar(snap)

    sells = [sig for sig in signals if sig.action == "SELL" and sig.order_type == "MARKET"]
    assert len(sells) >= 1, "Should exit on time stop when no meaningful gain"


def test_qty_capped_to_cash():
    """Position size should never exceed available cash / price."""
    s = create_strategy({"risk_per_trade": 0.5})  # Aggressive sizing
    seed_daily(s)
    force_adx(s, "TEST", 30.0)

    # Low cash scenario
    snap = make_snapshot(close_15m=120.0, cash=1000.0)
    signals = s.on_bar(snap)
    buys = [sig for sig in signals if sig.action == "BUY"]

    if buys:
        max_possible = int(1000.0 / 120.0)
        assert buys[0].quantity <= max_possible, \
            f"Qty {buys[0].quantity} exceeds max affordable {max_possible}"
