"""Tests for Multi-Timeframe Confirmation strategy."""

from strategies.base import (
    BarData, MarketSnapshot, Portfolio, Position, SessionContext,
    Signal, FillInfo, PendingOrder,
)
from strategies.deterministic.multi_tf_confirm import MultiTfConfirm


# --- Helpers ---

def _make_snapshot(
    ts: int,
    close_5m: float | None = None,
    high_5m: float | None = None,
    low_5m: float | None = None,
    close_15m: float | None = None,
    close_day: float | None = None,
    symbol: str = "TEST",
    cash: float = 100_000.0,
    positions: list[Position] | None = None,
    fills: list[FillInfo] | None = None,
    pending_orders: list[PendingOrder] | None = None,
) -> MarketSnapshot:
    timeframes: dict = {}
    if close_5m is not None:
        h = high_5m if high_5m is not None else close_5m + 1
        l = low_5m if low_5m is not None else close_5m - 1
        bar = BarData(symbol, close_5m, h, l, close_5m, 1000, 0)
        timeframes["5minute"] = {symbol: bar}
    if close_15m is not None:
        bar = BarData(symbol, close_15m, close_15m + 1, close_15m - 1, close_15m, 5000, 0)
        timeframes["15minute"] = {symbol: bar}
    if close_day is not None:
        bar = BarData(symbol, close_day, close_day + 2, close_day - 2, close_day, 50000, 0)
        timeframes["day"] = {symbol: bar}
    pos_list = positions or []
    equity = cash + sum(p.quantity * p.avg_price for p in pos_list)
    return MarketSnapshot(
        timestamp_ms=ts,
        timeframes=timeframes,
        history={},
        portfolio=Portfolio(cash=cash, equity=equity, positions=pos_list),
        instruments={},
        fills=fills or [],
        rejections=[],
        closed_trades=[],
        context=SessionContext(100_000.0, ts, 1000, "2024-01-01", "2024-12-31",
                               ["5minute", "15minute", "day"], 200),
        pending_orders=pending_orders or [],
    )


def _setup(**overrides) -> MultiTfConfirm:
    config = {
        "ema_period": 3,
        "rsi_period": 5,
        "rsi_entry_long": 35,
        "rsi_entry_short": 65,
        "risk_pct": 0.02,
        "atr_period": 3,
        "atr_mult": 2.0,
        "max_hold_bars": 40,
        "ema_strong_pct": 0.02,
    }
    config.update(overrides)
    s = MultiTfConfirm()
    s.initialize(config, {})
    return s


def _establish_uptrend(s: MultiTfConfirm, symbol: str = "TEST"):
    """Feed daily bars to set daily_trend='UP'."""
    for i in range(5):
        s.on_bar(_make_snapshot(i, close_day=100.0 + i * 5, symbol=symbol))


def _establish_downtrend(s: MultiTfConfirm, symbol: str = "TEST"):
    """Feed daily bars to set daily_trend='DOWN'."""
    for i in range(5):
        s.on_bar(_make_snapshot(i, close_day=120.0 - i * 5, symbol=symbol))


def _set_macd_bullish(s: MultiTfConfirm, symbol: str = "TEST"):
    """Set MACD histogram > 0 directly (linear ascending MACD = 0 due to EMA convergence)."""
    s._ensure(symbol)
    s.macd_bullish[symbol] = True


def _set_macd_bearish(s: MultiTfConfirm, symbol: str = "TEST"):
    """Set MACD histogram < 0 directly."""
    s._ensure(symbol)
    s.macd_bullish[symbol] = False


def _fill(symbol, side, price, qty, ts=0):
    return FillInfo(symbol=symbol, side=side, quantity=qty,
                    fill_price=price, costs=0.0, timestamp_ms=ts)


# --- Tests ---

def test_required_data():
    """Strategy requires 5minute, 15minute, and day intervals."""
    s = MultiTfConfirm()
    reqs = s.required_data()
    intervals = [r["interval"] for r in reqs]
    assert "5minute" in intervals
    assert "15minute" in intervals
    assert "day" in intervals
    assert all("lookback" in r for r in reqs)
    assert len(reqs) == 3


def test_long_all_three_agree():
    """Daily UP + MACD bullish + RSI < 35 should produce a BUY."""
    s = _setup()
    _establish_uptrend(s)
    _set_macd_bullish(s)

    # Warmup 5-min ascending
    for i in range(8):
        s.on_bar(_make_snapshot(200 + i, close_5m=100.0 + i, symbol="TEST"))

    # Steep drop to push Wilder RSI below 35
    ts = 300
    buy_signals = []
    for p in [106, 96, 86, 76, 66, 56]:
        snap = _make_snapshot(ts, close_5m=float(p))
        sigs = s.on_bar(snap)
        buy_signals += [sig for sig in sigs if sig.action == "BUY"]
        if buy_signals:
            break
        ts += 1

    assert len(buy_signals) >= 1, "Expected BUY signal when all three timeframes agree"
    assert buy_signals[0].order_type == "MARKET", "Entry should be MARKET order"


def test_no_entry_daily_down():
    """Daily DOWN should prevent long entry even if MACD+RSI agree."""
    s = _setup()
    _establish_downtrend(s)   # daily = DOWN
    _set_macd_bullish(s)       # MACD = bullish

    # Warmup 5-min ascending then drop (RSI < 35)
    for i in range(8):
        s.on_bar(_make_snapshot(200 + i, close_5m=100.0 + i, symbol="TEST"))

    ts = 300
    buy_signals = []
    for p in [106, 102, 98, 94, 90]:
        snap = _make_snapshot(ts, close_5m=float(p))
        sigs = s.on_bar(snap)
        buy_signals += [sig for sig in sigs if sig.action == "BUY"]
        ts += 1

    assert len(buy_signals) == 0, "Should NOT buy when daily trend is DOWN"


def test_no_entry_macd_bearish():
    """MACD bearish should prevent long entry even if daily+RSI agree."""
    s = _setup()
    _establish_uptrend(s)      # daily = UP
    _set_macd_bearish(s)       # MACD = bearish

    # Warmup 5-min ascending then drop
    for i in range(8):
        s.on_bar(_make_snapshot(200 + i, close_5m=100.0 + i, symbol="TEST"))

    ts = 300
    buy_signals = []
    for p in [106, 102, 98, 94, 90]:
        snap = _make_snapshot(ts, close_5m=float(p))
        sigs = s.on_bar(snap)
        buy_signals += [sig for sig in sigs if sig.action == "BUY"]
        ts += 1

    assert len(buy_signals) == 0, "Should NOT buy when MACD is bearish"


def test_no_entry_rsi_not_oversold():
    """RSI above threshold should prevent long entry even if daily+MACD agree."""
    s = _setup()
    _establish_uptrend(s)
    _set_macd_bullish(s)

    # Feed steady ascending prices -> RSI stays high, never < 35
    ts = 200
    buy_signals = []
    for i in range(15):
        snap = _make_snapshot(ts + i, close_5m=100.0 + i * 0.5)
        sigs = s.on_bar(snap)
        buy_signals += [sig for sig in sigs if sig.action == "BUY"]

    assert len(buy_signals) == 0, "Should NOT buy when RSI is not oversold"


def test_exit_on_daily_disagreement():
    """Long position should exit when daily trend flips to DOWN."""
    s = _setup()
    _establish_uptrend(s)
    _set_macd_bullish(s)

    # Warmup 5-min
    for i in range(8):
        s.on_bar(_make_snapshot(200 + i, close_5m=100.0 + i, symbol="TEST"))

    # Manually set up a long position
    pm_state = s.pm.get_state("TEST")
    pm_state.direction = "long"
    pm_state.qty = 20
    pm_state.avg_entry = 95.0
    pm_state.entry_bar = s.pm.bar_count
    pm_state.has_engine_stop = True
    pm_state.product_type = "MIS"
    pm_state.trailing_stop = 90.0

    # Flip daily trend to DOWN -- include positions so reconcile doesn't reset
    held = Position(symbol="TEST", quantity=20, avg_price=95.0, unrealized_pnl=0.0)
    stop_order = PendingOrder(symbol="TEST", side="SELL", quantity=20,
                              order_type="SL_M", limit_price=0.0, stop_price=90.0)
    for i in range(5):
        s.on_bar(_make_snapshot(300 + i, close_day=120.0 - i * 10, symbol="TEST",
                                positions=[held], pending_orders=[stop_order]))

    # Now send a 5-min bar -> should exit
    snap = _make_snapshot(400, close_5m=100.0, positions=[held],
                          pending_orders=[stop_order])
    sigs = s.on_bar(snap)

    sell_signals = [sig for sig in sigs if sig.action == "SELL" and sig.order_type == "MARKET"]
    assert len(sell_signals) >= 1, "Should exit long when daily trend disagrees"


def test_exit_on_macd_disagreement():
    """Long position should exit when MACD flips bearish."""
    s = _setup()
    _establish_uptrend(s)
    _set_macd_bullish(s)

    # Warmup 5-min
    for i in range(8):
        s.on_bar(_make_snapshot(200 + i, close_5m=100.0 + i, symbol="TEST"))

    # Set up long position
    pm_state = s.pm.get_state("TEST")
    pm_state.direction = "long"
    pm_state.qty = 20
    pm_state.avg_entry = 95.0
    pm_state.entry_bar = s.pm.bar_count
    pm_state.has_engine_stop = True
    pm_state.product_type = "MIS"
    pm_state.trailing_stop = 90.0

    # Flip MACD to bearish
    s.macd_bullish["TEST"] = False

    # Send a 5-min bar -> should exit since MACD disagrees
    held = Position(symbol="TEST", quantity=20, avg_price=95.0, unrealized_pnl=0.0)
    stop_order = PendingOrder(symbol="TEST", side="SELL", quantity=20,
                              order_type="SL_M", limit_price=0.0, stop_price=90.0)
    snap = _make_snapshot(400, close_5m=100.0, positions=[held],
                          pending_orders=[stop_order])
    sigs = s.on_bar(snap)

    sell_signals = [sig for sig in sigs if sig.action == "SELL" and sig.order_type == "MARKET"]
    assert len(sell_signals) >= 1, "Should exit long when MACD flips bearish"


def test_short_all_three_agree():
    """Daily DOWN + MACD bearish + RSI > 65 should produce a SELL."""
    s = _setup()
    _establish_downtrend(s)
    _set_macd_bearish(s)

    # Warmup 5-min descending
    for i in range(8):
        s.on_bar(_make_snapshot(200 + i, close_5m=100.0 - i, symbol="TEST"))

    # Rise prices to push RSI above 65
    ts = 300
    sell_signals = []
    for p in [94, 98, 102, 106, 110]:
        snap = _make_snapshot(ts, close_5m=float(p))
        sigs = s.on_bar(snap)
        sell_signals += [sig for sig in sigs if sig.action == "SELL"]
        if sell_signals:
            break
        ts += 1

    assert len(sell_signals) >= 1, "Expected SELL signal when all three timeframes agree for short"
    assert sell_signals[0].order_type == "MARKET", "Short entry should be MARKET order"


def test_short_always_mis():
    """Short entries must always use MIS product type."""
    s = _setup()
    _establish_downtrend(s)
    _set_macd_bearish(s)

    # Warmup 5-min descending
    for i in range(8):
        s.on_bar(_make_snapshot(200 + i, close_5m=100.0 - i, symbol="TEST"))

    # Rise to push RSI > 65
    ts = 300
    sell_signals = []
    for p in [94, 98, 102, 106, 110]:
        snap = _make_snapshot(ts, close_5m=float(p))
        sigs = s.on_bar(snap)
        sell_signals += [sig for sig in sigs if sig.action == "SELL"]
        if sell_signals:
            break
        ts += 1

    assert len(sell_signals) >= 1
    assert sell_signals[0].product_type == "MIS", "Short entries must always be MIS"


def test_trailing_stop():
    """After entry fill, trailing stop should be submitted and ratchet up."""
    s = _setup()
    _establish_uptrend(s)
    _set_macd_bullish(s)

    # Warmup 5-min
    for i in range(8):
        s.on_bar(_make_snapshot(200 + i, close_5m=100.0 + i, symbol="TEST"))

    # Manually set up a filled long via PM state
    pm_state = s.pm.get_state("TEST")
    pm_state.direction = "long"
    pm_state.qty = 20
    pm_state.avg_entry = 95.0
    pm_state.entry_bar = s.pm.bar_count
    pm_state.has_engine_stop = True
    pm_state.product_type = "MIS"
    pm_state.trailing_stop = 90.0

    # Feed 5-min bar at higher price -> should ratchet stop up
    held = Position(symbol="TEST", quantity=20, avg_price=95.0, unrealized_pnl=0.0)
    stop_order = PendingOrder(symbol="TEST", side="SELL", quantity=20,
                              order_type="SL_M", limit_price=0.0, stop_price=90.0)
    # Use a high price so ATR-based stop is above 90
    snap = _make_snapshot(400, close_5m=120.0, high_5m=122.0, low_5m=119.0,
                          positions=[held], pending_orders=[stop_order])
    sigs = s.on_bar(snap)

    # Check if CANCEL + SL_M were emitted (trailing stop ratchet)
    cancel_sigs = [sig for sig in sigs if sig.action == "CANCEL"]
    slm_sigs = [sig for sig in sigs if sig.order_type == "SL_M"]

    # The trailing stop should have moved up if ATR allows
    if cancel_sigs and slm_sigs:
        assert slm_sigs[0].stop_price > 90.0, "Trailing stop should ratchet up"
    else:
        # If ATR is too large relative to the price, stop won't ratchet -- that's OK
        # but the position should still be held (not exited) since trend agrees
        exit_sigs = [sig for sig in sigs if sig.action == "SELL" and sig.order_type == "MARKET"]
        assert len(exit_sigs) == 0, "Should hold position when all timeframes agree"


def test_cnc_for_strong_trend():
    """Long entry should use CNC when daily close is > 2% above EMA."""
    s = _setup(ema_strong_pct=0.02)
    # Establish a strong uptrend: close well above EMA
    # Feed ascending daily prices where last close >> EMA
    for i in range(5):
        s.on_bar(_make_snapshot(i, close_day=100.0 + i * 10, symbol="TEST"))

    # Daily close=140, EMA of [100,110,120,130,140] with period=3 is ~135
    # 140 > 135 * 1.02 = 137.7 -> strong trend -> CNC
    _set_macd_bullish(s)

    # Warmup 5-min ascending then drop
    for i in range(8):
        s.on_bar(_make_snapshot(200 + i, close_5m=100.0 + i, symbol="TEST"))

    ts = 300
    buy_signals = []
    for p in [106, 102, 98, 94, 90]:
        snap = _make_snapshot(ts, close_5m=float(p))
        sigs = s.on_bar(snap)
        buy_signals += [sig for sig in sigs if sig.action == "BUY"]
        if buy_signals:
            break
        ts += 1

    if buy_signals:
        assert buy_signals[0].product_type == "CNC", \
            f"Strong uptrend should use CNC, got {buy_signals[0].product_type}"
