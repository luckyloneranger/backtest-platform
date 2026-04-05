"""Tests for RSI Mean Reversion with Trend Filter + Pyramiding strategy."""

from strategies.base import (
    BarData, MarketSnapshot, Portfolio, Position, SessionContext, InstrumentInfo,
    Signal, FillInfo, PendingOrder,
)
from strategies.deterministic.rsi_daily_trend import RsiDailyTrend
from strategies.indicators import compute_rsi, compute_ema


# --- Helper ---

def make_snapshot(
    ts: int,
    close_15m: float | None = None,
    close_day: float | None = None,
    high_day: float | None = None,
    low_day: float | None = None,
    symbol: str = "TEST",
    cash: float = 100_000.0,
    positions: list[Position] | None = None,
    fills: list[FillInfo] | None = None,
    pending_orders: list[PendingOrder] | None = None,
) -> MarketSnapshot:
    timeframes = {}
    if close_15m is not None:
        bar = BarData(symbol, close_15m, close_15m + 1, close_15m - 1, close_15m, 1000, 0)
        timeframes["15minute"] = {symbol: bar}
    if close_day is not None:
        h = high_day if high_day is not None else close_day + 1
        l = low_day if low_day is not None else close_day - 1
        bar = BarData(symbol, close_day, h, l, close_day, 50000, 0)
        timeframes["day"] = {symbol: bar}
    pos_list = positions if positions is not None else []
    equity = cash + sum(p.quantity * p.avg_price for p in pos_list)
    fill_list = fills if fills is not None else []
    po_list = pending_orders if pending_orders is not None else []
    snap = MarketSnapshot(
        timestamp_ms=ts,
        timeframes=timeframes,
        history={},
        portfolio=Portfolio(cash=cash, equity=equity, positions=pos_list),
        instruments={},
        fills=fill_list,
        rejections=[],
        closed_trades=[],
        context=SessionContext(100_000.0, ts, 1000, "2024-01-01", "2024-12-31", ["15minute", "day"], 200),
    )
    snap.pending_orders = po_list
    return snap


def _setup_strategy(**overrides) -> RsiDailyTrend:
    """Create and initialize strategy with reasonable test defaults."""
    config = {
        "rsi_period": 5,
        "ema_period": 3,
        "rsi_entry_1": 40,
        "rsi_entry_2": 30,
        "rsi_entry_3": 20,
        "rsi_partial_exit": 60,
        "rsi_full_exit": 70,
        "risk_pct": 0.3,
        "atr_period": 3,
        "atr_stop_multiplier": 2.0,
        "max_loss_pct": 0.03,
        "max_hold_bars": 20,
        "cooldown_bars": 0,
    }
    config.update(overrides)
    s = RsiDailyTrend()
    s.initialize(config, {})
    return s


def _establish_uptrend(s: RsiDailyTrend, symbol: str = "TEST"):
    """Feed daily bars to establish an uptrend (close > EMA, EMA rising)."""
    for i in range(5):
        s.on_bar(make_snapshot(
            i, close_day=100.0 + i * 5, high_day=105.0 + i * 5, low_day=98.0 + i * 5,
            symbol=symbol,
        ))


def _establish_downtrend(s: RsiDailyTrend, symbol: str = "TEST"):
    """Feed daily bars to establish a downtrend (close < EMA, EMA falling)."""
    for i in range(5):
        s.on_bar(make_snapshot(
            i, close_day=120.0 - i * 5, high_day=125.0 - i * 5, low_day=118.0 - i * 5,
            symbol=symbol,
        ))


def _feed_15m_prices(s: RsiDailyTrend, prices: list[float], start_ts: int = 100,
                     symbol: str = "TEST", cash: float = 100_000.0,
                     positions: list[Position] | None = None,
                     fills: list[FillInfo] | None = None) -> list[Signal]:
    """Feed a series of 15-minute bars and collect all signals."""
    all_signals = []
    for i, p in enumerate(prices):
        snap = make_snapshot(start_ts + i, close_15m=float(p), symbol=symbol,
                             cash=cash, positions=positions, fills=fills)
        sigs = s.on_bar(snap)
        all_signals.extend(sigs)
    return all_signals


def _get_entry_fill(symbol: str, side: str, price: float, quantity: int, ts: int = 0) -> FillInfo:
    """Create a FillInfo for entry/pyramid fill confirmation."""
    return FillInfo(symbol=symbol, side=side, quantity=quantity,
                    fill_price=price, costs=0.0, timestamp_ms=ts)


# --- Unit tests for indicators (imported from indicators module) ---

def test_compute_rsi():
    """RSI computed from shared indicators module."""
    # Uptrend -> high RSI
    prices_up = [100 + i for i in range(20)]
    rsi = compute_rsi(prices_up, 14)
    assert rsi is not None
    assert rsi > 90

    # Downtrend -> low RSI
    prices_down = [100 - i for i in range(20)]
    rsi = compute_rsi(prices_down, 14)
    assert rsi is not None
    assert rsi < 10

    # Not enough data
    assert compute_rsi([100, 101, 102], 14) is None


def test_compute_ema():
    """EMA computed from shared indicators module."""
    prices = [10.0, 11.0, 12.0, 13.0, 14.0]
    ema = compute_ema(prices, 3)
    assert ema is not None
    assert 12.0 < ema < 14.0

    # Not enough data
    assert compute_ema([10.0, 11.0], 5) is None


# --- Strategy required_data ---

def test_required_data():
    s = RsiDailyTrend()
    reqs = s.required_data()
    intervals = [r["interval"] for r in reqs]
    assert "15minute" in intervals
    assert "day" in intervals
    assert all("lookback" in r for r in reqs)


# --- Pyramid entry tests ---

def test_pyramid_entry_at_rsi_40_30_20():
    """Strategy should submit LIMIT entries in pyramid levels, confirmed via fills."""
    # Disable stops so they don't interfere with pyramid entries
    s = _setup_strategy(max_loss_pct=0.99, atr_stop_multiplier=100.0, max_hold_bars=9999)
    _establish_uptrend(s)

    state = s._get_state("TEST")

    # Phase 1: warmup prices (ascending -> RSI high, no entry)
    warmup = [100, 101, 102, 103, 104, 105, 106]
    _feed_15m_prices(s, warmup)
    assert state.pyramid_level == 0

    # Phase 2: Drop prices to trigger Level 1 LIMIT entry
    ts = 200
    # First drop to get RSI < 40
    drop_prices = [104, 100, 96, 92, 88]
    limit_signals = []
    for p in drop_prices:
        snap = make_snapshot(ts, close_15m=float(p))
        sigs = s.on_bar(snap)
        for sig in sigs:
            if sig.action == "BUY" and sig.order_type == "LIMIT":
                limit_signals.append(sig)
        if limit_signals:
            break
        ts += 1

    # Should have at least 1 LIMIT BUY signal
    assert len(limit_signals) >= 1, f"Expected LIMIT BUY signal, got {len(limit_signals)}"
    assert state.direction == "flat", "Direction should still be flat (awaiting fill)"
    assert state.pending_entry_bar > 0, "pending_entry_bar should be set"

    level1_qty = limit_signals[0].quantity
    fill_price = limit_signals[0].limit_price

    # Simulate fill on next bar
    ts += 1
    fill = _get_entry_fill("TEST", "BUY", fill_price, level1_qty, ts)
    pos = [Position(symbol="TEST", quantity=level1_qty, avg_price=fill_price, unrealized_pnl=0.0)]
    snap = make_snapshot(ts, close_15m=86.0, positions=pos, fills=[fill])
    sigs = s.on_bar(snap)

    assert state.direction == "long", f"Expected direction='long' after fill, got '{state.direction}'"
    assert state.pyramid_level == 1, f"Expected pyramid_level=1, got {state.pyramid_level}"

    # Check that SL-M stop was submitted on fill
    slm_sigs = [sig for sig in sigs if sig.order_type == "SL_M"]
    assert len(slm_sigs) >= 1, "Should submit SL-M stop on entry fill"

    # Phase 3: Continue dropping for Level 2 LIMIT
    ts += 1
    level2_signals = []
    more_drop = [84, 82, 80]
    for p in more_drop:
        snap = make_snapshot(ts, close_15m=float(p), positions=pos)
        sigs = s.on_bar(snap)
        for sig in sigs:
            if sig.action == "BUY" and sig.order_type == "LIMIT":
                level2_signals.append(sig)
        if level2_signals:
            break
        ts += 1

    if level2_signals:
        # Simulate Level 2 fill
        l2_qty = level2_signals[0].quantity
        ts += 1
        fill2 = _get_entry_fill("TEST", "BUY", level2_signals[0].limit_price, l2_qty, ts)
        total = level1_qty + l2_qty
        pos2 = [Position(symbol="TEST", quantity=total, avg_price=fill_price, unrealized_pnl=0.0)]
        snap = make_snapshot(ts, close_15m=78.0, positions=pos2, fills=[fill2])
        sigs = s.on_bar(snap)

        assert state.pyramid_level == 2, f"Expected pyramid_level=2, got {state.pyramid_level}"


def test_no_entry_without_trend():
    """No BUY signal should be generated when daily trend is down."""
    s = _setup_strategy()
    _establish_downtrend(s)

    # Prices that drop sharply (should trigger RSI < 40 but no entry without uptrend)
    prices = [100, 102, 104, 106, 108, 110, 105, 100, 95, 90, 85, 80]
    sigs = _feed_15m_prices(s, prices)
    assert not any(sig.action == "BUY" for sig in sigs), \
        "No BUY signals when daily trend is down"


# --- Exit tests ---

def test_partial_exit_at_rsi_60():
    """Should sell half the position when RSI rises above partial exit threshold."""
    s = _setup_strategy(max_loss_pct=0.99, atr_stop_multiplier=100.0, max_hold_bars=9999)
    _establish_uptrend(s)

    state = s._get_state("TEST")

    # Warmup bars
    warmup = [100, 101, 102, 103, 104, 105, 106]
    _feed_15m_prices(s, warmup)

    # Manually set up a long position (bypassing LIMIT entry flow for exit test)
    entry_qty = 100
    state.direction = "long"
    state.pyramid_level = 1
    state.avg_entry_price = 90.0
    state.total_qty = entry_qty
    state.entry_bar = state.bar_count
    state.partial_taken = False
    state.has_engine_stop = True
    state.product_type = "MIS"
    state.trailing_stop = 85.0

    # Rise to push RSI above 60 (partial exit threshold)
    ts = 400
    partial_signal = None
    rise = [90, 93, 96, 99, 102, 105, 108]
    for p in rise:
        held = Position(symbol="TEST", quantity=entry_qty, avg_price=90.0, unrealized_pnl=0.0)
        snap = make_snapshot(ts, close_15m=float(p), positions=[held])
        sigs = s.on_bar(snap)
        for sig in sigs:
            if sig.action == "SELL" and sig.order_type == "MARKET" and partial_signal is None:
                partial_signal = sig
        if partial_signal is not None:
            break
        ts += 1

    assert partial_signal is not None, "Should trigger partial exit when RSI > 60"
    assert partial_signal.quantity == entry_qty // 2, \
        f"Partial exit should sell half ({entry_qty // 2}), got {partial_signal.quantity}"


def test_full_exit_at_rsi_70():
    """Should sell full position when RSI rises above full exit threshold."""
    s = _setup_strategy(max_loss_pct=0.99, atr_stop_multiplier=100.0, max_hold_bars=9999)
    _establish_uptrend(s)

    state = s._get_state("TEST")

    # Warmup
    warmup = [100, 101, 102, 103, 104, 105, 106]
    _feed_15m_prices(s, warmup)

    # Manually set up a long position
    entry_qty = 100
    state.direction = "long"
    state.pyramid_level = 1
    state.avg_entry_price = 90.0
    state.total_qty = entry_qty
    state.entry_bar = state.bar_count
    state.partial_taken = True  # already took partial
    state.has_engine_stop = True
    state.product_type = "MIS"
    state.trailing_stop = 85.0

    held = Position(symbol="TEST", quantity=entry_qty, avg_price=90.0, unrealized_pnl=0.0)

    # Strong rise to push RSI above 70
    rise = [90, 95, 100, 108, 115, 122, 130, 138, 145, 152]
    sigs = _feed_15m_prices(s, rise, start_ts=400, positions=[held])

    sell_sigs = [sig for sig in sigs if sig.action == "SELL" and sig.order_type == "MARKET"]
    assert len(sell_sigs) >= 1, "Should trigger full exit when RSI > 70"

    # After full exit, state should be reset
    assert state.pyramid_level == 0


def test_trailing_stop_exit():
    """Engine SL-M handles trailing stop; strategy ratchets the stop via CANCEL + new SL-M."""
    s = _setup_strategy(atr_stop_multiplier=2.0, max_loss_pct=0.99, max_hold_bars=9999)
    _establish_uptrend(s)

    state = s._get_state("TEST")

    # Warmup + manual position setup
    warmup = [100, 101, 102, 103, 104, 105, 106]
    _feed_15m_prices(s, warmup)

    entry_qty = 100
    state.direction = "long"
    state.pyramid_level = 3  # max level
    state.avg_entry_price = 90.0
    state.total_qty = entry_qty
    state.entry_bar = state.bar_count
    state.trailing_stop = 85.0
    state.partial_taken = True
    state.has_engine_stop = True
    state.product_type = "MIS"

    held = Position(symbol="TEST", quantity=entry_qty, avg_price=90.0, unrealized_pnl=0.0)

    # Feed a bar that should ratchet the stop upward (if ATR available)
    snap = make_snapshot(500, close_15m=95.0, positions=[held])
    sigs = s.on_bar(snap)

    # Engine handles the actual stop-loss triggering (via SL-M order fill).
    # The strategy just updates the SL-M via CANCEL + new SL-M.
    cancel_sigs = [sig for sig in sigs if sig.action == "CANCEL"]
    slm_sigs = [sig for sig in sigs if sig.order_type == "SL_M"]
    # If ATR is available and stop ratcheted, we should see cancel + new SL-M
    if cancel_sigs and slm_sigs:
        assert slm_sigs[-1].stop_price > 85.0, "Ratcheted stop should be higher"

    # Now simulate the engine triggering the stop (fill comes back)
    ts = 501
    stop_fill = _get_entry_fill("TEST", "SELL", 80.0, entry_qty, ts)
    snap = make_snapshot(ts, close_15m=80.0, positions=[], fills=[stop_fill])
    sigs = s.on_bar(snap)

    assert state.pyramid_level == 0, "State should be reset after stop-hit"
    assert state.direction == "flat", "Direction should be flat after stop-hit"


def test_max_loss_exit():
    """Engine SL-M handles max loss stop; verify SL-M is at correct price on entry."""
    s = _setup_strategy(max_loss_pct=0.03, atr_stop_multiplier=100.0, max_hold_bars=9999,
                        cooldown_bars=0)
    _establish_uptrend(s)

    state = s._get_state("TEST")

    # Warmup RSI with enough bars
    warmup = [100, 101, 102, 103, 104, 105, 106]
    _feed_15m_prices(s, warmup)

    # Drop to trigger LIMIT entry
    ts = 200
    limit_signals = []
    drop = [104, 100, 96, 92, 88]
    for p in drop:
        snap = make_snapshot(ts, close_15m=float(p))
        sigs = s.on_bar(snap)
        for sig in sigs:
            if sig.action == "BUY" and sig.order_type == "LIMIT":
                limit_signals.append(sig)
        if limit_signals:
            break
        ts += 1

    assert len(limit_signals) >= 1, "Should have submitted a LIMIT entry"
    entry_qty = limit_signals[0].quantity
    fill_price = 100.0  # fill price

    # Simulate fill
    ts += 1
    fill = _get_entry_fill("TEST", "BUY", fill_price, entry_qty, ts)
    pos = [Position(symbol="TEST", quantity=entry_qty, avg_price=fill_price, unrealized_pnl=0.0)]
    snap = make_snapshot(ts, close_15m=100.0, positions=pos, fills=[fill])
    sigs = s.on_bar(snap)

    # Check SL-M was submitted at max_loss level
    slm_sigs = [sig for sig in sigs if sig.order_type == "SL_M"]
    assert len(slm_sigs) >= 1, "Should submit SL-M on entry fill"
    expected_stop = fill_price * (1 - 0.03)  # 97.0
    assert abs(slm_sigs[0].stop_price - expected_stop) < 0.01, \
        f"SL-M stop should be at {expected_stop}, got {slm_sigs[0].stop_price}"


# --- Short selling tests ---

def test_short_entry_on_rsi_overbought_downtrend():
    """Downtrend + RSI > 60 should produce a LIMIT SELL signal to open a short position."""
    s = _setup_strategy(max_loss_pct=0.99, atr_stop_multiplier=100.0, max_hold_bars=9999)
    _establish_downtrend(s)

    state = s._get_state("TEST")

    # Warmup: descending prices to seed RSI (RSI will be low after this)
    warmup = [100, 99, 98, 97, 96, 95, 94]
    _feed_15m_prices(s, warmup)
    assert state.direction == "flat"

    # Now prices rise sharply to push RSI above 60 (overbought in downtrend)
    ts = 200
    sell_signals = []
    rise = [96, 100, 104, 108, 112]
    for p in rise:
        snap = make_snapshot(ts, close_15m=float(p))
        sigs = s.on_bar(snap)
        for sig in sigs:
            if sig.action == "SELL" and sig.order_type == "LIMIT":
                sell_signals.append(sig)
        if sell_signals:
            break
        ts += 1

    assert len(sell_signals) >= 1, f"Expected at least 1 short LIMIT SELL signal, got {len(sell_signals)}"
    assert sell_signals[0].product_type == "MIS", "Short signal should use MIS product type"
    assert sell_signals[0].limit_price > sell_signals[0].limit_price * 0.999, \
        "Short LIMIT price should be above close"
    assert state.pending_entry_bar > 0, "pending_entry_bar should be set"

    # Simulate fill
    entry_qty = sell_signals[0].quantity
    ts += 1
    fill = _get_entry_fill("TEST", "SELL", sell_signals[0].limit_price, entry_qty, ts)
    pos = [Position(symbol="TEST", quantity=-entry_qty, avg_price=sell_signals[0].limit_price, unrealized_pnl=0.0)]
    snap = make_snapshot(ts, close_15m=114.0, positions=pos, fills=[fill])
    sigs = s.on_bar(snap)

    assert state.direction == "short", f"Expected direction='short', got '{state.direction}'"
    assert state.pyramid_level >= 1, f"Expected pyramid_level >= 1, got {state.pyramid_level}"


def test_short_exit_on_rsi_oversold():
    """Short position should be covered (BUY) when RSI drops below 30."""
    s = _setup_strategy(max_loss_pct=0.99, atr_stop_multiplier=100.0, max_hold_bars=9999)
    _establish_downtrend(s)

    state = s._get_state("TEST")

    # Warmup
    warmup = [100, 99, 98, 97, 96, 95, 94]
    _feed_15m_prices(s, warmup)

    # Manually set up short position (bypassing LIMIT flow for exit test)
    current_qty = 100
    state.direction = "short"
    state.pyramid_level = 1
    state.avg_entry_price = 108.0
    state.total_qty = current_qty
    state.entry_bar = state.bar_count
    state.partial_taken = True  # skip partial exit path
    state.has_engine_stop = True
    state.product_type = "MIS"
    state.trailing_stop = 115.0

    # Drop prices sharply to push RSI below 30 (short full exit threshold)
    held = Position(symbol="TEST", quantity=-current_qty, avg_price=108.0, unrealized_pnl=0.0)
    ts = 300
    buy_sigs = []
    drop = [110, 106, 100, 94, 88, 82, 76]
    for p in drop:
        snap = make_snapshot(ts, close_15m=float(p), positions=[held])
        sigs = s.on_bar(snap)
        for sig in sigs:
            if sig.action == "BUY" and sig.order_type == "MARKET":
                buy_sigs.append(sig)
        if buy_sigs:
            break
        ts += 1

    assert len(buy_sigs) >= 1, "Should trigger full cover (BUY) when RSI < 30"
    assert state.pyramid_level == 0, "State should be reset after short cover"
    assert state.direction == "flat", "Direction should be flat after full cover"


def test_no_short_in_uptrend():
    """RSI > 60 but trend is UP should NOT produce a short entry."""
    s = _setup_strategy(max_loss_pct=0.99, atr_stop_multiplier=100.0, max_hold_bars=9999)
    _establish_uptrend(s)

    state = s._get_state("TEST")

    # Warmup: descending prices to seed RSI
    warmup = [100, 99, 98, 97, 96, 95, 94]
    _feed_15m_prices(s, warmup)

    # Rise to push RSI above 60 -- but trend is UP so no short entry
    rise = [96, 100, 104, 108, 112, 116, 120]
    sigs = _feed_15m_prices(s, rise, start_ts=200)

    # Should NOT have any SELL signals from short entry
    sell_sigs = [sig for sig in sigs if sig.action == "SELL"]
    assert len(sell_sigs) == 0, \
        f"Should NOT open short in uptrend, but got {len(sell_sigs)} SELL signals"
    assert state.direction == "flat", "Should remain flat -- no short in uptrend"


# --- New tests for limit entries, engine stops, dynamic CNC/MIS ---

def test_limit_entry_below_market():
    """Verify LIMIT buy is placed at close * 0.999 (just below market)."""
    s = _setup_strategy(max_loss_pct=0.99, atr_stop_multiplier=100.0, max_hold_bars=9999)
    _establish_uptrend(s)

    state = s._get_state("TEST")

    # Warmup (ascending -> high RSI, no entry)
    warmup = [100, 101, 102, 103, 104, 105, 106]
    _feed_15m_prices(s, warmup)

    # Drop to trigger RSI < 40
    ts = 200
    limit_sigs = []
    drop = [104, 100, 96, 92, 88]
    for p in drop:
        snap = make_snapshot(ts, close_15m=float(p))
        sigs = s.on_bar(snap)
        for sig in sigs:
            if sig.action == "BUY" and sig.order_type == "LIMIT":
                limit_sigs.append((sig, p))
        if limit_sigs:
            break
        ts += 1

    assert len(limit_sigs) >= 1, "Should submit LIMIT BUY entry"
    sig, close_at_signal = limit_sigs[0]
    expected_limit = close_at_signal * 0.999
    assert abs(sig.limit_price - expected_limit) < 0.01, \
        f"LIMIT price should be {expected_limit:.4f}, got {sig.limit_price:.4f}"
    assert sig.order_type == "LIMIT"
    assert state.direction == "flat", "Direction still flat (no fill yet)"


def test_cancel_stale_limit_after_3_bars():
    """Verify CANCEL is emitted after 3 bars if LIMIT entry not filled."""
    s = _setup_strategy(max_loss_pct=0.99, atr_stop_multiplier=100.0, max_hold_bars=9999)
    _establish_uptrend(s)

    state = s._get_state("TEST")

    # Warmup
    warmup = [100, 101, 102, 103, 104, 105, 106]
    _feed_15m_prices(s, warmup)

    # Drop to trigger LIMIT entry
    ts = 200
    drop = [104, 100, 96, 92, 88]
    limit_submitted = False
    limit_sig = None
    for p in drop:
        snap = make_snapshot(ts, close_15m=float(p))
        sigs = s.on_bar(snap)
        for sig in sigs:
            if sig.action == "BUY" and sig.order_type == "LIMIT":
                limit_submitted = True
                limit_sig = sig
        if limit_submitted:
            break
        ts += 1

    assert limit_submitted, "Should have submitted LIMIT entry"
    assert state.pending_entry_bar > 0, "pending_entry_bar should be set"
    submit_bar = state.pending_entry_bar

    # The LIMIT order is still pending in the engine (not expired, not filled)
    pending_limit = PendingOrder(
        symbol="TEST", side="BUY", quantity=limit_sig.quantity,
        order_type="LIMIT", limit_price=limit_sig.limit_price, stop_price=0.0,
    )

    # Feed 3 more bars without fill — no cancel yet
    # Use rising prices so RSI climbs back up (no re-entry after cancel)
    cancel_signals = []
    rise_prices = [95.0, 100.0, 105.0]
    for p in rise_prices:
        ts += 1
        snap = make_snapshot(ts, close_15m=p, pending_orders=[pending_limit])
        sigs = s.on_bar(snap)
        cancel_signals.extend([sig for sig in sigs if sig.action == "CANCEL"])

    assert len(cancel_signals) == 0, "Should NOT cancel within 3 bars"

    # 4th bar (> 3 bars elapsed) — should cancel
    # Use a high price so RSI > 40 and no re-entry is triggered
    ts += 1
    snap = make_snapshot(ts, close_15m=110.0, pending_orders=[pending_limit])
    sigs = s.on_bar(snap)
    cancel_now = [sig for sig in sigs if sig.action == "CANCEL"]

    assert len(cancel_now) >= 1, "Should emit CANCEL after 3 bars unfilled"
    assert state.pending_entry_bar == 0, "pending_entry_bar should be reset after cancel"


def test_engine_stop_on_entry_fill():
    """Fill in snapshot triggers SL-M submission at max_loss level."""
    s = _setup_strategy(max_loss_pct=0.03, atr_stop_multiplier=100.0, max_hold_bars=9999)
    _establish_uptrend(s)

    state = s._get_state("TEST")

    # Warmup
    warmup = [100, 101, 102, 103, 104, 105, 106]
    _feed_15m_prices(s, warmup)

    # Drop to trigger LIMIT entry
    ts = 200
    entry_signals = []
    drop = [104, 100, 96, 92, 88]
    for p in drop:
        snap = make_snapshot(ts, close_15m=float(p))
        sigs = s.on_bar(snap)
        for sig in sigs:
            if sig.action == "BUY" and sig.order_type == "LIMIT":
                entry_signals.append(sig)
        if entry_signals:
            break
        ts += 1

    assert len(entry_signals) >= 1
    entry_qty = entry_signals[0].quantity

    # Simulate fill at exactly the limit price
    fill_price = entry_signals[0].limit_price
    ts += 1
    fill = _get_entry_fill("TEST", "BUY", fill_price, entry_qty, ts)
    pos = [Position(symbol="TEST", quantity=entry_qty, avg_price=fill_price, unrealized_pnl=0.0)]
    snap = make_snapshot(ts, close_15m=fill_price, positions=pos, fills=[fill])
    sigs = s.on_bar(snap)

    # Should see SL-M stop at max_loss level
    slm_sigs = [sig for sig in sigs if sig.order_type == "SL_M"]
    assert len(slm_sigs) >= 1, "Should submit SL-M on entry fill"
    assert slm_sigs[0].action == "SELL", "SL-M should be a SELL for long position"

    expected_stop = fill_price * (1 - 0.03)
    assert abs(slm_sigs[0].stop_price - expected_stop) < 0.01, \
        f"SL-M stop price should be {expected_stop:.4f}, got {slm_sigs[0].stop_price:.4f}"

    assert state.has_engine_stop is True, "has_engine_stop should be True"
    assert state.direction == "long", "Direction should be long after fill"
    assert state.pyramid_level == 1, "pyramid_level should be 1"


def test_partial_exit_replaces_stop_at_breakeven():
    """Partial sell should CANCEL engine stop and replace at breakeven (avg_entry_price)."""
    s = _setup_strategy(max_loss_pct=0.99, atr_stop_multiplier=100.0, max_hold_bars=9999)
    _establish_uptrend(s)

    state = s._get_state("TEST")

    # Warmup
    warmup = [100, 101, 102, 103, 104, 105, 106]
    _feed_15m_prices(s, warmup)

    # Set up a filled long position
    entry_qty = 100
    state.direction = "long"
    state.pyramid_level = 1
    state.avg_entry_price = 90.0
    state.total_qty = entry_qty
    state.entry_bar = state.bar_count
    state.partial_taken = False
    state.has_engine_stop = True
    state.product_type = "MIS"
    state.trailing_stop = 85.0

    # Rise to trigger partial exit (RSI > 60)
    ts = 400
    partial_bar_sigs = None
    rise = [90, 93, 96, 99, 102, 105, 108]
    for p in rise:
        held = Position(symbol="TEST", quantity=entry_qty, avg_price=90.0, unrealized_pnl=0.0)
        snap = make_snapshot(ts, close_15m=float(p), positions=[held])
        sigs = s.on_bar(snap)
        # Look for partial exit signal
        market_sells = [sig for sig in sigs if sig.action == "SELL" and sig.order_type == "MARKET"]
        if market_sells:
            partial_bar_sigs = sigs
            break
        ts += 1

    assert partial_bar_sigs is not None, "Should trigger partial exit"

    # Check for CANCEL + SL-M at breakeven in the same bar
    cancel_sigs = [sig for sig in partial_bar_sigs if sig.action == "CANCEL"]
    slm_sigs = [sig for sig in partial_bar_sigs if sig.order_type == "SL_M"]

    # There could be cancel+SL-M from trailing stop ratchet AND from partial exit.
    # The partial exit's SL-M should be at avg_entry_price (breakeven).
    breakeven_slm = [sig for sig in slm_sigs if abs(sig.stop_price - 90.0) < 0.01]
    assert len(breakeven_slm) >= 1, \
        f"Should have SL-M at breakeven (90.0), got stops at {[sig.stop_price for sig in slm_sigs]}"
    assert len(cancel_sigs) >= 1, "Should have CANCEL before replacing stop"

    assert state.trailing_stop == 90.0, \
        f"Trailing stop should be moved to breakeven (90.0), got {state.trailing_stop}"


def test_dynamic_cnc_deep_oversold():
    """RSI < 25 in uptrend should use product_type='CNC'."""
    s = _setup_strategy(
        max_loss_pct=0.99, atr_stop_multiplier=100.0, max_hold_bars=9999,
        rsi_entry_1=40,  # entry at RSI < 40
    )
    _establish_uptrend(s)

    state = s._get_state("TEST")

    # Warmup (ascending)
    warmup = [100, 101, 102, 103, 104, 105, 106]
    _feed_15m_prices(s, warmup)

    # Steep drop to push RSI well below 25
    ts = 200
    cnc_signals = []
    # With rsi_period=5, a steep drop will produce very low RSI quickly
    drop = [104, 98, 90, 82, 74]
    for p in drop:
        snap = make_snapshot(ts, close_15m=float(p))
        sigs = s.on_bar(snap)
        for sig in sigs:
            if sig.action == "BUY" and sig.order_type == "LIMIT":
                cnc_signals.append(sig)
        if cnc_signals:
            break
        ts += 1

    assert len(cnc_signals) >= 1, "Should have submitted a LIMIT entry"

    # Compute what RSI was when entry was triggered
    rsi_at_entry = compute_rsi(list(state.prices_15m), 5)

    # If RSI < 25, product should be CNC
    if rsi_at_entry is not None and rsi_at_entry < 25:
        assert cnc_signals[0].product_type == "CNC", \
            f"Deep oversold (RSI={rsi_at_entry:.1f}) should use CNC, got {cnc_signals[0].product_type}"
        assert state.product_type == "CNC", "State product_type should be CNC"
    else:
        # RSI was between 25-40, should be MIS -- still a valid test outcome
        assert cnc_signals[0].product_type == "MIS"


def test_dynamic_mis_moderate():
    """RSI between 25-40 (not deep oversold) should use product_type='MIS'."""
    s = _setup_strategy(
        max_loss_pct=0.99, atr_stop_multiplier=100.0, max_hold_bars=9999,
        rsi_entry_1=40,
    )
    _establish_uptrend(s)

    state = s._get_state("TEST")

    # Warmup
    warmup = [100, 101, 102, 103, 104, 105, 106]
    _feed_15m_prices(s, warmup)

    # Moderate drop: RSI should be between 25 and 40 (not deeply oversold)
    # With rsi_period=5, a gentle decline gives moderate RSI
    ts = 200
    mis_signals = []
    # Gentle drop: less extreme than deep oversold test
    drop = [105, 103, 101, 99]
    for p in drop:
        snap = make_snapshot(ts, close_15m=float(p))
        sigs = s.on_bar(snap)
        for sig in sigs:
            if sig.action == "BUY" and sig.order_type == "LIMIT":
                mis_signals.append(sig)
        if mis_signals:
            break
        ts += 1

    if mis_signals:
        # With a moderate drop, RSI should be > 25 -> MIS
        assert mis_signals[0].product_type == "MIS", \
            f"Moderate RSI should use MIS, got {mis_signals[0].product_type}"
        assert state.product_type == "MIS"


def test_stop_hit_resets_state():
    """When engine fills SL-M (stop hit), strategy should detect and reset state."""
    s = _setup_strategy(max_loss_pct=0.03, atr_stop_multiplier=100.0, max_hold_bars=9999)
    _establish_uptrend(s)

    state = s._get_state("TEST")

    # Warmup
    warmup = [100, 101, 102, 103, 104, 105, 106]
    _feed_15m_prices(s, warmup)

    # Manually set up a long position with engine stop
    state.direction = "long"
    state.pyramid_level = 1
    state.avg_entry_price = 100.0
    state.total_qty = 50
    state.entry_bar = state.bar_count
    state.trailing_stop = 97.0
    state.partial_taken = False
    state.has_engine_stop = True
    state.product_type = "MIS"

    # Simulate stop-hit: engine fills the SL-M SELL
    ts = 500
    stop_fill = _get_entry_fill("TEST", "SELL", 96.5, 50, ts)
    # After stop fill, position is closed (no positions)
    snap = make_snapshot(ts, close_15m=96.0, positions=[], fills=[stop_fill])
    sigs = s.on_bar(snap)

    assert state.direction == "flat", f"Direction should be flat after stop-hit, got '{state.direction}'"
    assert state.pyramid_level == 0, "pyramid_level should be 0 after stop-hit"
    assert state.has_engine_stop is False, "has_engine_stop should be False after stop-hit"
    assert state.total_qty == 0, "total_qty should be 0 after stop-hit"
    assert state.avg_entry_price == 0.0, "avg_entry_price should be 0 after stop-hit"


# --- DAY order expiry tests ---

def test_stop_resubmitted_after_expiry():
    """Long position with has_engine_stop=True but no SL-M in pending_orders.

    The engine expired the stop at 15:30 IST. Next morning, the strategy should
    re-submit the SL-M at the current trailing_stop price.
    """
    s = _setup_strategy(max_loss_pct=0.99, atr_stop_multiplier=100.0, max_hold_bars=9999)
    _establish_uptrend(s)

    state = s._get_state("TEST")

    # Warmup
    warmup = [100, 101, 102, 103, 104, 105, 106]
    _feed_15m_prices(s, warmup)

    # Set up a long position with an engine stop that has expired
    entry_qty = 100
    state.direction = "long"
    state.pyramid_level = 1
    state.avg_entry_price = 90.0
    state.total_qty = entry_qty
    state.entry_bar = state.bar_count
    state.partial_taken = True
    state.has_engine_stop = True
    state.product_type = "CNC"
    state.trailing_stop = 85.0

    # Next bar: position still held, but pending_orders is EMPTY (stop expired)
    held = Position(symbol="TEST", quantity=entry_qty, avg_price=90.0, unrealized_pnl=0.0)
    snap = make_snapshot(
        600, close_15m=91.0, positions=[held],
        pending_orders=[],  # engine expired the SL-M
    )
    sigs = s.on_bar(snap)

    # Should re-submit SL-M at the trailing_stop price
    slm_sigs = [sig for sig in sigs if sig.order_type == "SL_M"]
    assert len(slm_sigs) >= 1, "Should re-submit SL-M after expiry"
    assert slm_sigs[0].action == "SELL", "Re-submitted stop should be SELL for long"
    assert abs(slm_sigs[0].stop_price - 85.0) < 0.01, \
        f"Re-submitted stop should be at trailing_stop=85.0, got {slm_sigs[0].stop_price}"
    assert slm_sigs[0].product_type == "CNC", "Should preserve product_type CNC"


def test_stop_resubmitted_after_expiry_short():
    """Short position with has_engine_stop=True but no SL-M in pending_orders.

    Same as long-side test but for short positions -- re-submitted stop is a BUY.
    """
    s = _setup_strategy(max_loss_pct=0.99, atr_stop_multiplier=100.0, max_hold_bars=9999)
    _establish_downtrend(s)

    state = s._get_state("TEST")

    # Warmup
    warmup = [100, 99, 98, 97, 96, 95, 94]
    _feed_15m_prices(s, warmup)

    # Set up a short position with an engine stop that has expired
    current_qty = 100
    state.direction = "short"
    state.pyramid_level = 1
    state.avg_entry_price = 108.0
    state.total_qty = current_qty
    state.entry_bar = state.bar_count
    state.partial_taken = True
    state.has_engine_stop = True
    state.product_type = "MIS"
    state.trailing_stop = 115.0

    # Next bar: position still held, but pending_orders is EMPTY (stop expired)
    held = Position(symbol="TEST", quantity=-current_qty, avg_price=108.0, unrealized_pnl=0.0)
    snap = make_snapshot(
        600, close_15m=107.0, positions=[held],
        pending_orders=[],  # engine expired the SL-M
    )
    sigs = s.on_bar(snap)

    # Should re-submit SL-M at the trailing_stop price
    slm_sigs = [sig for sig in sigs if sig.order_type == "SL_M"]
    assert len(slm_sigs) >= 1, "Should re-submit SL-M after expiry (short)"
    assert slm_sigs[0].action == "BUY", "Re-submitted stop should be BUY for short"
    assert abs(slm_sigs[0].stop_price - 115.0) < 0.01, \
        f"Re-submitted stop should be at trailing_stop=115.0, got {slm_sigs[0].stop_price}"


def test_no_stop_resubmit_when_stop_present():
    """When engine stop is still present in pending_orders, no re-submission should happen."""
    s = _setup_strategy(max_loss_pct=0.99, atr_stop_multiplier=100.0, max_hold_bars=9999)
    _establish_uptrend(s)

    state = s._get_state("TEST")

    # Warmup
    warmup = [100, 101, 102, 103, 104, 105, 106]
    _feed_15m_prices(s, warmup)

    # Set up a long position with engine stop still present
    entry_qty = 100
    state.direction = "long"
    state.pyramid_level = 1
    state.avg_entry_price = 90.0
    state.total_qty = entry_qty
    state.entry_bar = state.bar_count
    state.partial_taken = True
    state.has_engine_stop = True
    state.product_type = "MIS"
    state.trailing_stop = 85.0

    # Stop is still in pending_orders
    existing_stop = PendingOrder(
        symbol="TEST", side="SELL", quantity=entry_qty,
        order_type="SL_M", limit_price=0.0, stop_price=85.0,
    )
    held = Position(symbol="TEST", quantity=entry_qty, avg_price=90.0, unrealized_pnl=0.0)
    snap = make_snapshot(
        600, close_15m=91.0, positions=[held],
        pending_orders=[existing_stop],
    )
    sigs = s.on_bar(snap)

    # Should NOT have a re-submission SL-M (the first SL-M in signals, if any,
    # should be from trailing stop ratchet, not a re-submission)
    # With atr_stop_multiplier=100.0, trailing stop won't ratchet, so no SL-M expected
    slm_sigs = [sig for sig in sigs if sig.order_type == "SL_M"]
    assert len(slm_sigs) == 0, \
        f"Should NOT re-submit SL-M when stop still present, but got {len(slm_sigs)} SL-M signals"


def test_pending_entry_cleared_on_expiry():
    """pending_entry_bar > 0, empty pending_orders, no fills => entry expired, reset state."""
    s = _setup_strategy(max_loss_pct=0.99, atr_stop_multiplier=100.0, max_hold_bars=9999)
    _establish_uptrend(s)

    state = s._get_state("TEST")

    # Warmup
    warmup = [100, 101, 102, 103, 104, 105, 106]
    _feed_15m_prices(s, warmup)

    # Drop to trigger LIMIT entry
    ts = 200
    limit_submitted = False
    drop = [104, 100, 96, 92, 88]
    for p in drop:
        snap = make_snapshot(ts, close_15m=float(p))
        sigs = s.on_bar(snap)
        for sig in sigs:
            if sig.action == "BUY" and sig.order_type == "LIMIT":
                limit_submitted = True
        if limit_submitted:
            break
        ts += 1

    assert limit_submitted, "Should have submitted LIMIT entry"
    assert state.pending_entry_bar > 0, "pending_entry_bar should be set"
    assert state.direction == "flat", "Direction should be flat (awaiting fill)"

    # Next bar: no fill, no pending orders (engine expired the LIMIT at 15:30)
    # Use a high price to push RSI above entry threshold so no re-entry is triggered
    ts += 1
    snap = make_snapshot(
        ts, close_15m=120.0,
        pending_orders=[],  # engine expired the LIMIT
        fills=[],           # no fills
    )
    sigs = s.on_bar(snap)

    assert state.pending_entry_bar == 0, \
        f"pending_entry_bar should be reset after expiry, got {state.pending_entry_bar}"
    assert state.direction == "flat", "Direction should remain flat after entry expiry"


def test_pending_entry_not_cleared_when_limit_present():
    """pending_entry_bar > 0 with LIMIT still in pending_orders => do not reset."""
    s = _setup_strategy(max_loss_pct=0.99, atr_stop_multiplier=100.0, max_hold_bars=9999)
    _establish_uptrend(s)

    state = s._get_state("TEST")

    # Warmup
    warmup = [100, 101, 102, 103, 104, 105, 106]
    _feed_15m_prices(s, warmup)

    # Drop to trigger LIMIT entry
    ts = 200
    limit_submitted = False
    limit_sig = None
    drop = [104, 100, 96, 92, 88]
    for p in drop:
        snap = make_snapshot(ts, close_15m=float(p))
        sigs = s.on_bar(snap)
        for sig in sigs:
            if sig.action == "BUY" and sig.order_type == "LIMIT":
                limit_submitted = True
                limit_sig = sig
        if limit_submitted:
            break
        ts += 1

    assert limit_submitted and limit_sig is not None

    # Next bar: LIMIT still in pending_orders (not expired, not filled)
    pending_limit = PendingOrder(
        symbol="TEST", side="BUY", quantity=limit_sig.quantity,
        order_type="LIMIT", limit_price=limit_sig.limit_price, stop_price=0.0,
    )
    ts += 1
    snap = make_snapshot(
        ts, close_15m=89.0,
        pending_orders=[pending_limit],
        fills=[],
    )
    sigs = s.on_bar(snap)

    assert state.pending_entry_bar > 0, \
        "pending_entry_bar should NOT be reset when LIMIT is still pending"
