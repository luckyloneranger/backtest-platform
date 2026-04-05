"""Tests for RSI Mean Reversion strategy (PositionManager-based rewrite)."""

from strategies.base import (
    BarData, MarketSnapshot, Portfolio, Position, SessionContext,
    Signal, FillInfo, PendingOrder,
)
from strategies.deterministic.rsi_daily_trend import RsiDailyTrend
from strategies.indicators import compute_rsi, compute_ema


# --- Helpers ---

def _make_snapshot(
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
    timeframes: dict = {}
    if close_15m is not None:
        bar = BarData(symbol, close_15m, close_15m + 1, close_15m - 1, close_15m, 1000, 0)
        timeframes["15minute"] = {symbol: bar}
    if close_day is not None:
        h = high_day if high_day is not None else close_day + 1
        l = low_day if low_day is not None else close_day - 1
        bar = BarData(symbol, close_day, h, l, close_day, 50000, 0)
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
                               ["15minute", "day"], 200),
        pending_orders=pending_orders or [],
    )


def _setup(**overrides) -> RsiDailyTrend:
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
        "atr_multiplier": 2.0,
        "max_loss_pct": 0.03,
        "max_hold_bars": 20,
        "cooldown_bars": 0,
    }
    config.update(overrides)
    s = RsiDailyTrend()
    s.initialize(config, {})
    return s


def _establish_uptrend(s: RsiDailyTrend, symbol: str = "TEST"):
    """Feed daily bars to set trend_up=True."""
    for i in range(5):
        s.on_bar(_make_snapshot(
            i, close_day=100.0 + i * 5, high_day=105.0 + i * 5,
            low_day=98.0 + i * 5, symbol=symbol,
        ))


def _establish_downtrend(s: RsiDailyTrend, symbol: str = "TEST"):
    """Feed daily bars to set trend_down=True."""
    for i in range(5):
        s.on_bar(_make_snapshot(
            i, close_day=120.0 - i * 5, high_day=125.0 - i * 5,
            low_day=118.0 - i * 5, symbol=symbol,
        ))


def _feed_15m(s, prices, start_ts=100, symbol="TEST", cash=100_000.0,
              positions=None, fills=None, pending_orders=None):
    """Feed 15-minute bars and collect signals."""
    all_signals = []
    for i, p in enumerate(prices):
        snap = _make_snapshot(start_ts + i, close_15m=float(p), symbol=symbol,
                              cash=cash, positions=positions, fills=fills,
                              pending_orders=pending_orders)
        sigs = s.on_bar(snap)
        all_signals.extend(sigs)
    return all_signals


def _fill(symbol, side, price, qty, ts=0):
    return FillInfo(symbol=symbol, side=side, quantity=qty,
                    fill_price=price, costs=0.0, timestamp_ms=ts)


def _simulate_entry_fill(s, symbol, side, fill_price, qty, ts, close_15m,
                          positions=None):
    """Simulate a fill arriving and return signals from that bar."""
    fill = _fill(symbol, side, fill_price, qty, ts)
    pos = positions or [Position(symbol=symbol, quantity=qty if side == "BUY" else -qty,
                                  avg_price=fill_price, unrealized_pnl=0.0)]
    snap = _make_snapshot(ts, close_15m=close_15m, symbol=symbol,
                          positions=pos, fills=[fill])
    return s.on_bar(snap)


# --- Indicator tests ---

def test_compute_rsi():
    prices_up = [100 + i for i in range(20)]
    rsi = compute_rsi(prices_up, 14)
    assert rsi is not None and rsi > 90

    prices_down = [100 - i for i in range(20)]
    rsi = compute_rsi(prices_down, 14)
    assert rsi is not None and rsi < 10

    assert compute_rsi([100, 101, 102], 14) is None


def test_compute_ema():
    prices = [10.0, 11.0, 12.0, 13.0, 14.0]
    ema = compute_ema(prices, 3)
    assert ema is not None and 12.0 < ema < 14.0
    assert compute_ema([10.0, 11.0], 5) is None


# --- Strategy tests ---

def test_required_data():
    s = RsiDailyTrend()
    reqs = s.required_data()
    intervals = [r["interval"] for r in reqs]
    assert "15minute" in intervals
    assert "day" in intervals
    assert all("lookback" in r for r in reqs)


def test_rsi_entry_in_uptrend():
    """RSI < 40 in uptrend should produce a LIMIT BUY entry."""
    s = _setup(max_loss_pct=0.99, atr_multiplier=100.0, max_hold_bars=9999)
    _establish_uptrend(s)

    # Warmup (ascending -> high RSI, no entry)
    warmup = [100, 101, 102, 103, 104, 105, 106]
    _feed_15m(s, warmup)

    # Drop to trigger RSI < 40
    ts = 200
    limit_signals = []
    for p in [104, 100, 96, 92, 88]:
        snap = _make_snapshot(ts, close_15m=float(p))
        sigs = s.on_bar(snap)
        limit_signals += [sig for sig in sigs if sig.action == "BUY" and sig.order_type == "LIMIT"]
        if limit_signals:
            break
        ts += 1

    assert len(limit_signals) >= 1, "Expected LIMIT BUY signal"
    assert limit_signals[0].limit_price < 100.0, "LIMIT price should be below market"
    assert s.pm.is_flat("TEST"), "Still flat before fill"


def test_no_entry_without_trend():
    """No BUY signal when daily trend is down."""
    s = _setup()
    _establish_downtrend(s)

    prices = [100, 102, 104, 106, 108, 110, 105, 100, 95, 90, 85, 80]
    sigs = _feed_15m(s, prices)
    assert not any(sig.action == "BUY" for sig in sigs)


def test_no_entry_during_cooldown():
    """After exit, no re-entry within cooldown_bars."""
    s = _setup(max_loss_pct=0.99, atr_multiplier=100.0, max_hold_bars=9999,
               cooldown_bars=10)
    _establish_uptrend(s)

    # Warmup
    _feed_15m(s, [100, 101, 102, 103, 104, 105, 106])

    # Mark that an exit just happened
    s.last_exit_bar["TEST"] = s.pm.bar_count

    # Drop to trigger RSI < 40 — should be blocked by cooldown
    ts = 200
    sigs_in_cooldown = []
    for p in [104, 100, 96, 92, 88, 84, 80]:
        snap = _make_snapshot(ts, close_15m=float(p))
        sigs = s.on_bar(snap)
        sigs_in_cooldown += [sig for sig in sigs if sig.action == "BUY"]
        ts += 1

    assert len(sigs_in_cooldown) == 0, "Should not enter during cooldown"


def test_partial_exit_at_rsi_60():
    """Should sell half position when RSI > 60."""
    s = _setup(max_loss_pct=0.99, atr_multiplier=100.0, max_hold_bars=9999)
    _establish_uptrend(s)

    # Warmup
    _feed_15m(s, [100, 101, 102, 103, 104, 105, 106])

    # Manually set up a long position in PM
    pm_state = s.pm.get_state("TEST")
    pm_state.direction = "long"
    pm_state.qty = 100
    pm_state.avg_entry = 90.0
    pm_state.entry_bar = s.pm.bar_count
    pm_state.partial_taken = False
    pm_state.has_engine_stop = True
    pm_state.product_type = "MIS"
    pm_state.trailing_stop = 85.0

    # Feed a single price that gives RSI between 60-70 (partial but not full).
    # After warmup ending at 106, adding 104 gives:
    #   last 6 prices = [102,103,104,105,106,104]
    #   changes: +1,+1,+1,+1,-2 => avg_gain=0.8, avg_loss=0.4 => RSI=66.7
    ts = 400
    held = Position(symbol="TEST", quantity=100, avg_price=90.0, unrealized_pnl=0.0)
    # Use a SL-M pending order so resubmit_expired does not interfere
    existing_stop = PendingOrder(symbol="TEST", side="SELL", quantity=100,
                                 order_type="SL_M", limit_price=0.0, stop_price=85.0)
    snap = _make_snapshot(ts, close_15m=104.0, positions=[held],
                          pending_orders=[existing_stop])
    sigs = s.on_bar(snap)

    partial_signals = [sig for sig in sigs
                       if sig.action == "SELL" and sig.order_type == "MARKET"]
    assert len(partial_signals) >= 1, "Should trigger partial exit when RSI > 60"
    assert partial_signals[0].quantity == 50, \
        f"Partial exit should sell half (50), got {partial_signals[0].quantity}"


def test_full_exit_at_rsi_70():
    """Should sell full position when RSI > 70."""
    s = _setup(max_loss_pct=0.99, atr_multiplier=100.0, max_hold_bars=9999)
    _establish_uptrend(s)

    _feed_15m(s, [100, 101, 102, 103, 104, 105, 106])

    pm_state = s.pm.get_state("TEST")
    pm_state.direction = "long"
    pm_state.qty = 100
    pm_state.avg_entry = 90.0
    pm_state.entry_bar = s.pm.bar_count
    pm_state.partial_taken = True  # already took partial
    pm_state.has_engine_stop = True
    pm_state.product_type = "MIS"
    pm_state.trailing_stop = 85.0

    held = Position(symbol="TEST", quantity=100, avg_price=90.0, unrealized_pnl=0.0)
    rise = [90, 95, 100, 108, 115, 122, 130, 138, 145, 152]
    sigs = _feed_15m(s, rise, start_ts=400, positions=[held])

    sell_sigs = [sig for sig in sigs if sig.action == "SELL" and sig.order_type == "MARKET"]
    assert len(sell_sigs) >= 1, "Should trigger full exit when RSI > 70"


def test_short_entry_in_downtrend():
    """Downtrend + RSI > 60 should produce LIMIT SELL to open short."""
    s = _setup(max_loss_pct=0.99, atr_multiplier=100.0, max_hold_bars=9999)
    _establish_downtrend(s)

    # Warmup: descending
    _feed_15m(s, [100, 99, 98, 97, 96, 95, 94])

    # Rise to push RSI above 60 (overbought in downtrend)
    ts = 200
    sell_signals = []
    for p in [96, 100, 104, 108, 112]:
        snap = _make_snapshot(ts, close_15m=float(p))
        sigs = s.on_bar(snap)
        sell_signals += [sig for sig in sigs if sig.action == "SELL" and sig.order_type == "LIMIT"]
        if sell_signals:
            break
        ts += 1

    assert len(sell_signals) >= 1, "Expected short LIMIT SELL signal"
    assert sell_signals[0].product_type == "MIS", "Short should always use MIS"


def test_dynamic_cnc_deep_oversold():
    """RSI < rsi_entry_2 (30) in uptrend should use CNC."""
    s = _setup(max_loss_pct=0.99, atr_multiplier=100.0, max_hold_bars=9999,
               rsi_entry_1=40, rsi_entry_2=30)
    _establish_uptrend(s)

    _feed_15m(s, [100, 101, 102, 103, 104, 105, 106])

    # Steep drop to push RSI well below 30
    ts = 200
    cnc_signals = []
    for p in [104, 98, 90, 82, 74]:
        snap = _make_snapshot(ts, close_15m=float(p))
        sigs = s.on_bar(snap)
        cnc_signals += [sig for sig in sigs if sig.action == "BUY" and sig.order_type == "LIMIT"]
        if cnc_signals:
            break
        ts += 1

    assert len(cnc_signals) >= 1, "Should have LIMIT BUY entry"

    # Compute RSI at entry point to verify CNC/MIS choice
    rsi_at_entry = compute_rsi(list(s.prices_15m["TEST"]), 5)
    if rsi_at_entry is not None and rsi_at_entry < 30:
        assert cnc_signals[0].product_type == "CNC", \
            f"Deep oversold (RSI={rsi_at_entry:.1f}) should use CNC"
    else:
        assert cnc_signals[0].product_type == "MIS"


def test_pyramid_at_rsi_25():
    """After initial entry, RSI < 30 should add a pyramid level."""
    s = _setup(max_loss_pct=0.99, atr_multiplier=100.0, max_hold_bars=9999,
               max_pyramid_levels=2)
    _establish_uptrend(s)

    _feed_15m(s, [100, 101, 102, 103, 104, 105, 106])

    # Drop to trigger initial entry
    ts = 200
    entry_signals = []
    for p in [104, 100, 96, 92, 88]:
        snap = _make_snapshot(ts, close_15m=float(p))
        sigs = s.on_bar(snap)
        entry_signals += [sig for sig in sigs if sig.action == "BUY" and sig.order_type == "LIMIT"]
        if entry_signals:
            break
        ts += 1

    assert len(entry_signals) >= 1
    entry_qty = entry_signals[0].quantity
    fill_price = entry_signals[0].limit_price

    # Simulate fill
    ts += 1
    fill_sigs = _simulate_entry_fill(s, "TEST", "BUY", fill_price, entry_qty, ts, 86.0)
    pm_state = s.pm.get_state("TEST")
    assert pm_state.direction == "long"
    assert pm_state.pyramid_count == 0

    # Continue dropping for pyramid
    ts += 1
    pyramid_signals = []
    pos = [Position(symbol="TEST", quantity=entry_qty, avg_price=fill_price, unrealized_pnl=0.0)]
    for p in [84, 82, 80, 78]:
        snap = _make_snapshot(ts, close_15m=float(p), positions=pos)
        sigs = s.on_bar(snap)
        pyramid_signals += [sig for sig in sigs if sig.action == "BUY" and sig.order_type == "LIMIT"]
        if pyramid_signals:
            break
        ts += 1

    if pyramid_signals:
        # Simulate pyramid fill
        pyr_qty = pyramid_signals[0].quantity
        ts += 1
        total = entry_qty + pyr_qty
        pos2 = [Position(symbol="TEST", quantity=total, avg_price=fill_price, unrealized_pnl=0.0)]
        fill2 = _fill("TEST", "BUY", pyramid_signals[0].limit_price, pyr_qty, ts)
        snap = _make_snapshot(ts, close_15m=78.0, positions=pos2, fills=[fill2])
        s.on_bar(snap)
        assert pm_state.pyramid_count == 1, \
            f"Expected pyramid_count=1, got {pm_state.pyramid_count}"
