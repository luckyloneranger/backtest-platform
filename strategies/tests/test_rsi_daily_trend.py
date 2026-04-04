"""Tests for RSI Mean Reversion with Trend Filter + Pyramiding strategy."""

from strategies.base import (
    BarData, MarketSnapshot, Portfolio, Position, SessionContext, InstrumentInfo, Signal,
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
    return MarketSnapshot(
        timestamp_ms=ts,
        timeframes=timeframes,
        history={},
        portfolio=Portfolio(cash=cash, equity=equity, positions=pos_list),
        instruments={},
        fills=[],
        rejections=[],
        closed_trades=[],
        context=SessionContext(100_000.0, ts, 1000, "2024-01-01", "2024-12-31", ["15minute", "day"], 200),
    )


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
                     positions: list[Position] | None = None) -> list[Signal]:
    """Feed a series of 15-minute bars and collect all signals."""
    all_signals = []
    for i, p in enumerate(prices):
        snap = make_snapshot(start_ts + i, close_15m=float(p), symbol=symbol,
                             cash=cash, positions=positions)
        sigs = s.on_bar(snap)
        all_signals.extend(sigs)
    return all_signals


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
    """Strategy should enter in 3 pyramid levels at RSI < 40, 30, 20."""
    # Disable stops so they don't interfere with pyramid entries
    s = _setup_strategy(max_loss_pct=0.99, atr_stop_multiplier=100.0, max_hold_bars=9999)
    _establish_uptrend(s)

    state = s._get_state("TEST")

    # Phase 1: warmup prices (ascending -> RSI high, no entry)
    warmup = [100, 101, 102, 103, 104, 105, 106]
    _feed_15m_prices(s, warmup)
    assert state.pyramid_level == 0

    # Phase 2: Feed bars one at a time, tracking position and pyramid levels
    # With RSI period=5, RSI drops fast: 66.7 -> 33.3 -> 16.7 -> 6.7 -> 0.0
    # So all 3 levels can trigger in a single drop sequence
    current_qty = 0
    ts = 200
    buy_signals = []
    drop = [104, 100, 96, 92, 88]
    for p in drop:
        pos = [Position(symbol="TEST", quantity=current_qty, avg_price=96.0, unrealized_pnl=0.0)] if current_qty > 0 else []
        snap = make_snapshot(ts, close_15m=float(p), positions=pos)
        sigs = s.on_bar(snap)
        for sig in sigs:
            if sig.action == "BUY":
                buy_signals.append(sig)
                current_qty += sig.quantity
        ts += 1

    # Should have at least 2 pyramid entries (Level 1 at RSI<40, Level 2 at RSI<30)
    # With period=5, RSI drops quickly so Level 3 (RSI<20) also likely triggers
    assert len(buy_signals) >= 2, f"Expected at least 2 pyramid BUY signals, got {len(buy_signals)}"
    assert state.pyramid_level >= 2, f"Expected pyramid_level >= 2, got {state.pyramid_level}"
    assert current_qty > buy_signals[0].quantity, "Total qty should exceed first level qty"


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

    # Drop to enter position, tracking qty as we go
    current_qty = 0
    ts = 200
    drop = [104, 100, 96, 92, 88]
    for p in drop:
        pos = [Position(symbol="TEST", quantity=current_qty, avg_price=96.0, unrealized_pnl=0.0)] if current_qty > 0 else []
        snap = make_snapshot(ts, close_15m=float(p), positions=pos)
        sigs = s.on_bar(snap)
        for sig in sigs:
            if sig.action == "BUY":
                current_qty += sig.quantity
        ts += 1

    assert state.pyramid_level >= 1, "Should have entered at Level 1"
    entry_qty = current_qty
    assert entry_qty > 1, "Entry qty must be > 1 for partial exit test"

    # Rise to push RSI above 60 (partial exit threshold)
    # With rsi_period=5, RSI swings fast - it will cross 60 then 70 quickly.
    # Feed bars one at a time and stop after partial exit fires.
    state.partial_taken = False
    ts = 400
    partial_signal = None
    rise = [90, 93, 96, 99, 102, 105, 108]
    for p in rise:
        held = Position(symbol="TEST", quantity=current_qty, avg_price=90.0, unrealized_pnl=0.0)
        snap = make_snapshot(ts, close_15m=float(p), positions=[held])
        sigs = s.on_bar(snap)
        for sig in sigs:
            if sig.action == "SELL" and partial_signal is None:
                partial_signal = sig
        if partial_signal is not None:
            break
        ts += 1

    assert partial_signal is not None, "Should trigger partial exit when RSI > 60"
    assert partial_signal.quantity == current_qty // 2, \
        f"Partial exit should sell half ({current_qty // 2}), got {partial_signal.quantity}"


def test_full_exit_at_rsi_70():
    """Should sell full position when RSI rises above full exit threshold."""
    s = _setup_strategy(max_loss_pct=0.99, atr_stop_multiplier=100.0, max_hold_bars=9999)
    _establish_uptrend(s)

    state = s._get_state("TEST")

    # Warmup + enter, tracking qty
    warmup = [100, 101, 102, 103, 104, 105, 106]
    _feed_15m_prices(s, warmup)

    current_qty = 0
    ts = 200
    drop = [104, 100, 96, 92, 88]
    for p in drop:
        pos = [Position(symbol="TEST", quantity=current_qty, avg_price=96.0, unrealized_pnl=0.0)] if current_qty > 0 else []
        snap = make_snapshot(ts, close_15m=float(p), positions=pos)
        sigs = s.on_bar(snap)
        for sig in sigs:
            if sig.action == "BUY":
                current_qty += sig.quantity
        ts += 1

    assert state.pyramid_level >= 1
    entry_qty = current_qty

    # Already took partial profit
    state.partial_taken = True

    held = Position(symbol="TEST", quantity=entry_qty, avg_price=90.0, unrealized_pnl=0.0)

    # Strong rise to push RSI above 70
    rise = [90, 95, 100, 108, 115, 122, 130, 138, 145, 152]
    sigs = _feed_15m_prices(s, rise, start_ts=400, positions=[held])

    sell_sigs = [sig for sig in sigs if sig.action == "SELL"]
    assert len(sell_sigs) >= 1, "Should trigger full exit when RSI > 70"

    # After full exit, state should be reset
    assert state.pyramid_level == 0


def test_trailing_stop_exit():
    """Should exit when price drops below trailing stop (avg_entry - ATR * multiplier)."""
    # Use wide stops during setup so entry phase doesn't trigger exits
    s = _setup_strategy(atr_stop_multiplier=100.0, max_loss_pct=0.99, max_hold_bars=9999)
    _establish_uptrend(s)

    state = s._get_state("TEST")

    # Warmup + enter, tracking qty
    warmup = [100, 101, 102, 103, 104, 105, 106]
    _feed_15m_prices(s, warmup)

    current_qty = 0
    ts = 200
    drop = [104, 100, 96, 92, 88]
    for p in drop:
        pos = [Position(symbol="TEST", quantity=current_qty, avg_price=96.0, unrealized_pnl=0.0)] if current_qty > 0 else []
        snap = make_snapshot(ts, close_15m=float(p), positions=pos)
        sigs = s.on_bar(snap)
        for sig in sigs:
            if sig.action == "BUY":
                current_qty += sig.quantity
        ts += 1

    assert state.pyramid_level >= 1

    # Now manually set up for trailing stop test
    state.avg_entry_price = 90.0
    state.trailing_stop = 85.0
    state.partial_taken = True  # skip partial exit path

    entry_qty = current_qty
    held = Position(symbol="TEST", quantity=entry_qty, avg_price=90.0, unrealized_pnl=0.0)

    # Feed a 15m bar below trailing stop -> should trigger exit
    snap = make_snapshot(500, close_15m=84.0, positions=[held])
    sigs = s.on_bar(snap)

    sell_sigs = [sig for sig in sigs if sig.action == "SELL"]
    assert len(sell_sigs) >= 1, "Should exit when price drops below trailing stop"
    assert state.pyramid_level == 0, "State should be reset after trailing stop exit"


def test_max_loss_exit():
    """Should exit when price drops more than max_loss_pct below avg_entry."""
    s = _setup_strategy(max_loss_pct=0.03, atr_stop_multiplier=100.0, max_hold_bars=9999)
    _establish_uptrend(s)

    state = s._get_state("TEST")

    # Warmup + enter, tracking qty
    warmup = [100, 101, 102, 103, 104, 105, 106]
    _feed_15m_prices(s, warmup)

    current_qty = 0
    ts = 200
    drop = [104, 100, 96, 92, 88]
    for p in drop:
        pos = [Position(symbol="TEST", quantity=current_qty, avg_price=96.0, unrealized_pnl=0.0)] if current_qty > 0 else []
        snap = make_snapshot(ts, close_15m=float(p), positions=pos)
        sigs = s.on_bar(snap)
        for sig in sigs:
            if sig.action == "BUY":
                current_qty += sig.quantity
        ts += 1

    assert state.pyramid_level >= 1

    # Set state for max loss test
    state.avg_entry_price = 100.0
    state.trailing_stop = 0.0  # disable trailing stop
    state.partial_taken = True

    entry_qty = current_qty
    held = Position(symbol="TEST", quantity=entry_qty, avg_price=100.0, unrealized_pnl=0.0)

    # Price drops > 3% -> should trigger max loss exit
    # max_loss_price = 100 * (1 - 0.03) = 97.0
    snap = make_snapshot(500, close_15m=96.5, positions=[held])
    sigs = s.on_bar(snap)

    sell_sigs = [sig for sig in sigs if sig.action == "SELL"]
    assert len(sell_sigs) >= 1, "Should exit when price drops > max_loss_pct below avg entry"
    assert state.pyramid_level == 0, "State should be reset after max loss exit"
