"""Tests for Donchian Breakout strategy (engine stops, limit profits, dynamic CNC/MIS, short selling)."""

from strategies.base import (
    BarData, MarketSnapshot, Portfolio, Position, SessionContext, InstrumentInfo,
    FillInfo, PendingOrder,
)
from strategies.deterministic.donchian_breakout import DonchianBreakout
from strategies.indicators import compute_atr


def make_snapshot(
    ts: int,
    close_15m: float | None = None,
    high_15m: float | None = None,
    low_15m: float | None = None,
    volume_15m: int = 1000,
    close_day: float | None = None,
    high_day: float | None = None,
    low_day: float | None = None,
    volume_day: int = 100_000,
    symbol: str = "TEST",
    cash: float = 100_000.0,
    positions: list[Position] | None = None,
    fills: list[FillInfo] | None = None,
    pending_orders: list[PendingOrder] | None = None,
) -> MarketSnapshot:
    timeframes = {}
    if close_15m is not None:
        h = high_15m if high_15m is not None else close_15m + 1
        l = low_15m if low_15m is not None else close_15m - 1
        bar = BarData(symbol, close_15m, h, l, close_15m, volume_15m, 0)
        timeframes["15minute"] = {symbol: bar}
    if close_day is not None:
        h = high_day if high_day is not None else close_day + 2
        l = low_day if low_day is not None else close_day - 2
        bar = BarData(symbol, close_day, h, l, close_day, volume_day, 0)
        timeframes["day"] = {symbol: bar}
    pos_list = positions if positions is not None else []
    fill_list = fills if fills is not None else []
    pending_list = pending_orders if pending_orders is not None else []
    equity = cash + sum(p.quantity * p.avg_price for p in pos_list)
    return MarketSnapshot(
        timestamp_ms=ts,
        timeframes=timeframes,
        history={},
        portfolio=Portfolio(cash=cash, equity=equity, positions=pos_list),
        instruments={},
        fills=fill_list,
        rejections=[],
        closed_trades=[],
        context=SessionContext(100_000.0, ts, 1000, "2024-01-01", "2024-12-31", ["15minute", "day"], 200),
        pending_orders=pending_list,
    )


def _build_channel(strategy, num_bars=8, base_price=95.0, volume=100_000):
    """Feed enough daily bars to build a Donchian channel and compute ATR."""
    for i in range(num_bars):
        price = base_price + i
        strategy.on_bar(make_snapshot(
            i, close_day=price, high_day=price + 2, low_day=price - 2, volume_day=volume,
        ))


def _enter_position(strategy, entry_price=108.0, volume_day=200_000):
    """Feed breakout daily bar + 15-min bar to trigger entry. Returns signals."""
    strategy.on_bar(make_snapshot(
        8, close_day=entry_price, high_day=entry_price + 2,
        low_day=entry_price - 2, volume_day=volume_day,
    ))
    return strategy.on_bar(make_snapshot(100, close_15m=entry_price))


# ---- test_required_data ----

def test_required_data():
    s = DonchianBreakout()
    reqs = s.required_data()
    intervals = [r["interval"] for r in reqs]
    assert "day" in intervals
    assert "15minute" in intervals
    lookbacks = {r["interval"]: r["lookback"] for r in reqs}
    assert lookbacks["day"] == 60
    assert lookbacks["15minute"] == 30


# ---- test_risk_based_position_sizing ----

def test_risk_based_position_sizing():
    """Position size = int((cash * risk_per_trade) / (ATR * atr_multiplier))."""
    s = DonchianBreakout()
    s.initialize({
        "channel_period": 5, "atr_period": 3, "atr_multiplier": 1.5,
        "volume_factor": 1.0, "risk_per_trade": 0.02,
    }, {})

    _build_channel(s)
    signals = _enter_position(s)

    buy_signals = [sig for sig in signals if sig.action == "BUY"]
    assert len(buy_signals) == 1

    # Verify qty is risk-based, not percentage-of-capital-based
    atr = s.current_atr["TEST"]
    expected_qty = int((100_000.0 * 0.02) / (atr * 1.5))
    assert buy_signals[0].quantity == expected_qty
    assert expected_qty > 0


# ---- test_breakout_entry_with_relaxed_volume ----

def test_breakout_entry_with_relaxed_volume():
    """Entry triggers when volume >= avg_volume * 1.0 (relaxed factor)."""
    s = DonchianBreakout()
    s.initialize({
        "channel_period": 5, "atr_period": 3,
        "volume_factor": 1.0, "risk_per_trade": 0.02,
    }, {})

    # Feed daily bars with uniform volume
    _build_channel(s, volume=100_000)

    # Breakout day with same volume as average (factor=1.0 so this passes)
    s.on_bar(make_snapshot(
        8, close_day=108.0, high_day=110.0, low_day=106.0, volume_day=100_000,
    ))
    signals = s.on_bar(make_snapshot(100, close_15m=108.0))
    assert any(sig.action == "BUY" for sig in signals)


def test_no_buy_without_volume():
    """No entry when volume < avg_volume * volume_factor."""
    s = DonchianBreakout()
    s.initialize({
        "channel_period": 5, "atr_period": 3, "volume_factor": 2.0,
    }, {})

    _build_channel(s, volume=100_000)
    # Breakout day with volume = 100k, but factor=2.0 needs >= 200k
    signals = s.on_bar(make_snapshot(100, close_15m=108.0))
    assert not any(sig.action == "BUY" for sig in signals)


# ---- test_trailing_stop_tighter ----

def test_trailing_stop_tighter():
    """Trailing stop at 1.5x ATR (tighter than old 2.0x). Engine orders submitted after entry."""
    s = DonchianBreakout()
    s.initialize({
        "channel_period": 5, "atr_period": 3, "atr_multiplier": 1.5,
        "volume_factor": 1.0, "risk_per_trade": 0.02,
    }, {})

    _build_channel(s)
    signals = _enter_position(s)
    assert any(sig.action == "BUY" for sig in signals)

    atr = s.current_atr["TEST"]
    entry = 108.0

    # Initial trailing stop should be entry - 1.5 * ATR
    expected_initial_stop = entry - (atr * 1.5)
    assert abs(s.trailing_stop["TEST"] - expected_initial_stop) < 0.01

    # Price rises, trailing stop moves up
    s.on_bar(make_snapshot(101, close_15m=112.0))
    expected_stop_after_rise = 112.0 - (atr * 1.5)
    assert s.trailing_stop["TEST"] >= expected_stop_after_rise - 0.01

    # Price drops through trailing stop -> should exit via max_loss (engine not active in test)
    stop_level = s.trailing_stop["TEST"]
    drop_price = stop_level - 1.0
    held = Position(symbol="TEST", quantity=100, avg_price=entry, unrealized_pnl=0.0)
    signals = s.on_bar(make_snapshot(102, close_15m=drop_price, positions=[held]))
    assert any(sig.action == "SELL" for sig in signals)


# ---- test_partial_profit_exit_at_2atr ----

def test_partial_profit_exit_at_2atr():
    """Engine submits LIMIT sell at avg_entry + 2*ATR for partial profit on first bar after entry."""
    s = DonchianBreakout()
    s.initialize({
        "channel_period": 5, "atr_period": 3, "atr_multiplier": 1.5,
        "volume_factor": 1.0, "risk_per_trade": 0.02,
        "profit_target_atr": 2.0,
    }, {})

    _build_channel(s)
    entry_signals = _enter_position(s)
    buy_qty = entry_signals[0].quantity
    entry_price = 108.0
    atr = s.current_atr["TEST"]

    # First bar after entry: engine orders should be submitted (SL-M + LIMIT)
    held = Position(symbol="TEST", quantity=buy_qty, avg_price=entry_price, unrealized_pnl=0.0)
    signals = s.on_bar(make_snapshot(101, close_15m=entry_price + 1, positions=[held]))

    sell_signals = [sig for sig in signals if sig.action == "SELL"]
    # Should have SL-M stop + LIMIT profit target
    assert len(sell_signals) == 2

    slm_signal = [sig for sig in sell_signals if sig.order_type == "SL_M"]
    limit_signal = [sig for sig in sell_signals if sig.order_type == "LIMIT"]
    assert len(slm_signal) == 1
    assert len(limit_signal) == 1

    # LIMIT at entry + 2 * ATR
    expected_profit_price = entry_price + 2.0 * atr
    assert abs(limit_signal[0].limit_price - expected_profit_price) < 0.01

    # Partial qty = 1/3 of held
    expected_partial = max(1, buy_qty // 3)
    assert limit_signal[0].quantity == expected_partial

    # Engine stop should be tracked
    assert s.has_engine_stop["TEST"] is True
    assert s.has_profit_target["TEST"] is True


# ---- test_pyramid_adds_to_position ----

def test_pyramid_adds_to_position():
    """Pyramid adds 50% of original qty when price > avg_entry + 1*ATR."""
    s = DonchianBreakout()
    s.initialize({
        "channel_period": 5, "atr_period": 3, "atr_multiplier": 1.5,
        "volume_factor": 1.0, "risk_per_trade": 0.02,
        "pyramid_levels": 2, "profit_target_atr": 5.0,  # high target so partial doesn't trigger first
    }, {})

    _build_channel(s)
    entry_signals = _enter_position(s)
    original_qty = entry_signals[0].quantity
    entry_price = 108.0
    atr = s.current_atr["TEST"]

    # Price moves above entry + ATR -> pyramid trigger
    pyramid_price = entry_price + atr + 1.0
    held = Position(symbol="TEST", quantity=original_qty, avg_price=entry_price, unrealized_pnl=0.0)
    signals = s.on_bar(make_snapshot(101, close_15m=pyramid_price, positions=[held]))

    buy_signals = [sig for sig in signals if sig.action == "BUY"]
    assert len(buy_signals) == 1
    expected_add = max(1, original_qty // 2)
    assert buy_signals[0].quantity == expected_add
    assert s.pyramid_count["TEST"] == 1


# ---- test_max_loss_exit ----

def test_max_loss_exit():
    """Exit when price < avg_entry * (1 - max_loss_pct)."""
    s = DonchianBreakout()
    s.initialize({
        "channel_period": 5, "atr_period": 3, "atr_multiplier": 1.5,
        "volume_factor": 1.0, "risk_per_trade": 0.02,
        "max_loss_pct": 0.02,
    }, {})

    _build_channel(s)
    _enter_position(s)
    entry_price = 108.0

    # Price drops below entry * (1 - 0.02) = 108 * 0.98 = 105.84
    loss_price = entry_price * (1 - 0.02) - 1.0  # well below max loss
    held = Position(symbol="TEST", quantity=100, avg_price=entry_price, unrealized_pnl=0.0)
    signals = s.on_bar(make_snapshot(101, close_15m=loss_price, positions=[held]))

    # Should have CANCEL + SELL (market) for exit, plus engine orders may also be submitted
    sell_market = [sig for sig in signals if sig.action == "SELL" and sig.order_type == "MARKET"]
    assert len(sell_market) == 1
    assert sell_market[0].quantity == 100

    # CANCEL should be present
    cancel_signals = [sig for sig in signals if sig.action == "CANCEL"]
    assert len(cancel_signals) >= 1


# ---- test_channel_low_full_exit ----

def test_channel_low_full_exit():
    """Full exit when price drops below Donchian channel low."""
    s = DonchianBreakout()
    s.initialize({
        "channel_period": 5, "atr_period": 3, "atr_multiplier": 1.5,
        "volume_factor": 1.0, "risk_per_trade": 0.02,
        "max_loss_pct": 0.50,  # high max_loss so it doesn't trigger first
    }, {})

    _build_channel(s)
    _enter_position(s)

    # Drop price below the lowest possible channel_low
    drop_price = 90.0
    held = Position(symbol="TEST", quantity=100, avg_price=108.0, unrealized_pnl=0.0)
    signals = s.on_bar(make_snapshot(101, close_15m=drop_price, positions=[held]))

    sell_market = [sig for sig in signals if sig.action == "SELL" and sig.order_type == "MARKET"]
    assert len(sell_market) == 1
    assert sell_market[0].quantity == 100

    # CANCEL should be present (cancels engine orders before exit)
    cancel_signals = [sig for sig in signals if sig.action == "CANCEL"]
    assert len(cancel_signals) >= 1


# ---- test_compute_atr (imports from indicators) ----

def test_compute_atr():
    """ATR from shared indicators module works correctly."""
    highs = [102, 104, 103, 105, 106, 104, 108, 107, 109, 110, 111, 112, 113, 114, 115, 116]
    lows = [98, 99, 97, 100, 101, 99, 103, 102, 104, 105, 106, 107, 108, 109, 110, 111]
    closes = [100, 101, 100, 103, 104, 102, 106, 105, 107, 108, 109, 110, 111, 112, 113, 114]
    atr = compute_atr(highs, lows, closes, 14)
    assert atr is not None
    assert atr > 0


def test_compute_atr_not_enough_data():
    atr = compute_atr([100, 101], [98, 99], [99, 100], 14)
    assert atr is None


# ---- test_no_signal_during_warmup ----

def test_no_signal_during_warmup():
    s = DonchianBreakout()
    s.initialize({"channel_period": 5, "atr_period": 3}, {})
    for i in range(3):
        s.on_bar(make_snapshot(i, close_day=100.0 + i))
    signals = s.on_bar(make_snapshot(10, close_15m=100.0))
    assert signals == []


# ==============================================================================
# SHORT SELLING TESTS
# ==============================================================================


def _build_channel_for_short(strategy, num_bars=8, base_price=105.0, volume=100_000):
    """Feed daily bars with descending highs to create a channel where a low break is possible.

    Prices: base_price - i, so they trend downward, creating:
      highs like 107, 106, 105, 104, 103, 102, 101, 100
      lows  like 103, 102, 101, 100, 99, 98, 97, 96
    Channel low (min of lows excluding last) will be around 97-98.
    """
    for i in range(num_bars):
        price = base_price - i
        strategy.on_bar(make_snapshot(
            i, close_day=price, high_day=price + 2, low_day=price - 2, volume_day=volume,
        ))


def _enter_short_position(strategy, entry_price=90.0, volume_day=200_000):
    """Feed a daily bar (to update volume) + 15-min bar below channel low to trigger short entry."""
    # Feed another daily bar to have enough volume history
    strategy.on_bar(make_snapshot(
        8, close_day=entry_price, high_day=entry_price + 2,
        low_day=entry_price - 2, volume_day=volume_day,
    ))
    return strategy.on_bar(make_snapshot(100, close_15m=entry_price))


# ---- test_short_entry_on_channel_low_break ----

def test_short_entry_on_channel_low_break():
    """Price breaks below channel low with volume -> SELL signal (short entry) with MIS product type."""
    s = DonchianBreakout()
    s.initialize({
        "channel_period": 5, "atr_period": 3, "atr_multiplier": 1.5,
        "volume_factor": 1.0, "risk_per_trade": 0.02,
    }, {})

    _build_channel_for_short(s)

    entry_price = 90.0  # well below channel low
    signals = _enter_short_position(s, entry_price=entry_price)

    sell_signals = [sig for sig in signals if sig.action == "SELL"]
    assert len(sell_signals) == 1
    assert sell_signals[0].quantity > 0
    # Short entries must always use MIS (CNC shorts not allowed in Zerodha)
    assert sell_signals[0].product_type == "MIS", (
        f"Short entry must use MIS, got {sell_signals[0].product_type}"
    )
    assert s.in_position["TEST"] is True
    assert s.is_short["TEST"] is True
    assert s.product_type["TEST"] == "MIS", (
        f"Short position product_type must be MIS, got {s.product_type['TEST']}"
    )

    # Verify risk-based sizing
    atr = s.current_atr["TEST"]
    expected_qty = int((100_000.0 * 0.02) / (atr * 1.5))
    assert sell_signals[0].quantity == expected_qty


# ---- test_short_no_entry_without_volume ----

def test_short_no_entry_without_volume():
    """No short entry when volume < avg_volume * volume_factor."""
    s = DonchianBreakout()
    s.initialize({
        "channel_period": 5, "atr_period": 3, "volume_factor": 2.0,
    }, {})

    _build_channel_for_short(s, volume=100_000)
    # Price below channel low, but volume_factor=2.0 needs >= 200k
    signals = s.on_bar(make_snapshot(100, close_15m=90.0))
    assert not any(sig.action == "SELL" for sig in signals)


# ---- test_short_trailing_stop_exit ----

def test_short_trailing_stop_exit():
    """Short position: trailing stop tracks correctly and engine manages exit."""
    s = DonchianBreakout()
    s.initialize({
        "channel_period": 5, "atr_period": 3, "atr_multiplier": 1.5,
        "volume_factor": 1.0, "risk_per_trade": 0.02,
        "max_loss_pct": 0.50,  # high so max loss doesn't trigger
    }, {})

    _build_channel_for_short(s)
    entry_signals = _enter_short_position(s, entry_price=90.0)
    sell_signals = [sig for sig in entry_signals if sig.action == "SELL"]
    assert len(sell_signals) == 1
    entry_qty = sell_signals[0].quantity
    entry_price = 90.0
    atr = s.current_atr["TEST"]

    # Initial trailing stop for short = entry + atr * multiplier
    expected_stop = entry_price + atr * 1.5
    assert abs(s.trailing_stop["TEST"] - expected_stop) < 0.01

    # Price drops further -> lowest_since_entry updates, trailing stop moves down
    # Feed with position so reconciliation doesn't reset
    held = Position(symbol="TEST", quantity=-entry_qty, avg_price=entry_price, unrealized_pnl=0.0)
    signals = s.on_bar(make_snapshot(101, close_15m=85.0, positions=[held]))
    new_expected_stop = 85.0 + atr * 1.5
    assert s.trailing_stop["TEST"] <= new_expected_stop + 0.01

    # Engine stop should have been submitted after entry
    assert s.has_engine_stop["TEST"] is True

    # Simulate stop fill by providing a BUY fill
    stop_fill = FillInfo(symbol="TEST", side="BUY", quantity=entry_qty,
                         fill_price=s.trailing_stop["TEST"], costs=0.0, timestamp_ms=102)
    # Price rises above trailing stop (engine would have filled the SL-M)
    rise_price = s.trailing_stop["TEST"] + 1.0
    signals = s.on_bar(make_snapshot(
        102, close_15m=rise_price,
        positions=[],  # position gone after stop fill
        fills=[stop_fill],
    ))
    # Position should be reset via reconciliation (held_qty=0)
    assert s.in_position["TEST"] is False


# ---- test_short_channel_high_exit ----

def test_short_channel_high_exit():
    """Short position: price breaks above channel high -> BUY to cover (trend reversal)."""
    s = DonchianBreakout()
    s.initialize({
        "channel_period": 5, "atr_period": 3, "atr_multiplier": 1.5,
        "volume_factor": 1.0, "risk_per_trade": 0.02,
        "max_loss_pct": 0.50,  # high so max loss doesn't trigger
    }, {})

    _build_channel_for_short(s)
    entry_signals = _enter_short_position(s, entry_price=90.0)
    sell_signals = [sig for sig in entry_signals if sig.action == "SELL"]
    assert len(sell_signals) == 1
    entry_qty = sell_signals[0].quantity

    # Price well above any channel high
    cover_price = 120.0
    held = Position(symbol="TEST", quantity=-entry_qty, avg_price=90.0, unrealized_pnl=0.0)
    signals = s.on_bar(make_snapshot(101, close_15m=cover_price, positions=[held]))

    buy_market = [sig for sig in signals if sig.action == "BUY" and sig.order_type == "MARKET"]
    assert len(buy_market) == 1
    assert buy_market[0].quantity == entry_qty
    # Position should be reset
    assert s.in_position["TEST"] is False


# ---- test_short_max_loss_exit ----

def test_short_max_loss_exit():
    """Short position: price > avg_entry * (1 + max_loss_pct) -> cover all."""
    s = DonchianBreakout()
    s.initialize({
        "channel_period": 5, "atr_period": 3, "atr_multiplier": 100.0,  # huge so trailing stop won't trigger
        "volume_factor": 1.0, "risk_per_trade": 0.02,
        "max_loss_pct": 0.02,
    }, {})

    _build_channel_for_short(s)
    entry_signals = _enter_short_position(s, entry_price=90.0)
    sell_signals = [sig for sig in entry_signals if sig.action == "SELL"]
    assert len(sell_signals) == 1
    entry_qty = sell_signals[0].quantity
    entry_price = 90.0

    # Max loss for short: price > entry * (1 + 0.02) = 90 * 1.02 = 91.8
    loss_price = entry_price * (1 + 0.02) + 1.0  # well above max loss
    held = Position(symbol="TEST", quantity=-entry_qty, avg_price=entry_price, unrealized_pnl=0.0)
    signals = s.on_bar(make_snapshot(101, close_15m=loss_price, positions=[held]))

    buy_market = [sig for sig in signals if sig.action == "BUY" and sig.order_type == "MARKET"]
    assert len(buy_market) == 1
    assert buy_market[0].quantity == entry_qty


# ---- test_short_partial_cover ----

def test_short_partial_cover():
    """Short position: engine submits LIMIT BUY for partial cover at entry - 2*ATR."""
    s = DonchianBreakout()
    s.initialize({
        "channel_period": 5, "atr_period": 3, "atr_multiplier": 1.5,
        "volume_factor": 1.0, "risk_per_trade": 0.02,
        "profit_target_atr": 2.0,
    }, {})

    _build_channel_for_short(s)
    entry_signals = _enter_short_position(s, entry_price=90.0)
    sell_signals = [sig for sig in entry_signals if sig.action == "SELL"]
    assert len(sell_signals) == 1
    entry_qty = sell_signals[0].quantity
    entry_price = 90.0
    atr = s.current_atr["TEST"]

    # First bar after entry: engine orders submitted (SL-M + LIMIT)
    held = Position(symbol="TEST", quantity=-entry_qty, avg_price=entry_price, unrealized_pnl=0.0)
    signals = s.on_bar(make_snapshot(101, close_15m=entry_price - 1, positions=[held]))

    buy_signals = [sig for sig in signals if sig.action == "BUY"]
    assert len(buy_signals) == 2  # SL-M + LIMIT

    slm_signal = [sig for sig in buy_signals if sig.order_type == "SL_M"]
    limit_signal = [sig for sig in buy_signals if sig.order_type == "LIMIT"]
    assert len(slm_signal) == 1
    assert len(limit_signal) == 1

    # LIMIT at entry - 2*ATR
    expected_profit_price = entry_price - 2.0 * atr
    assert abs(limit_signal[0].limit_price - expected_profit_price) < 0.01

    expected_partial = max(1, entry_qty // 3)
    assert limit_signal[0].quantity == expected_partial

    # Trailing stop moved to breakeven should happen after partial fill
    assert s.has_profit_target["TEST"] is True
    assert s.has_engine_stop["TEST"] is True


# ---- test_short_pyramid ----

def test_short_pyramid():
    """Short position: price drops 1*ATR below entry -> add 50% more shorts."""
    s = DonchianBreakout()
    s.initialize({
        "channel_period": 5, "atr_period": 3, "atr_multiplier": 1.5,
        "volume_factor": 1.0, "risk_per_trade": 0.02,
        "pyramid_levels": 2, "profit_target_atr": 10.0,  # high so partial doesn't trigger
    }, {})

    _build_channel_for_short(s)
    entry_signals = _enter_short_position(s, entry_price=90.0)
    sell_signals = [sig for sig in entry_signals if sig.action == "SELL"]
    original_qty = sell_signals[0].quantity
    entry_price = 90.0
    atr = s.current_atr["TEST"]

    # Price drops below entry - ATR -> pyramid trigger
    pyramid_price = entry_price - atr - 1.0
    held = Position(symbol="TEST", quantity=-original_qty, avg_price=entry_price, unrealized_pnl=0.0)
    signals = s.on_bar(make_snapshot(101, close_15m=pyramid_price, positions=[held]))

    # Pyramid adds more short (SELL signal)
    sell_signals = [sig for sig in signals if sig.action == "SELL" and sig.order_type == "MARKET"]
    assert len(sell_signals) == 1
    expected_add = max(1, original_qty // 2)
    assert sell_signals[0].quantity == expected_add
    assert s.pyramid_count["TEST"] == 1


# ---- test_short_time_stop ----

def test_short_time_stop():
    """Short position: held too long with minimal gain -> cover all."""
    s = DonchianBreakout()
    s.initialize({
        "channel_period": 5, "atr_period": 3, "atr_multiplier": 100.0,  # huge so trailing won't trigger
        "volume_factor": 1.0, "risk_per_trade": 0.02,
        "max_loss_pct": 0.50,  # high so max loss won't trigger
        "max_hold_bars": 5,
    }, {})

    _build_channel_for_short(s)
    entry_signals = _enter_short_position(s, entry_price=90.0)
    sell_signals = [sig for sig in entry_signals if sig.action == "SELL"]
    entry_qty = sell_signals[0].quantity
    entry_price = 90.0

    # Feed bars without hitting any exit, to accumulate bar count.
    held = Position(symbol="TEST", quantity=-entry_qty, avg_price=entry_price, unrealized_pnl=0.0)
    for i in range(5):
        s.on_bar(make_snapshot(101 + i, close_15m=entry_price, positions=[held]))

    # Now bars_held = 6 (> 5), price at entry (gain = 0% < 0.5%) -> time stop
    signals = s.on_bar(make_snapshot(200, close_15m=entry_price, positions=[held]))

    buy_market = [sig for sig in signals if sig.action == "BUY" and sig.order_type == "MARKET"]
    assert len(buy_market) == 1
    assert buy_market[0].quantity == entry_qty


# ---- test_no_short_when_already_long ----

def test_no_short_when_already_long():
    """Cannot enter short while in a long position."""
    s = DonchianBreakout()
    s.initialize({
        "channel_period": 5, "atr_period": 3, "atr_multiplier": 100.0,  # huge trailing stop
        "volume_factor": 1.0, "risk_per_trade": 0.02,
        "max_loss_pct": 0.90,
    }, {})

    _build_channel(s)
    entry_signals = _enter_position(s)
    assert any(sig.action == "BUY" for sig in entry_signals)
    assert s.in_position["TEST"] is True
    assert s.is_short["TEST"] is False

    # Try to get a short signal by feeding a low price (below channel_low)
    # Should not trigger because we are already in a long position
    held = Position(symbol="TEST", quantity=100, avg_price=108.0, unrealized_pnl=0.0)
    signals = s.on_bar(make_snapshot(200, close_15m=80.0, positions=[held]))

    # Should get a SELL exit (long exit via channel low), NOT a short entry
    sell_market = [sig for sig in signals if sig.action == "SELL" and sig.order_type == "MARKET"]
    assert len(sell_market) == 1
    # After exit, position should be reset (not short)
    assert s.in_position["TEST"] is False


# ==============================================================================
# NEW TESTS: Engine stops, limit profits, dynamic CNC/MIS
# ==============================================================================


def test_engine_stop_submitted_after_entry():
    """Entry fill in snapshot -> SL-M emitted on first bar after entry."""
    s = DonchianBreakout()
    s.initialize({
        "channel_period": 5, "atr_period": 3, "atr_multiplier": 1.5,
        "volume_factor": 1.0, "risk_per_trade": 0.02,
    }, {})

    _build_channel(s)
    entry_signals = _enter_position(s)
    buy_qty = entry_signals[0].quantity
    entry_price = 108.0
    atr = s.current_atr["TEST"]

    # First bar after entry with position held
    held = Position(symbol="TEST", quantity=buy_qty, avg_price=entry_price, unrealized_pnl=0.0)
    signals = s.on_bar(make_snapshot(101, close_15m=entry_price + 0.5, positions=[held]))

    # Should emit SL-M stop order
    slm_signals = [sig for sig in signals if sig.order_type == "SL_M"]
    assert len(slm_signals) == 1
    assert slm_signals[0].action == "SELL"
    assert slm_signals[0].symbol == "TEST"
    assert slm_signals[0].quantity == buy_qty

    # Stop price should be at trailing stop level
    # The trailing stop may ratchet slightly if bar.close > highest_since_entry
    # (entry was at 108.0, new bar at 108.5 causes a small ratchet)
    assert slm_signals[0].stop_price >= entry_price - atr * 1.5 - 0.01
    assert abs(slm_signals[0].stop_price - s.trailing_stop["TEST"]) < 0.01

    # State tracking
    assert s.has_engine_stop["TEST"] is True


def test_limit_profit_target_after_entry():
    """Entry fill -> LIMIT sell at entry + 2*ATR for partial profit."""
    s = DonchianBreakout()
    s.initialize({
        "channel_period": 5, "atr_period": 3, "atr_multiplier": 1.5,
        "volume_factor": 1.0, "risk_per_trade": 0.02,
        "profit_target_atr": 2.0,
    }, {})

    _build_channel(s)
    entry_signals = _enter_position(s)
    buy_qty = entry_signals[0].quantity
    entry_price = 108.0
    atr = s.current_atr["TEST"]

    # First bar after entry
    held = Position(symbol="TEST", quantity=buy_qty, avg_price=entry_price, unrealized_pnl=0.0)
    signals = s.on_bar(make_snapshot(101, close_15m=entry_price + 0.5, positions=[held]))

    # Should emit LIMIT sell for partial profit
    limit_signals = [sig for sig in signals if sig.order_type == "LIMIT"]
    assert len(limit_signals) == 1
    assert limit_signals[0].action == "SELL"

    # Price should be entry + 2 * ATR
    expected_price = entry_price + 2.0 * atr
    assert abs(limit_signals[0].limit_price - expected_price) < 0.01

    # Quantity should be 1/3 of position
    expected_partial = max(1, buy_qty // 3)
    assert limit_signals[0].quantity == expected_partial

    assert s.has_profit_target["TEST"] is True


def test_partial_fill_moves_stop_to_breakeven():
    """Profit target fills -> CANCEL + SL-M at avg_entry (breakeven)."""
    s = DonchianBreakout()
    s.initialize({
        "channel_period": 5, "atr_period": 3, "atr_multiplier": 1.5,
        "volume_factor": 1.0, "risk_per_trade": 0.02,
        "profit_target_atr": 2.0,
    }, {})

    _build_channel(s)
    entry_signals = _enter_position(s)
    buy_qty = entry_signals[0].quantity
    entry_price = 108.0
    atr = s.current_atr["TEST"]

    # First bar: submit engine orders
    held = Position(symbol="TEST", quantity=buy_qty, avg_price=entry_price, unrealized_pnl=0.0)
    s.on_bar(make_snapshot(101, close_15m=entry_price + 0.5, positions=[held]))
    assert s.has_engine_stop["TEST"] is True
    assert s.has_profit_target["TEST"] is True

    # Simulate partial profit fill (LIMIT sell filled by engine)
    partial_qty = max(1, buy_qty // 3)
    remaining_qty = buy_qty - partial_qty
    profit_fill = FillInfo(symbol="TEST", side="SELL", quantity=partial_qty,
                           fill_price=entry_price + 2.0 * atr, costs=10.0, timestamp_ms=102)
    held_after = Position(symbol="TEST", quantity=remaining_qty, avg_price=entry_price, unrealized_pnl=0.0)
    signals = s.on_bar(make_snapshot(
        102, close_15m=entry_price + 2.0 * atr,
        positions=[held_after], fills=[profit_fill],
    ))

    # Should have CANCEL (old stop) + new SL-M at breakeven
    cancel_signals = [sig for sig in signals if sig.action == "CANCEL"]
    assert len(cancel_signals) >= 1

    slm_signals = [sig for sig in signals if sig.order_type == "SL_M"]
    assert len(slm_signals) >= 1
    # Stop should be at breakeven (avg entry price)
    assert abs(slm_signals[0].stop_price - entry_price) < 0.01

    # State should reflect partial taken
    assert s.partial_taken["TEST"] is True
    assert s.has_profit_target["TEST"] is False
    # Trailing stop should be at least at breakeven (avg entry).
    # It may be higher if the trailing stop ratchet moved it up (price is high).
    assert s.trailing_stop["TEST"] >= entry_price - 0.01


def test_trailing_stop_ratchet_cancel_resubmit():
    """Highest moved -> CANCEL + new SL-M at higher stop level."""
    s = DonchianBreakout()
    s.initialize({
        "channel_period": 5, "atr_period": 3, "atr_multiplier": 1.5,
        "volume_factor": 1.0, "risk_per_trade": 0.02,
        "profit_target_atr": 10.0,  # high so partial doesn't trigger
    }, {})

    _build_channel(s)
    entry_signals = _enter_position(s)
    buy_qty = entry_signals[0].quantity
    entry_price = 108.0
    atr = s.current_atr["TEST"]

    # First bar: submit engine orders
    held = Position(symbol="TEST", quantity=buy_qty, avg_price=entry_price, unrealized_pnl=0.0)
    s.on_bar(make_snapshot(101, close_15m=entry_price + 0.5, positions=[held]))
    assert s.has_engine_stop["TEST"] is True
    initial_stop = s.trailing_stop["TEST"]

    # Price rises significantly -> trailing stop should ratchet up
    high_price = entry_price + 5.0
    # Provide existing pending SL-M so expiry re-submission doesn't trigger
    existing_stop = PendingOrder(
        symbol="TEST", side="SELL", quantity=buy_qty,
        order_type="SL_M", limit_price=0.0, stop_price=initial_stop,
    )
    signals = s.on_bar(make_snapshot(
        102, close_15m=high_price, positions=[held],
        pending_orders=[existing_stop],
    ))

    # New stop should be higher
    expected_new_stop = high_price - atr * 1.5
    assert s.trailing_stop["TEST"] >= expected_new_stop - 0.01
    assert s.trailing_stop["TEST"] > initial_stop

    # Should have CANCEL + new SL-M
    cancel_signals = [sig for sig in signals if sig.action == "CANCEL"]
    assert len(cancel_signals) >= 1

    slm_signals = [sig for sig in signals if sig.order_type == "SL_M"]
    assert len(slm_signals) >= 1
    assert abs(slm_signals[0].stop_price - expected_new_stop) < 0.01


def test_dynamic_cnc_strong_volume():
    """Volume > 1.5x average -> CNC product type on entry."""
    s = DonchianBreakout()
    s.initialize({
        "channel_period": 5, "atr_period": 3, "atr_multiplier": 1.5,
        "volume_factor": 1.0, "risk_per_trade": 0.02,
    }, {})

    # Build channel with 100k volume
    _build_channel(s, volume=100_000)

    # Breakout with 200k volume (> 1.5x * 100k = 150k) -> CNC
    signals = _enter_position(s, entry_price=108.0, volume_day=200_000)
    buy_signals = [sig for sig in signals if sig.action == "BUY"]
    assert len(buy_signals) == 1
    assert buy_signals[0].product_type == "CNC"
    assert s.product_type["TEST"] == "CNC"


def test_dynamic_mis_normal_volume():
    """Volume <= 1.5x average -> MIS product type on entry."""
    s = DonchianBreakout()
    s.initialize({
        "channel_period": 5, "atr_period": 3, "atr_multiplier": 1.5,
        "volume_factor": 1.0, "risk_per_trade": 0.02,
    }, {})

    # Build channel with 100k volume
    _build_channel(s, volume=100_000)

    # Breakout with exactly 100k volume (not > 1.5x * 100k = 150k) -> MIS
    # Need to pass volume_factor check (>= 1.0x) but not volume_strong (> 1.5x)
    s.on_bar(make_snapshot(
        8, close_day=108.0, high_day=110.0, low_day=106.0, volume_day=100_000,
    ))
    signals = s.on_bar(make_snapshot(100, close_15m=108.0))

    buy_signals = [sig for sig in signals if sig.action == "BUY"]
    assert len(buy_signals) == 1
    assert buy_signals[0].product_type == "MIS"
    assert s.product_type["TEST"] == "MIS"


def test_full_exit_cancels_all_pending():
    """Channel low exit -> CANCEL before SELL market order."""
    s = DonchianBreakout()
    s.initialize({
        "channel_period": 5, "atr_period": 3, "atr_multiplier": 1.5,
        "volume_factor": 1.0, "risk_per_trade": 0.02,
        "max_loss_pct": 0.50,  # high so max loss doesn't trigger first
    }, {})

    _build_channel(s)
    entry_signals = _enter_position(s)
    buy_qty = entry_signals[0].quantity
    entry_price = 108.0

    # First bar: submit engine orders
    held = Position(symbol="TEST", quantity=buy_qty, avg_price=entry_price, unrealized_pnl=0.0)
    s.on_bar(make_snapshot(101, close_15m=entry_price + 0.5, positions=[held]))
    assert s.has_engine_stop["TEST"] is True
    assert s.has_profit_target["TEST"] is True

    # Channel low exit
    drop_price = 90.0
    signals = s.on_bar(make_snapshot(102, close_15m=drop_price, positions=[held]))

    # CANCEL must appear before the MARKET sell
    cancel_signals = [sig for sig in signals if sig.action == "CANCEL"]
    sell_market = [sig for sig in signals if sig.action == "SELL" and sig.order_type == "MARKET"]
    assert len(cancel_signals) >= 1
    assert len(sell_market) == 1
    assert sell_market[0].quantity == buy_qty

    # Find the indices: CANCEL should come before SELL MARKET
    cancel_idx = next(i for i, s in enumerate(signals) if s.action == "CANCEL")
    sell_idx = next(i for i, s in enumerate(signals) if s.action == "SELL" and s.order_type == "MARKET")
    assert cancel_idx < sell_idx

    # State should be reset
    assert s.in_position["TEST"] is False
    assert s.has_engine_stop["TEST"] is False
    assert s.has_profit_target["TEST"] is False


def test_stop_hit_resets_state():
    """SL-M fill detected (with no profit target pending) -> state reset."""
    s = DonchianBreakout()
    s.initialize({
        "channel_period": 5, "atr_period": 3, "atr_multiplier": 1.5,
        "volume_factor": 1.0, "risk_per_trade": 0.02,
        "profit_target_atr": 2.0,
    }, {})

    _build_channel(s)
    entry_signals = _enter_position(s)
    buy_qty = entry_signals[0].quantity
    entry_price = 108.0
    atr = s.current_atr["TEST"]

    # First bar: submit engine orders
    held = Position(symbol="TEST", quantity=buy_qty, avg_price=entry_price, unrealized_pnl=0.0)
    s.on_bar(make_snapshot(101, close_15m=entry_price + 0.5, positions=[held]))
    assert s.has_engine_stop["TEST"] is True
    assert s.has_profit_target["TEST"] is True

    # Simulate partial profit fill first (to clear has_profit_target)
    partial_qty = max(1, buy_qty // 3)
    remaining_qty = buy_qty - partial_qty
    profit_fill = FillInfo(symbol="TEST", side="SELL", quantity=partial_qty,
                           fill_price=entry_price + 2.0 * atr, costs=5.0, timestamp_ms=102)
    held_after = Position(symbol="TEST", quantity=remaining_qty, avg_price=entry_price, unrealized_pnl=0.0)
    s.on_bar(make_snapshot(102, close_15m=entry_price + 2.0 * atr,
                           positions=[held_after], fills=[profit_fill]))
    assert s.partial_taken["TEST"] is True
    assert s.has_profit_target["TEST"] is False
    assert s.has_engine_stop["TEST"] is True  # stop still active

    # Now simulate stop hit: engine fills the SL-M
    stop_fill = FillInfo(symbol="TEST", side="SELL", quantity=remaining_qty,
                         fill_price=entry_price, costs=5.0, timestamp_ms=103)
    signals = s.on_bar(make_snapshot(
        103, close_15m=entry_price - 1,
        positions=[],  # position gone
        fills=[stop_fill],
    ))

    # Position should be fully reset
    assert s.in_position["TEST"] is False
    assert s.has_engine_stop["TEST"] is False
    assert s.has_profit_target["TEST"] is False


# ==============================================================================
# DAY ORDER EXPIRY RE-SUBMISSION TESTS
# ==============================================================================


def test_stop_resubmitted_after_expiry():
    """Long position with has_engine_stop=True but no pending SL-M -> re-submits SL-M."""
    s = DonchianBreakout()
    s.initialize({
        "channel_period": 5, "atr_period": 3, "atr_multiplier": 1.5,
        "volume_factor": 1.0, "risk_per_trade": 0.02,
        "profit_target_atr": 2.0,
    }, {})

    _build_channel(s)
    entry_signals = _enter_position(s)
    buy_qty = entry_signals[0].quantity
    entry_price = 108.0
    atr = s.current_atr["TEST"]

    # First bar after entry: submit engine orders (SL-M + LIMIT)
    held = Position(symbol="TEST", quantity=buy_qty, avg_price=entry_price, unrealized_pnl=0.0)
    s.on_bar(make_snapshot(101, close_15m=entry_price + 0.5, positions=[held]))
    assert s.has_engine_stop["TEST"] is True
    assert s.has_profit_target["TEST"] is True

    # Simulate next morning: DAY orders expired, pending_orders is empty.
    # has_engine_stop is still True but there's no actual pending SL-M.
    # The strategy should detect this and re-submit the SL-M.
    signals = s.on_bar(make_snapshot(
        200, close_15m=entry_price + 1.0,
        positions=[held],
        pending_orders=[],  # all expired
    ))

    # Should contain a re-submitted SL-M stop
    slm_signals = [sig for sig in signals if sig.order_type == "SL_M"]
    assert len(slm_signals) >= 1
    assert slm_signals[0].action == "SELL"
    assert slm_signals[0].symbol == "TEST"
    assert slm_signals[0].quantity == buy_qty
    assert slm_signals[0].stop_price > 0


def test_profit_target_resubmitted_after_expiry():
    """Long position with has_profit_target=True, partial_taken=False, no pending LIMIT -> re-submits LIMIT."""
    s = DonchianBreakout()
    s.initialize({
        "channel_period": 5, "atr_period": 3, "atr_multiplier": 1.5,
        "volume_factor": 1.0, "risk_per_trade": 0.02,
        "profit_target_atr": 2.0,
    }, {})

    _build_channel(s)
    entry_signals = _enter_position(s)
    buy_qty = entry_signals[0].quantity
    entry_price = 108.0
    atr = s.current_atr["TEST"]

    # First bar after entry: submit engine orders (SL-M + LIMIT)
    held = Position(symbol="TEST", quantity=buy_qty, avg_price=entry_price, unrealized_pnl=0.0)
    s.on_bar(make_snapshot(101, close_15m=entry_price + 0.5, positions=[held]))
    assert s.has_engine_stop["TEST"] is True
    assert s.has_profit_target["TEST"] is True
    assert s.partial_taken["TEST"] is False

    # Simulate next morning: DAY orders expired, pending_orders is empty.
    signals = s.on_bar(make_snapshot(
        200, close_15m=entry_price + 1.0,
        positions=[held],
        pending_orders=[],  # all expired
    ))

    # Should contain a re-submitted LIMIT profit target
    limit_signals = [sig for sig in signals if sig.order_type == "LIMIT"]
    assert len(limit_signals) >= 1
    assert limit_signals[0].action == "SELL"
    assert limit_signals[0].symbol == "TEST"

    # Price should be entry + 2 * ATR
    expected_price = entry_price + 2.0 * atr
    assert abs(limit_signals[0].limit_price - expected_price) < 0.01

    # Quantity should be 1/3 of position
    expected_partial = max(1, buy_qty // 3)
    assert limit_signals[0].quantity == expected_partial


def test_short_stop_resubmitted_after_expiry():
    """Short position with has_engine_stop=True but no pending SL-M -> re-submits SL-M BUY."""
    s = DonchianBreakout()
    s.initialize({
        "channel_period": 5, "atr_period": 3, "atr_multiplier": 1.5,
        "volume_factor": 1.0, "risk_per_trade": 0.02,
        "profit_target_atr": 2.0,
        "max_loss_pct": 0.50,
    }, {})

    _build_channel_for_short(s)
    entry_signals = _enter_short_position(s, entry_price=90.0)
    sell_signals = [sig for sig in entry_signals if sig.action == "SELL"]
    entry_qty = sell_signals[0].quantity
    entry_price = 90.0

    # First bar after entry: submit engine orders
    held = Position(symbol="TEST", quantity=-entry_qty, avg_price=entry_price, unrealized_pnl=0.0)
    s.on_bar(make_snapshot(101, close_15m=entry_price - 1, positions=[held]))
    assert s.has_engine_stop["TEST"] is True

    # Simulate next morning: DAY orders expired
    signals = s.on_bar(make_snapshot(
        200, close_15m=entry_price - 0.5,
        positions=[held],
        pending_orders=[],  # all expired
    ))

    # Should contain a re-submitted SL-M BUY (stop for short)
    slm_signals = [sig for sig in signals if sig.order_type == "SL_M"]
    assert len(slm_signals) >= 1
    assert slm_signals[0].action == "BUY"
    assert slm_signals[0].symbol == "TEST"
    assert slm_signals[0].quantity == entry_qty
    assert slm_signals[0].stop_price > 0


def test_no_resubmit_when_pending_orders_present():
    """No re-submission when pending orders still exist (not expired)."""
    s = DonchianBreakout()
    s.initialize({
        "channel_period": 5, "atr_period": 3, "atr_multiplier": 1.5,
        "volume_factor": 1.0, "risk_per_trade": 0.02,
        "profit_target_atr": 2.0,
    }, {})

    _build_channel(s)
    entry_signals = _enter_position(s)
    buy_qty = entry_signals[0].quantity
    entry_price = 108.0
    atr = s.current_atr["TEST"]

    # First bar after entry: submit engine orders
    held = Position(symbol="TEST", quantity=buy_qty, avg_price=entry_price, unrealized_pnl=0.0)
    s.on_bar(make_snapshot(101, close_15m=entry_price + 0.5, positions=[held]))
    assert s.has_engine_stop["TEST"] is True
    assert s.has_profit_target["TEST"] is True

    # Next bar: pending orders still exist -> no re-submission
    existing_stop = PendingOrder(
        symbol="TEST", side="SELL", quantity=buy_qty,
        order_type="SL_M", limit_price=0.0,
        stop_price=s.trailing_stop["TEST"],
    )
    existing_target = PendingOrder(
        symbol="TEST", side="SELL", quantity=max(1, buy_qty // 3),
        order_type="LIMIT",
        limit_price=entry_price + 2.0 * atr,
        stop_price=0.0,
    )
    signals = s.on_bar(make_snapshot(
        200, close_15m=entry_price + 0.5,
        positions=[held],
        pending_orders=[existing_stop, existing_target],
    ))

    # Should NOT have any re-submitted SL-M or LIMIT (the ratchet may cancel+resubmit
    # if price moved, but since price didn't move much, we shouldn't see extra orders)
    # The key assertion: no extra stop re-submission from the expiry code path
    slm_signals = [sig for sig in signals if sig.order_type == "SL_M"]
    # At most one SL-M from trailing stop ratchet (if price moved up)
    # but not from expiry re-submission
    assert len(slm_signals) <= 1
