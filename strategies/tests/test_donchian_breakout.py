"""Tests for Donchian Breakout strategy (risk sizing, partial profits, pyramiding, short selling)."""

from strategies.base import (
    BarData, MarketSnapshot, Portfolio, Position, SessionContext, InstrumentInfo,
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
    """Trailing stop at 1.5x ATR (tighter than old 2.0x)."""
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

    # Price drops through trailing stop -> should exit
    stop_level = s.trailing_stop["TEST"]
    drop_price = stop_level - 1.0
    held = Position(symbol="TEST", quantity=100, avg_price=entry, unrealized_pnl=0.0)
    signals = s.on_bar(make_snapshot(102, close_15m=drop_price, positions=[held]))
    assert any(sig.action == "SELL" for sig in signals)


# ---- test_partial_profit_exit_at_2atr ----

def test_partial_profit_exit_at_2atr():
    """Partial exit of 1/3 position at avg_entry + 2*ATR, then stop moves to breakeven."""
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

    # Price reaches profit target: entry + 2 * ATR
    target_price = entry_price + 2.0 * atr + 1.0  # slightly above target
    held = Position(symbol="TEST", quantity=buy_qty, avg_price=entry_price, unrealized_pnl=0.0)
    signals = s.on_bar(make_snapshot(101, close_15m=target_price, positions=[held]))

    sell_signals = [sig for sig in signals if sig.action == "SELL"]
    assert len(sell_signals) == 1
    # Should sell 1/3 of held qty
    expected_partial = max(1, buy_qty // 3)
    assert sell_signals[0].quantity == expected_partial

    # Trailing stop should now be at breakeven (avg entry price)
    assert s.partial_taken["TEST"] is True
    assert abs(s.trailing_stop["TEST"] - entry_price) < 0.01


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

    sell_signals = [sig for sig in signals if sig.action == "SELL"]
    assert len(sell_signals) == 1
    assert sell_signals[0].quantity == 100


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

    # Channel low = min of lows over last 5 days (excluding current)
    # With base_price=95, lows are 93,94,95,...,100 for 8 bars
    # Channel uses last 6 bars [:-1] = indices 2..7, lows = 95,96,97,98,99
    # min = 95.0 (approx — depends on exact window)
    # We drop price below the lowest possible channel_low
    drop_price = 90.0
    held = Position(symbol="TEST", quantity=100, avg_price=108.0, unrealized_pnl=0.0)
    signals = s.on_bar(make_snapshot(101, close_15m=drop_price, positions=[held]))

    sell_signals = [sig for sig in signals if sig.action == "SELL"]
    assert len(sell_signals) == 1
    assert sell_signals[0].quantity == 100


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
    """Price breaks below channel low with volume -> SELL signal (short entry)."""
    s = DonchianBreakout()
    s.initialize({
        "channel_period": 5, "atr_period": 3, "atr_multiplier": 1.5,
        "volume_factor": 1.0, "risk_per_trade": 0.02,
    }, {})

    _build_channel_for_short(s)

    # Channel low = min of lows over last 5 days (excluding current).
    # With descending prices from 105, lows are: 103,102,101,100,99,98,97,96
    # Channel period=5, so we look at lows[-(5+1):-1] = last 6 excluding last = indices 2..7
    # lows at indices 2..6 = 101,100,99,98,97 -> channel_low = 97
    # We need price < 97 to trigger short entry
    entry_price = 90.0  # well below channel low
    signals = _enter_short_position(s, entry_price=entry_price)

    sell_signals = [sig for sig in signals if sig.action == "SELL"]
    assert len(sell_signals) == 1
    assert sell_signals[0].quantity > 0
    assert s.in_position["TEST"] is True
    assert s.is_short["TEST"] is True

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
    """Short position: price rises above trailing stop -> BUY to cover."""
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
    s.on_bar(make_snapshot(101, close_15m=85.0))
    new_expected_stop = 85.0 + atr * 1.5
    assert s.trailing_stop["TEST"] <= new_expected_stop + 0.01

    # Price rises above trailing stop -> should cover
    stop_level = s.trailing_stop["TEST"]
    rise_price = stop_level + 1.0
    held = Position(symbol="TEST", quantity=-entry_qty, avg_price=entry_price, unrealized_pnl=0.0)
    signals = s.on_bar(make_snapshot(102, close_15m=rise_price, positions=[held]))

    buy_signals = [sig for sig in signals if sig.action == "BUY"]
    assert len(buy_signals) == 1
    assert buy_signals[0].quantity == entry_qty


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

    # Channel high = max of highs over last 5+1 days excluding last
    # highs descending: 107,106,105,104,103,102,101,100 + breakout day
    # channel_high is around 105-107 depending on window
    # We set price well above any channel high
    cover_price = 120.0
    held = Position(symbol="TEST", quantity=-entry_qty, avg_price=90.0, unrealized_pnl=0.0)
    signals = s.on_bar(make_snapshot(101, close_15m=cover_price, positions=[held]))

    buy_signals = [sig for sig in signals if sig.action == "BUY"]
    assert len(buy_signals) == 1
    assert buy_signals[0].quantity == entry_qty
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

    buy_signals = [sig for sig in signals if sig.action == "BUY"]
    assert len(buy_signals) == 1
    assert buy_signals[0].quantity == entry_qty


# ---- test_short_partial_cover ----

def test_short_partial_cover():
    """Short position: price drops to profit target -> cover 1/3, stop moves to breakeven."""
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

    # Profit target for short: price <= entry - profit_target_atr * ATR
    target_price = entry_price - 2.0 * atr - 1.0  # slightly below target
    held = Position(symbol="TEST", quantity=-entry_qty, avg_price=entry_price, unrealized_pnl=0.0)
    signals = s.on_bar(make_snapshot(101, close_15m=target_price, positions=[held]))

    buy_signals = [sig for sig in signals if sig.action == "BUY"]
    assert len(buy_signals) == 1
    expected_partial = max(1, entry_qty // 3)
    assert buy_signals[0].quantity == expected_partial

    # Trailing stop moved to breakeven
    assert s.partial_taken["TEST"] is True
    assert abs(s.trailing_stop["TEST"] - entry_price) < 0.01


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
    sell_signals = [sig for sig in signals if sig.action == "SELL"]
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
    # Must pass position so reconciliation doesn't reset the short.
    # entry_bar = bar_counter at entry. We need bars_held > max_hold_bars (5).
    # Feed exactly 5 bars (bars_held = 5, not yet > 5), then the 6th triggers.
    held = Position(symbol="TEST", quantity=-entry_qty, avg_price=entry_price, unrealized_pnl=0.0)
    for i in range(5):
        s.on_bar(make_snapshot(101 + i, close_15m=entry_price, positions=[held]))

    # Now bars_held = 6 (> 5), price at entry (gain = 0% < 0.5%) -> time stop
    signals = s.on_bar(make_snapshot(200, close_15m=entry_price, positions=[held]))

    buy_signals = [sig for sig in signals if sig.action == "BUY"]
    assert len(buy_signals) == 1
    assert buy_signals[0].quantity == entry_qty


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
    sell_signals = [sig for sig in signals if sig.action == "SELL"]
    assert len(sell_signals) == 1
    # After exit, position should be reset (not short)
    assert s.in_position["TEST"] is False
