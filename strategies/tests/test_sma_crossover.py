import pytest

from strategies.base import (
    BarData, FillInfo, MarketSnapshot, PendingOrder, Portfolio,
    Signal, SessionContext, InstrumentInfo, Position,
)
from strategies.deterministic.sma_crossover import SmaCrossover


DEFAULT_CAPITAL = 1_000_000.0


def make_snapshot(
    ts: int,
    close: float,
    symbol: str = "TEST",
    high: float | None = None,
    low: float | None = None,
    positions: list[Position] | None = None,
    capital: float = DEFAULT_CAPITAL,
    fills: list[FillInfo] | None = None,
    pending_orders: list[PendingOrder] | None = None,
) -> MarketSnapshot:
    h = high if high is not None else close
    l = low if low is not None else close
    bar = BarData(symbol, close, h, l, close, 1000, 0)
    return MarketSnapshot(
        timestamp_ms=ts,
        timeframes={"day": {symbol: bar}},
        history={},
        portfolio=Portfolio(cash=capital, equity=capital, positions=positions or []),
        instruments={},
        fills=fills or [],
        rejections=[],
        closed_trades=[],
        context=SessionContext(capital, ts, 200, "2024-01-01", "2024-12-31", ["day"], 200),
        pending_orders=pending_orders or [],
    )


def _warmup_and_enter(
    s: SmaCrossover,
    symbol: str = "TEST",
    slow_period: int = 5,
    atr_period: int = 3,
    extra_config: dict | None = None,
) -> tuple[list[Signal], float]:
    """Feed enough bars to warm up indicators then trigger a golden cross entry.

    Returns (all_signals, entry_price).

    With limit orders, entry is a two-phase process:
    1. Golden cross bar: strategy emits LIMIT BUY → pending_entry=True
    2. Next bar with fill: strategy confirms entry and emits SL-M stop

    After this helper returns, the strategy has a confirmed open position
    reflected in its internal state. Subsequent bars fed to on_bar MUST
    include a matching Position in the snapshot so that _reconcile_position
    does not reset.
    """
    config = {
        "fast_period": 2,
        "slow_period": slow_period,
        "atr_period": atr_period,
        "atr_stop_multiplier": 2.0,
        "min_spread": 0.005,
        "risk_per_trade": 0.02,
        "max_hold_bars": 50,
        "pyramid_levels": 0,  # disable pyramiding during warmup for cleaner state
        "cnc_spread_threshold": 0.01,
    }
    if extra_config:
        config.update(extra_config)
    s.initialize(config, {})

    all_signals: list[Signal] = []
    entry_price: float = 0.0
    position_qty: int = 0
    avg_entry: float = 0.0
    pending_entry = False
    pending_qty = 0
    pending_limit_price = 0.0

    # Phase 1: warmup with declining prices
    warmup_bars = max(slow_period, atr_period + 1)
    for i in range(warmup_bars):
        price = 100.0 - i * 2
        positions = []
        if position_qty > 0:
            positions = [Position(symbol, position_qty, avg_entry, 0.0)]
        fills = []
        if pending_entry and position_qty == 0:
            # Simulate fill of pending limit order
            fills = [FillInfo(symbol, "BUY", pending_qty, pending_limit_price, 0.0, i)]
            position_qty = pending_qty
            avg_entry = pending_limit_price
            entry_price = pending_limit_price
            positions = [Position(symbol, position_qty, avg_entry, 0.0)]
            pending_entry = False
        snap = make_snapshot(i, price, symbol, high=price + 5, low=price - 5,
                            positions=positions, fills=fills)
        sigs = s.on_bar(snap)
        for sig in sigs:
            if sig.action == "BUY" and sig.order_type == "LIMIT" and position_qty == 0:
                pending_entry = True
                pending_qty = sig.quantity
                pending_limit_price = sig.limit_price
        all_signals.extend(sigs)

    # Phase 2: sharp upswing to trigger golden cross with strong spread
    base = 100.0 - (warmup_bars - 1) * 2
    for i in range(6):
        price = base + (i + 1) * 15
        positions = []
        fills = []
        if pending_entry:
            # Simulate fill of pending limit order
            fills = [FillInfo(symbol, "BUY", pending_qty, pending_limit_price, 0.0, warmup_bars + i)]
            position_qty = pending_qty
            avg_entry = pending_limit_price
            entry_price = pending_limit_price
            positions = [Position(symbol, position_qty, avg_entry, 0.0)]
            pending_entry = False
        elif position_qty > 0:
            positions = [Position(symbol, position_qty, avg_entry, 0.0)]
        snap = make_snapshot(warmup_bars + i, price, symbol, high=price + 5, low=price - 5,
                            positions=positions, fills=fills)
        sigs = s.on_bar(snap)
        for sig in sigs:
            if sig.action == "BUY" and sig.order_type == "LIMIT" and position_qty == 0:
                pending_entry = True
                pending_qty = sig.quantity
                pending_limit_price = sig.limit_price
            elif sig.action == "BUY" and sig.order_type == "MARKET" and position_qty > 0:
                # Pyramid add (market order, immediate)
                avg_entry = (avg_entry * position_qty + price * sig.quantity) / (position_qty + sig.quantity)
                position_qty += sig.quantity
        all_signals.extend(sigs)

    return all_signals, entry_price


class TestRequiredData:
    def test_required_data(self):
        s = SmaCrossover()
        reqs = s.required_data()
        assert len(reqs) == 1
        assert reqs[0]["interval"] == "day"
        assert reqs[0]["lookback"] == 200


class TestValidation:
    def test_fast_ge_slow_raises(self):
        """initialize raises ValueError when fast_period >= slow_period."""
        s = SmaCrossover()
        with pytest.raises(ValueError, match="fast_period"):
            s.initialize({"fast_period": 30, "slow_period": 10}, {})

    def test_fast_equal_slow_raises(self):
        s = SmaCrossover()
        with pytest.raises(ValueError, match="fast_period"):
            s.initialize({"fast_period": 10, "slow_period": 10}, {})


class TestPositionSizing:
    def test_position_sizing_uses_atr(self):
        """Entry quantity is ATR-based, not quantity=1.

        qty = int((capital * risk_per_trade) / (ATR * atr_stop_multiplier))
        With capital=1_000_000, risk=0.02, ATR~13, multiplier=2 -> qty~750.
        """
        s = SmaCrossover()
        all_signals, _ = _warmup_and_enter(s)

        buy_signals = [sig for sig in all_signals if sig.action == "BUY" and sig.order_type == "LIMIT"]
        assert len(buy_signals) > 0, "Expected at least one LIMIT BUY signal"
        assert buy_signals[0].quantity > 1, (
            f"Expected ATR-based sizing > 1, got {buy_signals[0].quantity}"
        )


class TestTrendStrengthFilter:
    def test_trend_strength_filter_blocks_weak_crossover(self):
        """A crossover with spread below min_spread should NOT trigger a BUY."""
        s = SmaCrossover()
        all_signals, _ = _warmup_and_enter(s, extra_config={"min_spread": 10.0})

        buy_signals = [sig for sig in all_signals if sig.action == "BUY"]
        assert len(buy_signals) == 0, (
            "Expected no BUY when min_spread is extremely high"
        )


class TestTrailingStopExit:
    def test_trailing_stop_triggers_engine_stop(self):
        """After entry, if engine SL-M stop fires (fill appears), state resets."""
        s = SmaCrossover()
        all_signals, entry_price = _warmup_and_enter(s)

        buy_signals = [sig for sig in all_signals if sig.action == "BUY" and sig.order_type == "LIMIT"]
        assert len(buy_signals) > 0, "Need a LIMIT BUY to test trailing stop"

        state = s.states["TEST"]
        total_qty = state.position_qty
        avg = state.avg_entry
        assert total_qty > 0, "Strategy should have an open position"
        assert state.has_engine_stop, "Strategy should have an engine SL-M stop"

        # Feed a few rising bars to push the trailing stop higher
        current_highest = state.highest_since_entry
        for i in range(3):
            rise_price = current_highest + (i + 1) * 10
            snap = make_snapshot(
                100 + i, rise_price, "TEST", high=rise_price + 5, low=rise_price - 5,
                positions=[Position("TEST", total_qty, avg, 0.0)],
            )
            sigs = s.on_bar(snap)

        # Trailing stop should have risen
        new_stop = state.trailing_stop
        assert new_stop > 0, "Trailing stop should be positive after rising bars"

        # Simulate engine stop hit: position gone, SELL fill present
        snap = make_snapshot(
            200, new_stop - 10, "TEST", high=new_stop - 5, low=new_stop - 15,
            positions=[],  # position closed by engine
            fills=[FillInfo("TEST", "SELL", total_qty, new_stop, 0.0, 200)],
        )
        sigs = s.on_bar(snap)
        # State should be reset
        assert state.position_qty == 0, "Position should be closed after stop hit"
        assert not state.has_engine_stop, "Engine stop flag should be cleared"


class TestDeathCrossExit:
    def test_death_cross_exit(self):
        """After entry, a death cross (fast drops below slow) should trigger SELL."""
        s = SmaCrossover()
        all_signals, entry_price = _warmup_and_enter(s)

        buy_signals = [sig for sig in all_signals if sig.action == "BUY" and sig.order_type == "LIMIT"]
        assert len(buy_signals) > 0, "Need a BUY to test death cross"

        state = s.states["TEST"]
        total_qty = state.position_qty
        avg = state.avg_entry
        assert total_qty > 0, "Strategy should have an open position"

        # Feed sharply declining prices to trigger death cross
        high_price = state.highest_since_entry
        found_sell = False
        for i in range(10):
            price = high_price - (i + 1) * 25
            if price < 10:
                price = 10.0
            snap = make_snapshot(
                200 + i, price, "TEST", high=price + 5, low=price - 5,
                positions=[Position("TEST", total_qty, avg, 0.0)],
            )
            sigs = s.on_bar(snap)
            # Look for CANCEL + SELL (market) pair
            sell_sigs = [sig for sig in sigs if sig.action == "SELL" and sig.order_type == "MARKET"]
            cancel_sigs = [sig for sig in sigs if sig.action == "CANCEL"]
            if sell_sigs:
                assert len(cancel_sigs) > 0, "Should CANCEL pending before SELL on death cross"
                assert sell_sigs[0].quantity == total_qty, "Should sell entire position"
                found_sell = True
                break

        assert found_sell, "Expected a SELL from death cross during sharp decline"


class TestPyramiding:
    def test_pyramid_adds_to_position(self):
        """When price rises above avg_entry + ATR, strategy should add to position."""
        s = SmaCrossover()
        # Enter with pyramid_levels=0 during warmup, then enable pyramiding
        all_signals, entry_price = _warmup_and_enter(s)

        buy_signals = [sig for sig in all_signals if sig.action == "BUY" and sig.order_type == "LIMIT"]
        assert len(buy_signals) > 0, "Need a BUY to test pyramiding"

        state = s.states["TEST"]
        total_qty = state.position_qty
        avg = state.avg_entry
        original_qty = state.original_qty
        assert total_qty > 0, "Strategy should have an open position"

        # Now enable pyramiding
        s.pyramid_levels = 2
        state.pyramid_count = 0

        # Price well above avg_entry + ATR to trigger pyramid
        target_price = avg + 60  # way above avg + ATR

        found_pyramid = False
        for i in range(3):
            price = target_price + i * 5
            snap = make_snapshot(
                300 + i, price, "TEST", high=price + 5, low=price - 5,
                positions=[Position("TEST", state.position_qty, avg, 0.0)],
            )
            sigs = s.on_bar(snap)
            pyramid_buys = [sig for sig in sigs if sig.action == "BUY" and sig.order_type == "MARKET"]
            if pyramid_buys:
                expected_add = max(1, int(original_qty * 0.5))
                assert pyramid_buys[0].quantity == expected_add, (
                    f"Pyramid qty should be {expected_add}, got {pyramid_buys[0].quantity}"
                )
                found_pyramid = True
                break

        assert found_pyramid, "Expected a pyramid BUY when price is well above avg_entry + ATR"


def _warmup_and_enter_short(
    s: SmaCrossover,
    symbol: str = "TEST",
    slow_period: int = 5,
    atr_period: int = 3,
    extra_config: dict | None = None,
) -> tuple[list[Signal], float]:
    """Feed enough bars to warm up indicators then trigger a death cross short entry.

    Returns (all_signals, entry_price).

    With limit orders, short entry is a two-phase process:
    1. Death cross bar: strategy emits LIMIT SELL -> pending_entry=True
    2. Next bar with fill: strategy confirms entry and emits SL-M BUY stop

    After this helper returns, the strategy has a confirmed open SHORT position
    (negative position_qty) reflected in its internal state.
    """
    config = {
        "fast_period": 2,
        "slow_period": slow_period,
        "atr_period": atr_period,
        "atr_stop_multiplier": 2.0,
        "min_spread": 0.005,
        "risk_per_trade": 0.02,
        "max_hold_bars": 50,
        "pyramid_levels": 0,  # disable pyramiding during warmup for cleaner state
        "cnc_spread_threshold": 0.01,
    }
    if extra_config:
        config.update(extra_config)
    s.initialize(config, {})

    all_signals: list[Signal] = []
    entry_price: float = 0.0
    position_qty: int = 0
    avg_entry: float = 0.0
    pending_entry = False
    pending_side = ""
    pending_qty = 0
    pending_limit_price = 0.0

    # Phase 1: warmup with RISING prices (so fast > slow is established)
    warmup_bars = max(slow_period, atr_period + 1)
    for i in range(warmup_bars):
        price = 100.0 + i * 2
        positions = []
        fills = []
        if pending_entry:
            if pending_side == "SELL" and position_qty == 0:
                fills = [FillInfo(symbol, "SELL", pending_qty, pending_limit_price, 0.0, i)]
                position_qty = -pending_qty
                avg_entry = pending_limit_price
                entry_price = pending_limit_price
                positions = [Position(symbol, position_qty, avg_entry, 0.0)]
                pending_entry = False
                pending_side = ""
            elif pending_side == "BUY" and position_qty == 0:
                fills = [FillInfo(symbol, "BUY", pending_qty, pending_limit_price, 0.0, i)]
                position_qty = pending_qty
                avg_entry = pending_limit_price
                entry_price = pending_limit_price
                positions = [Position(symbol, position_qty, avg_entry, 0.0)]
                pending_entry = False
                pending_side = ""
        if not fills and position_qty != 0:
            positions = [Position(symbol, position_qty, avg_entry, 0.0)]
        snap = make_snapshot(i, price, symbol, high=price + 5, low=price - 5,
                            positions=positions, fills=fills)
        sigs = s.on_bar(snap)
        for sig in sigs:
            if sig.order_type == "LIMIT" and position_qty == 0 and not pending_entry:
                pending_entry = True
                pending_side = sig.action
                pending_qty = sig.quantity
                pending_limit_price = sig.limit_price
            elif sig.action == "SELL" and sig.order_type == "MARKET" and position_qty > 0:
                position_qty -= sig.quantity
            elif sig.action == "BUY" and sig.order_type == "MARKET" and position_qty < 0:
                position_qty += sig.quantity
        all_signals.extend(sigs)

    # Phase 2: sharp downswing to trigger death cross with strong spread
    base = 100.0 + (warmup_bars - 1) * 2
    for i in range(6):
        price = base - (i + 1) * 15
        if price < 10:
            price = 10.0
        positions = []
        fills = []
        if pending_entry:
            if pending_side == "SELL" and position_qty == 0:
                fills = [FillInfo(symbol, "SELL", pending_qty, pending_limit_price, 0.0, warmup_bars + i)]
                position_qty = -pending_qty
                avg_entry = pending_limit_price
                entry_price = pending_limit_price
                positions = [Position(symbol, position_qty, avg_entry, 0.0)]
                pending_entry = False
                pending_side = ""
            elif pending_side == "BUY" and position_qty == 0:
                fills = [FillInfo(symbol, "BUY", pending_qty, pending_limit_price, 0.0, warmup_bars + i)]
                position_qty = pending_qty
                avg_entry = pending_limit_price
                entry_price = pending_limit_price
                positions = [Position(symbol, position_qty, avg_entry, 0.0)]
                pending_entry = False
                pending_side = ""
        if not fills and position_qty != 0:
            positions = [Position(symbol, position_qty, avg_entry, 0.0)]
        snap = make_snapshot(warmup_bars + i, price, symbol, high=price + 5, low=price - 5,
                            positions=positions, fills=fills)
        sigs = s.on_bar(snap)
        for sig in sigs:
            if sig.order_type == "LIMIT" and position_qty == 0 and not pending_entry:
                pending_entry = True
                pending_side = sig.action
                pending_qty = sig.quantity
                pending_limit_price = sig.limit_price
            elif sig.action == "SELL" and sig.order_type == "MARKET" and position_qty > 0:
                position_qty -= sig.quantity
            elif sig.action == "BUY" and sig.order_type == "MARKET" and position_qty < 0:
                position_qty += sig.quantity
        all_signals.extend(sigs)

    return all_signals, entry_price


class TestShortEntryOnDeathCross:
    def test_short_entry_on_death_cross(self):
        """Death cross with sufficient spread should produce a LIMIT SELL signal to go short."""
        s = SmaCrossover()
        all_signals, entry_price = _warmup_and_enter_short(s)

        # There should be at least one LIMIT SELL signal for short entry
        sell_signals = [sig for sig in all_signals if sig.action == "SELL" and sig.order_type == "LIMIT"]
        assert len(sell_signals) > 0, "Expected at least one LIMIT SELL signal on death cross"
        assert sell_signals[-1].quantity > 1, (
            f"Expected ATR-based sizing > 1, got {sell_signals[-1].quantity}"
        )
        # Short entries must always use MIS (CNC shorts not allowed in Zerodha)
        assert sell_signals[-1].product_type == "MIS", (
            f"Short entry must use MIS, got {sell_signals[-1].product_type}"
        )

        # Strategy should now be short (negative position_qty)
        state = s.states["TEST"]
        assert state.position_qty < 0, (
            f"Expected negative position_qty for short, got {state.position_qty}"
        )
        assert state.product_type == "MIS", (
            f"Short position product_type must be MIS, got {state.product_type}"
        )
        assert state.has_engine_stop, "Should have engine SL-M stop after short entry fill"
        # Trailing stop should be above the lowest price seen (stop moves down with price)
        assert state.trailing_stop > state.lowest_since_entry, (
            f"Short trailing stop ({state.trailing_stop}) should be above lowest since entry ({state.lowest_since_entry})"
        )
        assert state.lowest_since_entry > 0, (
            "lowest_since_entry should be set for short positions"
        )
        assert state.lowest_since_entry <= state.entry_price, (
            "lowest_since_entry should be at or below entry_price"
        )

    def test_short_entry_blocked_by_high_min_spread(self):
        """A death cross with spread below min_spread should NOT trigger a short."""
        s = SmaCrossover()
        all_signals, _ = _warmup_and_enter_short(s, extra_config={"min_spread": 10.0})

        sell_signals = [sig for sig in all_signals if sig.action == "SELL"]
        assert len(sell_signals) == 0, (
            "Expected no SELL when min_spread is extremely high"
        )


class TestShortExitOnGoldenCross:
    def test_short_exit_on_golden_cross(self):
        """While short, a golden cross should produce a BUY signal to cover."""
        s = SmaCrossover()
        all_signals, entry_price = _warmup_and_enter_short(s)

        state = s.states["TEST"]
        assert state.position_qty < 0, "Must be short to test golden cross exit"
        abs_qty = abs(state.position_qty)
        avg = state.avg_entry

        # Feed sharply rising prices to trigger golden cross while short
        low_price = state.lowest_since_entry
        found_buy = False
        for i in range(10):
            price = low_price + (i + 1) * 25
            snap = make_snapshot(
                200 + i, price, "TEST", high=price + 5, low=price - 5,
                positions=[Position("TEST", state.position_qty, avg, 0.0)],
            )
            sigs = s.on_bar(snap)
            buy_sigs = [sig for sig in sigs if sig.action == "BUY" and sig.order_type == "MARKET"]
            if buy_sigs:
                assert buy_sigs[0].quantity == abs_qty, (
                    f"Should cover entire short position ({abs_qty}), got {buy_sigs[0].quantity}"
                )
                # Should have CANCEL before cover
                cancel_sigs = [sig for sig in sigs if sig.action == "CANCEL"]
                assert len(cancel_sigs) > 0, "Should CANCEL engine stop before covering"
                found_buy = True
                break

        assert found_buy, "Expected a BUY to cover short on golden cross during sharp rise"

        # After covering, position should be flat
        assert state.position_qty == 0, "Position should be flat after covering short"


class TestShortTrailingStop:
    def test_short_trailing_stop(self):
        """While short, if engine SL-M fires (BUY fill), state resets."""
        s = SmaCrossover()
        all_signals, entry_price = _warmup_and_enter_short(s)

        state = s.states["TEST"]
        assert state.position_qty < 0, "Must be short to test trailing stop"
        abs_qty = abs(state.position_qty)
        avg = state.avg_entry
        assert state.has_engine_stop, "Must have engine stop"

        # Feed a few declining bars to push the trailing stop lower
        current_lowest = state.lowest_since_entry
        for i in range(3):
            drop_price = current_lowest - (i + 1) * 10
            if drop_price < 5:
                drop_price = 5.0
            snap = make_snapshot(
                100 + i, drop_price, "TEST", high=drop_price + 5, low=drop_price - 5,
                positions=[Position("TEST", state.position_qty, avg, 0.0)],
            )
            sigs = s.on_bar(snap)
            assert not any(
                sig.action == "BUY" and sig.order_type == "MARKET" for sig in sigs
            ), f"Should not cover on declining price bar {i}"

        # Trailing stop should have lowered
        new_stop = state.trailing_stop
        assert new_stop > 0, "Trailing stop should be positive"

        # Simulate engine stop hit: position gone, BUY fill present
        spike_price = new_stop + 10
        snap = make_snapshot(
            200, spike_price, "TEST", high=spike_price + 1, low=spike_price - 1,
            positions=[],  # position closed by engine
            fills=[FillInfo("TEST", "BUY", abs_qty, new_stop, 0.0, 200)],
        )
        sigs = s.on_bar(snap)
        assert state.position_qty == 0, "Position should be flat after stop hit"
        assert not state.has_engine_stop, "Engine stop flag should be cleared"


# ============================================================
# New tests for limit entries, engine stops, and dynamic CNC/MIS
# ============================================================


class TestLimitEntryOnGoldenCross:
    def test_limit_entry_on_golden_cross(self):
        """Golden cross should produce a LIMIT BUY at bar.close, not a MARKET order."""
        s = SmaCrossover()
        config = {
            "fast_period": 2, "slow_period": 5, "atr_period": 3,
            "atr_stop_multiplier": 2.0, "min_spread": 0.005,
            "risk_per_trade": 0.02, "max_hold_bars": 50,
            "pyramid_levels": 0, "cnc_spread_threshold": 0.01,
        }
        s.initialize(config, {})

        # Warmup with declining prices
        warmup_bars = 5
        for i in range(warmup_bars):
            price = 100.0 - i * 2
            snap = make_snapshot(i, price, high=price + 5, low=price - 5)
            s.on_bar(snap)

        # Sharp upswing to trigger golden cross
        base = 100.0 - (warmup_bars - 1) * 2
        entry_signals = []
        for i in range(4):
            price = base + (i + 1) * 15
            snap = make_snapshot(warmup_bars + i, price, high=price + 5, low=price - 5)
            sigs = s.on_bar(snap)
            entry_signals.extend(sigs)

        limit_buys = [sig for sig in entry_signals if sig.action == "BUY" and sig.order_type == "LIMIT"]
        assert len(limit_buys) > 0, "Expected LIMIT BUY on golden cross"
        assert limit_buys[0].limit_price > 0, "Limit price should be set"

        state = s.states["TEST"]
        assert state.pending_entry, "Should be waiting for fill"
        assert state.pending_side == "BUY"


class TestCancelUnfilledLimitOnReversal:
    def test_cancel_unfilled_limit_on_reversal(self):
        """If crossover reverses before limit fills, CANCEL should be emitted."""
        s = SmaCrossover()
        config = {
            "fast_period": 2, "slow_period": 5, "atr_period": 3,
            "atr_stop_multiplier": 2.0, "min_spread": 0.005,
            "risk_per_trade": 0.02, "max_hold_bars": 50,
            "pyramid_levels": 0, "cnc_spread_threshold": 0.01,
        }
        s.initialize(config, {})

        # Warmup with declining prices
        warmup_bars = 5
        for i in range(warmup_bars):
            price = 100.0 - i * 2
            snap = make_snapshot(i, price, high=price + 5, low=price - 5)
            s.on_bar(snap)

        # Sharp upswing to trigger golden cross
        base = 100.0 - (warmup_bars - 1) * 2
        for i in range(4):
            price = base + (i + 1) * 15
            snap = make_snapshot(warmup_bars + i, price, high=price + 5, low=price - 5)
            s.on_bar(snap)

        state = s.states["TEST"]
        assert state.pending_entry, "Should be pending entry after golden cross"

        # Now crash prices to reverse the crossover (fast_above becomes False)
        reversal_signals = []
        for i in range(6):
            price = 30.0 - i * 10  # sharp decline
            if price < 10:
                price = 10.0
            snap = make_snapshot(warmup_bars + 4 + i, price, high=price + 5, low=price - 5)
            sigs = s.on_bar(snap)
            reversal_signals.extend(sigs)
            if not state.pending_entry:
                break

        cancel_sigs = [sig for sig in reversal_signals if sig.action == "CANCEL"]
        assert len(cancel_sigs) > 0, "Expected CANCEL when crossover reverses before fill"
        assert not state.pending_entry, "pending_entry should be cleared after cancellation"


class TestEngineStopSubmittedOnFill:
    def test_engine_stop_submitted_on_fill(self):
        """When limit entry fills, an SL-M SELL stop should be submitted."""
        s = SmaCrossover()
        config = {
            "fast_period": 2, "slow_period": 5, "atr_period": 3,
            "atr_stop_multiplier": 2.0, "min_spread": 0.005,
            "risk_per_trade": 0.02, "max_hold_bars": 50,
            "pyramid_levels": 0, "cnc_spread_threshold": 0.01,
        }
        s.initialize(config, {})

        # Warmup with declining prices
        warmup_bars = 5
        for i in range(warmup_bars):
            price = 100.0 - i * 2
            snap = make_snapshot(i, price, high=price + 5, low=price - 5)
            s.on_bar(snap)

        # Sharp upswing to trigger golden cross
        base = 100.0 - (warmup_bars - 1) * 2
        limit_price = 0.0
        limit_qty = 0
        for i in range(4):
            price = base + (i + 1) * 15
            snap = make_snapshot(warmup_bars + i, price, high=price + 5, low=price - 5)
            sigs = s.on_bar(snap)
            for sig in sigs:
                if sig.action == "BUY" and sig.order_type == "LIMIT":
                    limit_price = sig.limit_price
                    limit_qty = sig.quantity

        state = s.states["TEST"]
        assert state.pending_entry, "Should be pending"
        assert limit_qty > 0, "Should have a limit order"

        # Simulate fill on next bar
        fill_price = limit_price
        next_price = limit_price + 5
        snap = make_snapshot(
            warmup_bars + 4, next_price, high=next_price + 5, low=next_price - 5,
            positions=[Position("TEST", limit_qty, fill_price, 0.0)],
            fills=[FillInfo("TEST", "BUY", limit_qty, fill_price, 0.0, warmup_bars + 4)],
        )
        sigs = s.on_bar(snap)

        slm_sigs = [sig for sig in sigs if sig.order_type == "SL_M"]
        assert len(slm_sigs) == 1, "Expected exactly one SL-M stop after fill"
        assert slm_sigs[0].action == "SELL", "SL-M should be SELL for long position"
        assert slm_sigs[0].stop_price > 0, "Stop price should be set"
        assert slm_sigs[0].stop_price < fill_price, "Stop should be below fill price"
        assert state.has_engine_stop, "has_engine_stop should be True"
        assert not state.pending_entry, "pending_entry should be cleared"
        assert state.position_qty == limit_qty, "position_qty should match filled qty"


class TestTrailingStopRatchetCancelResubmit:
    def test_trailing_stop_ratchet_cancel_resubmit(self):
        """When trailing stop moves up, CANCEL + new SL-M should be emitted."""
        s = SmaCrossover()
        all_signals, entry_price = _warmup_and_enter(s)

        state = s.states["TEST"]
        total_qty = state.position_qty
        avg = state.avg_entry
        assert total_qty > 0, "Should be long"
        assert state.has_engine_stop, "Should have engine stop"

        initial_stop = state.trailing_stop

        # Feed a rising bar that raises the stop
        new_high = state.highest_since_entry + 50
        snap = make_snapshot(
            300, new_high, "TEST", high=new_high + 5, low=new_high - 5,
            positions=[Position("TEST", total_qty, avg, 0.0)],
        )
        sigs = s.on_bar(snap)

        assert state.trailing_stop > initial_stop, "Trailing stop should have ratcheted up"

        cancel_sigs = [sig for sig in sigs if sig.action == "CANCEL"]
        slm_sigs = [sig for sig in sigs if sig.order_type == "SL_M"]
        assert len(cancel_sigs) >= 1, "Should CANCEL old SL-M"
        assert len(slm_sigs) == 1, "Should resubmit new SL-M"
        assert slm_sigs[0].stop_price == state.trailing_stop, "New SL-M should use updated stop"


class TestDynamicCncStrongTrend:
    def test_dynamic_cnc_strong_trend(self):
        """Spread > cnc_spread_threshold (1%) should produce product_type='CNC'."""
        s = SmaCrossover()
        config = {
            "fast_period": 2, "slow_period": 5, "atr_period": 3,
            "atr_stop_multiplier": 2.0, "min_spread": 0.005,
            "risk_per_trade": 0.02, "max_hold_bars": 50,
            "pyramid_levels": 0, "cnc_spread_threshold": 0.01,
        }
        s.initialize(config, {})

        # Warmup with declining prices
        warmup_bars = 5
        for i in range(warmup_bars):
            price = 100.0 - i * 2
            snap = make_snapshot(i, price, high=price + 5, low=price - 5)
            s.on_bar(snap)

        # Very sharp upswing for strong spread (> 1%)
        base = 100.0 - (warmup_bars - 1) * 2
        all_sigs = []
        for i in range(4):
            price = base + (i + 1) * 20  # bigger jumps for stronger spread
            snap = make_snapshot(warmup_bars + i, price, high=price + 5, low=price - 5)
            sigs = s.on_bar(snap)
            all_sigs.extend(sigs)

        limit_buys = [sig for sig in all_sigs if sig.action == "BUY" and sig.order_type == "LIMIT"]
        assert len(limit_buys) > 0, "Expected LIMIT BUY"
        assert limit_buys[0].product_type == "CNC", (
            f"Expected CNC for strong trend, got {limit_buys[0].product_type}"
        )


class TestDynamicMisWeakTrend:
    def test_dynamic_mis_weak_trend(self):
        """Spread < cnc_spread_threshold (1%) should produce product_type='MIS'."""
        s = SmaCrossover()
        config = {
            "fast_period": 2, "slow_period": 5, "atr_period": 3,
            "atr_stop_multiplier": 2.0, "min_spread": 0.001,  # very low threshold to allow entry
            "risk_per_trade": 0.02, "max_hold_bars": 50,
            "pyramid_levels": 0, "cnc_spread_threshold": 10.0,  # very high CNC threshold -> always MIS
        }
        s.initialize(config, {})

        # Warmup with declining prices
        warmup_bars = 5
        for i in range(warmup_bars):
            price = 100.0 - i * 2
            snap = make_snapshot(i, price, high=price + 5, low=price - 5)
            s.on_bar(snap)

        # Gentle upswing (spread < 10.0 easily)
        base = 100.0 - (warmup_bars - 1) * 2
        all_sigs = []
        for i in range(4):
            price = base + (i + 1) * 15
            snap = make_snapshot(warmup_bars + i, price, high=price + 5, low=price - 5)
            sigs = s.on_bar(snap)
            all_sigs.extend(sigs)

        limit_buys = [sig for sig in all_sigs if sig.action == "BUY" and sig.order_type == "LIMIT"]
        assert len(limit_buys) > 0, "Expected LIMIT BUY"
        assert limit_buys[0].product_type == "MIS", (
            f"Expected MIS for weak trend, got {limit_buys[0].product_type}"
        )


class TestDeathCrossCancelsPending:
    def test_death_cross_cancels_pending(self):
        """On death cross exit from long, CANCEL should be emitted before SELL."""
        s = SmaCrossover()
        all_signals, entry_price = _warmup_and_enter(s)

        state = s.states["TEST"]
        total_qty = state.position_qty
        avg = state.avg_entry
        assert total_qty > 0, "Should be long"
        assert state.has_engine_stop, "Should have engine stop"

        # Feed sharply declining prices to trigger death cross
        high_price = state.highest_since_entry
        found_exit = False
        for i in range(10):
            price = high_price - (i + 1) * 25
            if price < 10:
                price = 10.0
            snap = make_snapshot(
                200 + i, price, "TEST", high=price + 5, low=price - 5,
                positions=[Position("TEST", total_qty, avg, 0.0)],
            )
            sigs = s.on_bar(snap)
            sell_sigs = [sig for sig in sigs if sig.action == "SELL" and sig.order_type == "MARKET"]
            cancel_sigs = [sig for sig in sigs if sig.action == "CANCEL"]
            if sell_sigs:
                assert len(cancel_sigs) > 0, "Must CANCEL pending orders before exit SELL"
                # CANCEL should come before SELL in the signal list
                cancel_idx = next(j for j, sig in enumerate(sigs) if sig.action == "CANCEL")
                sell_idx = next(j for j, sig in enumerate(sigs) if sig.action == "SELL" and sig.order_type == "MARKET")
                assert cancel_idx < sell_idx, "CANCEL must precede SELL in signal list"
                found_exit = True
                break

        assert found_exit, "Expected death cross exit"
        assert not state.has_engine_stop, "Engine stop flag should be cleared after exit"


class TestStopHitResetsState:
    def test_stop_hit_resets_state(self):
        """When SL-M fill appears in snapshot, state must fully reset."""
        s = SmaCrossover()
        all_signals, entry_price = _warmup_and_enter(s)

        state = s.states["TEST"]
        total_qty = state.position_qty
        assert total_qty > 0, "Should be long"
        assert state.has_engine_stop, "Should have engine stop"

        # Simulate engine stop hit
        snap = make_snapshot(
            400, state.trailing_stop - 10, "TEST",
            high=state.trailing_stop - 5, low=state.trailing_stop - 15,
            positions=[],  # engine closed position
            fills=[FillInfo("TEST", "SELL", total_qty, state.trailing_stop, 0.0, 400)],
        )
        sigs = s.on_bar(snap)

        # Verify full state reset
        assert state.position_qty == 0
        assert state.pyramid_count == 0
        assert state.original_qty == 0
        assert state.entry_price == 0.0
        assert state.avg_entry == 0.0
        assert state.highest_since_entry == 0.0
        assert state.trailing_stop == 0.0
        assert state.bars_in_position == 0
        assert not state.has_engine_stop
        assert not state.pending_entry
