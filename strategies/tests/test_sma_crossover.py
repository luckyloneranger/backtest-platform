import pytest

from strategies.base import BarData, MarketSnapshot, Portfolio, Signal, SessionContext, InstrumentInfo, Position
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
        fills=[],
        rejections=[],
        closed_trades=[],
        context=SessionContext(capital, ts, 200, "2024-01-01", "2024-12-31", ["day"], 200),
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

    After this helper returns, the strategy has an open position reflected in
    its internal state.  Subsequent bars fed to on_bar MUST include a matching
    Position in the snapshot so that _reconcile_position does not reset.
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
    }
    if extra_config:
        config.update(extra_config)
    s.initialize(config, {})

    all_signals: list[Signal] = []
    entry_price: float = 0.0
    position_qty: int = 0
    avg_entry: float = 0.0

    # Phase 1: warmup with declining prices
    warmup_bars = max(slow_period, atr_period + 1)
    for i in range(warmup_bars):
        price = 100.0 - i * 2
        positions = []
        if position_qty > 0:
            positions = [Position(symbol, position_qty, avg_entry, 0.0)]
        snap = make_snapshot(i, price, symbol, high=price + 5, low=price - 5, positions=positions)
        sigs = s.on_bar(snap)
        for sig in sigs:
            if sig.action == "BUY":
                position_qty += sig.quantity
                avg_entry = price
                entry_price = price
        all_signals.extend(sigs)

    # Phase 2: sharp upswing to trigger golden cross with strong spread
    base = 100.0 - (warmup_bars - 1) * 2
    for i in range(4):
        price = base + (i + 1) * 15
        positions = []
        if position_qty > 0:
            positions = [Position(symbol, position_qty, avg_entry, 0.0)]
        snap = make_snapshot(warmup_bars + i, price, symbol, high=price + 5, low=price - 5, positions=positions)
        sigs = s.on_bar(snap)
        for sig in sigs:
            if sig.action == "BUY":
                if position_qty == 0:
                    avg_entry = price
                    entry_price = price
                else:
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

        buy_signals = [sig for sig in all_signals if sig.action == "BUY"]
        assert len(buy_signals) > 0, "Expected at least one BUY signal"
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
    def test_trailing_stop_exit(self):
        """After entry, if price drops below trailing stop, a SELL should fire."""
        s = SmaCrossover()
        all_signals, entry_price = _warmup_and_enter(s)

        buy_signals = [sig for sig in all_signals if sig.action == "BUY"]
        assert len(buy_signals) > 0, "Need a BUY to test trailing stop"

        state = s.states["TEST"]
        total_qty = state.position_qty
        avg = state.avg_entry
        assert total_qty > 0, "Strategy should have an open position"

        # Feed a few rising bars to push the trailing stop higher
        current_highest = state.highest_since_entry
        for i in range(3):
            rise_price = current_highest + (i + 1) * 10
            snap = make_snapshot(
                100 + i, rise_price, "TEST", high=rise_price + 5, low=rise_price - 5,
                positions=[Position("TEST", total_qty, avg, 0.0)],
            )
            sigs = s.on_bar(snap)
            assert not any(sig.action == "SELL" for sig in sigs), (
                f"Should not sell on rising price bar {i}"
            )

        # Trailing stop should have risen
        new_stop = state.trailing_stop
        assert new_stop > 0, "Trailing stop should be positive after rising bars"

        # Crash below trailing stop
        crash_price = new_stop - 10
        snap = make_snapshot(
            200, crash_price, "TEST", high=crash_price + 1, low=crash_price - 1,
            positions=[Position("TEST", total_qty, avg, 0.0)],
        )
        sigs = s.on_bar(snap)
        sell_signals = [sig for sig in sigs if sig.action == "SELL"]
        assert len(sell_signals) == 1, "Expected SELL on trailing stop breach"
        assert sell_signals[0].quantity == total_qty, "Should sell entire position"


class TestDeathCrossExit:
    def test_death_cross_exit(self):
        """After entry, a death cross (fast drops below slow) should trigger SELL."""
        s = SmaCrossover()
        all_signals, entry_price = _warmup_and_enter(s)

        buy_signals = [sig for sig in all_signals if sig.action == "BUY"]
        assert len(buy_signals) > 0, "Need a BUY to test death cross"

        state = s.states["TEST"]
        total_qty = state.position_qty
        avg = state.avg_entry
        assert total_qty > 0, "Strategy should have an open position"

        # Keep trailing stop very low so it doesn't interfere
        state.trailing_stop = 0.0

        # Feed sharply declining prices to trigger death cross
        # Start from the current highest and drop hard
        high_price = state.highest_since_entry
        found_sell = False
        for i in range(10):
            price = high_price - (i + 1) * 25
            if price < 10:
                price = 10.0
            # Disable trailing stop every bar so only death cross triggers exit
            state.trailing_stop = 0.0
            snap = make_snapshot(
                200 + i, price, "TEST", high=price + 5, low=price - 5,
                positions=[Position("TEST", total_qty, avg, 0.0)],
            )
            sigs = s.on_bar(snap)
            sell_sigs = [sig for sig in sigs if sig.action == "SELL"]
            if sell_sigs:
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

        buy_signals = [sig for sig in all_signals if sig.action == "BUY"]
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
            # Keep trailing stop very low to prevent sell
            state.trailing_stop = 0.0
            snap = make_snapshot(
                300 + i, price, "TEST", high=price + 5, low=price - 5,
                positions=[Position("TEST", state.position_qty, avg, 0.0)],
            )
            sigs = s.on_bar(snap)
            pyramid_buys = [sig for sig in sigs if sig.action == "BUY"]
            if pyramid_buys:
                expected_add = max(1, int(original_qty * 0.5))
                assert pyramid_buys[0].quantity == expected_add, (
                    f"Pyramid qty should be {expected_add}, got {pyramid_buys[0].quantity}"
                )
                found_pyramid = True
                break

        assert found_pyramid, "Expected a pyramid BUY when price is well above avg_entry + ATR"
