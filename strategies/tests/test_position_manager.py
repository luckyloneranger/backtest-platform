"""Tests for PositionManager order lifecycle management."""

import pytest

from strategies.base import (
    BarData, FillInfo, MarketSnapshot, PendingOrder, Portfolio,
    Position, SessionContext, Signal,
)
from strategies.position_manager import PositionManager, PositionState


DEFAULT_CAPITAL = 1_000_000.0


def make_snapshot(
    fills: list[FillInfo] | None = None,
    pending_orders: list[PendingOrder] | None = None,
    positions: list[Position] | None = None,
    capital: float = DEFAULT_CAPITAL,
    ts: int = 1_000_000,
) -> MarketSnapshot:
    """Minimal snapshot builder for PositionManager tests."""
    bar = BarData("TEST", 100.0, 105.0, 95.0, 100.0, 10000, 0, ts)
    return MarketSnapshot(
        timestamp_ms=ts,
        timeframes={"day": {"TEST": bar}},
        history={},
        portfolio=Portfolio(
            cash=capital,
            equity=capital,
            positions=positions or [],
        ),
        instruments={},
        fills=fills or [],
        rejections=[],
        closed_trades=[],
        context=SessionContext(capital, 1, 200, "2024-01-01", "2024-12-31", ["day"], 200),
        pending_orders=pending_orders or [],
    )


class TestEntryMethods:
    def test_enter_long_emits_limit_buy(self):
        pm = PositionManager()
        signals = pm.enter_long("TEST", 100, limit_price=500.0,
                                product_type="CNC", stop_price=480.0)
        assert len(signals) == 1
        s = signals[0]
        assert s.action == "BUY"
        assert s.symbol == "TEST"
        assert s.quantity == 100
        assert s.order_type == "LIMIT"
        assert s.limit_price == 500.0
        assert s.product_type == "CNC"
        # State should be pending
        state = pm.get_state("TEST")
        assert state.pending_entry is True
        assert state.pending_side == "BUY"
        assert state.trailing_stop == 480.0

    def test_enter_long_market(self):
        pm = PositionManager()
        signals = pm.enter_long("TEST", 50, limit_price=0,
                                product_type="MIS", stop_price=95.0)
        assert len(signals) == 1
        s = signals[0]
        assert s.order_type == "MARKET"
        assert s.limit_price == 0
        assert s.product_type == "MIS"

    def test_enter_short_forces_mis(self):
        pm = PositionManager()
        signals = pm.enter_short("TEST", 100, limit_price=500.0, stop_price=520.0)
        assert len(signals) == 1
        s = signals[0]
        assert s.action == "SELL"
        assert s.product_type == "MIS"
        assert s.order_type == "LIMIT"
        state = pm.get_state("TEST")
        assert state.product_type == "MIS"
        assert state.trailing_stop == 520.0

    def test_no_double_entry(self):
        pm = PositionManager()
        pm.enter_long("TEST", 100, limit_price=500.0,
                      product_type="CNC", stop_price=480.0)
        # Second entry should be blocked because pending_entry is True
        signals = pm.enter_long("TEST", 50, limit_price=490.0,
                                product_type="CNC", stop_price=470.0)
        assert signals == []

        # Also blocked for short entry on the same symbol
        signals = pm.enter_short("TEST", 50, limit_price=490.0, stop_price=510.0)
        assert signals == []


class TestExitMethods:
    def test_exit_emits_cancel_plus_market(self):
        pm = PositionManager()
        # Set up an active long position
        state = pm.get_state("TEST")
        state.direction = "long"
        state.qty = 100
        state.product_type = "CNC"
        state.has_engine_stop = True

        signals = pm.exit_position("TEST")
        assert len(signals) == 2
        assert signals[0].action == "CANCEL"
        assert signals[0].symbol == "TEST"
        assert signals[1].action == "SELL"
        assert signals[1].quantity == 100
        assert signals[1].order_type == "MARKET"
        # State should be reset
        assert state.direction == "flat"

    def test_partial_exit(self):
        pm = PositionManager()
        state = pm.get_state("TEST")
        state.direction = "long"
        state.qty = 100
        state.product_type = "CNC"
        state.has_engine_stop = True

        signals = pm.exit_position("TEST", qty=40)
        assert len(signals) == 2
        assert signals[1].quantity == 40
        # Position should still be open with remaining qty
        assert state.direction == "long"
        assert state.qty == 60
        assert state.partial_taken is True


class TestProcessFills:
    def test_process_fills_entry_fill(self):
        """BUY fill on pending entry sets direction and submits SL-M."""
        pm = PositionManager()
        pm.enter_long("TEST", 100, limit_price=500.0,
                      product_type="CNC", stop_price=480.0)

        fill = FillInfo("TEST", "BUY", 100, 500.0, 10.0, 1_000_000)
        snap = make_snapshot(fills=[fill])
        signals = pm.process_fills(snap)

        state = pm.get_state("TEST")
        assert state.direction == "long"
        assert state.qty == 100
        assert state.avg_entry == 500.0
        assert state.has_engine_stop is True

        # Should submit SL-M stop
        assert len(signals) == 1
        sl = signals[0]
        assert sl.action == "SELL"
        assert sl.order_type == "SL_M"
        assert sl.stop_price == 480.0
        assert sl.quantity == 100

    def test_process_fills_stop_hit(self):
        """SELL fill on engine stop resets state to flat."""
        pm = PositionManager()
        state = pm.get_state("TEST")
        state.direction = "long"
        state.qty = 100
        state.has_engine_stop = True
        state.trailing_stop = 480.0
        state.product_type = "CNC"

        fill = FillInfo("TEST", "SELL", 100, 479.0, 5.0, 2_000_000)
        snap = make_snapshot(fills=[fill])
        signals = pm.process_fills(snap)

        assert state.direction == "flat"
        assert state.qty == 0
        assert signals == []

    def test_process_fills_profit_target_fill(self):
        """Profit target fill reduces qty and moves stop to breakeven."""
        pm = PositionManager()
        state = pm.get_state("TEST")
        state.direction = "long"
        state.qty = 100
        state.avg_entry = 500.0
        state.has_engine_stop = True
        state.has_profit_target = True
        state.profit_target_price = 550.0
        state.profit_target_qty = 50
        state.trailing_stop = 480.0
        state.product_type = "CNC"

        fill = FillInfo("TEST", "SELL", 50, 550.0, 5.0, 3_000_000)
        snap = make_snapshot(fills=[fill])
        signals = pm.process_fills(snap)

        # Qty reduced, profit target cleared
        assert state.qty == 50
        assert state.has_profit_target is False
        assert state.partial_taken is True

        # Stop moved to breakeven: CANCEL + SL-M
        assert len(signals) == 2
        assert signals[0].action == "CANCEL"
        assert signals[1].order_type == "SL_M"
        assert signals[1].stop_price == 500.0  # avg_entry
        assert signals[1].quantity == 50

    def test_process_fills_pyramid_fill(self):
        """Pyramid fill updates avg entry and resubmits SL-M for total qty."""
        pm = PositionManager()
        state = pm.get_state("TEST")
        state.direction = "long"
        state.qty = 100
        state.avg_entry = 500.0
        state.has_engine_stop = True
        state.trailing_stop = 480.0
        state.product_type = "CNC"
        # Simulate pending pyramid entry
        state.pending_entry = True
        state.pending_side = "BUY"
        state.pending_qty = 50

        fill = FillInfo("TEST", "BUY", 50, 520.0, 8.0, 4_000_000)
        snap = make_snapshot(fills=[fill])
        signals = pm.process_fills(snap)

        # Avg entry updated: (500*100 + 520*50) / 150 = 76000/150 = 506.67
        assert state.qty == 150
        expected_avg = (500.0 * 100 + 520.0 * 50) / 150
        assert abs(state.avg_entry - expected_avg) < 0.01
        assert state.pyramid_count == 1

        # Should cancel old SL and resubmit for total qty
        assert len(signals) == 2
        assert signals[0].action == "CANCEL"
        assert signals[1].order_type == "SL_M"
        assert signals[1].quantity == 150


class TestStaleAndExpired:
    def test_stale_entry_cancelled(self):
        """Pending entry older than max_pending_bars gets cancelled."""
        pm = PositionManager(max_pending_bars=2)
        pm.enter_long("TEST", 100, limit_price=500.0,
                      product_type="CNC", stop_price=480.0)

        # Advance bar count past max_pending_bars
        pm.bar_count = 4

        pending = PendingOrder("TEST", "BUY", 100, "LIMIT", 500.0, 0.0)
        snap = make_snapshot(pending_orders=[pending])
        signals = pm.process_fills(snap)

        # Should emit CANCEL for stale order
        assert len(signals) == 1
        assert signals[0].action == "CANCEL"
        assert signals[0].symbol == "TEST"

        # State should be cleared
        state = pm.get_state("TEST")
        assert state.pending_entry is False

    def test_resubmit_expired_stop(self):
        """SL-M stop missing from pending_orders gets resubmitted."""
        pm = PositionManager()
        state = pm.get_state("TEST")
        state.direction = "long"
        state.qty = 100
        state.has_engine_stop = True
        state.trailing_stop = 480.0
        state.product_type = "CNC"

        # No pending orders -- stop expired
        snap = make_snapshot(pending_orders=[])
        signals = pm.resubmit_expired(snap)

        assert len(signals) == 1
        s = signals[0]
        assert s.action == "SELL"
        assert s.order_type == "SL_M"
        assert s.stop_price == 480.0
        assert s.quantity == 100

    def test_resubmit_expired_profit_target(self):
        """LIMIT profit target missing from pending_orders gets resubmitted."""
        pm = PositionManager()
        state = pm.get_state("TEST")
        state.direction = "long"
        state.qty = 100
        state.has_engine_stop = True
        state.trailing_stop = 480.0
        state.has_profit_target = True
        state.profit_target_price = 550.0
        state.profit_target_qty = 50
        state.product_type = "CNC"

        # Only SL-M present, no LIMIT profit target
        pending = PendingOrder("TEST", "SELL", 100, "SL_M", 0.0, 480.0)
        snap = make_snapshot(pending_orders=[pending])
        signals = pm.resubmit_expired(snap)

        assert len(signals) == 1
        s = signals[0]
        assert s.action == "SELL"
        assert s.order_type == "LIMIT"
        assert s.limit_price == 550.0
        assert s.quantity == 50

    def test_resubmit_no_action_when_present(self):
        """No resubmission when all engine orders are present."""
        pm = PositionManager()
        state = pm.get_state("TEST")
        state.direction = "long"
        state.qty = 100
        state.has_engine_stop = True
        state.trailing_stop = 480.0
        state.product_type = "CNC"

        # SL-M is present
        pending = PendingOrder("TEST", "SELL", 100, "SL_M", 0.0, 480.0)
        snap = make_snapshot(pending_orders=[pending])
        signals = pm.resubmit_expired(snap)

        assert signals == []


class TestReconcile:
    def test_reconcile_resets_disappeared_position(self):
        """Position gone from portfolio resets internal state."""
        pm = PositionManager()
        state = pm.get_state("TEST")
        state.direction = "long"
        state.qty = 100
        state.has_engine_stop = True

        # Empty portfolio -- position disappeared (e.g., MIS squareoff)
        snap = make_snapshot(positions=[])
        pm.reconcile(snap)

        assert state.direction == "flat"
        assert state.qty == 0
        assert state.has_engine_stop is False


class TestTrailingStop:
    def test_update_trailing_stop_ratchets_up(self):
        """Trailing stop for long position ratchets up."""
        pm = PositionManager()
        state = pm.get_state("TEST")
        state.direction = "long"
        state.qty = 100
        state.has_engine_stop = True
        state.trailing_stop = 480.0
        state.product_type = "CNC"

        signals = pm.update_trailing_stop("TEST", 490.0)
        assert len(signals) == 2
        assert signals[0].action == "CANCEL"
        assert signals[1].order_type == "SL_M"
        assert signals[1].stop_price == 490.0
        assert signals[1].quantity == 100
        assert state.trailing_stop == 490.0

    def test_update_trailing_stop_no_change(self):
        """Trailing stop for long position does not move down."""
        pm = PositionManager()
        state = pm.get_state("TEST")
        state.direction = "long"
        state.qty = 100
        state.has_engine_stop = True
        state.trailing_stop = 490.0
        state.product_type = "CNC"

        signals = pm.update_trailing_stop("TEST", 485.0)
        assert signals == []
        assert state.trailing_stop == 490.0  # unchanged


class TestQueryMethods:
    def test_is_flat_long_short_queries(self):
        """Query methods reflect current direction."""
        pm = PositionManager()
        assert pm.is_flat("TEST") is True
        assert pm.is_long("TEST") is False
        assert pm.is_short("TEST") is False

        state = pm.get_state("TEST")
        state.direction = "long"
        assert pm.is_flat("TEST") is False
        assert pm.is_long("TEST") is True
        assert pm.is_short("TEST") is False

        state.direction = "short"
        assert pm.is_flat("TEST") is False
        assert pm.is_long("TEST") is False
        assert pm.is_short("TEST") is True


class TestPyramidAndProfitTarget:
    def test_add_pyramid_emits_signal(self):
        """Pyramid on existing long emits BUY signal."""
        pm = PositionManager()
        state = pm.get_state("TEST")
        state.direction = "long"
        state.qty = 100
        state.product_type = "CNC"

        signals = pm.add_pyramid("TEST", 50, limit_price=510.0)
        assert len(signals) == 1
        s = signals[0]
        assert s.action == "BUY"
        assert s.quantity == 50
        assert s.order_type == "LIMIT"
        assert s.limit_price == 510.0
        assert s.product_type == "CNC"

        # Should set pending entry
        assert state.pending_entry is True
        assert state.pending_side == "BUY"

    def test_set_profit_target(self):
        """Set profit target on long emits SELL LIMIT."""
        pm = PositionManager()
        state = pm.get_state("TEST")
        state.direction = "long"
        state.qty = 100
        state.product_type = "CNC"

        signals = pm.set_profit_target("TEST", 50, limit_price=550.0)
        assert len(signals) == 1
        s = signals[0]
        assert s.action == "SELL"
        assert s.quantity == 50
        assert s.order_type == "LIMIT"
        assert s.limit_price == 550.0

        assert state.has_profit_target is True
        assert state.profit_target_price == 550.0
        assert state.profit_target_qty == 50
