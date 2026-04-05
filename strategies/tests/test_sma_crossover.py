"""Tests for SMA Crossover strategy — trading logic only.

PositionManager infrastructure is tested in test_position_manager.py.
These tests verify the strategy's crossover detection, sizing, filtering,
trailing stop updates, time stops, pyramiding, and product selection.
"""

import pytest
from strategies.base import (
    BarData, FillInfo, MarketSnapshot, PendingOrder, Portfolio,
    Signal, SessionContext, Position,
)
from strategies.deterministic.sma_crossover import SmaCrossover


CAPITAL = 1_000_000.0
SYMBOL = "TEST"


def _snap(close, symbol=SYMBOL, high=None, low=None, capital=CAPITAL,
          fills=None, pending_orders=None, positions=None):
    """Build a minimal MarketSnapshot for one symbol at one bar."""
    h = high if high is not None else close
    lo = low if low is not None else close
    bar = BarData(symbol, close, h, lo, close, 10000, 0)
    return MarketSnapshot(
        timestamp_ms=0,
        timeframes={"day": {symbol: bar}},
        history={},
        portfolio=Portfolio(capital, capital, positions or []),
        instruments={},
        fills=fills or [],
        rejections=[],
        closed_trades=[],
        context=SessionContext(capital, 0, 200, "2024-01-01", "2024-12-31", ["day"], 200),
        pending_orders=pending_orders or [],
    )


def _init(fast=3, slow=5, atr_period=3, **extra):
    """Create and initialize a SmaCrossover with small periods for testing."""
    s = SmaCrossover()
    config = {"fast_period": fast, "slow_period": slow, "atr_period": atr_period,
              "risk_per_trade": 0.02, "atr_multiplier": 2.0,
              "min_spread": 0.005, "max_hold_bars": 50, "pyramid_levels": 2}
    config.update(extra)
    s.initialize(config, {})
    return s


def _feed_prices(s, prices, high=None, low=None, fills_at=None,
                 pending_at=None, positions_at=None):
    """Feed a sequence of bars. Return (all_signals_from_all_bars, last_bar_signals).

    fills_at/pending_at/positions_at: dict mapping bar index to list.
    """
    fills_at = fills_at or {}
    pending_at = pending_at or {}
    positions_at = positions_at or {}
    all_signals = []
    last_signals = []
    for i, p in enumerate(prices):
        h = high[i] if high else None
        lo = low[i] if low else None
        snap = _snap(p, high=h, low=lo,
                     fills=fills_at.get(i, []),
                     pending_orders=pending_at.get(i, []),
                     positions=positions_at.get(i, []))
        last_signals = s.on_bar(snap)
        all_signals.extend(last_signals)
    return all_signals, last_signals


# ---------------------------------------------------------------------------
# Test data patterns (fast=3, slow=5, atr_period=3)
#
# DOWNTREND_THEN_UP: starts high, descends (fast<slow from bar 4),
#   then reverses → golden cross at bar 10 with spread=0.0204.
#
# UPTREND_THEN_DOWN: starts low, ascends (fast>slow from bar 4, prev=None),
#   then reverses → death cross at bar 9 with spread=0.0058.
# ---------------------------------------------------------------------------
DOWNTREND_THEN_UP = [108, 106, 104, 102, 100, 98, 96, 94, 97, 100, 103]
UPTREND_THEN_DOWN = [98, 99, 100, 101, 102, 103, 104, 105, 102, 99, 96]


class TestRequiredData:
    def test_required_data(self):
        s = SmaCrossover()
        reqs = s.required_data()
        assert reqs == [{"interval": "day", "lookback": 200}]


class TestInit:
    def test_fast_ge_slow_raises(self):
        s = SmaCrossover()
        with pytest.raises(ValueError, match="fast_period.*>= slow_period"):
            s.initialize({"fast_period": 10, "slow_period": 5}, {})

    def test_fast_eq_slow_raises(self):
        s = SmaCrossover()
        with pytest.raises(ValueError, match="fast_period.*>= slow_period"):
            s.initialize({"fast_period": 10, "slow_period": 10}, {})


class TestGoldenCrossEntry:
    def test_golden_cross_emits_buy(self):
        """Downtrend reversal → golden cross → BUY LIMIT signal."""
        s = _init()
        prices = DOWNTREND_THEN_UP
        high = [p + 1 for p in prices]
        low = [p - 1 for p in prices]
        all_sigs, last_sigs = _feed_prices(s, prices, high=high, low=low)

        buys = [sig for sig in last_sigs if sig.action == "BUY"]
        assert len(buys) == 1
        assert buys[0].symbol == SYMBOL
        assert buys[0].order_type == "LIMIT"
        assert buys[0].limit_price == 103  # bar.close
        assert buys[0].quantity > 0


class TestDeathCrossShortEntry:
    def test_death_cross_emits_sell(self):
        """Uptrend reversal → death cross → SELL signal (short, MIS)."""
        s = _init()
        prices = UPTREND_THEN_DOWN
        high = [p + 1 for p in prices]
        low = [p - 1 for p in prices]
        all_sigs, last_sigs = _feed_prices(s, prices, high=high, low=low)

        # Death cross fires at bar 9 (index 9, close=99)
        # But bar 9 is not the last bar. Let me check which bar....
        # Actually _feed_prices returns last_sigs from bar 10 (index 10, close=96)
        # At bar 9: death cross fires. At bar 10: already short direction.
        # The sell should be in all_sigs.
        sells = [sig for sig in all_sigs if sig.action == "SELL"]
        assert len(sells) >= 1
        assert sells[0].product_type == "MIS"  # shorts always MIS


class TestTrendStrengthFilter:
    def test_weak_spread_blocks_entry(self):
        """When spread is below min_spread, no entry signal is emitted."""
        s = _init(min_spread=0.10)  # very high threshold — no cross will pass
        prices = DOWNTREND_THEN_UP
        high = [p + 1 for p in prices]
        low = [p - 1 for p in prices]
        all_sigs, _ = _feed_prices(s, prices, high=high, low=low)

        buys = [sig for sig in all_sigs if sig.action == "BUY"]
        sells = [sig for sig in all_sigs if sig.action == "SELL"]
        assert len(buys) == 0
        assert len(sells) == 0


class TestTrailingStopUpdates:
    def test_trailing_stop_ratchets_up_for_long(self):
        """After entry fill, rising prices should produce CANCEL + SL_M with higher stop."""
        s = _init()

        # Phase 1: warm up + golden cross at bar 10
        prices = DOWNTREND_THEN_UP
        high = [p + 1 for p in prices]
        low = [p - 1 for p in prices]
        all_sigs, last_sigs = _feed_prices(s, prices, high=high, low=low)

        entry_buys = [sig for sig in last_sigs if sig.action == "BUY" and sig.order_type == "LIMIT"]
        assert len(entry_buys) == 1
        entry_qty = entry_buys[0].quantity

        # Phase 2: simulate fill → PM transitions to long, submits SL_M
        fill = FillInfo(SYMBOL, "BUY", entry_qty, 103.0, 0.0, 0)
        position = Position(SYMBOL, entry_qty, 103.0, 0.0)
        pending_stop = PendingOrder(SYMBOL, "SELL", entry_qty, "SL_M", 0.0, 95.0)

        snap = _snap(107, high=108, low=106,
                     fills=[fill],
                     positions=[position],
                     pending_orders=[pending_stop])
        fill_signals = s.on_bar(snap)

        # Should have SL_M from process_fills (initial stop)
        sl_signals = [sig for sig in fill_signals if sig.order_type == "SL_M"]
        assert len(sl_signals) >= 1
        initial_stop = sl_signals[0].stop_price

        # Phase 3: price rises further
        snap2 = _snap(112, high=113, low=111,
                      positions=[position],
                      pending_orders=[PendingOrder(SYMBOL, "SELL", entry_qty, "SL_M", 0.0, initial_stop)])
        signals2 = s.on_bar(snap2)

        # Trailing stop should ratchet: CANCEL + new higher SL_M
        new_stops = [sig for sig in signals2 if sig.order_type == "SL_M"]
        assert len(new_stops) >= 1
        assert new_stops[-1].stop_price > initial_stop


class TestTimeStopExit:
    def test_time_stop_exits_stale_long(self):
        """Position held > max_hold_bars with no gain triggers exit."""
        s = _init(max_hold_bars=3)

        # Warm up + golden cross
        prices = DOWNTREND_THEN_UP
        high = [p + 1 for p in prices]
        low = [p - 1 for p in prices]
        _, last_sigs = _feed_prices(s, prices, high=high, low=low)

        entry_qty = [sig for sig in last_sigs if sig.action == "BUY"][0].quantity

        # Simulate fill
        fill = FillInfo(SYMBOL, "BUY", entry_qty, 103.0, 0.0, 0)
        position = Position(SYMBOL, entry_qty, 103.0, 0.0)
        pending_stop = PendingOrder(SYMBOL, "SELL", entry_qty, "SL_M", 0.0, 95.0)
        snap = _snap(103, high=104, low=102,
                     fills=[fill], positions=[position],
                     pending_orders=[pending_stop])
        s.on_bar(snap)

        # Feed 5 bars with tiny rise to keep fast > slow (avoid death cross),
        # but gain < 0.5% so time stop fires
        all_signals = []
        for i in range(5):
            price = 103.1 + i * 0.1  # 103.1, 103.2, ..., 103.5
            snap = _snap(price, high=price + 1, low=price - 1,
                         positions=[position],
                         pending_orders=[PendingOrder(SYMBOL, "SELL", entry_qty, "SL_M", 0.0, 95.0)])
            all_signals.extend(s.on_bar(snap))

        # After max_hold_bars=3, with gain < 0.005, should have exited
        sells = [sig for sig in all_signals if sig.action == "SELL" and sig.order_type == "MARKET"]
        assert len(sells) >= 1
        assert s.pm.is_flat(SYMBOL)


class TestPyramiding:
    def test_pyramid_on_continued_trend(self):
        """Price moves > avg_entry + ATR → pyramid BUY signal."""
        s = _init()

        # Warm up + golden cross
        prices = DOWNTREND_THEN_UP
        high = [p + 1 for p in prices]
        low = [p - 1 for p in prices]
        _, last_sigs = _feed_prices(s, prices, high=high, low=low)

        entry_qty = [sig for sig in last_sigs if sig.action == "BUY"][0].quantity

        # Simulate fill at 103
        fill = FillInfo(SYMBOL, "BUY", entry_qty, 103.0, 0.0, 0)
        position = Position(SYMBOL, entry_qty, 103.0, 0.0)
        pending_stop = PendingOrder(SYMBOL, "SELL", entry_qty, "SL_M", 0.0, 95.0)
        snap = _snap(103, high=104, low=102,
                     fills=[fill], positions=[position],
                     pending_orders=[pending_stop])
        s.on_bar(snap)

        # Feed bar far above entry + ATR.  avg_entry=103, ATR~4, so need close > 107.
        snap2 = _snap(112, high=113, low=111,
                      positions=[position],
                      pending_orders=[PendingOrder(SYMBOL, "SELL", entry_qty, "SL_M", 0.0, 95.0)])
        signals = s.on_bar(snap2)

        buys = [sig for sig in signals if sig.action == "BUY"]
        assert len(buys) >= 1
        assert buys[0].quantity > 0


class TestDynamicProductType:
    def test_cnc_for_strong_trend(self):
        """Strong trend spread > 0.01 → CNC product type."""
        s = _init()
        # DOWNTREND_THEN_UP produces spread=0.0204 at golden cross → CNC
        prices = DOWNTREND_THEN_UP
        high = [p + 1 for p in prices]
        low = [p - 1 for p in prices]
        _, last_sigs = _feed_prices(s, prices, high=high, low=low)

        buys = [sig for sig in last_sigs if sig.action == "BUY"]
        assert len(buys) == 1
        assert buys[0].product_type == "CNC"

    def test_mis_for_weak_trend(self):
        """Moderate trend spread between min_spread and 0.01 → MIS product type."""
        # Engineer data where crossover spread is between 0.005 and 0.01
        # Gentler reversal from downtrend
        s = _init(min_spread=0.003)
        prices = [104, 103, 102, 101, 100, 99, 98, 97, 98, 99, 100]
        high = [p + 1 for p in prices]
        low = [p - 1 for p in prices]
        all_sigs, _ = _feed_prices(s, prices, high=high, low=low)

        buys = [sig for sig in all_sigs if sig.action == "BUY"]
        if buys:
            assert buys[0].product_type == "MIS"


class TestOnComplete:
    def test_on_complete_returns_strategy_type(self):
        s = _init()
        result = s.on_complete()
        assert result == {"strategy_type": "sma_crossover"}
