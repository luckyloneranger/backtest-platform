"""Tests for Multi-Indicator Confluence strategy.

Tests verify scoring logic, entry/exit decisions, ADX amplification,
trailing stop management, time stops, and ATR-based position sizing.
"""

import pytest
from unittest.mock import patch
from strategies.base import (
    BarData, FillInfo, MarketSnapshot, PendingOrder, Portfolio,
    Signal, SessionContext, Position,
)
from strategies.deterministic.confluence import Confluence


CAPITAL = 1_000_000.0
SYMBOL = "TEST"


def _snap(close, symbol=SYMBOL, high=None, low=None, volume=10000,
          capital=CAPITAL, fills=None, pending_orders=None, positions=None):
    """Build a minimal MarketSnapshot for one symbol at one bar."""
    h = high if high is not None else close
    lo = low if low is not None else close
    bar = BarData(symbol, close, h, lo, close, volume, 0)
    return MarketSnapshot(
        timestamp_ms=0,
        timeframes={"day": {symbol: bar}},
        history={},
        portfolio=Portfolio(capital, capital, positions or []),
        instruments={},
        fills=fills or [],
        rejections=[],
        closed_trades=[],
        context=SessionContext(capital, 0, 200, "2024-01-01", "2024-12-31", ["day"], 50),
        pending_orders=pending_orders or [],
    )


def _init(**overrides):
    """Create and initialize a Confluence strategy with small periods for testing."""
    s = Confluence()
    config = {
        "rsi_period": 5, "rsi_oversold": 35, "rsi_overbought": 65,
        "macd_fast": 3, "macd_slow": 5, "macd_signal": 2,
        "bb_period": 5, "bb_std": 2.0,
        "adx_period": 5, "adx_trend_threshold": 25,
        "obv_period": 5,
        "atr_period": 3, "atr_multiplier": 2.0,
        "risk_per_trade": 0.02, "threshold": 3,
        "max_hold_bars": 30, "min_gain_for_hold": 0.005,
    }
    config.update(overrides)
    s.initialize(config, {})
    return s


def _warm_up(s, n=15):
    """Feed n bars of stable data to fill indicator buffers."""
    for i in range(n):
        price = 100 + (i % 3) * 0.5  # slight oscillation
        snap = _snap(price, high=price + 1, low=price - 1)
        s.on_bar(snap)


class TestRequiredData:
    def test_required_data(self):
        s = Confluence()
        reqs = s.required_data()
        assert reqs == [{"interval": "day", "lookback": 50}]


class TestHighConfluenceLongEntry:
    def test_high_confluence_long_entry(self):
        """When _compute_score returns score >= threshold, a BUY signal is emitted."""
        s = _init()
        _warm_up(s)

        with patch.object(s, "_compute_score", return_value=(3, 30.0)):
            snap = _snap(100, high=101, low=99)
            signals = s.on_bar(snap)

        buys = [sig for sig in signals if sig.action == "BUY"]
        assert len(buys) == 1
        assert buys[0].symbol == SYMBOL
        assert buys[0].quantity > 0
        assert buys[0].order_type == "MARKET"


class TestHighConfluenceShortEntry:
    def test_high_confluence_short_entry(self):
        """When score <= -threshold, a SELL signal (short) is emitted."""
        s = _init()
        _warm_up(s)

        with patch.object(s, "_compute_score", return_value=(-3, 20.0)):
            snap = _snap(100, high=101, low=99)
            signals = s.on_bar(snap)

        sells = [sig for sig in signals if sig.action == "SELL"]
        assert len(sells) == 1
        assert sells[0].symbol == SYMBOL
        assert sells[0].product_type == "MIS"  # shorts always MIS


class TestLowConfluenceNoEntry:
    def test_low_confluence_no_entry(self):
        """When score is between -2 and +2 (below threshold), no entry signals."""
        s = _init()
        _warm_up(s)

        for score_val in [-2, -1, 0, 1, 2]:
            with patch.object(s, "_compute_score", return_value=(score_val, 20.0)):
                snap = _snap(100, high=101, low=99)
                signals = s.on_bar(snap)

            entry_signals = [sig for sig in signals
                             if sig.action in ("BUY", "SELL") and sig.order_type == "MARKET"]
            assert len(entry_signals) == 0, f"Unexpected entry at score={score_val}"


class TestScoreFlipExits:
    def test_score_flip_exits_long(self):
        """Enter long at score=3, then score drops to 0 -> exit."""
        s = _init()
        _warm_up(s)

        # Enter long
        with patch.object(s, "_compute_score", return_value=(3, 30.0)):
            snap = _snap(100, high=101, low=99)
            s.on_bar(snap)

        # Simulate fill
        state = s.pm.get_state(SYMBOL)
        qty = state.pending_qty
        fill = FillInfo(SYMBOL, "BUY", qty, 100.0, 0.0, 0)
        position = Position(SYMBOL, qty, 100.0, 0.0)
        pending_stop = PendingOrder(SYMBOL, "SELL", qty, "SL_M", 0.0, 96.0)

        with patch.object(s, "_compute_score", return_value=(3, 30.0)):
            snap = _snap(100, high=101, low=99,
                         fills=[fill], positions=[position],
                         pending_orders=[pending_stop])
            s.on_bar(snap)

        assert s.pm.is_long(SYMBOL)

        # Score flips to 0 -> exit
        with patch.object(s, "_compute_score", return_value=(0, 20.0)):
            snap = _snap(100, high=101, low=99,
                         positions=[position],
                         pending_orders=[PendingOrder(SYMBOL, "SELL", qty, "SL_M", 0.0, 96.0)])
            signals = s.on_bar(snap)

        sells = [sig for sig in signals if sig.action == "SELL" and sig.order_type == "MARKET"]
        assert len(sells) >= 1
        assert s.pm.is_flat(SYMBOL)


class TestAdxAmplifiesMacd:
    def test_adx_amplifies_macd(self):
        """With high ADX, MACD vote is doubled, increasing the score."""
        s = _init(adx_trend_threshold=25)
        _warm_up(s)

        # Mock individual indicators:
        # RSI=50 (neutral=0), MACD hist>0 (+1 or +2), BB mid (0), OBV slope>0 (+1)
        # With low ADX (<25): MACD vote = +1, total = 0+1+0+1 = 2 (below threshold=3)
        # With high ADX (>25): MACD vote = +2, total = 0+2+0+1 = 3 (meets threshold)
        mock_indicators = {
            "strategies.deterministic.confluence.compute_rsi": 50.0,       # neutral
            "strategies.deterministic.confluence.compute_macd": (1.0, 0.5, 0.5),  # hist > 0
            "strategies.deterministic.confluence.compute_bollinger": (110.0, 100.0, 90.0),  # close in middle
            "strategies.deterministic.confluence.compute_obv_slope": 1.0,  # positive
        }

        # Low ADX -> score = 2, no entry
        with patch("strategies.deterministic.confluence.compute_rsi", return_value=mock_indicators["strategies.deterministic.confluence.compute_rsi"]), \
             patch("strategies.deterministic.confluence.compute_macd", return_value=mock_indicators["strategies.deterministic.confluence.compute_macd"]), \
             patch("strategies.deterministic.confluence.compute_bollinger", return_value=mock_indicators["strategies.deterministic.confluence.compute_bollinger"]), \
             patch("strategies.deterministic.confluence.compute_adx", return_value=20.0), \
             patch("strategies.deterministic.confluence.compute_obv_slope", return_value=mock_indicators["strategies.deterministic.confluence.compute_obv_slope"]):
            snap = _snap(100, high=101, low=99)
            signals_low_adx = s.on_bar(snap)

        buys_low = [sig for sig in signals_low_adx if sig.action == "BUY"]
        assert len(buys_low) == 0, "Low ADX should not produce entry (score=2)"

        # High ADX -> score = 3, entry
        with patch("strategies.deterministic.confluence.compute_rsi", return_value=mock_indicators["strategies.deterministic.confluence.compute_rsi"]), \
             patch("strategies.deterministic.confluence.compute_macd", return_value=mock_indicators["strategies.deterministic.confluence.compute_macd"]), \
             patch("strategies.deterministic.confluence.compute_bollinger", return_value=mock_indicators["strategies.deterministic.confluence.compute_bollinger"]), \
             patch("strategies.deterministic.confluence.compute_adx", return_value=30.0), \
             patch("strategies.deterministic.confluence.compute_obv_slope", return_value=mock_indicators["strategies.deterministic.confluence.compute_obv_slope"]):
            snap = _snap(100, high=101, low=99)
            signals_high_adx = s.on_bar(snap)

        buys_high = [sig for sig in signals_high_adx if sig.action == "BUY"]
        assert len(buys_high) == 1, "High ADX should amplify MACD and produce entry (score=3)"


class TestTrailingStopUpdates:
    def test_trailing_stop_updates(self):
        """After entry fill, rising prices should trigger trailing stop updates."""
        s = _init()
        _warm_up(s)

        # Enter long
        with patch.object(s, "_compute_score", return_value=(4, 30.0)):
            snap = _snap(100, high=101, low=99)
            s.on_bar(snap)

        state = s.pm.get_state(SYMBOL)
        qty = state.pending_qty

        # Simulate fill
        fill = FillInfo(SYMBOL, "BUY", qty, 100.0, 0.0, 0)
        position = Position(SYMBOL, qty, 100.0, 0.0)
        pending_stop = PendingOrder(SYMBOL, "SELL", qty, "SL_M", 0.0, 96.0)

        with patch.object(s, "_compute_score", return_value=(4, 30.0)):
            snap = _snap(100, high=101, low=99,
                         fills=[fill], positions=[position],
                         pending_orders=[pending_stop])
            fill_signals = s.on_bar(snap)

        sl_signals = [sig for sig in fill_signals if sig.order_type == "SL_M"]
        assert len(sl_signals) >= 1
        initial_stop = sl_signals[0].stop_price

        # Price rises -> trailing stop should ratchet up
        with patch.object(s, "_compute_score", return_value=(3, 30.0)):
            snap = _snap(110, high=111, low=109,
                         positions=[position],
                         pending_orders=[PendingOrder(SYMBOL, "SELL", qty, "SL_M", 0.0, initial_stop)])
            signals = s.on_bar(snap)

        new_stops = [sig for sig in signals if sig.order_type == "SL_M"]
        assert len(new_stops) >= 1
        assert new_stops[-1].stop_price > initial_stop


class TestTimeStopExit:
    def test_time_stop_exit(self):
        """Position held > max_hold_bars with low gain triggers exit."""
        s = _init(max_hold_bars=3, min_gain_for_hold=0.005)
        _warm_up(s)

        # Enter long
        with patch.object(s, "_compute_score", return_value=(4, 30.0)):
            snap = _snap(100, high=101, low=99)
            s.on_bar(snap)

        state = s.pm.get_state(SYMBOL)
        qty = state.pending_qty

        # Simulate fill
        fill = FillInfo(SYMBOL, "BUY", qty, 100.0, 0.0, 0)
        position = Position(SYMBOL, qty, 100.0, 0.0)
        pending_stop = PendingOrder(SYMBOL, "SELL", qty, "SL_M", 0.0, 96.0)

        with patch.object(s, "_compute_score", return_value=(4, 30.0)):
            snap = _snap(100, high=101, low=99,
                         fills=[fill], positions=[position],
                         pending_orders=[pending_stop])
            s.on_bar(snap)

        # Feed bars with score > 0 (so score-flip exit doesn't fire),
        # low gain, past max_hold_bars
        all_signals = []
        for i in range(5):
            price = 100.1 + i * 0.05  # tiny gain < 0.5%
            with patch.object(s, "_compute_score", return_value=(1, 30.0)):
                snap = _snap(price, high=price + 1, low=price - 1,
                             positions=[position],
                             pending_orders=[PendingOrder(SYMBOL, "SELL", qty, "SL_M", 0.0, 96.0)])
                all_signals.extend(s.on_bar(snap))

        sells = [sig for sig in all_signals if sig.action == "SELL" and sig.order_type == "MARKET"]
        assert len(sells) >= 1
        assert s.pm.is_flat(SYMBOL)


class TestPositionSizingAtr:
    def test_position_sizing_atr(self):
        """Verify qty = capital * risk_per_trade / (ATR * multiplier)."""
        s = _init(risk_per_trade=0.02, atr_multiplier=2.0)
        _warm_up(s)

        # Mock _compute_score for entry, and mock compute_atr to return a known value
        with patch.object(s, "_compute_score", return_value=(4, 30.0)), \
             patch("strategies.deterministic.confluence.compute_atr", return_value=5.0):
            snap = _snap(100, high=101, low=99)
            signals = s.on_bar(snap)

        buys = [sig for sig in signals if sig.action == "BUY"]
        assert len(buys) == 1
        # qty = 1_000_000 * 0.02 / (5.0 * 2.0) = 20_000 / 10.0 = 2000
        assert buys[0].quantity == 2000


class TestProductTypeSelection:
    def test_cnc_for_strong_adx(self):
        """ADX > 30 -> CNC product type for longs."""
        s = _init()
        _warm_up(s)

        with patch.object(s, "_compute_score", return_value=(3, 35.0)):
            snap = _snap(100, high=101, low=99)
            signals = s.on_bar(snap)

        buys = [sig for sig in signals if sig.action == "BUY"]
        assert len(buys) == 1
        assert buys[0].product_type == "CNC"

    def test_mis_for_weak_adx(self):
        """ADX <= 30 -> MIS product type for longs."""
        s = _init()
        _warm_up(s)

        with patch.object(s, "_compute_score", return_value=(3, 20.0)):
            snap = _snap(100, high=101, low=99)
            signals = s.on_bar(snap)

        buys = [sig for sig in signals if sig.action == "BUY"]
        assert len(buys) == 1
        assert buys[0].product_type == "MIS"
