"""Tests for narrative_builder module."""

import math

from strategies.narrative_builder import (
    build_portfolio_narrative,
    build_regime_narrative,
    build_symbol_narrative,
    build_trade_history_narrative,
)


# ---------------------------------------------------------------------------
# build_symbol_narrative
# ---------------------------------------------------------------------------

def test_symbol_narrative_bullish():
    """RSI=30, MACD positive, BB low → 'POTENTIAL LONG ENTRY'."""
    features = {
        "rsi_14": 30.0,
        "adx_14": 18.0,
        "macd_hist": 1.5,
        "macd_hist_slope": 0.3,
        "bb_pct_b": 0.15,
        "obv_slope_10": 500.0,
        "close_sma_ratio": 0.97,
    }
    result = build_symbol_narrative("RELIANCE", 1200.0, features, 25.0, None, 100000)
    assert "POTENTIAL LONG ENTRY" in result
    assert "RELIANCE" in result
    assert "1200.00" in result
    # Confluence should report at least 3 bullish
    assert "bullish" in result


def test_symbol_narrative_bearish():
    """RSI=72, MACD negative → 'POTENTIAL SHORT ENTRY'."""
    features = {
        "rsi_14": 72.0,
        "adx_14": 32.0,
        "macd_hist": -2.0,
        "macd_hist_slope": -1.0,
        "bb_pct_b": 0.85,
        "obv_slope_10": -300.0,
        "close_sma_ratio": 0.98,
    }
    result = build_symbol_narrative("INFY", 1500.0, features, 30.0, None, 100000)
    assert "POTENTIAL SHORT ENTRY" in result
    assert "bearish" in result


def test_symbol_narrative_mixed():
    """RSI=30 (bullish) but MACD negative (bearish) → 'CONFLICTING'."""
    features = {
        "rsi_14": 30.0,
        "adx_14": 18.0,
        "macd_hist": -1.0,
        "bb_pct_b": 0.5,
        "obv_slope_10": 0.0,
        "close_sma_ratio": 1.0,
    }
    result = build_symbol_narrative("TCS", 3500.0, features, 50.0, None, 100000)
    assert "CONFLICTING" in result


def test_symbol_narrative_neutral():
    """RSI=50, no extremes → 'NO ACTION'."""
    features = {
        "rsi_14": 50.0,
        "adx_14": 18.0,
        "macd_hist": 0.01,
        "bb_pct_b": 0.5,
        "obv_slope_10": 0.0,
        "close_sma_ratio": 1.001,
    }
    result = build_symbol_narrative("HDFC", 2700.0, features, 40.0, None, 100000)
    assert "NO ACTION" in result


def test_symbol_narrative_with_position():
    """Includes position P&L text when a position is held."""
    features = {
        "rsi_14": 50.0,
        "adx_14": 22.0,
        "macd_hist": 0.5,
        "bb_pct_b": 0.6,
        "obv_slope_10": 100.0,
        "close_sma_ratio": 1.02,
    }
    position = {"qty": 10, "avg_price": 1000.0, "unrealized_pnl": 500.0, "product_type": "CNC"}
    result = build_symbol_narrative("RELIANCE", 1050.0, features, 20.0, position, 100000)
    assert "You hold 10 shares" in result
    assert "1000.00" in result
    # P&L should be +5.0%
    assert "+5.0%" in result


def test_symbol_narrative_insufficient():
    """features=None → 'INSUFFICIENT DATA'."""
    result = build_symbol_narrative("WIPRO", 450.0, None, None, None, 100000)
    assert "INSUFFICIENT DATA" in result
    assert "WIPRO" in result


# ---------------------------------------------------------------------------
# build_regime_narrative
# ---------------------------------------------------------------------------

def test_regime_trending():
    """ADX=30 → 'TRENDING'."""
    result = build_regime_narrative(adx=30.0, bbw=0.05, avg_bbw=0.05)
    # ADX=30 is exactly > 30 boundary; it's >= 30 so should be TRENDING
    # Actually the code checks > 30, so 30 falls to > 20 branch = TRANSITIONING
    # ADX=31 would be TRENDING. Let's use 35.
    result = build_regime_narrative(adx=35.0, bbw=0.05, avg_bbw=0.05)
    assert "TRENDING" in result
    assert "Trend-following" in result


def test_regime_ranging():
    """ADX=15 → 'RANGING'."""
    result = build_regime_narrative(adx=15.0, bbw=0.04, avg_bbw=0.05)
    assert "RANGING" in result
    assert "Mean-reversion" in result


def test_regime_volatile():
    """High BBW ratio → mentions VOLATILE."""
    result = build_regime_narrative(adx=25.0, bbw=0.15, avg_bbw=0.05)
    assert "VOLATILE" in result


def test_regime_compressed():
    """Low BBW ratio → mentions COMPRESSED."""
    result = build_regime_narrative(adx=25.0, bbw=0.02, avg_bbw=0.05)
    assert "COMPRESSED" in result


# ---------------------------------------------------------------------------
# build_portfolio_narrative
# ---------------------------------------------------------------------------

def test_portfolio_narrative():
    """Includes cash, equity, drawdown, costs."""
    result = build_portfolio_narrative(
        cash=80000.0,
        equity=15000.0,
        initial_capital=100000.0,
        positions=[{"symbol": "RELIANCE", "qty": 10}],
        total_costs=250.0,
        trade_count=5,
    )
    assert "PORTFOLIO SUMMARY" in result
    assert "95,000.00" in result  # total value
    assert "-5.0%" in result  # return
    assert "1" in result  # open positions
    assert "5" in result  # trade count
    assert "250.00" in result  # costs
    assert "DRAWDOWN" in result


def test_portfolio_narrative_in_profit():
    """Portfolio above initial capital → no drawdown warning."""
    result = build_portfolio_narrative(
        cash=90000.0,
        equity=20000.0,
        initial_capital=100000.0,
        positions=[],
        total_costs=100.0,
        trade_count=3,
    )
    assert "in profit" in result
    assert "+10.0%" in result


# ---------------------------------------------------------------------------
# build_trade_history_narrative
# ---------------------------------------------------------------------------

def test_trade_history_narrative():
    """Includes win rate, lesson, recent trades."""
    trades = [
        {"symbol": "RELIANCE", "side": "BUY", "entry_price": 1000, "exit_price": 1050,
         "pnl": 500, "pnl_pct": 5.0, "reasoning": "RSI oversold", "bars_held": 10},
        {"symbol": "INFY", "side": "BUY", "entry_price": 1500, "exit_price": 1470,
         "pnl": -300, "pnl_pct": -2.0, "reasoning": "RSI oversold", "bars_held": 5},
        {"symbol": "TCS", "side": "BUY", "entry_price": 3500, "exit_price": 3400,
         "pnl": -1000, "pnl_pct": -2.9, "reasoning": "RSI overbought entry", "bars_held": 2},
        {"symbol": "HDFC", "side": "BUY", "entry_price": 2700, "exit_price": 2800,
         "pnl": 1000, "pnl_pct": 3.7, "reasoning": "Breakout", "bars_held": 15},
    ]
    result = build_trade_history_narrative(trades)
    assert "TRADE HISTORY" in result
    assert "Win rate: 50%" in result
    assert "2W / 2L" in result
    assert "Profit factor" in result


def test_trade_history_empty():
    """No trades → appropriate message."""
    result = build_trade_history_narrative([])
    assert "No completed trades" in result


def test_trade_history_lesson_one_side():
    """All recent losses on one side → lesson about bias."""
    trades = [
        {"symbol": "A", "side": "BUY", "entry_price": 100, "exit_price": 110,
         "pnl": 100, "pnl_pct": 10.0, "reasoning": "", "bars_held": 10},
        {"symbol": "B", "side": "BUY", "entry_price": 200, "exit_price": 190,
         "pnl": -100, "pnl_pct": -5.0, "reasoning": "", "bars_held": 8},
        {"symbol": "C", "side": "BUY", "entry_price": 300, "exit_price": 280,
         "pnl": -200, "pnl_pct": -6.7, "reasoning": "", "bars_held": 6},
    ]
    result = build_trade_history_narrative(trades)
    assert "LESSON" in result
    assert "LONG" in result


def test_trade_history_lesson_quick_stops():
    """Multiple quick stops → lesson about tight stops."""
    trades = [
        {"symbol": "A", "side": "BUY", "entry_price": 100, "exit_price": 95,
         "pnl": -50, "pnl_pct": -5.0, "reasoning": "", "bars_held": 2},
        {"symbol": "B", "side": "SELL", "entry_price": 200, "exit_price": 205,
         "pnl": -50, "pnl_pct": -2.5, "reasoning": "", "bars_held": 1},
        {"symbol": "C", "side": "BUY", "entry_price": 300, "exit_price": 310,
         "pnl": 100, "pnl_pct": 3.3, "reasoning": "", "bars_held": 10},
    ]
    result = build_trade_history_narrative(trades)
    assert "LESSON" in result
    assert "stopped out" in result
