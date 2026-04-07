"""Tests for narrative_builder module — thesis-driven facts-only approach."""

from strategies.narrative_builder import (
    build_portfolio_narrative,
    build_regime_narrative,
    build_symbol_narrative,
    build_intraday_narrative,
    build_cross_stock_narrative,
    build_trade_history_narrative,
)


# ---------------------------------------------------------------------------
# build_symbol_narrative — facts only, no suggestions
# ---------------------------------------------------------------------------

def test_symbol_narrative_no_suggestions():
    """Verify text does NOT contain interpretive keywords."""
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
    # These interpretive terms must NOT appear
    assert "SUGGESTION" not in result
    assert "POTENTIAL LONG" not in result
    assert "POTENTIAL SHORT" not in result
    assert "CONFLUENCE" not in result
    assert "oversold" not in result.lower()
    assert "overbought" not in result.lower()
    assert "buy signal" not in result.lower()
    assert "sell signal" not in result.lower()
    assert "NO ACTION" not in result
    assert "favor" not in result.lower()


def test_symbol_narrative_has_facts():
    """Verify text contains raw factual values without interpretation."""
    features = {
        "rsi_14": 32.0,
        "adx_14": 28.0,
        "macd_hist": -1.1,
        "macd_hist_slope": -0.5,
        "bb_pct_b": 0.15,
        "obv_slope_10": -300.0,
        "close_sma_ratio": 0.979,
        "volume_zscore": 1.5,
        "ret_5": -0.052,
        "ret_1": -0.01,
    }
    result = build_symbol_narrative("INFY", 1520.0, features, 22.0, None, 100000)
    assert "INFY" in result
    assert "1520.00" in result
    # RSI as a number, not a label
    assert "RSI 32" in result
    # ADX as a number
    assert "ADX 28" in result
    # MACD histogram value
    assert "-1.1" in result
    # BB position as percentile
    assert "15th percentile" in result
    # OBV factual direction
    assert "OBV declining" in result
    # Volume context
    assert "z-score" in result.lower() or "Volume" in result
    # Not held
    assert "Not held" in result


def test_symbol_narrative_relative_context():
    """Verify text mentions relative context like week change, SMA distance."""
    features = {
        "rsi_14": 50.0,
        "adx_14": 22.0,
        "macd_hist": 0.5,
        "bb_pct_b": 0.5,
        "obv_slope_10": 100.0,
        "close_sma_ratio": 0.97,
        "ret_5": -0.03,
        "ret_1": 0.005,
        "ret_10": -0.05,
    }
    result = build_symbol_narrative("TCS", 3500.0, features, 50.0, None, 100000)
    # Weekly return context
    assert "week" in result.lower() or "5-day" in result.lower()
    # SMA distance
    assert "20-day" in result
    assert "below" in result.lower()


def test_symbol_narrative_with_position():
    """Includes factual position P&L when held."""
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
    assert "10 shares" in result
    assert "1000.00" in result
    # P&L should be +5.0%
    assert "+5.0%" in result
    # No interpretive advice about holding/exiting
    assert "HOLD" not in result
    assert "CONSIDER EXIT" not in result


def test_symbol_narrative_no_position_shows_not_held():
    """When no position, shows 'Not held'."""
    features = {
        "rsi_14": 50.0,
        "adx_14": 22.0,
        "macd_hist": 0.5,
        "bb_pct_b": 0.5,
        "obv_slope_10": 0.0,
        "close_sma_ratio": 1.0,
    }
    result = build_symbol_narrative("HDFC", 2700.0, features, 40.0, None, 100000)
    assert "Not held" in result


def test_symbol_narrative_insufficient():
    """features=None -> 'INSUFFICIENT DATA'."""
    result = build_symbol_narrative("WIPRO", 450.0, None, None, None, 100000)
    assert "INSUFFICIENT DATA" in result
    assert "WIPRO" in result


def test_symbol_narrative_risk_math():
    """Risk calculation is present and factual."""
    features = {
        "rsi_14": 50.0,
        "adx_14": 22.0,
        "macd_hist": 0.5,
        "bb_pct_b": 0.5,
        "obv_slope_10": 0.0,
        "close_sma_ratio": 1.0,
    }
    result = build_symbol_narrative("SBIN", 500.0, features, 10.0, None, 100000)
    assert "2x ATR stop" in result
    assert "3% capital risk" in result
    # 2 * 10 = 20; 100000 * 0.03 / 20 = 150
    assert "150 shares" in result


def test_symbol_narrative_with_zscore():
    """Z-score is included when provided."""
    features = {
        "rsi_14": 50.0,
        "adx_14": 22.0,
        "macd_hist": 0.5,
        "bb_pct_b": 0.5,
        "obv_slope_10": 0.0,
        "close_sma_ratio": 1.0,
    }
    result = build_symbol_narrative("SBIN", 500.0, features, 10.0, None, 100000, zscore=1.8)
    assert "z-score" in result.lower()
    assert "+1.8" in result
    assert "above" in result.lower()
    assert "standard deviations" in result


def test_symbol_narrative_zscore_negative():
    """Negative z-score uses 'below'."""
    features = {
        "rsi_14": 50.0,
        "adx_14": 22.0,
        "macd_hist": 0.5,
        "bb_pct_b": 0.5,
        "obv_slope_10": 0.0,
        "close_sma_ratio": 1.0,
    }
    result = build_symbol_narrative("ITC", 400.0, features, 8.0, None, 100000, zscore=-2.1)
    assert "-2.1" in result
    assert "below" in result.lower()


def test_symbol_narrative_no_zscore():
    """No z-score -> no statistical position section."""
    features = {
        "rsi_14": 50.0,
        "adx_14": 22.0,
        "macd_hist": 0.5,
        "bb_pct_b": 0.5,
        "obv_slope_10": 0.0,
        "close_sma_ratio": 1.0,
    }
    result = build_symbol_narrative("TCS", 3500.0, features, 50.0, None, 100000)
    assert "Statistical position" not in result


# ---------------------------------------------------------------------------
# build_intraday_narrative — VWAP, 15-min indicators
# ---------------------------------------------------------------------------

def test_intraday_narrative_vwap():
    """VWAP and VWAP bands appear in intraday narrative."""
    # Simulate 10 bars of intraday data
    closes = [100.0 + i * 0.5 for i in range(10)]
    highs = [c + 1.0 for c in closes]
    lows = [c - 1.0 for c in closes]
    volumes = [50000] * 10

    result = build_intraday_narrative(
        "RELIANCE", closes, highs, lows, volumes,
        closes, highs, lows,
        daily_prev_close=99.0,
    )
    assert "RELIANCE INTRADAY" in result
    assert "VWAP" in result


def test_intraday_narrative_session_gap():
    """Session gap from previous daily close."""
    closes = [105.0, 106.0, 107.0]
    highs = [106.0, 107.0, 108.0]
    lows = [104.0, 105.0, 106.0]
    volumes = [50000] * 3

    result = build_intraday_narrative(
        "TCS", closes, highs, lows, volumes,
        closes, highs, lows,
        daily_prev_close=100.0,
    )
    assert "Session gap" in result
    assert "+5.0%" in result


def test_intraday_narrative_no_data():
    """Empty intraday data -> appropriate message."""
    result = build_intraday_narrative(
        "INFY", [], [], [], [],
        [], [], [],
        daily_prev_close=1500.0,
    )
    assert "No intraday data" in result


def test_intraday_narrative_no_suggestions():
    """Intraday narrative should not contain interpretive keywords."""
    closes = [100.0 + i for i in range(20)]
    highs = [c + 2.0 for c in closes]
    lows = [c - 2.0 for c in closes]
    volumes = [60000] * 20

    result = build_intraday_narrative(
        "SBIN", closes, highs, lows, volumes,
        closes, highs, lows,
        daily_prev_close=99.0,
    )
    assert "buy" not in result.lower() or "above" in result.lower()
    assert "sell" not in result.lower()
    assert "POTENTIAL" not in result
    assert "SUGGESTION" not in result


def test_intraday_narrative_volume_profile():
    """Volume profile shows bar count and average volume."""
    closes = [100.0, 101.0, 102.0, 103.0, 104.0]
    highs = [c + 1.0 for c in closes]
    lows = [c - 1.0 for c in closes]
    volumes = [40000, 50000, 60000, 45000, 55000]

    result = build_intraday_narrative(
        "HDFC", closes, highs, lows, volumes,
        closes, highs, lows,
        daily_prev_close=99.0,
    )
    assert "5 bars today" in result
    assert "Volume profile" in result


# ---------------------------------------------------------------------------
# build_cross_stock_narrative — correlations, cointegration, z-scores
# ---------------------------------------------------------------------------

def test_cross_stock_narrative_correlations():
    """Cross-stock shows correlation pairs."""
    import math
    # Create correlated price series
    n = 50
    base = [100.0 + i * 0.5 + math.sin(i * 0.3) * 5 for i in range(n)]
    series_a = base[:]
    series_b = [x + 10.0 + (i % 3) * 0.1 for i, x in enumerate(base)]  # highly correlated
    series_c = [200.0 - x for x in base]  # negatively correlated

    all_closes = {"A": series_a, "B": series_b, "C": series_c}
    result = build_cross_stock_narrative(all_closes, ["A", "B", "C"])
    assert "CROSS-STOCK ANALYSIS" in result
    assert "correlations" in result.lower()
    # Should show the ρ symbol
    assert "ρ=" in result


def test_cross_stock_narrative_z_scores():
    """Cross-stock shows relative z-scores."""
    n = 50
    # Stock trending up (positive z-score)
    series_up = [100.0 + i * 2.0 for i in range(n)]
    # Stock trending down (negative z-score)
    series_down = [200.0 - i * 2.0 for i in range(n)]
    # Stock flat (near zero z-score)
    series_flat = [150.0 + (i % 5 - 2) * 0.5 for i in range(n)]

    all_closes = {"UP": series_up, "DOWN": series_down, "FLAT": series_flat}
    result = build_cross_stock_narrative(all_closes, ["UP", "DOWN", "FLAT"])
    assert "z-scores" in result.lower()


def test_cross_stock_narrative_insufficient_data():
    """Less than 2 symbols with enough data -> appropriate message."""
    all_closes = {"A": [100.0] * 5}  # too few bars
    result = build_cross_stock_narrative(all_closes, ["A"])
    assert "Insufficient data" in result


def test_cross_stock_narrative_no_suggestions():
    """Cross-stock narrative has no trading suggestions."""
    n = 50
    base = [100.0 + i for i in range(n)]
    all_closes = {"X": base[:], "Y": [x + 5 for x in base]}
    result = build_cross_stock_narrative(all_closes, ["X", "Y"])
    assert "SUGGESTION" not in result
    assert "POTENTIAL" not in result
    assert "consider" not in result.lower()
    assert "recommend" not in result.lower()


# ---------------------------------------------------------------------------
# build_regime_narrative — facts, no strategy advice
# ---------------------------------------------------------------------------

def test_regime_no_strategy_advice():
    """Verify regime text does NOT contain strategy recommendations."""
    result = build_regime_narrative(adx=35.0, bbw=0.05, avg_bbw=0.05)
    assert "Trend-following" not in result
    assert "Mean-reversion" not in result
    assert "favor" not in result.lower()
    assert "preferred" not in result.lower()
    assert "strategies" not in result.lower()
    assert "consider" not in result.lower()


def test_regime_has_facts():
    """Verify regime text contains ADX values and BBW facts."""
    result = build_regime_narrative(adx=28.4, bbw=0.05, avg_bbw=0.05)
    assert "28.4" in result
    assert "ADX" in result
    assert "Bollinger Band Width" in result


def test_regime_high_volatility():
    """High BBW ratio -> mentions elevated volatility factually."""
    result = build_regime_narrative(adx=25.0, bbw=0.15, avg_bbw=0.05)
    assert "elevated volatility" in result.lower() or "above average" in result.lower()
    # No advice
    assert "reducing" not in result.lower()
    assert "consider" not in result.lower()


def test_regime_compressed():
    """Low BBW ratio -> mentions compressed volatility factually."""
    result = build_regime_narrative(adx=25.0, bbw=0.02, avg_bbw=0.05)
    assert "compressed" in result.lower()
    # No prediction
    assert "imminent" not in result.lower()


def test_regime_no_adx():
    """ADX=None -> says unavailable."""
    result = build_regime_narrative(adx=None, bbw=0.04, avg_bbw=0.05)
    assert "unavailable" in result.lower()


# ---------------------------------------------------------------------------
# build_portfolio_narrative — already mostly factual
# ---------------------------------------------------------------------------

def test_portfolio_narrative():
    """Includes cash, equity, drawdown, costs — all factual."""
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
    assert "Drawdown" in result
    # No advice like "consider reducing"
    assert "consider" not in result.lower()
    assert "DANGER" not in result
    assert "caution" not in result.lower()


def test_portfolio_narrative_in_profit():
    """Portfolio above initial capital -> no drawdown."""
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
# build_trade_history_narrative — outcomes without lessons
# ---------------------------------------------------------------------------

def test_trade_history_no_lessons():
    """Verify text does NOT contain LESSON or advice."""
    trades = [
        {"symbol": "A", "side": "BUY", "entry_price": 100, "exit_price": 110,
         "pnl": 100, "pnl_pct": 10.0, "reasoning": "", "bars_held": 10},
        {"symbol": "B", "side": "BUY", "entry_price": 200, "exit_price": 190,
         "pnl": -100, "pnl_pct": -5.0, "reasoning": "", "bars_held": 8},
        {"symbol": "C", "side": "BUY", "entry_price": 300, "exit_price": 280,
         "pnl": -200, "pnl_pct": -6.7, "reasoning": "", "bars_held": 6},
    ]
    result = build_trade_history_narrative(trades)
    assert "LESSON" not in result
    assert "Consider" not in result
    assert "reducing" not in result.lower()
    assert "confirmation" not in result.lower()
    assert "stopped out" not in result.lower()
    assert "widening" not in result.lower()


def test_trade_history_has_outcomes():
    """Verify individual trade results are shown with P&L."""
    trades = [
        {"symbol": "RELIANCE", "side": "BUY", "entry_price": 1000, "exit_price": 1050,
         "pnl": 500, "pnl_pct": 5.0, "reasoning": "test", "bars_held": 10},
        {"symbol": "INFY", "side": "BUY", "entry_price": 1500, "exit_price": 1470,
         "pnl": -300, "pnl_pct": -2.0, "reasoning": "test", "bars_held": 5},
    ]
    result = build_trade_history_narrative(trades)
    assert "TRADE HISTORY" in result
    assert "RELIANCE" in result
    assert "INFY" in result
    assert "+5.0%" in result
    assert "-2.0%" in result
    assert "Win rate: 50%" in result
    assert "1W / 1L" in result
    assert "Profit factor" in result


def test_trade_history_empty():
    """No trades -> appropriate message."""
    result = build_trade_history_narrative([])
    assert "No completed trades" in result


def test_trade_history_statistics():
    """Verify win rate, avg win/loss, profit factor are factual."""
    trades = [
        {"symbol": "A", "side": "BUY", "entry_price": 100, "exit_price": 110,
         "pnl": 100, "pnl_pct": 10.0, "reasoning": "", "bars_held": 10},
        {"symbol": "B", "side": "BUY", "entry_price": 200, "exit_price": 190,
         "pnl": -100, "pnl_pct": -5.0, "reasoning": "", "bars_held": 8},
        {"symbol": "C", "side": "BUY", "entry_price": 300, "exit_price": 280,
         "pnl": -200, "pnl_pct": -6.7, "reasoning": "", "bars_held": 6},
        {"symbol": "D", "side": "BUY", "entry_price": 400, "exit_price": 440,
         "pnl": 400, "pnl_pct": 10.0, "reasoning": "", "bars_held": 12},
    ]
    result = build_trade_history_narrative(trades)
    assert "Win rate: 50%" in result
    assert "2W / 2L out of 4" in result
    assert "Avg win:" in result
    assert "Avg loss:" in result
    assert "Profit factor:" in result
