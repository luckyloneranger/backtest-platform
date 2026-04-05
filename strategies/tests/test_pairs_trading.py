"""Tests for Statistical Pairs Trading strategy."""

import math
from unittest.mock import patch
from collections import deque

from strategies.base import (
    BarData, MarketSnapshot, Portfolio, Position, SessionContext, InstrumentInfo,
    FillInfo, PendingOrder,
)
from strategies.deterministic.pairs_trading import PairsTrading


# --- Helpers ---

DEFAULT_INSTRUMENTS = {
    "SYM_A": InstrumentInfo(
        symbol="SYM_A", exchange="NSE", instrument_type="EQ", lot_size=1,
        tick_size=0.05, expiry="", strike=0.0, option_type="",
        circuit_limit_upper=0.0, circuit_limit_lower=0.0,
    ),
    "SYM_B": InstrumentInfo(
        symbol="SYM_B", exchange="NSE", instrument_type="EQ", lot_size=1,
        tick_size=0.05, expiry="", strike=0.0, option_type="",
        circuit_limit_upper=0.0, circuit_limit_lower=0.0,
    ),
    "SYM_C": InstrumentInfo(
        symbol="SYM_C", exchange="NSE", instrument_type="EQ", lot_size=1,
        tick_size=0.05, expiry="", strike=0.0, option_type="",
        circuit_limit_upper=0.0, circuit_limit_lower=0.0,
    ),
}

DEFAULT_CONTEXT = SessionContext(
    initial_capital=1_000_000.0, bar_number=0, total_bars=200,
    start_date="2024-01-01", end_date="2024-12-31",
    intervals=["day"], lookback_window=60,
)


def make_snapshot(
    ts=0,
    day_bars=None,
    cash=1_000_000.0,
    positions=None,
    fills=None,
    pending_orders=None,
    equity=None,
):
    """Build a MarketSnapshot with daily bars for multiple symbols."""
    timeframes = {}
    if day_bars:
        timeframes["day"] = {
            sym: BarData(sym, bar["close"], bar.get("high", bar["close"] + 1),
                         bar.get("low", bar["close"] - 1), bar["close"],
                         bar.get("volume", 100_000), 0)
            for sym, bar in day_bars.items()
        }
    pos_list = positions or []
    eq = equity if equity is not None else cash + sum(p.quantity * p.avg_price for p in pos_list)
    return MarketSnapshot(
        timestamp_ms=ts,
        timeframes=timeframes,
        history={},
        portfolio=Portfolio(cash=cash, equity=eq, positions=pos_list),
        instruments=DEFAULT_INSTRUMENTS,
        fills=fills or [],
        rejections=[],
        closed_trades=[],
        context=DEFAULT_CONTEXT,
        pending_orders=pending_orders or [],
    )


def create_strategy(config=None):
    s = PairsTrading()
    s.initialize(config or {}, DEFAULT_INSTRUMENTS)
    return s


def seed_cointegrated(strategy, n=65, base_a=100.0, base_b=50.0):
    """Feed n daily bars with cointegrated prices (A ~ 2*B + noise).

    Returns the last prices used for each symbol.
    """
    import numpy as np
    rng = np.random.RandomState(42)
    # Random walk
    walk = np.cumsum(rng.randn(n) * 0.5)
    prices_a = base_a + walk
    prices_b = base_b + walk * 0.5 + rng.randn(n) * 0.1  # B ~ A/2 + small noise

    for i in range(n):
        snap = make_snapshot(
            ts=i,
            day_bars={
                "SYM_A": {"close": float(prices_a[i])},
                "SYM_B": {"close": float(prices_b[i])},
            },
        )
        strategy.on_bar(snap)
    return float(prices_a[-1]), float(prices_b[-1])


def seed_uncorrelated(strategy, n=65):
    """Feed n daily bars with uncorrelated random prices."""
    import numpy as np
    rng = np.random.RandomState(99)
    prices_a = 100 + np.cumsum(rng.randn(n) * 2.0)
    prices_b = 200 + np.cumsum(rng.randn(n) * 3.0)
    prices_c = 50 + np.cumsum(rng.randn(n) * 1.5)

    for i in range(n):
        snap = make_snapshot(
            ts=i,
            day_bars={
                "SYM_A": {"close": float(prices_a[i])},
                "SYM_B": {"close": float(prices_b[i])},
                "SYM_C": {"close": float(prices_c[i])},
            },
        )
        strategy.on_bar(snap)


def force_pair_selected(strategy, symbol_a="SYM_A", symbol_b="SYM_B",
                        hedge_ratio=2.0, spread_values=None):
    """Directly set pair state so we can test signal logic without needing
    genuine cointegration from statistical tests."""
    strategy.pair_selected = True
    strategy.symbol_a = symbol_a
    strategy.symbol_b = symbol_b
    strategy.hedge_ratio = hedge_ratio
    if spread_values:
        strategy.spread_history = deque(spread_values, maxlen=strategy.lookback_period)


# --- Tests ---


def test_required_data():
    s = create_strategy()
    reqs = s.required_data()
    assert len(reqs) == 1
    assert reqs[0]["interval"] == "day"
    assert reqs[0]["lookback"] == 60


def test_pair_selection():
    """Cointegrated prices should result in pair being selected."""
    s = create_strategy({"lookback_period": 60, "min_pvalue": 0.10})
    seed_cointegrated(s, n=65)
    assert s.pair_selected is True
    assert s.symbol_a in ("SYM_A", "SYM_B")
    assert s.symbol_b in ("SYM_A", "SYM_B")
    assert s.symbol_a != s.symbol_b
    assert s.hedge_ratio != 0.0


def test_no_pair_uncorrelated():
    """Uncorrelated random prices should NOT result in pair selection."""
    s = create_strategy({"lookback_period": 60, "min_pvalue": 0.01})
    seed_uncorrelated(s, n=65)
    # With strict p-value threshold and truly random walks, pair should not be selected
    # (random walks are not cointegrated)
    assert s.pair_selected is False


def test_long_spread_entry():
    """zscore < -entry_threshold should trigger long A + short B."""
    s = create_strategy({"entry_threshold": 2.0, "zscore_period": 20})

    # Set up pair with known hedge ratio
    force_pair_selected(s, hedge_ratio=2.0)
    # Pre-populate prices so the strategy has them
    s._ensure_prices("SYM_A")
    s._ensure_prices("SYM_B")

    # Build spread history: mean 0, std 1, then a value at -2.5
    import numpy as np
    rng = np.random.RandomState(10)
    base_spreads = list(rng.randn(25) * 1.0)  # mean ~0, std ~1
    force_pair_selected(s, hedge_ratio=2.0, spread_values=base_spreads)

    # Now create a bar where spread = price_A - 2.0 * price_B is very negative
    # We need z-score < -2. Mean of base_spreads ~ 0, std ~ 1
    # So spread needs to be around -2.5
    mean_s = float(np.mean(base_spreads[-20:]))
    std_s = float(np.std(base_spreads[-20:], ddof=1))
    target_spread = mean_s - 2.5 * std_s
    # price_A - 2.0 * price_B = target_spread
    # Let price_B = 50, then price_A = target_spread + 100
    price_b = 50.0
    price_a = target_spread + 2.0 * price_b

    snap = make_snapshot(
        ts=100,
        day_bars={
            "SYM_A": {"close": price_a},
            "SYM_B": {"close": price_b},
        },
        equity=1_000_000.0,
    )
    signals = s.on_bar(snap)

    # Should have: BUY on SYM_A (long A), SELL on SYM_B (short B)
    buys = [sig for sig in signals if sig.action == "BUY" and sig.symbol == "SYM_A"]
    sells = [sig for sig in signals if sig.action == "SELL" and sig.symbol == "SYM_B"]
    assert len(buys) >= 1, f"Expected BUY SYM_A, got signals: {signals}"
    assert len(sells) >= 1, f"Expected SELL SYM_B, got signals: {signals}"
    assert s.in_trade is True
    assert s.trade_direction == "long_spread"


def test_short_spread_entry():
    """zscore > +entry_threshold should trigger short A + long B."""
    s = create_strategy({"entry_threshold": 2.0, "zscore_period": 20})

    force_pair_selected(s, hedge_ratio=2.0)
    s._ensure_prices("SYM_A")
    s._ensure_prices("SYM_B")

    import numpy as np
    rng = np.random.RandomState(10)
    base_spreads = list(rng.randn(25) * 1.0)
    force_pair_selected(s, hedge_ratio=2.0, spread_values=base_spreads)

    mean_s = float(np.mean(base_spreads[-20:]))
    std_s = float(np.std(base_spreads[-20:], ddof=1))
    target_spread = mean_s + 2.5 * std_s
    price_b = 50.0
    price_a = target_spread + 2.0 * price_b

    snap = make_snapshot(
        ts=100,
        day_bars={
            "SYM_A": {"close": price_a},
            "SYM_B": {"close": price_b},
        },
        equity=1_000_000.0,
    )
    signals = s.on_bar(snap)

    # Should have: SELL on SYM_A (short A), BUY on SYM_B (long B)
    sells = [sig for sig in signals if sig.action == "SELL" and sig.symbol == "SYM_A"]
    buys = [sig for sig in signals if sig.action == "BUY" and sig.symbol == "SYM_B"]
    assert len(sells) >= 1, f"Expected SELL SYM_A, got signals: {signals}"
    assert len(buys) >= 1, f"Expected BUY SYM_B, got signals: {signals}"
    assert s.in_trade is True
    assert s.trade_direction == "short_spread"


def test_exit_on_mean_reversion():
    """zscore crossing exit_threshold (0) should close both legs."""
    s = create_strategy({"entry_threshold": 2.0, "zscore_period": 20, "exit_threshold": 0.0})
    force_pair_selected(s, hedge_ratio=2.0)
    s._ensure_prices("SYM_A")
    s._ensure_prices("SYM_B")

    import numpy as np
    rng = np.random.RandomState(10)
    base_spreads = list(rng.randn(25) * 1.0)

    # Enter a short_spread trade by setting state directly
    force_pair_selected(s, hedge_ratio=2.0, spread_values=base_spreads)
    s.in_trade = True
    s.trade_direction = "short_spread"
    s.bars_in_trade = 5

    # Set PM states to simulate open positions
    state_a = s.pm_a.get_state("SYM_A")
    state_a.direction = "short"
    state_a.qty = 10
    state_a.product_type = "MIS"
    state_a.has_engine_stop = True
    state_a.trailing_stop = 50000.0

    state_b = s.pm_b.get_state("SYM_B")
    state_b.direction = "long"
    state_b.qty = 20
    state_b.product_type = "MIS"
    state_b.has_engine_stop = True
    state_b.trailing_stop = 0.01

    # Now provide a bar where z-score is near 0 (mean-reverted)
    mean_s = float(np.mean(base_spreads[-20:]))
    std_s = float(np.std(base_spreads[-20:], ddof=1))
    # For short_spread trade: exit when zscore <= exit_threshold (0)
    # target z-score = -0.5 (below 0)
    target_spread = mean_s - 0.5 * std_s
    price_b = 50.0
    price_a = target_spread + 2.0 * price_b

    snap = make_snapshot(
        ts=200,
        day_bars={
            "SYM_A": {"close": price_a},
            "SYM_B": {"close": price_b},
        },
        positions=[
            Position("SYM_A", -10, price_a, 0.0),
            Position("SYM_B", 20, price_b, 0.0),
        ],
        pending_orders=[
            PendingOrder("SYM_A", "BUY", 10, "SL_M", 0.0, 50000.0),
            PendingOrder("SYM_B", "SELL", 20, "SL_M", 0.0, 0.01),
        ],
    )
    signals = s.on_bar(snap)

    # Should have exit signals for both legs
    # pm_a short exit: CANCEL + BUY MARKET
    # pm_b long exit: CANCEL + SELL MARKET
    cancel_signals = [sig for sig in signals if sig.action == "CANCEL"]
    buy_a = [sig for sig in signals if sig.action == "BUY" and sig.symbol == "SYM_A"
             and sig.order_type == "MARKET"]
    sell_b = [sig for sig in signals if sig.action == "SELL" and sig.symbol == "SYM_B"
              and sig.order_type == "MARKET"]

    assert len(cancel_signals) >= 2, f"Expected CANCEL signals, got: {signals}"
    assert len(buy_a) >= 1, f"Expected BUY SYM_A (cover short), got: {signals}"
    assert len(sell_b) >= 1, f"Expected SELL SYM_B (exit long), got: {signals}"
    assert s.in_trade is False


def test_stop_on_divergence():
    """|zscore| > stop_threshold should trigger stop-loss on both legs."""
    s = create_strategy({"entry_threshold": 2.0, "stop_threshold": 3.0, "zscore_period": 20})
    force_pair_selected(s, hedge_ratio=2.0)
    s._ensure_prices("SYM_A")
    s._ensure_prices("SYM_B")

    import numpy as np
    rng = np.random.RandomState(10)
    base_spreads = list(rng.randn(25) * 1.0)
    force_pair_selected(s, hedge_ratio=2.0, spread_values=base_spreads)

    # Simulate being in a short_spread trade
    s.in_trade = True
    s.trade_direction = "short_spread"
    s.bars_in_trade = 3

    state_a = s.pm_a.get_state("SYM_A")
    state_a.direction = "short"
    state_a.qty = 10
    state_a.product_type = "MIS"
    state_a.has_engine_stop = True
    state_a.trailing_stop = 50000.0

    state_b = s.pm_b.get_state("SYM_B")
    state_b.direction = "long"
    state_b.qty = 20
    state_b.product_type = "MIS"
    state_b.has_engine_stop = True
    state_b.trailing_stop = 0.01

    # Z-score > 3 (diverging further for short_spread = bad)
    # Need a large multiplier because the extreme value inflates the rolling std
    mean_s = float(np.mean(base_spreads[-20:]))
    std_s = float(np.std(base_spreads[-20:], ddof=1))
    target_spread = mean_s + 6.0 * std_s  # yields |z| > 3 after inclusion in window
    price_b = 50.0
    price_a = target_spread + 2.0 * price_b

    snap = make_snapshot(
        ts=200,
        day_bars={
            "SYM_A": {"close": price_a},
            "SYM_B": {"close": price_b},
        },
        positions=[
            Position("SYM_A", -10, price_a, 0.0),
            Position("SYM_B", 20, price_b, 0.0),
        ],
        pending_orders=[
            PendingOrder("SYM_A", "BUY", 10, "SL_M", 0.0, 50000.0),
            PendingOrder("SYM_B", "SELL", 20, "SL_M", 0.0, 0.01),
        ],
    )
    signals = s.on_bar(snap)

    # Both legs should exit
    buy_a = [sig for sig in signals if sig.action == "BUY" and sig.symbol == "SYM_A"
             and sig.order_type == "MARKET"]
    sell_b = [sig for sig in signals if sig.action == "SELL" and sig.symbol == "SYM_B"
              and sig.order_type == "MARKET"]
    assert len(buy_a) >= 1, f"Expected stop-loss BUY SYM_A, got: {signals}"
    assert len(sell_b) >= 1, f"Expected stop-loss SELL SYM_B, got: {signals}"
    assert s.in_trade is False


def test_time_stop():
    """bars_in_trade > max_hold_bars should exit both legs."""
    s = create_strategy({"entry_threshold": 2.0, "max_hold_bars": 10, "zscore_period": 20})
    force_pair_selected(s, hedge_ratio=2.0)
    s._ensure_prices("SYM_A")
    s._ensure_prices("SYM_B")

    import numpy as np
    rng = np.random.RandomState(10)
    base_spreads = list(rng.randn(25) * 1.0)
    force_pair_selected(s, hedge_ratio=2.0, spread_values=base_spreads)

    s.in_trade = True
    s.trade_direction = "short_spread"
    s.bars_in_trade = 11  # > max_hold_bars

    state_a = s.pm_a.get_state("SYM_A")
    state_a.direction = "short"
    state_a.qty = 10
    state_a.product_type = "MIS"
    state_a.has_engine_stop = True
    state_a.trailing_stop = 50000.0

    state_b = s.pm_b.get_state("SYM_B")
    state_b.direction = "long"
    state_b.qty = 20
    state_b.product_type = "MIS"
    state_b.has_engine_stop = True
    state_b.trailing_stop = 0.01

    # Z-score in the middle (not triggering mean reversion or stop)
    mean_s = float(np.mean(base_spreads[-20:]))
    std_s = float(np.std(base_spreads[-20:], ddof=1))
    target_spread = mean_s + 1.5 * std_s  # z ~ 1.5, not triggering entry or stop
    price_b = 50.0
    price_a = target_spread + 2.0 * price_b

    snap = make_snapshot(
        ts=200,
        day_bars={
            "SYM_A": {"close": price_a},
            "SYM_B": {"close": price_b},
        },
        positions=[
            Position("SYM_A", -10, price_a, 0.0),
            Position("SYM_B", 20, price_b, 0.0),
        ],
        pending_orders=[
            PendingOrder("SYM_A", "BUY", 10, "SL_M", 0.0, 50000.0),
            PendingOrder("SYM_B", "SELL", 20, "SL_M", 0.0, 0.01),
        ],
    )
    signals = s.on_bar(snap)

    buy_a = [sig for sig in signals if sig.action == "BUY" and sig.symbol == "SYM_A"
             and sig.order_type == "MARKET"]
    sell_b = [sig for sig in signals if sig.action == "SELL" and sig.symbol == "SYM_B"
              and sig.order_type == "MARKET"]
    assert len(buy_a) >= 1, f"Expected time-stop BUY SYM_A, got: {signals}"
    assert len(sell_b) >= 1, f"Expected time-stop SELL SYM_B, got: {signals}"
    assert s.in_trade is False


def test_equal_dollar_sizing():
    """Both legs should have similar dollar amounts."""
    s = create_strategy({"entry_threshold": 2.0, "risk_pct": 0.02, "zscore_period": 20})
    force_pair_selected(s, hedge_ratio=2.0)
    s._ensure_prices("SYM_A")
    s._ensure_prices("SYM_B")

    import numpy as np
    rng = np.random.RandomState(10)
    base_spreads = list(rng.randn(25) * 1.0)
    force_pair_selected(s, hedge_ratio=2.0, spread_values=base_spreads)

    # Trigger entry with zscore > +2
    mean_s = float(np.mean(base_spreads[-20:]))
    std_s = float(np.std(base_spreads[-20:], ddof=1))
    target_spread = mean_s + 2.5 * std_s
    price_b = 50.0
    price_a = target_spread + 2.0 * price_b

    snap = make_snapshot(
        ts=100,
        day_bars={
            "SYM_A": {"close": price_a},
            "SYM_B": {"close": price_b},
        },
        equity=1_000_000.0,
    )
    signals = s.on_bar(snap)

    sell_a = [sig for sig in signals if sig.action == "SELL" and sig.symbol == "SYM_A"]
    buy_b = [sig for sig in signals if sig.action == "BUY" and sig.symbol == "SYM_B"]
    assert len(sell_a) >= 1
    assert len(buy_b) >= 1

    dollar_a = sell_a[0].quantity * price_a
    dollar_b = buy_b[0].quantity * price_b

    # Both should be approximately capital * risk_pct / 2 = 1M * 0.02 / 2 = 10,000
    target = 1_000_000.0 * 0.02 / 2.0
    assert abs(dollar_a - target) / target < 0.5, \
        f"Dollar A ({dollar_a}) too far from target ({target})"
    assert abs(dollar_b - target) / target < 0.5, \
        f"Dollar B ({dollar_b}) too far from target ({target})"


def test_both_legs_mis():
    """Both legs should use MIS product type."""
    s = create_strategy({"entry_threshold": 2.0, "zscore_period": 20})
    force_pair_selected(s, hedge_ratio=2.0)
    s._ensure_prices("SYM_A")
    s._ensure_prices("SYM_B")

    import numpy as np
    rng = np.random.RandomState(10)
    base_spreads = list(rng.randn(25) * 1.0)
    force_pair_selected(s, hedge_ratio=2.0, spread_values=base_spreads)

    # Test short_spread entry: short A, long B
    mean_s = float(np.mean(base_spreads[-20:]))
    std_s = float(np.std(base_spreads[-20:], ddof=1))
    target_spread = mean_s + 2.5 * std_s
    price_b = 50.0
    price_a = target_spread + 2.0 * price_b

    snap = make_snapshot(
        ts=100,
        day_bars={
            "SYM_A": {"close": price_a},
            "SYM_B": {"close": price_b},
        },
        equity=1_000_000.0,
    )
    signals = s.on_bar(snap)

    sell_a = [sig for sig in signals if sig.action == "SELL" and sig.symbol == "SYM_A"]
    buy_b = [sig for sig in signals if sig.action == "BUY" and sig.symbol == "SYM_B"]
    assert len(sell_a) >= 1
    assert len(buy_b) >= 1

    # Short leg is always MIS (enter_short enforces it)
    assert sell_a[0].product_type == "MIS"
    # Long leg should also be MIS (we pass "MIS" to enter_long)
    assert buy_b[0].product_type == "MIS"
