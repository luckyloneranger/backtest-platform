"""Tests for Relative Strength Rotation strategy.

Covers: required_data, warmup gating, ranking logic (long/short split),
shorts always MIS, equal dollar allocation, exit at 15:00,
no re-entry after ranking, market neutrality (same longs and shorts).
"""

import pytest
from datetime import datetime, timezone, timedelta

from strategies.base import (
    BarData, FillInfo, MarketSnapshot, PendingOrder, Portfolio,
    Signal, SessionContext, Position,
)
from strategies.deterministic.relative_strength import RelativeStrength, _ist_time


IST = timezone(timedelta(hours=5, minutes=30))
CAPITAL = 1_000_000.0

# 6 symbols needed for default n_long=3, n_short=3
SYMBOLS = ["SYM_A", "SYM_B", "SYM_C", "SYM_D", "SYM_E", "SYM_F"]


def _ts(hour: int, minute: int = 0, day: int = 2) -> int:
    """Build epoch ms for a given IST hour:minute on 2024-01-{day}."""
    dt = datetime(2024, 1, day, hour, minute, tzinfo=IST)
    return int(dt.timestamp() * 1000)


def _multi_snap(symbol_prices: dict[str, float], timestamp_ms: int,
                capital: float = CAPITAL, fills=None,
                pending_orders=None, positions=None,
                open_prices: dict[str, float] | None = None) -> MarketSnapshot:
    """Build a 15-minute MarketSnapshot with multiple symbols.

    symbol_prices: {symbol: close_price}
    open_prices: {symbol: open_price} -- if None, open = close
    """
    bars = {}
    for sym, close in symbol_prices.items():
        op = close if open_prices is None else open_prices.get(sym, close)
        bars[sym] = BarData(sym, op, close + 1, close - 1, close,
                            100000, 0, timestamp_ms)
    return MarketSnapshot(
        timestamp_ms=timestamp_ms,
        timeframes={"15minute": bars},
        history={},
        portfolio=Portfolio(capital, capital, positions or []),
        instruments={},
        fills=fills or [],
        rejections=[],
        closed_trades=[],
        context=SessionContext(capital, 0, 200, "2024-01-01", "2024-12-31",
                               ["15minute"], 30),
        pending_orders=pending_orders or [],
    )


def _init(**overrides) -> RelativeStrength:
    """Create and initialize a RelativeStrength with defaults."""
    s = RelativeStrength()
    config = {
        "n_long": 3,
        "n_short": 3,
        "risk_pct": 0.20,
        "warmup_bars": 2,
        "exit_time_hour": 15,
        "atr_period": 14,
        "atr_stop_mult": 1.5,
    }
    config.update(overrides)
    s.initialize(config, {})
    return s


def _warmup_prices(base: float = 100.0) -> dict[str, float]:
    """Return consistent prices for all 6 symbols during warmup."""
    return {sym: base for sym in SYMBOLS}


def _feed_warmup(s: RelativeStrength, n: int = 2):
    """Feed n warmup bars (9:15, 9:30) with uniform prices.

    Returns all signals generated during warmup.
    """
    all_signals = []
    for i in range(n):
        minute = 15 + i * 15  # 9:15, 9:30 for 15-min bars
        ts = _ts(9, minute)
        prices = _warmup_prices()
        snap = _multi_snap(prices, ts)
        sigs = s.on_bar(snap)
        all_signals.extend(sigs)
    return all_signals


def _ranking_prices() -> dict[str, float]:
    """Symbol prices at ranking time that produce clear 30-min returns.

    All symbols opened at 100.0 during warmup. Returns at 9:45:
      SYM_A: 105 -> +5%
      SYM_B: 103 -> +3%
      SYM_C: 101 -> +1%
      SYM_D:  99 -> -1%
      SYM_E:  97 -> -3%
      SYM_F:  95 -> -5%

    Expected: long A, B, C; short D, E, F
    """
    return {
        "SYM_A": 105.0,
        "SYM_B": 103.0,
        "SYM_C": 101.0,
        "SYM_D": 99.0,
        "SYM_E": 97.0,
        "SYM_F": 95.0,
    }


class TestRequiredData:
    def test_required_data(self):
        s = RelativeStrength()
        reqs = s.required_data()
        assert reqs == [{"interval": "15minute", "lookback": 30}]


class TestNoTradeDuringWarmup:
    def test_no_trade_during_warmup(self):
        """No entry signals during the warmup period (first 2 bars)."""
        s = _init()
        signals = _feed_warmup(s, n=2)

        buys = [sig for sig in signals if sig.action == "BUY"]
        sells = [sig for sig in signals if sig.action == "SELL"]
        assert len(buys) == 0
        assert len(sells) == 0


class TestRankingProducesLongsAndShorts:
    def test_ranking_top3_long_bottom3_short(self):
        """Top 3 by 30-min return get BUY, bottom 3 get SELL."""
        s = _init()
        _feed_warmup(s, n=2)

        # Bar 3 at 9:45 -- ranking time
        ts = _ts(9, 45)
        prices = _ranking_prices()
        snap = _multi_snap(prices, ts)
        signals = s.on_bar(snap)

        buys = [sig for sig in signals if sig.action == "BUY"]
        sells = [sig for sig in signals if sig.action == "SELL"]

        buy_syms = {sig.symbol for sig in buys}
        sell_syms = {sig.symbol for sig in sells}

        # Top 3 should be long
        assert buy_syms == {"SYM_A", "SYM_B", "SYM_C"}, f"Got buys for {buy_syms}"

        # Bottom 3 should be short
        assert sell_syms == {"SYM_D", "SYM_E", "SYM_F"}, f"Got sells for {sell_syms}"


class TestShortsAlwaysMIS:
    def test_shorts_are_mis(self):
        """All short entries must use MIS product type."""
        s = _init()
        _feed_warmup(s, n=2)

        ts = _ts(9, 45)
        prices = _ranking_prices()
        snap = _multi_snap(prices, ts)
        signals = s.on_bar(snap)

        sells = [sig for sig in signals if sig.action == "SELL"]
        for sig in sells:
            assert sig.product_type == "MIS", \
                f"{sig.symbol} short has product_type={sig.product_type}, expected MIS"

    def test_longs_also_mis(self):
        """All entries are MIS since this is an intraday strategy."""
        s = _init()
        _feed_warmup(s, n=2)

        ts = _ts(9, 45)
        prices = _ranking_prices()
        snap = _multi_snap(prices, ts)
        signals = s.on_bar(snap)

        buys = [sig for sig in signals if sig.action == "BUY"]
        for sig in buys:
            assert sig.product_type == "MIS", \
                f"{sig.symbol} long has product_type={sig.product_type}, expected MIS"


class TestEqualDollarAllocation:
    def test_equal_dollar_allocation(self):
        """All positions have roughly the same dollar value."""
        s = _init()
        _feed_warmup(s, n=2)

        ts = _ts(9, 45)
        prices = _ranking_prices()
        snap = _multi_snap(prices, ts)
        signals = s.on_bar(snap)

        entries = [sig for sig in signals if sig.action in ("BUY", "SELL")]
        assert len(entries) == 6  # 3 long + 3 short

        # Expected allocation per position: 1M * 0.20 / 6 = ~33,333
        expected_alloc = CAPITAL * 0.20 / 6

        for sig in entries:
            dollar_value = sig.quantity * prices[sig.symbol]
            # Should be within 2% of expected allocation (rounding)
            assert abs(dollar_value - expected_alloc) < expected_alloc * 0.02, \
                f"{sig.symbol}: ${dollar_value:.0f} vs expected ${expected_alloc:.0f}"


class TestExitAt1500:
    def test_exit_at_1500(self):
        """All positions closed when hour >= 15."""
        s = _init()
        _feed_warmup(s, n=2)

        # Trigger entry at 9:45
        ts_entry = _ts(9, 45)
        prices = _ranking_prices()
        snap = _multi_snap(prices, ts_entry)
        entry_sigs = s.on_bar(snap)

        buys = [sig for sig in entry_sigs if sig.action == "BUY"]
        sells = [sig for sig in entry_sigs if sig.action == "SELL"]
        assert len(buys) == 3
        assert len(sells) == 3

        # Simulate fills for long positions (pick first long symbol)
        long_sym = buys[0].symbol
        long_qty = buys[0].quantity
        fill = FillInfo(long_sym, "BUY", long_qty, prices[long_sym], 0.0, ts_entry)
        position = Position(long_sym, long_qty, prices[long_sym], 0.0)
        pending_stop = PendingOrder(long_sym, "SELL", long_qty, "SL_M", 0.0,
                                     prices[long_sym] * 0.95)

        # Feed a mid-day bar with the fill
        ts_mid = _ts(10, 0)
        snap_mid = _multi_snap(prices, ts_mid, fills=[fill],
                                positions=[position],
                                pending_orders=[pending_stop])
        s.on_bar(snap_mid)

        # Jump to 15:00 -- should exit
        ts_1500 = _ts(15, 0)
        snap_exit = _multi_snap(prices, ts_1500,
                                 positions=[position],
                                 pending_orders=[pending_stop])
        exit_sigs = s.on_bar(snap_exit)

        # Should have CANCEL + SELL MARKET for exit
        sell_exits = [sig for sig in exit_sigs
                      if sig.symbol == long_sym and sig.action == "SELL"
                      and sig.order_type == "MARKET"]
        assert len(sell_exits) >= 1, \
            f"Expected SELL MARKET exit for {long_sym}, got {exit_sigs}"


class TestNoReentryAfterRanking:
    def test_no_reentry_after_ranking(self):
        """After ranking bar, no new entries on subsequent bars."""
        s = _init()
        _feed_warmup(s, n=2)

        # Ranking bar at 9:45
        ts_rank = _ts(9, 45)
        prices = _ranking_prices()
        snap = _multi_snap(prices, ts_rank)
        rank_sigs = s.on_bar(snap)

        entries_at_rank = [sig for sig in rank_sigs
                           if sig.action in ("BUY", "SELL")]
        assert len(entries_at_rank) == 6

        # Next bar at 10:00 -- should NOT produce new entries
        ts_next = _ts(10, 0)
        snap_next = _multi_snap(prices, ts_next)
        next_sigs = s.on_bar(snap_next)

        new_entries = [sig for sig in next_sigs
                       if sig.action in ("BUY", "SELL")
                       and sig.order_type == "MARKET"]
        assert len(new_entries) == 0, \
            f"Expected no new entries after ranking, got {len(new_entries)}"

        # Another bar at 11:00 -- also no entries
        ts_late = _ts(11, 0)
        snap_late = _multi_snap(prices, ts_late)
        late_sigs = s.on_bar(snap_late)

        late_entries = [sig for sig in late_sigs
                        if sig.action in ("BUY", "SELL")
                        and sig.order_type == "MARKET"]
        assert len(late_entries) == 0


class TestMarketNeutral:
    def test_same_number_longs_and_shorts(self):
        """Strategy produces equal number of long and short positions."""
        s = _init()
        _feed_warmup(s, n=2)

        ts = _ts(9, 45)
        prices = _ranking_prices()
        snap = _multi_snap(prices, ts)
        signals = s.on_bar(snap)

        buys = [sig for sig in signals if sig.action == "BUY"]
        sells = [sig for sig in signals if sig.action == "SELL"]

        assert len(buys) == len(sells) == 3

    def test_fewer_symbols_skips_trading(self):
        """If fewer symbols than n_long + n_short, skip trading."""
        s = _init(n_long=3, n_short=3)
        # Only feed 4 symbols (need 6)
        few_symbols = {"SYM_A": 105.0, "SYM_B": 103.0,
                        "SYM_C": 101.0, "SYM_D": 99.0}

        # Warmup with 4 symbols
        for i in range(2):
            minute = 15 + i * 15
            ts = _ts(9, minute)
            snap = _multi_snap({sym: 100.0 for sym in few_symbols}, ts)
            s.on_bar(snap)

        # Ranking bar with only 4 symbols
        ts_rank = _ts(9, 45)
        snap_rank = _multi_snap(few_symbols, ts_rank)
        sigs = s.on_bar(snap_rank)

        entries = [sig for sig in sigs if sig.action in ("BUY", "SELL")]
        assert len(entries) == 0, \
            f"Expected no entries with only 4 symbols, got {len(entries)}"
