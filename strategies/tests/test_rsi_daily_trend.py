"""Tests for RSI + Daily EMA Trend strategy."""

from strategies.base import (
    BarData, MarketSnapshot, Portfolio, Position, SessionContext, InstrumentInfo, Signal,
)
from strategies.deterministic.rsi_daily_trend import RsiDailyTrend, compute_rsi, compute_ema


# --- Helper ---

def make_snapshot(
    ts: int,
    close_15m: float | None = None,
    close_day: float | None = None,
    symbol: str = "TEST",
    cash: float = 100_000.0,
    positions: list[Position] | None = None,
) -> MarketSnapshot:
    timeframes = {}
    if close_15m is not None:
        bar = BarData(symbol, close_15m, close_15m + 1, close_15m - 1, close_15m, 1000, 0)
        timeframes["15minute"] = {symbol: bar}
    if close_day is not None:
        bar = BarData(symbol, close_day, close_day + 1, close_day - 1, close_day, 50000, 0)
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


# --- Unit tests for indicators ---

def test_compute_rsi_uptrend():
    prices = [100 + i for i in range(20)]
    rsi = compute_rsi(prices, 14)
    assert rsi is not None
    assert rsi > 90


def test_compute_rsi_downtrend():
    prices = [100 - i for i in range(20)]
    rsi = compute_rsi(prices, 14)
    assert rsi is not None
    assert rsi < 10


def test_compute_rsi_not_enough_data():
    prices = [100, 101, 102]
    rsi = compute_rsi(prices, 14)
    assert rsi is None


def test_compute_ema_basic():
    prices = [10.0, 11.0, 12.0, 13.0, 14.0]
    ema = compute_ema(prices, 3)
    assert ema is not None
    assert 12.0 < ema < 14.0


def test_compute_ema_not_enough_data():
    prices = [10.0, 11.0]
    ema = compute_ema(prices, 5)
    assert ema is None


# --- Strategy tests ---

def test_no_signal_during_warmup():
    s = RsiDailyTrend()
    s.initialize({"rsi_period": 5, "ema_period": 3}, {})

    for i in range(4):
        signals = s.on_bar(make_snapshot(i, close_15m=100.0, close_day=100.0))
        assert signals == []


def test_buy_on_rsi_oversold_with_uptrend():
    s = RsiDailyTrend()
    s.initialize({"rsi_period": 5, "rsi_oversold": 30, "ema_period": 3}, {})

    # Establish daily uptrend
    for i in range(5):
        s.on_bar(make_snapshot(i, close_day=100.0 + i * 2))

    # Feed 15-min bars: rise then sharp drop to trigger RSI oversold
    prices = [100, 102, 104, 106, 108, 110, 105, 100, 95, 90]
    all_signals = []
    for i, p in enumerate(prices):
        signals = s.on_bar(make_snapshot(100 + i, close_15m=float(p)))
        all_signals.extend(signals)

    assert any(sig.action == "BUY" for sig in all_signals)


def test_no_buy_when_daily_trend_is_down():
    s = RsiDailyTrend()
    s.initialize({"rsi_period": 5, "rsi_oversold": 30, "ema_period": 3}, {})

    # Establish daily downtrend
    for i in range(5):
        s.on_bar(make_snapshot(i, close_day=100.0 - i * 5))

    prices = [100, 102, 104, 106, 108, 110, 105, 100, 95, 90]
    all_signals = []
    for i, p in enumerate(prices):
        signals = s.on_bar(make_snapshot(100 + i, close_15m=float(p)))
        all_signals.extend(signals)

    assert not any(sig.action == "BUY" for sig in all_signals)


def test_sell_on_rsi_overbought():
    s = RsiDailyTrend()
    s.initialize({"rsi_period": 5, "rsi_oversold": 30, "rsi_overbought": 70, "ema_period": 3}, {})

    # Establish daily uptrend
    for i in range(5):
        s.on_bar(make_snapshot(i, close_day=100.0 + i * 2))

    # Drop to trigger buy
    prices_down = [100, 102, 104, 106, 108, 110, 105, 100, 95, 90]
    for i, p in enumerate(prices_down):
        s.on_bar(make_snapshot(100 + i, close_15m=float(p)))

    # Rise to trigger overbought sell
    prices_up = [92, 95, 100, 105, 110, 115, 120, 125, 130, 135]
    held = Position(symbol="TEST", quantity=200, avg_price=90.0, unrealized_pnl=0.0)
    all_signals = []
    for i, p in enumerate(prices_up):
        signals = s.on_bar(make_snapshot(200 + i, close_15m=float(p), positions=[held]))
        all_signals.extend(signals)

    assert any(sig.action == "SELL" for sig in all_signals)


def test_required_data():
    s = RsiDailyTrend()
    reqs = s.required_data()
    intervals = [r["interval"] for r in reqs]
    assert "15minute" in intervals
    assert "day" in intervals
    assert all("lookback" in r for r in reqs)


def test_on_complete_returns_metadata():
    s = RsiDailyTrend()
    s.initialize({"rsi_period": 14, "ema_period": 20, "risk_pct": 0.15}, {})
    result = s.on_complete()
    assert result["strategy_type"] == "rsi_daily_trend"
    assert result["rsi_period"] == 14
    assert result["ema_period"] == 20
    assert result["risk_pct"] == 0.15
