"""Tests for LLM Autonomous Trader strategy."""

import json
from unittest.mock import patch, MagicMock

from strategies.base import (
    BarData,
    FillInfo,
    MarketSnapshot,
    PendingOrder,
    Portfolio,
    Position,
    SessionContext,
    TradeInfo,
    InstrumentInfo,
)
from strategies.llm.llm_autonomous_trader import LLMAutonomousTrader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bar(symbol: str = "TEST", close: float = 100.0) -> BarData:
    return BarData(
        symbol=symbol,
        open=close - 1,
        high=close + 2,
        low=close - 2,
        close=close,
        volume=50_000,
        oi=0,
        timestamp_ms=0,
    )


def _make_snapshot(
    symbols: list[str] | None = None,
    cash: float = 100_000.0,
    close: float = 100.0,
    positions: list[Position] | None = None,
    fills: list[FillInfo] | None = None,
    closed_trades: list[TradeInfo] | None = None,
    pending_orders: list[PendingOrder] | None = None,
    bar_number: int = 1,
    initial_capital: float = 100_000.0,
    equity: float | None = None,
) -> MarketSnapshot:
    if symbols is None:
        symbols = ["TEST"]
    bars = {s: _make_bar(s, close) for s in symbols}
    return MarketSnapshot(
        timestamp_ms=1000,
        timeframes={"day": bars},
        history={},
        portfolio=Portfolio(
            cash=cash,
            equity=equity if equity is not None else cash,
            positions=positions or [],
        ),
        instruments={},
        fills=fills or [],
        rejections=[],
        closed_trades=closed_trades or [],
        context=SessionContext(
            initial_capital, bar_number, 252, "2024-01-01", "2024-12-31", ["day"], 200
        ),
        pending_orders=pending_orders or [],
    )


def _init_strategy(config: dict | None = None) -> LLMAutonomousTrader:
    """Create and initialize strategy with mocked AzureOpenAIClient."""
    s = LLMAutonomousTrader()
    with patch(
        "strategies.llm.llm_autonomous_trader.AzureOpenAIClient"
    ) as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        s.initialize(config or {"initial_capital": 100_000}, {})
    return s


def _warm_up(strategy: LLMAutonomousTrader, symbols: list[str] | None = None, bars: int = 55):
    """Feed enough bars to produce valid indicators (>= 50 required)."""
    if symbols is None:
        symbols = ["TEST"]
    for i in range(bars):
        snap = _make_snapshot(
            symbols=symbols,
            close=100.0 + (i % 10) * 0.5,
            bar_number=i,
        )
        strategy.pm.increment_bars()
        for symbol, bar in snap.timeframes["day"].items():
            strategy._update_buffers(symbol, bar)
        strategy.last_bar_number = i


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_required_data():
    s = LLMAutonomousTrader()
    reqs = s.required_data()
    assert len(reqs) == 1
    assert reqs[0]["interval"] == "day"
    assert reqs[0]["lookback"] == 200


def test_narrative_sent_to_llm():
    """Verify the LLM receives narrative text (not raw numbers),
    and check for keywords like 'oversold', 'TRENDING', 'CONFLUENCE'."""
    s = _init_strategy()
    _warm_up(s)

    captured_messages = []

    def _capture(messages, **kwargs):
        captured_messages.extend(messages)
        return json.dumps([])

    s.client.chat_completion = _capture

    snap = _make_snapshot(close=100.0, bar_number=100)
    s.on_bar(snap)

    assert len(captured_messages) == 2
    user_content = captured_messages[1]["content"]

    # Narrative builder keywords that appear in English interpretation
    assert "PORTFOLIO SUMMARY" in user_content
    assert "MARKET REGIME" in user_content
    assert "TEST" in user_content
    # Confluence section is always present in symbol narrative
    assert "CONFLUENCE" in user_content
    # Should contain narrative text, not raw arrays of numbers
    assert "SUGGESTION" in user_content


def test_guardrail_position_cap():
    """LLM suggests qty=1000 but cash only allows fewer shares."""
    s = _init_strategy({"initial_capital": 1000})
    _warm_up(s)

    s.client.chat_completion = MagicMock(
        return_value=json.dumps(
            [
                {
                    "action": "BUY",
                    "symbol": "TEST",
                    "quantity": 1000,
                    "order_type": "MARKET",
                    "product_type": "CNC",
                    "stop_price": 95.0,
                    "reasoning": "test position cap",
                }
            ]
        )
    )

    # Cash = 1000, price = 100 -> max 10 shares
    snap = _make_snapshot(cash=1000.0, close=100.0, bar_number=100, initial_capital=1000)
    signals = s.on_bar(snap)

    buy_signals = [sig for sig in signals if sig.action == "BUY"]
    assert len(buy_signals) == 1
    assert buy_signals[0].quantity <= 10


def test_guardrail_max_positions():
    """Already at max_positions, verify new entry rejected."""
    s = _init_strategy({"initial_capital": 100_000, "max_positions": 2})
    _warm_up(s, symbols=["A", "B", "C"])

    # Simulate 2 existing open positions
    s.pm.get_state("A").direction = "long"
    s.pm.get_state("A").qty = 10
    s.pm.get_state("B").direction = "long"
    s.pm.get_state("B").qty = 10

    s.client.chat_completion = MagicMock(
        return_value=json.dumps(
            [
                {
                    "action": "BUY",
                    "symbol": "C",
                    "quantity": 5,
                    "order_type": "MARKET",
                    "product_type": "CNC",
                    "stop_price": 95.0,
                    "reasoning": "should be rejected",
                }
            ]
        )
    )

    # Must include A and B in portfolio positions so reconcile() doesn't reset them
    snap = _make_snapshot(
        symbols=["A", "B", "C"],
        bar_number=100,
        positions=[
            Position("A", 10, 100.0, 0.0),
            Position("B", 10, 100.0, 0.0),
        ],
    )
    signals = s.on_bar(snap)

    # The BUY for C should be rejected; only PM lifecycle signals expected
    buy_c = [sig for sig in signals if sig.action == "BUY" and sig.symbol == "C"]
    assert len(buy_c) == 0


def test_guardrail_auto_stop():
    """LLM submits BUY without stop, verify SL_M auto-added."""
    s = _init_strategy()
    _warm_up(s)

    s.client.chat_completion = MagicMock(
        return_value=json.dumps(
            [
                {
                    "action": "BUY",
                    "symbol": "TEST",
                    "quantity": 5,
                    "order_type": "MARKET",
                    "product_type": "CNC",
                    "stop_price": 0,
                    "reasoning": "no stop provided",
                }
            ]
        )
    )

    snap = _make_snapshot(close=100.0, bar_number=100)
    signals = s.on_bar(snap)

    # enter_long returns a BUY signal; the SL_M is submitted after fill
    # via process_fills, but the guardrail should have computed a stop
    buy_signals = [sig for sig in signals if sig.action == "BUY"]
    assert len(buy_signals) == 1
    # Verify the PM state has a trailing_stop set (auto-computed)
    state = s.pm.get_state("TEST")
    assert state.trailing_stop > 0
    assert state.trailing_stop < 100.0  # stop should be below entry price


def test_guardrail_drawdown_scaling():
    """Drawdown > 10%, verify qty halved."""
    s = _init_strategy({"initial_capital": 100_000})
    _warm_up(s)
    s.peak_equity = 100_000.0

    s.client.chat_completion = MagicMock(
        return_value=json.dumps(
            [
                {
                    "action": "BUY",
                    "symbol": "TEST",
                    "quantity": 20,
                    "order_type": "MARKET",
                    "product_type": "CNC",
                    "stop_price": 90.0,
                    "reasoning": "drawdown test",
                }
            ]
        )
    )

    # Equity is 85000 while peak is 100000 => 15% drawdown
    snap = _make_snapshot(
        cash=85_000.0, close=100.0, bar_number=100, equity=85_000.0
    )
    signals = s.on_bar(snap)

    buy_signals = [sig for sig in signals if sig.action == "BUY"]
    assert len(buy_signals) == 1
    # qty=20 halved to 10
    assert buy_signals[0].quantity == 10


def test_llm_failure_returns_hold():
    """Mock LLM error, verify no crash and only PM lifecycle signals."""
    s = _init_strategy()
    _warm_up(s)

    s.client.chat_completion = MagicMock(
        side_effect=LLMClientError("API error")
    )

    snap = _make_snapshot(bar_number=100)
    signals = s.on_bar(snap)

    # Should not crash; returns only PM lifecycle signals (empty here)
    assert isinstance(signals, list)
    # No BUY/SELL from LLM
    buy_sell = [sig for sig in signals if sig.action in ("BUY", "SELL")]
    assert len(buy_sell) == 0


def test_trade_log_updated():
    """Verify closed trade appears in trade_log."""
    s = _init_strategy()
    _warm_up(s)

    s.client.chat_completion = MagicMock(return_value=json.dumps([]))

    trade = TradeInfo(
        symbol="TEST",
        quantity=10,
        entry_price=100.0,
        exit_price=110.0,
        entry_timestamp_ms=0,
        exit_timestamp_ms=1000,
        pnl=100.0,
        costs=5.0,
    )

    snap = _make_snapshot(bar_number=100, closed_trades=[trade])
    s.on_bar(snap)

    assert len(s.trade_log) == 1
    assert s.trade_log[0]["symbol"] == "TEST"
    assert s.trade_log[0]["pnl"] == 100.0
    assert s.trade_log[0]["side"] == "LONG"  # entry < exit


def test_short_always_mis():
    """Verify short entries use MIS product type."""
    s = _init_strategy()
    _warm_up(s)

    s.client.chat_completion = MagicMock(
        return_value=json.dumps(
            [
                {
                    "action": "SELL",
                    "symbol": "TEST",
                    "quantity": 5,
                    "order_type": "MARKET",
                    "product_type": "CNC",
                    "stop_price": 110.0,
                    "reasoning": "short entry",
                }
            ]
        )
    )

    snap = _make_snapshot(close=100.0, bar_number=100)
    signals = s.on_bar(snap)

    sell_signals = [sig for sig in signals if sig.action == "SELL"]
    assert len(sell_signals) == 1
    assert sell_signals[0].product_type == "MIS"


def test_parse_llm_with_reasoning():
    """Verify reasoning field extracted from LLM JSON."""
    s = _init_strategy()

    response = json.dumps(
        [
            {
                "action": "BUY",
                "symbol": "TEST",
                "quantity": 5,
                "order_type": "MARKET",
                "product_type": "CNC",
                "stop_price": 95.0,
                "reasoning": "RSI oversold at 22, strong buy signal",
            }
        ]
    )

    parsed = s._parse_llm_response(response)
    assert len(parsed) == 1
    assert parsed[0]["reasoning"] == "RSI oversold at 22, strong buy signal"


def test_parse_llm_markdown_code_block():
    """Verify parsing handles markdown code blocks."""
    s = _init_strategy()

    response = '```json\n[{"action": "BUY", "symbol": "TEST", "quantity": 5}]\n```'
    parsed = s._parse_llm_response(response)
    assert len(parsed) == 1
    assert parsed[0]["action"] == "BUY"


def test_on_complete():
    s = _init_strategy()
    result = s.on_complete()
    assert result["strategy_type"] == "llm_autonomous_trader"
    assert result["total_trades"] == 0
    assert result["total_costs"] == 0.0


# Import LLMClientError for LLM failure test
from strategies.llm_client import LLMClientError  # noqa: E402
