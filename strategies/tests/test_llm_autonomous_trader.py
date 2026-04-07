"""Tests for LLM Autonomous Trader strategy — 15-min multi-timeframe thesis-driven approach."""

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
    timeframes: dict | None = None,
    timestamp_ms: int = 1000,
) -> MarketSnapshot:
    if symbols is None:
        symbols = ["TEST"]
    if timeframes is None:
        # Default: both day and 15minute
        bars_day = {s: _make_bar(s, close) for s in symbols}
        bars_15m = {s: _make_bar(s, close) for s in symbols}
        timeframes = {"day": bars_day, "15minute": bars_15m}
    intervals = list(timeframes.keys())
    return MarketSnapshot(
        timestamp_ms=timestamp_ms,
        timeframes=timeframes,
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
            initial_capital, bar_number, 252, "2024-01-01", "2024-12-31",
            intervals, 200,
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


def _warm_up_daily(strategy: LLMAutonomousTrader, symbols: list[str] | None = None, bars: int = 55):
    """Feed enough daily bars to produce valid indicators (>= 50 required)."""
    if symbols is None:
        symbols = ["TEST"]
    for i in range(bars):
        snap = _make_snapshot(
            symbols=symbols,
            close=100.0 + (i % 10) * 0.5,
            bar_number=i,
            timeframes={"day": {s: _make_bar(s, 100.0 + (i % 10) * 0.5) for s in symbols}},
        )
        strategy.pm.increment_bars()
        for symbol, bar in snap.timeframes["day"].items():
            strategy._update_daily_buffers(symbol, bar)
        strategy.last_daily_date = f"2024-01-{i+1:02d}"

    # Trigger daily analysis recompute
    strategy._recompute_daily_analysis(snap)


def _warm_up_m15(strategy: LLMAutonomousTrader, symbols: list[str] | None = None, bars: int = 55):
    """Feed enough 15-min bars to build rolling buffers."""
    if symbols is None:
        symbols = ["TEST"]
    for i in range(bars):
        for symbol in symbols:
            bar = _make_bar(symbol, 100.0 + (i % 10) * 0.3)
            strategy._update_m15_buffers(symbol, bar)
            strategy._update_intraday_buffers(symbol, bar)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_required_data():
    s = LLMAutonomousTrader()
    reqs = s.required_data()
    assert len(reqs) == 2
    intervals = {r["interval"] for r in reqs}
    assert "15minute" in intervals
    assert "day" in intervals
    for r in reqs:
        assert r["lookback"] == 200


def test_narrative_sent_to_llm():
    """Verify the LLM receives multi-timeframe narrative with thesis-driven system prompt."""
    s = _init_strategy()
    _warm_up_daily(s)
    _warm_up_m15(s)

    captured_messages = []

    def _capture(messages, **kwargs):
        captured_messages.extend(messages)
        return json.dumps([])

    s.client.chat_completion = _capture
    # Set bar_count so throttle triggers
    s.bar_count = s.llm_interval_bars - 1

    snap = _make_snapshot(close=100.0, bar_number=100, timestamp_ms=1704067200000)
    s.on_bar(snap)

    assert len(captured_messages) == 2

    system_content = captured_messages[0]["content"]
    user_content = captured_messages[1]["content"]

    # System prompt should contain thesis-driven keywords
    assert "THESIS" in system_content
    assert "CONVICTION" in system_content
    assert "COUNTER-THESIS" in system_content
    assert "EVIDENCE" in system_content
    # Multi-timeframe keywords
    assert "multi-timeframe" in system_content.lower()
    assert "VWAP" in system_content
    # System prompt should NOT contain old rule-based keywords
    assert "CRITICAL TRADING RULES" not in system_content
    assert "ADX>25" not in system_content

    # Narrative (user content) should contain factual sections
    assert "PORTFOLIO SUMMARY" in user_content
    assert "TEST" in user_content
    # Narrative should NOT contain interpretive keywords
    assert "SUGGESTION" not in user_content
    assert "CONFLUENCE" not in user_content
    assert "POTENTIAL LONG" not in user_content


def test_guardrail_position_cap():
    """LLM suggests qty=1000 but cash only allows fewer shares."""
    s = _init_strategy({"initial_capital": 1000})
    _warm_up_daily(s)
    _warm_up_m15(s)
    s.bar_count = s.llm_interval_bars - 1

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
                    "reasoning": "CONVICTION: 9/10 test position cap",
                }
            ]
        )
    )

    snap = _make_snapshot(
        cash=1000.0, close=100.0, bar_number=100,
        initial_capital=1000, timestamp_ms=1704067200000,
    )
    signals = s.on_bar(snap)

    buy_signals = [sig for sig in signals if sig.action == "BUY"]
    assert len(buy_signals) == 1
    assert buy_signals[0].quantity <= 10


def test_guardrail_max_positions():
    """Already at max_positions, verify new entry rejected."""
    s = _init_strategy({"initial_capital": 100_000, "max_positions": 2})
    _warm_up_daily(s, symbols=["A", "B", "C"])
    _warm_up_m15(s, symbols=["A", "B", "C"])
    s.bar_count = s.llm_interval_bars - 1

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
                    "reasoning": "CONVICTION: 9/10 should be rejected",
                }
            ]
        )
    )

    snap = _make_snapshot(
        symbols=["A", "B", "C"],
        bar_number=100,
        positions=[
            Position("A", 10, 100.0, 0.0),
            Position("B", 10, 100.0, 0.0),
        ],
        timestamp_ms=1704067200000,
    )
    signals = s.on_bar(snap)

    buy_c = [sig for sig in signals if sig.action == "BUY" and sig.symbol == "C"]
    assert len(buy_c) == 0


def test_guardrail_auto_stop():
    """LLM submits BUY without stop, verify SL_M auto-added."""
    s = _init_strategy()
    _warm_up_daily(s)
    _warm_up_m15(s)
    s.bar_count = s.llm_interval_bars - 1

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
                    "reasoning": "CONVICTION: 8/10 no stop provided",
                }
            ]
        )
    )

    snap = _make_snapshot(close=100.0, bar_number=100, timestamp_ms=1704067200000)
    signals = s.on_bar(snap)

    buy_signals = [sig for sig in signals if sig.action == "BUY"]
    assert len(buy_signals) == 1
    state = s.pm.get_state("TEST")
    assert state.trailing_stop > 0
    assert state.trailing_stop < 100.0


def test_guardrail_drawdown_scaling():
    """Drawdown > 10%, verify qty halved."""
    s = _init_strategy({"initial_capital": 100_000})
    _warm_up_daily(s)
    _warm_up_m15(s)
    s.peak_equity = 100_000.0
    s.bar_count = s.llm_interval_bars - 1

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
                    "reasoning": "CONVICTION: 9/10 drawdown test",
                }
            ]
        )
    )

    snap = _make_snapshot(
        cash=85_000.0, close=100.0, bar_number=100,
        equity=85_000.0, timestamp_ms=1704067200000,
    )
    signals = s.on_bar(snap)

    buy_signals = [sig for sig in signals if sig.action == "BUY"]
    assert len(buy_signals) == 1
    # 20 * 0.75 (conviction 9) = 15, then // 2 (drawdown >10%) = 7
    assert buy_signals[0].quantity == 7
def test_llm_failure_returns_hold():
    """Mock LLM error, verify no crash and only PM lifecycle signals."""
    s = _init_strategy()
    _warm_up_daily(s)
    _warm_up_m15(s)
    s.bar_count = s.llm_interval_bars - 1

    s.client.chat_completion = MagicMock(
        side_effect=LLMClientError("API error")
    )

    snap = _make_snapshot(bar_number=100, timestamp_ms=1704067200000)
    signals = s.on_bar(snap)

    assert isinstance(signals, list)
    buy_sell = [sig for sig in signals if sig.action in ("BUY", "SELL")]
    assert len(buy_sell) == 0


def test_trade_log_updated():
    """Verify closed trade appears in trade_log."""
    s = _init_strategy()
    _warm_up_daily(s)
    _warm_up_m15(s)
    s.bar_count = s.llm_interval_bars - 1

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

    snap = _make_snapshot(bar_number=100, closed_trades=[trade], timestamp_ms=1704067200000)
    s.on_bar(snap)

    assert len(s.trade_log) == 1
    assert s.trade_log[0]["symbol"] == "TEST"
    assert s.trade_log[0]["pnl"] == 100.0
    assert s.trade_log[0]["side"] == "LONG"


def test_short_always_mis():
    """Verify short entries use MIS product type."""
    s = _init_strategy()
    _warm_up_daily(s)
    _warm_up_m15(s)
    s.bar_count = s.llm_interval_bars - 1

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
                    "reasoning": "CONVICTION: 8/10 short entry",
                }
            ]
        )
    )

    snap = _make_snapshot(close=100.0, bar_number=100, timestamp_ms=1704067200000)
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


def test_throttling_skips_non_interval_bars():
    """LLM is NOT called on every 15-min bar, only every llm_interval_bars."""
    s = _init_strategy({"initial_capital": 100_000, "llm_interval_bars": 4})
    _warm_up_daily(s)
    _warm_up_m15(s)

    call_count = 0
    def _track_calls(messages, **kwargs):
        nonlocal call_count
        call_count += 1
        return json.dumps([])

    s.client.chat_completion = _track_calls
    s.bar_count = 0

    # Send 8 bars — should trigger LLM on bar 4 and 8
    for i in range(8):
        snap = _make_snapshot(
            close=100.0, bar_number=100 + i,
            timeframes={"15minute": {"TEST": _make_bar("TEST", 100.0)}},
            timestamp_ms=1704067200000 + i * 900000,
        )
        s.on_bar(snap)

    assert call_count == 2


def test_intraday_buffers_reset_on_new_day():
    """Intraday buffers clear when a new daily bar arrives."""
    s = _init_strategy()
    _warm_up_daily(s)

    # Add some intraday data
    bar = _make_bar("TEST", 100.0)
    s._update_intraday_buffers("TEST", bar)
    s._update_intraday_buffers("TEST", bar)
    assert len(s.intraday_closes["TEST"]) == 2

    # Simulate new day by sending a day bar with new date
    s.last_daily_date = "2024-01-15"
    snap = _make_snapshot(
        close=101.0, bar_number=200,
        timeframes={
            "day": {"TEST": _make_bar("TEST", 101.0)},
            "15minute": {"TEST": _make_bar("TEST", 101.0)},
        },
        timestamp_ms=1705363200000,  # 2024-01-16
    )
    s.on_bar(snap)

    # Intraday should have been cleared and then re-populated with 1 bar
    assert len(s.intraday_closes["TEST"]) == 1


def test_daily_and_m15_buffers_separate():
    """Daily and 15-min buffers are independent."""
    s = _init_strategy()

    bar_daily = _make_bar("TEST", 200.0)
    bar_m15 = _make_bar("TEST", 201.0)

    s._update_daily_buffers("TEST", bar_daily)
    s._update_m15_buffers("TEST", bar_m15)

    assert list(s.daily_closes["TEST"]) == [200.0]
    assert list(s.m15_closes["TEST"]) == [201.0]


def test_beliefs_injected_into_system_prompt():
    """Verify regime-filtered beliefs appear in LLM system prompt."""
    s = _init_strategy({"initial_capital": 100_000, "reset_experience": True})
    _warm_up_daily(s)
    _warm_up_m15(s)
    s.bar_count = s.llm_interval_bars - 1

    # Inject structured beliefs
    s.experience.beliefs = [
        {"text": "Banks work in trends.", "regime": "all", "created_at": 0, "strength": 1.0},
        {"text": "Avoid ICICIBANK.", "regime": "ranging", "created_at": 0, "strength": 1.0},
    ]

    captured = []
    def _capture(messages, **kwargs):
        captured.extend(messages)
        return json.dumps([])
    s.client.chat_completion = _capture

    snap = _make_snapshot(close=100.0, bar_number=100, timestamp_ms=1704067200000)
    s.on_bar(snap)

    system_content = captured[0]["content"]
    assert "LEARNED BELIEFS" in system_content
    assert "Banks work in trends" in system_content


def test_experience_records_trade_entry():
    """When LLM places a BUY with conviction 8+, ExperienceManager records it."""
    s = _init_strategy({"initial_capital": 100_000, "reset_experience": True})
    _warm_up_daily(s)
    _warm_up_m15(s)
    s.bar_count = s.llm_interval_bars - 1

    s.client.chat_completion = MagicMock(
        return_value=json.dumps([{
            "action": "BUY", "symbol": "TEST", "quantity": 5,
            "order_type": "MARKET", "product_type": "CNC",
            "stop_price": 95.0,
            "reasoning": "THESIS: Strong momentum. EVIDENCE: RSI 35. CONVICTION: 9/10",
        }])
    )

    snap = _make_snapshot(close=100.0, bar_number=100, timestamp_ms=1704067200000)
    s.on_bar(snap)

    assert "TEST" in s.experience.open_trades
    assert "THESIS" in s.experience.open_trades["TEST"]["thesis"]


def test_on_complete_includes_experience_stats():
    """on_complete returns experience metadata."""
    s = _init_strategy({"initial_capital": 100_000, "reset_experience": True})
    s.experience.beliefs = [{"text": "Test belief", "regime": "all", "created_at": 0, "strength": 1.0}]
    result = s.on_complete()
    assert result["beliefs"] == [{"text": "Test belief", "regime": "all", "created_at": 0, "strength": 1.0}]
    assert result["reflections"] == 0
    assert result["missed_opportunities"] == 0


# ---------------------------------------------------------------------------
# Conviction parsing and scaling
# ---------------------------------------------------------------------------

def test_extract_conviction():
    """Parse conviction score from reasoning text."""
    s = _init_strategy({"reset_experience": True})
    assert s._extract_conviction("THESIS: blah. CONVICTION: 8/10") == 8
    assert s._extract_conviction("CONVICTION: 10/10") == 10
    assert s._extract_conviction("conviction: 6/10") == 6
    assert s._extract_conviction("some text conviction 9") == 9
    assert s._extract_conviction("no conviction mentioned") == 0
    assert s._extract_conviction("") == 0


def test_conviction_below_8_rejected():
    """Entry with conviction < 8 is rejected by guardrails."""
    s = _init_strategy({"initial_capital": 100_000, "reset_experience": True})
    _warm_up_daily(s)
    _warm_up_m15(s)
    s.bar_count = s.llm_interval_bars - 1

    s.client.chat_completion = MagicMock(
        return_value=json.dumps([{
            "action": "BUY", "symbol": "TEST", "quantity": 10,
            "order_type": "MARKET", "product_type": "CNC",
            "stop_price": 95.0,
            "reasoning": "THESIS: Okay setup. CONVICTION: 7/10",
        }])
    )

    snap = _make_snapshot(close=100.0, bar_number=100, timestamp_ms=1704067200000)
    signals = s.on_bar(snap)

    buy_signals = [sig for sig in signals if sig.action == "BUY"]
    assert len(buy_signals) == 0  # conviction 7 < threshold 8


def test_conviction_8_scales_to_50pct():
    """Conviction 8 → 50% of requested quantity."""
    s = _init_strategy({"initial_capital": 100_000, "reset_experience": True})
    _warm_up_daily(s)
    _warm_up_m15(s)
    s.bar_count = s.llm_interval_bars - 1

    s.client.chat_completion = MagicMock(
        return_value=json.dumps([{
            "action": "BUY", "symbol": "TEST", "quantity": 20,
            "order_type": "MARKET", "product_type": "CNC",
            "stop_price": 95.0,
            "reasoning": "THESIS: Good setup. CONVICTION: 8/10",
        }])
    )

    snap = _make_snapshot(close=100.0, bar_number=100, timestamp_ms=1704067200000)
    signals = s.on_bar(snap)

    buy_signals = [sig for sig in signals if sig.action == "BUY"]
    assert len(buy_signals) == 1
    assert buy_signals[0].quantity == 10  # 20 * 0.5 = 10


def test_conviction_10_full_size():
    """Conviction 10 → 100% of requested quantity."""
    s = _init_strategy({"initial_capital": 100_000, "reset_experience": True})
    _warm_up_daily(s)
    _warm_up_m15(s)
    s.bar_count = s.llm_interval_bars - 1

    s.client.chat_completion = MagicMock(
        return_value=json.dumps([{
            "action": "BUY", "symbol": "TEST", "quantity": 20,
            "order_type": "MARKET", "product_type": "CNC",
            "stop_price": 95.0,
            "reasoning": "THESIS: Perfect setup. CONVICTION: 10/10",
        }])
    )

    snap = _make_snapshot(close=100.0, bar_number=100, timestamp_ms=1704067200000)
    signals = s.on_bar(snap)

    buy_signals = [sig for sig in signals if sig.action == "BUY"]
    assert len(buy_signals) == 1
    assert buy_signals[0].quantity == 20  # 20 * 1.0 = 20


# Import LLMClientError for LLM failure test
from strategies.llm_client import LLMClientError  # noqa: E402
