"""Tests for LLM Signal Generator strategy."""

import json
from unittest.mock import patch, MagicMock

from strategies.base import (
    BarData, MarketSnapshot, Portfolio, SessionContext, InstrumentInfo,
)
from strategies.llm.llm_signal_generator import LLMSignalGenerator


def make_snapshot(
    ts: int = 1000,
    close: float = 100.0,
    symbol: str = "TEST",
    cash: float = 100_000.0,
) -> MarketSnapshot:
    bar = BarData(symbol, close, close + 1, close - 1, close, 5000, 0)
    return MarketSnapshot(
        timestamp_ms=ts,
        timeframes={"day": {symbol: bar}},
        history={},
        portfolio=Portfolio(cash=cash, equity=cash, positions=[]),
        instruments={},
        fills=[],
        rejections=[],
        closed_trades=[],
        context=SessionContext(cash, 1, 100, "2024-01-01", "2024-12-31", ["day"], 20),
    )


def test_required_data():
    s = LLMSignalGenerator()
    reqs = s.required_data()
    assert len(reqs) == 1
    assert reqs[0]["interval"] == "day"
    assert reqs[0]["lookback"] == 20


def test_initialize_strategy_defaults():
    with patch("strategies.llm_base.AzureOpenAIClient"):
        s = LLMSignalGenerator()
        s.initialize({}, {})
        assert s.risk_pct == 0.2
        assert s.system_prompt is not None


def test_initialize_strategy_custom_params():
    with patch("strategies.llm_base.AzureOpenAIClient"):
        s = LLMSignalGenerator()
        s.initialize({"risk_pct": 0.5, "system_prompt": "Custom prompt"}, {})
        assert s.risk_pct == 0.5
        assert s.system_prompt == "Custom prompt"


def test_build_prompt_has_system_and_user():
    with patch("strategies.llm_base.AzureOpenAIClient"):
        s = LLMSignalGenerator()
        s.initialize({}, {})
        snap = make_snapshot()
        messages = s.build_prompt(snap)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "TEST" in messages[1]["content"]


def test_build_prompt_system_mentions_json():
    """System prompt must instruct the LLM to return JSON."""
    with patch("strategies.llm_base.AzureOpenAIClient"):
        s = LLMSignalGenerator()
        s.initialize({}, {})
        snap = make_snapshot()
        messages = s.build_prompt(snap)
        assert "JSON" in messages[0]["content"] or "json" in messages[0]["content"]


def test_on_bar_returns_signals():
    """Full integration: on_bar calls LLM and returns parsed signals."""
    with patch("strategies.llm_base.AzureOpenAIClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = json.dumps([
            {"action": "BUY", "symbol": "TEST", "quantity": 50, "product_type": "CNC"},
        ])
        mock_cls.return_value = mock_client

        s = LLMSignalGenerator()
        s.initialize({}, {})
        signals = s.on_bar(make_snapshot())

        assert len(signals) == 1
        assert signals[0].action == "BUY"
        assert signals[0].symbol == "TEST"
        assert signals[0].quantity == 50


def test_on_bar_hold_returns_empty():
    """LLM returning HOLD or empty array means no signals."""
    with patch("strategies.llm_base.AzureOpenAIClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = "[]"
        mock_cls.return_value = mock_client

        s = LLMSignalGenerator()
        s.initialize({}, {})
        signals = s.on_bar(make_snapshot())
        assert signals == []


def test_on_complete():
    with patch("strategies.llm_base.AzureOpenAIClient"):
        s = LLMSignalGenerator()
        s.initialize({"risk_pct": 0.3}, {})
        result = s.on_complete()
        assert result["strategy_type"] == "llm_signal_generator"
        assert result["risk_pct"] == 0.3


def test_registered_name():
    """Strategy is registered under 'llm_signal_generator'."""
    from server.registry import get_strategy
    with patch("strategies.llm_base.AzureOpenAIClient"):
        # Import triggers registration
        import strategies.llm.llm_signal_generator  # noqa: F401
        s = get_strategy("llm_signal_generator")
        assert isinstance(s, LLMSignalGenerator)
