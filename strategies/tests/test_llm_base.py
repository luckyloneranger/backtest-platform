"""Tests for LLMStrategy base class."""

import json
from unittest.mock import patch, MagicMock

from strategies.base import (
    BarData, MarketSnapshot, Portfolio, Position, SessionContext, InstrumentInfo, Signal,
)
from strategies.llm_base import LLMStrategy


# --- Concrete subclass for testing ---

class MockLLMStrategy(LLMStrategy):
    def required_data(self):
        return [{"interval": "day", "lookback": 10}]

    def initialize_strategy(self, config, instruments):
        self.custom_param = config.get("custom_param", 42)

    def build_prompt(self, snapshot):
        return [
            {"role": "system", "content": "You are a trader."},
            {"role": "user", "content": self.format_snapshot(snapshot)},
        ]


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
        context=SessionContext(cash, 1, 100, "2024-01-01", "2024-12-31", ["day"], 10),
    )


def test_initialize_creates_client():
    """initialize() creates an AzureOpenAIClient."""
    with patch("strategies.llm_base.AzureOpenAIClient") as mock_cls:
        s = MockLLMStrategy()
        s.initialize({"custom_param": 99}, {})
        mock_cls.assert_called_once()
        assert s.custom_param == 99


def test_format_snapshot_includes_key_data():
    """format_snapshot produces text with symbol, OHLCV, and portfolio info."""
    with patch("strategies.llm_base.AzureOpenAIClient"):
        s = MockLLMStrategy()
        s.initialize({}, {})
        snap = make_snapshot(close=105.5, symbol="RELIANCE", cash=500_000.0)
        text = s.format_snapshot(snap)
        assert "RELIANCE" in text
        assert "105.5" in text
        assert "500000.00" in text


def test_parse_signals_valid_json():
    """parse_signals correctly parses a valid JSON signal array."""
    with patch("strategies.llm_base.AzureOpenAIClient"):
        s = MockLLMStrategy()
        s.initialize({}, {})
        response = json.dumps([
            {"action": "BUY", "symbol": "TEST", "quantity": 10, "product_type": "CNC"},
            {"action": "SELL", "symbol": "OTHER", "quantity": 5},
        ])
        signals = s.parse_signals(response)
        assert len(signals) == 2
        assert signals[0].action == "BUY"
        assert signals[0].symbol == "TEST"
        assert signals[0].quantity == 10
        assert signals[0].product_type == "CNC"
        assert signals[1].action == "SELL"
        assert signals[1].product_type == "CNC"  # default


def test_parse_signals_json_in_markdown():
    """parse_signals extracts JSON from a markdown code block."""
    with patch("strategies.llm_base.AzureOpenAIClient"):
        s = MockLLMStrategy()
        s.initialize({}, {})
        response = 'Here is my analysis:\n```json\n[{"action": "BUY", "symbol": "TEST", "quantity": 5}]\n```'
        signals = s.parse_signals(response)
        assert len(signals) == 1
        assert signals[0].action == "BUY"


def test_parse_signals_empty_array():
    """parse_signals returns empty list for []."""
    with patch("strategies.llm_base.AzureOpenAIClient"):
        s = MockLLMStrategy()
        s.initialize({}, {})
        signals = s.parse_signals("[]")
        assert signals == []


def test_parse_signals_garbage_returns_empty():
    """parse_signals returns empty list on unparseable response."""
    with patch("strategies.llm_base.AzureOpenAIClient"):
        s = MockLLMStrategy()
        s.initialize({}, {})
        signals = s.parse_signals("I don't know what to do")
        assert signals == []


def test_on_bar_calls_llm_and_returns_signals():
    """on_bar wires together build_prompt -> chat_completion -> parse_signals."""
    with patch("strategies.llm_base.AzureOpenAIClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = json.dumps([
            {"action": "BUY", "symbol": "TEST", "quantity": 10}
        ])
        mock_cls.return_value = mock_client

        s = MockLLMStrategy()
        s.initialize({}, {})
        snap = make_snapshot()
        signals = s.on_bar(snap)

        mock_client.chat_completion.assert_called_once()
        assert len(signals) == 1
        assert signals[0].action == "BUY"


def test_on_bar_llm_error_returns_empty():
    """on_bar returns empty signals when LLM call fails."""
    with patch("strategies.llm_base.AzureOpenAIClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.chat_completion.side_effect = Exception("API error")
        mock_cls.return_value = mock_client

        s = MockLLMStrategy()
        s.initialize({}, {})
        snap = make_snapshot()
        signals = s.on_bar(snap)
        assert signals == []


def test_parse_signals_invalid_order_type_defaults_to_market():
    """Invalid order_type from LLM is replaced with MARKET."""
    with patch("strategies.llm_base.AzureOpenAIClient"):
        s = MockLLMStrategy()
        s.initialize({}, {})
        response = json.dumps([
            {"action": "BUY", "symbol": "TEST", "quantity": 10, "order_type": "FOO"},
        ])
        signals = s.parse_signals(response)
        assert len(signals) == 1
        assert signals[0].order_type == "MARKET"


def test_parse_signals_invalid_product_type_defaults_to_cnc():
    """Invalid product_type from LLM is replaced with CNC."""
    with patch("strategies.llm_base.AzureOpenAIClient"):
        s = MockLLMStrategy()
        s.initialize({}, {})
        response = json.dumps([
            {"action": "BUY", "symbol": "TEST", "quantity": 10, "product_type": "INVALID"},
        ])
        signals = s.parse_signals(response)
        assert len(signals) == 1
        assert signals[0].product_type == "CNC"


def test_parse_signals_valid_order_and_product_types_preserved():
    """Valid order_type and product_type values are preserved."""
    with patch("strategies.llm_base.AzureOpenAIClient"):
        s = MockLLMStrategy()
        s.initialize({}, {})
        response = json.dumps([
            {"action": "BUY", "symbol": "TEST", "quantity": 10,
             "order_type": "LIMIT", "product_type": "MIS", "limit_price": 100.0},
        ])
        signals = s.parse_signals(response)
        assert len(signals) == 1
        assert signals[0].order_type == "LIMIT"
        assert signals[0].product_type == "MIS"
