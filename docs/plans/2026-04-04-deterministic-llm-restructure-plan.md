# Deterministic & LLM Strategy Restructure — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restructure `strategies/` to separate deterministic and LLM strategies, add an Azure OpenAI client wrapper, LLM base class, and a first LLM signal generator strategy.

**Architecture:** Move existing strategies from `examples/` into `deterministic/`, create `llm/` for LLM strategies. Add `llm_client.py` (Azure OpenAI REST wrapper) and `llm_base.py` (LLMStrategy ABC that handles client init, snapshot formatting, and signal parsing). All strategies continue using the same `@register` decorator and gRPC flow.

**Tech Stack:** Python 3.11+, requests (Azure OpenAI REST API), pytest, unittest.mock

---

### Task 1: Create `deterministic/` directory and move strategies

**Files:**
- Create: `strategies/strategies/deterministic/__init__.py`
- Move: `strategies/strategies/examples/sma_crossover.py` → `strategies/strategies/deterministic/sma_crossover.py`
- Move: `strategies/strategies/examples/rsi_daily_trend.py` → `strategies/strategies/deterministic/rsi_daily_trend.py`
- Move: `strategies/strategies/examples/donchian_breakout.py` → `strategies/strategies/deterministic/donchian_breakout.py`
- Delete: `strategies/strategies/examples/` (entire directory)

**Step 1: Create the deterministic directory and move files**

```bash
cd strategies
mkdir -p strategies/deterministic
mv strategies/examples/sma_crossover.py strategies/deterministic/
mv strategies/examples/rsi_daily_trend.py strategies/deterministic/
mv strategies/examples/donchian_breakout.py strategies/deterministic/
touch strategies/deterministic/__init__.py
rm -rf strategies/examples/
```

**Step 2: Update test imports**

In `tests/test_sma_crossover.py`, change:
```python
# Old:
from strategies.examples.sma_crossover import SmaCrossover
# New:
from strategies.deterministic.sma_crossover import SmaCrossover
```

In `tests/test_rsi_daily_trend.py`, change:
```python
# Old:
from strategies.examples.rsi_daily_trend import RsiDailyTrend, compute_rsi, compute_ema
# New:
from strategies.deterministic.rsi_daily_trend import RsiDailyTrend, compute_rsi, compute_ema
```

In `tests/test_donchian_breakout.py`, change:
```python
# Old:
from strategies.examples.donchian_breakout import DonchianBreakout, compute_atr
# New:
from strategies.deterministic.donchian_breakout import DonchianBreakout, compute_atr
```

**Step 3: Update server.py imports**

In `server/server.py`, change:
```python
# Old:
import strategies.examples.sma_crossover  # noqa: F401
import strategies.examples.rsi_daily_trend  # noqa: F401
import strategies.examples.donchian_breakout  # noqa: F401

# New:
import strategies.deterministic.sma_crossover  # noqa: F401
import strategies.deterministic.rsi_daily_trend  # noqa: F401
import strategies.deterministic.donchian_breakout  # noqa: F401
```

**Step 4: Run all existing tests to verify nothing broke**

```bash
cd strategies
source .venv/bin/activate
pytest tests/test_sma_crossover.py tests/test_rsi_daily_trend.py tests/test_donchian_breakout.py -v
```

Expected: All 21 tests pass.

**Step 5: Commit**

```bash
git add -A
git commit -m "refactor: move strategies from examples/ to deterministic/"
```

---

### Task 2: Create `llm/` directory and `llm_client.py`

**Files:**
- Create: `strategies/strategies/llm/__init__.py`
- Create: `strategies/strategies/llm_client.py`
- Create: `strategies/tests/test_llm_client.py`

**Step 1: Create the llm directory**

```bash
cd strategies
mkdir -p strategies/llm
touch strategies/llm/__init__.py
```

**Step 2: Write the failing tests for llm_client.py**

Create `strategies/tests/test_llm_client.py`:

```python
"""Tests for Azure OpenAI client wrapper."""

import json
from unittest.mock import patch, MagicMock

import pytest

from strategies.llm_client import AzureOpenAIClient, LLMClientError


def test_missing_env_vars_raises():
    """Client raises LLMClientError when env vars are not set."""
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(LLMClientError, match="AZURE_OPENAI_ENDPOINT"):
            AzureOpenAIClient()


def test_missing_api_key_raises():
    """Client raises LLMClientError when API key is missing."""
    with patch.dict("os.environ", {"AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com"}, clear=True):
        with pytest.raises(LLMClientError, match="AZURE_OPENAI_API_KEY"):
            AzureOpenAIClient()


def test_missing_deployment_raises():
    """Client raises LLMClientError when deployment is missing."""
    env = {
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
        "AZURE_OPENAI_API_KEY": "test-key",
    }
    with patch.dict("os.environ", env, clear=True):
        with pytest.raises(LLMClientError, match="AZURE_OPENAI_DEPLOYMENT"):
            AzureOpenAIClient()


def test_client_init_from_env():
    """Client initializes correctly from env vars."""
    env = {
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
        "AZURE_OPENAI_API_KEY": "test-key",
        "AZURE_OPENAI_DEPLOYMENT": "gpt-4o",
    }
    with patch.dict("os.environ", env, clear=True):
        client = AzureOpenAIClient()
        assert client.endpoint == "https://test.openai.azure.com"
        assert client.deployment == "gpt-4o"


def test_client_init_explicit_params():
    """Client accepts explicit params over env vars."""
    client = AzureOpenAIClient(
        endpoint="https://custom.openai.azure.com",
        api_key="custom-key",
        deployment="gpt-4o-mini",
    )
    assert client.endpoint == "https://custom.openai.azure.com"
    assert client.deployment == "gpt-4o-mini"


def test_chat_completion_success():
    """Successful API call returns assistant message content."""
    client = AzureOpenAIClient(
        endpoint="https://test.openai.azure.com",
        api_key="test-key",
        deployment="gpt-4o",
    )
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": '[{"action": "BUY", "symbol": "RELIANCE", "quantity": 10}]'}}]
    }

    with patch("strategies.llm_client.requests.post", return_value=mock_response) as mock_post:
        result = client.chat_completion(
            messages=[{"role": "user", "content": "test"}],
            temperature=0.0,
            max_tokens=512,
        )
        assert result == '[{"action": "BUY", "symbol": "RELIANCE", "quantity": 10}]'
        mock_post.assert_called_once()


def test_chat_completion_auth_error():
    """401 raises LLMClientError immediately (no retry)."""
    client = AzureOpenAIClient(
        endpoint="https://test.openai.azure.com",
        api_key="bad-key",
        deployment="gpt-4o",
    )
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.text = "Unauthorized"
    mock_response.raise_for_status.side_effect = Exception("401 Unauthorized")

    with patch("strategies.llm_client.requests.post", return_value=mock_response):
        with pytest.raises(LLMClientError, match="Authentication failed"):
            client.chat_completion(messages=[{"role": "user", "content": "test"}])


def test_chat_completion_retry_on_429():
    """429 triggers retry, succeeds on second attempt."""
    client = AzureOpenAIClient(
        endpoint="https://test.openai.azure.com",
        api_key="test-key",
        deployment="gpt-4o",
    )

    rate_limit_response = MagicMock()
    rate_limit_response.status_code = 429
    rate_limit_response.text = "Rate limited"

    success_response = MagicMock()
    success_response.status_code = 200
    success_response.json.return_value = {
        "choices": [{"message": {"content": "[]"}}]
    }

    with patch("strategies.llm_client.requests.post", side_effect=[rate_limit_response, success_response]):
        with patch("time.sleep"):  # skip actual sleep
            result = client.chat_completion(messages=[{"role": "user", "content": "test"}])
            assert result == "[]"


def test_chat_completion_max_retries_exceeded():
    """Exhausting retries raises LLMClientError."""
    client = AzureOpenAIClient(
        endpoint="https://test.openai.azure.com",
        api_key="test-key",
        deployment="gpt-4o",
        max_retries=2,
    )

    error_response = MagicMock()
    error_response.status_code = 500
    error_response.text = "Internal Server Error"

    with patch("strategies.llm_client.requests.post", return_value=error_response):
        with patch("time.sleep"):
            with pytest.raises(LLMClientError, match="after 2 retries"):
                client.chat_completion(messages=[{"role": "user", "content": "test"}])
```

**Step 3: Run tests to verify they fail**

```bash
pytest tests/test_llm_client.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'strategies.llm_client'`

**Step 4: Implement llm_client.py**

Create `strategies/strategies/llm_client.py`:

```python
"""Azure OpenAI client wrapper for LLM-based strategies."""

import os
import time

import requests


class LLMClientError(Exception):
    """Raised on Azure OpenAI client errors (auth, config, retries exhausted)."""


class AzureOpenAIClient:
    """Thin wrapper around Azure OpenAI chat completions REST API.

    Config priority: explicit params > env vars.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        deployment: str | None = None,
        api_version: str = "2024-02-01",
        max_retries: int = 3,
    ):
        self.endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not self.endpoint:
            raise LLMClientError("AZURE_OPENAI_ENDPOINT not set")

        self._api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        if not self._api_key:
            raise LLMClientError("AZURE_OPENAI_API_KEY not set")

        self.deployment = deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        if not self.deployment:
            raise LLMClientError("AZURE_OPENAI_DEPLOYMENT not set")

        self.api_version = api_version
        self.max_retries = max_retries

        # Strip trailing slash from endpoint
        self.endpoint = self.endpoint.rstrip("/")

    def chat_completion(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        """Send a chat completion request and return the assistant's message content.

        Retries on 429 (rate limit) and 5xx (server errors) with exponential backoff.
        Raises LLMClientError on auth errors or after exhausting retries.
        """
        url = (
            f"{self.endpoint}/openai/deployments/{self.deployment}"
            f"/chat/completions?api-version={self.api_version}"
        )
        headers = {
            "Content-Type": "application/json",
            "api-key": self._api_key,
        }
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        last_error = None
        for attempt in range(self.max_retries):
            response = requests.post(url, headers=headers, json=payload, timeout=60)

            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]

            if response.status_code == 401:
                raise LLMClientError(f"Authentication failed: {response.text}")

            if response.status_code in (429, 500, 502, 503, 504):
                last_error = f"HTTP {response.status_code}: {response.text}"
                wait = 2 ** attempt
                time.sleep(wait)
                continue

            raise LLMClientError(f"Unexpected HTTP {response.status_code}: {response.text}")

        raise LLMClientError(f"Request failed after {self.max_retries} retries: {last_error}")
```

**Step 5: Add `requests` to pyproject.toml**

In `strategies/pyproject.toml`, update:
```toml
dependencies = [
    "grpcio>=1.68",
    "grpcio-tools>=1.68",
    "protobuf>=5",
    "requests>=2.31",
]
```

**Step 6: Install updated deps and run tests**

```bash
cd strategies
pip install -e ".[dev]"
pytest tests/test_llm_client.py -v
```

Expected: All 9 tests pass.

**Step 7: Commit**

```bash
git add -A
git commit -m "feat: add Azure OpenAI client wrapper (llm_client.py)"
```

---

### Task 3: Create `llm_base.py` — LLMStrategy base class

**Files:**
- Create: `strategies/strategies/llm_base.py`
- Create: `strategies/tests/test_llm_base.py`

**Step 1: Write the failing tests for llm_base.py**

Create `strategies/tests/test_llm_base.py`:

```python
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
        assert "500000" in text or "500,000" in text or "500_000" in text


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
    """on_bar wires together build_prompt → chat_completion → parse_signals."""
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
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_llm_base.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'strategies.llm_base'`

**Step 3: Implement llm_base.py**

Create `strategies/strategies/llm_base.py`:

```python
"""LLM strategy base class.

Subclass LLMStrategy to build strategies that use an LLM for signal generation.
Override required_data(), initialize_strategy(), and build_prompt().
"""

import json
import logging
import re
from abc import abstractmethod

from strategies.base import Strategy, MarketSnapshot, InstrumentInfo, Signal
from strategies.llm_client import AzureOpenAIClient

logger = logging.getLogger(__name__)


class LLMStrategy(Strategy):
    """Base class for LLM-powered strategies.

    Handles Azure OpenAI client setup, snapshot formatting, and signal parsing.
    Subclasses implement build_prompt() to define the LLM interaction.
    """

    def initialize(self, config: dict, instruments: dict[str, InstrumentInfo]) -> None:
        self.client = AzureOpenAIClient()
        self.instruments = instruments
        self.llm_config = {
            "temperature": config.get("temperature", 0.0),
            "max_tokens": config.get("max_tokens", 512),
        }
        self.initialize_strategy(config, instruments)

    @abstractmethod
    def initialize_strategy(self, config: dict, instruments: dict[str, InstrumentInfo]) -> None:
        """Strategy-specific initialization. Called after LLM client is set up."""
        pass

    @abstractmethod
    def build_prompt(self, snapshot: MarketSnapshot) -> list[dict]:
        """Build the messages list for the LLM chat completion call."""
        pass

    def on_bar(self, snapshot: MarketSnapshot) -> list[Signal]:
        """Call LLM with the prompt and parse the response into signals."""
        try:
            messages = self.build_prompt(snapshot)
            response = self.client.chat_completion(messages, **self.llm_config)
            return self.parse_signals(response)
        except Exception:
            logger.exception("LLM call failed, returning no signals")
            return []

    def format_snapshot(self, snapshot: MarketSnapshot) -> str:
        """Convert a MarketSnapshot into a compact text summary for the LLM."""
        lines = [f"Timestamp: {snapshot.timestamp_ms}"]

        # Portfolio
        p = snapshot.portfolio
        lines.append(f"Portfolio: cash={p.cash:.0f}, equity={p.equity:.0f}")
        if p.positions:
            for pos in p.positions:
                lines.append(f"  Position: {pos.symbol} qty={pos.quantity} avg={pos.avg_price:.2f} pnl={pos.unrealized_pnl:.2f}")

        # Current bars
        for interval, bars in snapshot.timeframes.items():
            lines.append(f"Timeframe: {interval}")
            for symbol, bar in bars.items():
                lines.append(
                    f"  {symbol}: O={bar.open:.2f} H={bar.high:.2f} L={bar.low:.2f} "
                    f"C={bar.close:.2f} V={bar.volume} OI={bar.oi}"
                )

        # Recent history (last 5 bars per symbol/interval)
        if snapshot.history:
            lines.append("Recent history:")
            for (symbol, interval), bars in snapshot.history.items():
                recent = bars[-5:]
                closes = [f"{b.close:.2f}" for b in recent]
                lines.append(f"  {symbol}/{interval} last {len(recent)} closes: {', '.join(closes)}")

        # Context
        ctx = snapshot.context
        lines.append(f"Bar {ctx.bar_number}/{ctx.total_bars}, capital={ctx.initial_capital:.0f}")

        return "\n".join(lines)

    def parse_signals(self, llm_response: str) -> list[Signal]:
        """Parse LLM response into a list of Signal objects.

        Expects a JSON array. Handles markdown code blocks.
        Returns empty list on parse failure.
        """
        # Try to extract JSON from markdown code block
        md_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", llm_response, re.DOTALL)
        text = md_match.group(1).strip() if md_match else llm_response.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON: %s", llm_response[:200])
            return []

        if not isinstance(data, list):
            logger.warning("LLM response is not a JSON array: %s", type(data))
            return []

        signals = []
        for item in data:
            action = item.get("action", "HOLD").upper()
            if action not in ("BUY", "SELL", "HOLD"):
                continue
            if action == "HOLD":
                continue

            signals.append(Signal(
                action=action,
                symbol=item.get("symbol", ""),
                quantity=int(item.get("quantity", 0)),
                order_type=item.get("order_type", "MARKET").upper(),
                limit_price=float(item.get("limit_price", 0.0)),
                stop_price=float(item.get("stop_price", 0.0)),
                product_type=item.get("product_type", "CNC").upper(),
            ))

        return signals
```

**Step 4: Run tests**

```bash
pytest tests/test_llm_base.py -v
```

Expected: All 8 tests pass.

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: add LLMStrategy base class (llm_base.py)"
```

---

### Task 4: Create `llm_signal_generator.py` — first LLM strategy

**Files:**
- Create: `strategies/strategies/llm/llm_signal_generator.py`
- Create: `strategies/tests/test_llm_signal_generator.py`
- Modify: `strategies/server/server.py` (add import)

**Step 1: Write the failing tests**

Create `strategies/tests/test_llm_signal_generator.py`:

```python
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
        assert "system_prompt" not in dir(s) or s.system_prompt is not None


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
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_llm_signal_generator.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'strategies.llm.llm_signal_generator'`

**Step 3: Implement llm_signal_generator.py**

Create `strategies/strategies/llm/llm_signal_generator.py`:

```python
"""LLM Signal Generator — direct signal generation via Azure OpenAI.

Sends OHLCV data and portfolio state to the LLM each bar and asks it
to return BUY/SELL signals as a JSON array.
"""

from server.registry import register
from strategies.base import MarketSnapshot, InstrumentInfo
from strategies.llm_base import LLMStrategy

DEFAULT_SYSTEM_PROMPT = """\
You are a quantitative trading assistant. You receive market data and portfolio state, \
and you must decide whether to BUY, SELL, or HOLD each symbol.

Respond ONLY with a JSON array of signals. Each signal is an object with:
- "action": "BUY" or "SELL" (omit symbols where you want to HOLD)
- "symbol": the stock symbol
- "quantity": number of shares (integer, positive)
- "product_type": "CNC" (delivery), "MIS" (intraday), or "NRML" (F&O)

Rules:
- Only trade symbols present in the market data
- Position size should not exceed {risk_pct:.0%} of available cash
- Return [] if no action should be taken
- Do NOT include any text outside the JSON array

Example response:
[{{"action": "BUY", "symbol": "RELIANCE", "quantity": 10, "product_type": "CNC"}}]
"""


@register("llm_signal_generator")
class LLMSignalGenerator(LLMStrategy):
    """Direct LLM signal generator — asks the LLM for trading decisions each bar."""

    def required_data(self) -> list[dict]:
        return [{"interval": "day", "lookback": 20}]

    def initialize_strategy(self, config: dict, instruments: dict[str, InstrumentInfo]) -> None:
        self.risk_pct = config.get("risk_pct", 0.2)
        self.system_prompt = config.get(
            "system_prompt",
            DEFAULT_SYSTEM_PROMPT.format(risk_pct=self.risk_pct),
        )

    def build_prompt(self, snapshot: MarketSnapshot) -> list[dict]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.format_snapshot(snapshot)},
        ]

    def on_complete(self) -> dict:
        return {
            "strategy_type": "llm_signal_generator",
            "risk_pct": self.risk_pct,
        }
```

**Step 4: Add import to server.py**

In `server/server.py`, after the deterministic imports, add:
```python
import strategies.llm.llm_signal_generator  # noqa: F401
```

**Step 5: Run tests**

```bash
pytest tests/test_llm_signal_generator.py -v
```

Expected: All 9 tests pass.

**Step 6: Run ALL tests to verify nothing is broken**

```bash
pytest tests/ -v
```

Expected: All tests pass (21 existing + 9 llm_client + 8 llm_base + 9 llm_signal_generator = 47 tests).

**Step 7: Commit**

```bash
git add -A
git commit -m "feat: add LLM signal generator strategy with Azure OpenAI"
```

---

### Task 5: Update CLAUDE.md and README.md

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

**Step 1: Update CLAUDE.md**

Changes:
- Update Python section in Architecture to reflect `deterministic/` and `llm/` dirs
- Add `llm_client.py` and `llm_base.py` to the Python description
- Add Azure OpenAI env vars to Key Conventions
- Update "New strategies go in" path

In Architecture → Python section, replace:
```markdown
- `strategies/base.py` — Abstract `Strategy` class with `required_data()`, `initialize()`, `on_bar()`, `on_complete()`. Also defines `MarketSnapshot`, `BarData`, `InstrumentInfo`, `FillInfo`, `OrderRejection`, `TradeInfo`, `SessionContext`.
- `server/registry.py` — `@register("name")` decorator for strategy discovery
- `server/server.py` — gRPC server: handles `GetRequirements`, `Initialize`, `OnBar`, `OnComplete`
- New strategies go in `strategies/strategies/examples/`, decorated with `@register`
```

With:
```markdown
- `strategies/base.py` — Abstract `Strategy` class with `required_data()`, `initialize()`, `on_bar()`, `on_complete()`. Also defines `MarketSnapshot`, `BarData`, `InstrumentInfo`, `FillInfo`, `OrderRejection`, `TradeInfo`, `SessionContext`.
- `strategies/llm_base.py` — `LLMStrategy` subclass of `Strategy`. Handles Azure OpenAI client init, snapshot formatting, and signal parsing. LLM strategies subclass this and implement `build_prompt()`.
- `strategies/llm_client.py` — `AzureOpenAIClient` wrapper. Reads env vars, calls Azure OpenAI REST API, retry with backoff.
- `server/registry.py` — `@register("name")` decorator for strategy discovery
- `server/server.py` — gRPC server: handles `GetRequirements`, `Initialize`, `OnBar`, `OnComplete`
- Deterministic strategies go in `strategies/strategies/deterministic/`, LLM strategies in `strategies/strategies/llm/`, all decorated with `@register`
```

In Key Conventions, add after the Kite env vars line:
```markdown
- Azure OpenAI env vars for LLM strategies: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT`
- LLM strategies subclass `LLMStrategy` from `llm_base.py`, not `Strategy` directly
```

**Step 2: Update README.md**

Update the "Included strategies" table to show the directory split:
```markdown
### Included strategies

| Strategy | Type | Timeframes | Description |
|----------|------|-----------|-------------|
| `sma_crossover` | Deterministic | day | Simple Moving Average crossover (golden/death cross) |
| `rsi_daily_trend` | Deterministic | 15min + day | RSI for entry timing, daily EMA for trend filter, dynamic position sizing |
| `donchian_breakout` | Deterministic | 15min + day | Donchian channel breakout with volume confirmation and ATR trailing stop |
| `llm_signal_generator` | LLM | day | Direct signal generation via Azure OpenAI — sends market data, receives BUY/SELL |
```

Update the Project Structure tree to show `deterministic/` and `llm/` instead of `examples/`.

Update Prerequisites to mention Azure OpenAI (optional, for LLM strategies).

**Step 3: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: update CLAUDE.md and README for deterministic/LLM strategy split"
```

---

### Task 6: Final verification

**Step 1: Run full Python test suite**

```bash
cd strategies
source .venv/bin/activate
pytest tests/ -v
```

Expected: All 47 tests pass.

**Step 2: Verify server starts (without Azure creds — should start, just can't use LLM strategy)**

```bash
cd strategies
python -m server.server &
sleep 2
kill %1
```

Expected: Server starts, prints "Strategy server listening on port 50051", and shuts down cleanly.

**Step 3: Run Rust tests (sanity check — nothing should have changed)**

```bash
cd engine
cargo test
```

Expected: All 98 Rust tests pass.
