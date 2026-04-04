# Deterministic & LLM Strategy Restructure Design

Date: 2026-04-04

## Goal

Restructure the Python strategies directory to separate deterministic (rule-based) and LLM-based strategies, and implement a first LLM strategy using Azure OpenAI as a direct signal generator.

## Directory Layout

```
strategies/
├── pyproject.toml                    # + requests dependency
├── strategies/
│   ├── base.py                       # unchanged — shared Strategy ABC
│   ├── llm_base.py                   # NEW — LLMStrategy ABC
│   ├── llm_client.py                 # NEW — AzureOpenAIClient wrapper
│   ├── deterministic/
│   │   ├── __init__.py
│   │   ├── sma_crossover.py          # moved from examples/
│   │   ├── rsi_daily_trend.py        # moved from examples/
│   │   └── donchian_breakout.py      # moved from examples/
│   └── llm/
│       ├── __init__.py
│       └── llm_signal_generator.py   # NEW — direct signal generator
├── server/
│   └── server.py                     # updated imports
└── tests/
    ├── test_sma_crossover.py         # updated imports
    ├── test_rsi_daily_trend.py       # updated imports
    ├── test_donchian_breakout.py     # updated imports
    ├── test_llm_client.py            # NEW — mocked HTTP tests
    └── test_llm_signal_generator.py  # NEW — mocked LLM response tests
```

`strategies/strategies/examples/` is deleted after moving files to `deterministic/`.

## New Components

### llm_client.py — AzureOpenAIClient

- Reads `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT` from env vars
- Single `chat_completion(messages, temperature, max_tokens)` method
- POST to Azure OpenAI chat completions REST API (no OpenAI SDK — uses `requests`)
- Retry with exponential backoff on 429/5xx (max 3 retries)
- Raises `LLMClientError` on auth failures or missing env vars
- `temperature=0.0` default for reproducibility
- Configurable `api_version` (default `2024-02-01`)

### llm_base.py — LLMStrategy

Subclasses `Strategy` from `base.py`. Provides:

- `initialize()`: creates `AzureOpenAIClient`, reads LLM config from strategy params
- `format_snapshot(snapshot) -> str`: converts MarketSnapshot to compact text (timestamp, OHLCV per symbol, portfolio cash/equity/positions, recent history)
- `parse_signals(llm_response, symbols) -> list[Signal]`: extracts JSON array from LLM response, validates, converts to Signal objects. Returns empty list on parse failure (logged).
- `on_bar()`: calls `build_prompt()` → `chat_completion()` → `parse_signals()`

Subclasses implement:
- `required_data()` — same as base Strategy
- `initialize_strategy(config, instruments)` — strategy-specific init
- `build_prompt(snapshot) -> list[dict]` — build the messages list for the LLM

### llm_signal_generator.py — First LLM Strategy

Registered as `"llm_signal_generator"`. Direct signal generator pattern:

- `required_data()`: day interval, 20-bar lookback
- System prompt instructs LLM to act as a quantitative trader, analyze OHLCV + portfolio, return JSON signal array
- System prompt overridable via `--params '{"system_prompt": "..."}'`
- LLM params (`temperature`, `max_tokens`, `risk_pct`) also configurable via `--params`
- Called every bar

Expected LLM response format:
```json
[{"action": "BUY", "symbol": "RELIANCE", "quantity": 10, "product_type": "CNC"}]
```

## Changes to Existing Files

### server/server.py

Update imports:
```python
import strategies.deterministic.sma_crossover
import strategies.deterministic.rsi_daily_trend
import strategies.deterministic.donchian_breakout
import strategies.llm.llm_signal_generator
```

### pyproject.toml

Add `requests` to dependencies.

### Tests

Existing test logic unchanged — only import paths update from `strategies.examples.*` to `strategies.deterministic.*`. New LLM tests mock `AzureOpenAIClient.chat_completion` to return canned JSON (no API calls in tests).

## What Doesn't Change

- `base.py` — Strategy ABC unchanged
- `server/registry.py` — `@register` decorator unchanged
- gRPC proto definitions — unchanged
- Rust engine — unchanged
- Strategy discovery mechanism — same `@register` + manual imports in `server.py`
