# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

### Rust Engine
```bash
cd engine
cargo build                          # build all crates
cargo build --release -p backtest-cli # release build of CLI binary
cargo test                           # run all tests (108 tests across workspace)
cargo test -p backtest-core           # test single crate (85 tests)
cargo test -p backtest-core -- matching  # test single module
```

### Python Strategy Server
```bash
cd strategies
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
./generate_proto.sh                  # regenerate gRPC stubs from proto
pytest tests/ -v                     # run strategy tests (54 tests)
python -m server.server              # start gRPC server on port 50051
```

### End-to-End
```bash
./tests/e2e_test.sh                  # full pipeline: build → generate data → start server → backtest → results
```

### CLI (after `cargo build --release -p backtest-cli`)
```bash
CLI=./engine/target/release/backtest
$CLI data fetch-instruments                    # fetch all instruments from Kite (must run first)
$CLI data fetch --symbol RELIANCE --from 2024-01-01 --to 2025-03-31 --interval day
$CLI data fetch --symbol RELIANCE --from 2025-03-01 --to 2025-03-31 --interval 5minute  # auto-chunks
$CLI data list
$CLI run --strategy sma_crossover --symbols RELIANCE --from 2024-01-01 --to 2025-03-31 --capital 1000000 --interval day --exchange NSE --params '{"fast_period": 10, "slow_period": 30}'
$CLI results list
$CLI results show <backtest_id>
```

## Architecture

Rust engine + Python strategy server communicating via gRPC (port 50051).

### Rust Workspace (`engine/crates/`)

| Crate | Purpose | Key types |
|-------|---------|-----------|
| `backtest-proto` | gRPC proto definitions + codegen | `BarEvent`, `Signal`, `StrategyService`, `DataRequirements` |
| `backtest-core` | Engine, matching, portfolio, costs, metrics | `BacktestEngine`, `OrderMatcher`, `PortfolioManager`, `ZerodhaCostModel`, `MetricsReport`, `MarketSnapshot` |
| `backtest-data` | Kite API client, SQLite instruments, Parquet candles | `KiteClient`, `InstrumentStore`, `CandleStore`, `QuoteData` |
| `backtest-cli` | CLI binary (`backtest`) | Subcommands: `data`, `run`, `results` |

Dependency flow: `cli → data → core → proto`

### Python (`strategies/`)

- `strategies/base.py` — Abstract `Strategy` class with `required_data()`, `initialize()`, `on_bar()`, `on_complete()`. Also defines `MarketSnapshot`, `BarData`, `InstrumentInfo`, `FillInfo`, `OrderRejection`, `TradeInfo`, `SessionContext`.
- `strategies/llm_base.py` — `LLMStrategy` subclass of `Strategy`. Handles Azure OpenAI client init, snapshot formatting, and signal parsing. LLM strategies subclass this and implement `build_prompt()`.
- `strategies/llm_client.py` — `AzureOpenAIClient` wrapper. Reads env vars, calls Azure OpenAI REST API, retry with backoff.
- `server/registry.py` — `@register("name")` decorator for strategy discovery
- `server/server.py` — gRPC server: handles `GetRequirements`, `Initialize`, `OnBar`, `OnComplete`
- Deterministic strategies go in `strategies/strategies/deterministic/`, LLM strategies in `strategies/strategies/llm/`, all decorated with `@register`

### Strategy Data Flow

**Strategy declares requirements → CLI loads data with warmup → Engine ticks active bars → Strategy decides → Engine executes.**

1. Engine calls `GetRequirements` RPC → strategy returns intervals + lookback per interval
2. CLI loads warmup bars (before `--from`) into pre-populated lookback buffers, and active bars (`--from` to `--to`) for the engine to tick through
3. Engine ticks at the finest declared interval over active bars only
4. Each tick: `MarketSnapshot.timeframes` is `{interval → {symbol → BarData}}`. Coarser bars only appear when their candle closes. `history` contains lookback bars (pre-filled from warmup + accumulating during active period).
5. Strategy returns signals → engine processes orders

### Event Loop (`BacktestEngine::run`)

Per timestamp: process pending orders (with circuit limit + margin checks for buys only) → update portfolio prices → build `MarketSnapshot` with all timeframes, lookback, fills (with costs), rejections, trades, instruments → call strategy via gRPC → submit new orders from signals. Market orders fill at next bar's open. SL-M orders fill at market price with slippage when triggered. Costs applied per fill via `ZerodhaCostModel`. Sells without a position are rejected. Oversells are clamped to actual position size.

### Data Storage

- **SQLite** (`./data/instruments.db`): Instrument metadata (token, symbol, name, exchange, lot_size, tick_size, expiry, strike, option_type, segment)
- **Parquet** (`./data/{EXCHANGE}/{SYMBOL}/{INTERVAL}/data.parquet`): OHLCV+OI candles
- **Results** (`./results/{id}/`): `config.json`, `metrics.json`, `trades.parquet`, `equity_curve.parquet`

## Key Conventions

- Rust error handling: `anyhow::Result<T>` everywhere, propagate with `?`
- Rust tests: `#[cfg(test)] mod tests` inline in each module, `tempfile::TempDir` for file-based tests
- Proto changes require regenerating both Rust (`cargo build -p backtest-proto`) and Python (`cd strategies && ./generate_proto.sh`) stubs
- `generate_proto.sh` applies a `sed` fix for Python relative imports — don't remove it
- Kite access tokens expire daily; env vars `KITE_API_KEY` and `KITE_ACCESS_TOKEN` must be set for data fetching
- Azure OpenAI env vars for LLM strategies: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT`
- LLM strategies subclass `LLMStrategy` from `llm_base.py`, not `Strategy` directly
- Supported intervals: `minute`, `3minute`, `5minute`, `10minute`, `15minute`, `30minute`, `60minute`, `day`
- `fetch_candles_chunked` auto-splits requests to stay under Kite's 2000-candle limit
- The `StrategyClient` trait in `engine.rs` is the abstraction boundary — `GrpcStrategyClient` is the production impl, tests use mock impls
- `CircuitLimits` in `OrderMatcher` are optional — set via `set_circuit_limits()`, unset means no checking
- `BacktestConfig.margin_available` is optional — when `Some`, buy orders exceeding the margin are rejected (sells are never margin-blocked)
- Order rejections (circuit limit, margin) are tracked and sent to strategies via `MarketSnapshot.rejections`
- Strategies declare data needs via `required_data()` — CLI loads warmup bars to pre-populate lookback buffers, then only active bars are ticked through (no warmup `on_bar` calls)
- Multi-timeframe: engine ticks at finest interval, coarser bars appear only when their candle closes
- Lookback buffers are pre-populated from bars before `--from` date — strategies get full history from the first `on_bar` call
- Sells without a position are rejected with a warning. Oversells (quantity > position) are clamped to actual position size.
- `ClosedTrade.pnl` is net of costs (entry + exit costs subtracted from price-difference PnL)
- `Fill.costs` tracks per-fill transaction costs and is passed to strategies via gRPC
- `BarData` includes `timestamp_ms` for time-aware analysis in strategy history
- SL-M (stop-loss market) orders fill at market price (bar.open ± slippage) when triggered, not at stop price
- Metrics use sample standard deviation (N-1) for Sharpe/Sortino. CAGR derived from actual equity curve timestamps.
- `CandleStore.write()` merges new bars with existing data (deduplicates by timestamp). Does not overwrite.
- Test data generator uses IST (UTC+5:30) timestamps to match real Kite data
- Signals include `product_type`: `CNC` (equity delivery), `MIS` (equity intraday), `NRML` (F&O overnight). Engine derives `is_intraday` from this for cost calculation.
- Zerodha charges zero brokerage on all equity trades (CNC and MIS). ₹20/order (per side) applies to F&O only.
- Cost model (`ZerodhaCostModel`): zero equity brokerage, STT (0.1% delivery both sides, 0.025% intraday sell-only), transaction charges, GST, SEBI fees, stamp duty
- CLI `run` command supports `--exchange` flag (default NSE) for BSE/MCX backtesting
- Instrument metadata is loaded from `instruments.db` and passed to the engine for correct cost model (instrument type) and strategy use
- `run` command filters bars by `--from/--to` date range; lookback warmup bars are loaded separately and pre-populate the history buffer
