# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

### Rust Engine
```bash
cd engine
cargo build                          # build all crates
cargo build --release -p backtest-cli # release build of CLI binary
cargo test                           # run all tests (156 tests across workspace)
cargo test -p backtest-core           # test single crate (129 tests)
cargo test -p backtest-core -- matching  # test single module
```

### Python Strategy Server
```bash
cd strategies
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
./generate_proto.sh                  # regenerate gRPC stubs from proto
pytest tests/ -v                     # run strategy tests (76 tests)
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
$CLI data import-corporate-actions --file actions.csv  # import split/bonus/dividend data
$CLI run --strategy sma_crossover --symbols RELIANCE --from 2024-01-01 --to 2025-03-31 --capital 1000000 --interval day --exchange NSE --params '{"fast_period": 10, "slow_period": 30}'
$CLI run --strategy rsi_daily_trend --symbols RELIANCE,INFY,TCS --from 2024-01-01 --to 2024-12-31 --capital 1000000 --interval 15minute --max-drawdown 0.10 --max-volume-pct 0.10 --max-exposure 0.80 --reference-symbols "NIFTY 50"
$CLI results list
$CLI results show <backtest_id>
```

## Architecture

Rust engine + Python strategy server communicating via gRPC (port 50051).

### Rust Workspace (`engine/crates/`)

| Crate | Purpose | Key types |
|-------|---------|-----------|
| `backtest-proto` | gRPC proto definitions + codegen | `BarEvent`, `Signal`, `StrategyService`, `DataRequirements` |
| `backtest-core` | Engine, matching, portfolio, costs, metrics, calendar | `BacktestEngine`, `OrderMatcher`, `PortfolioManager`, `ZerodhaCostModel`, `MetricsReport`, `MarketSnapshot`, `TradingCalendar` |
| `backtest-data` | Kite API client, SQLite instruments, Parquet candles | `KiteClient`, `InstrumentStore`, `CandleStore`, `CorporateAction` |
| `backtest-cli` | CLI binary (`backtest`) | Subcommands: `data`, `run`, `results` |

Dependency flow: `cli → data → core → proto`

### Python (`strategies/`)

- `strategies/base.py` — Abstract `Strategy` class with `required_data()`, `initialize()`, `on_bar()`, `on_complete()`. Also defines `MarketSnapshot`, `BarData`, `InstrumentInfo`, `FillInfo`, `OrderRejection`, `TradeInfo`, `SessionContext`, `PendingOrder`.
- `strategies/indicators.py` — Shared technical indicators: `compute_sma`, `compute_ema`, `compute_rsi`, `compute_atr`.
- `strategies/llm_base.py` — `LLMStrategy` subclass of `Strategy`. Handles Azure OpenAI client init, snapshot formatting, and signal parsing. LLM strategies subclass this and implement `build_prompt()`.
- `strategies/llm_client.py` — `AzureOpenAIClient` wrapper. Reads env vars, calls Azure OpenAI REST API, retry with backoff on HTTP errors and network failures.
- `server/registry.py` — `@register("name")` decorator for strategy discovery
- `server/server.py` — gRPC server: handles `GetRequirements`, `Initialize`, `OnBar`, `OnComplete`
- Deterministic strategies go in `strategies/strategies/deterministic/`, LLM strategies in `strategies/strategies/llm/`, all decorated with `@register`

### Strategy Data Flow

**Strategy declares requirements → CLI loads data with warmup → Engine ticks active bars → Strategy decides → Engine executes.**

1. Engine calls `GetRequirements` RPC → strategy returns intervals + lookback per interval
2. CLI loads warmup bars (before `--from`) into pre-populated lookback buffers, and active bars (`--from` to `--to`) for the engine to tick through. Corporate action adjustments applied to both.
3. Engine ticks at the finest declared interval over active bars only
4. Each tick: `MarketSnapshot.timeframes` is `{interval → {symbol → BarData}}`. Coarser bars only appear when their candle closes. `history` contains lookback bars (pre-filled from warmup + accumulating during active period). `pending_orders` shows unfilled orders.
5. Strategy returns signals (BUY/SELL/CANCEL) → engine processes orders with risk controls

### Event Loop (`BacktestEngine::run`)

Per timestamp: check kill switch → process pending orders (with gap handling, volume constraints, circuit limits) → update portfolio prices → auto-squareoff MIS positions at 15:20 IST → check daily loss limit → build `MarketSnapshot` with all timeframes, lookback, fills (with costs), rejections, pending orders, trades, instruments → call strategy via gRPC → submit new orders with risk checks (margin, position limit, exposure limit). Market orders fill at next bar's open. Limit/SL orders handle gaps (fill at bar.open when gapped through). SL-M orders fill at market price with slippage. Costs applied per fill via `ZerodhaCostModel`. Short selling supported (negative positions).

### Data Storage

- **SQLite** (`./data/instruments.db`): Instrument metadata + corporate actions table (splits, bonuses, dividends)
- **Parquet** (`./data/{EXCHANGE}/{SYMBOL}/{INTERVAL}/data.parquet`): OHLCV+OI candles
- **Results** (`./results/{id}/`): `config.json`, `metrics.json` (includes per-symbol breakdown, monthly returns, benchmark), `trades.parquet` (with direction column), `equity_curve.parquet`

## Key Conventions

- Rust error handling: `anyhow::Result<T>` everywhere, propagate with `?`
- Rust tests: `#[cfg(test)] mod tests` inline in each module, `tempfile::TempDir` for file-based tests
- Proto changes require regenerating both Rust (`cargo build -p backtest-proto`) and Python (`cd strategies && ./generate_proto.sh`) stubs
- `generate_proto.sh` applies a `sed` fix for Python relative imports — don't remove it
- Kite access tokens expire daily; env vars `KITE_API_KEY` and `KITE_ACCESS_TOKEN` must be set for data fetching
- Azure OpenAI env vars for LLM strategies: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT`
- LLM strategies subclass `LLMStrategy` from `llm_base.py`, not `Strategy` directly. Use `max_completion_tokens` (not `max_tokens`) for newer Azure models.
- Supported intervals: `minute`, `3minute`, `5minute`, `10minute`, `15minute`, `30minute`, `60minute`, `day`
- `fetch_candles_chunked` auto-splits requests to stay under Kite's 2000-candle limit
- The `StrategyClient` trait in `engine.rs` is the abstraction boundary — `GrpcStrategyClient` is the production impl, tests use mock impls
- Shared technical indicators in `strategies/indicators.py` — all strategies import from here, no duplication
- Short selling: positions can be negative. Selling without a position creates a short. Buying covers a short. `ClosedTrade.direction` is `Long` or `Short`.
- MIS auto-squareoff: all MIS positions are force-closed at 15:20 IST each trading day
- Order cancellation: strategies can send `action=CANCEL` to remove pending limit/SL orders. Pending orders visible in `MarketSnapshot.pending_orders`.
- Gap handling: limit/SL orders that gap through fill at `bar.open` (not the limit/stop price)
- Volume constraints: fills clamped to `max_volume_pct` of bar volume (default 1.0 = unconstrained). Remainder stays pending.
- Risk controls (all optional via config/CLI flags):
  - `max_drawdown_pct`: kill switch — force-close all positions and stop trading when drawdown exceeds threshold
  - `daily_loss_limit`: reject new buys and close MIS after daily loss exceeds limit, resets next day
  - `max_position_qty`: clamp per-symbol position size
  - `max_exposure_pct`: reject buys that would push total exposure above % of capital
- Trading calendar: `TradingCalendar::nse()` with NSE holidays 2020-2027. Test data generator skips holidays.
- `CandleStore.write()` merges new bars with existing data (deduplicates by timestamp). Does not overwrite.
- Corporate actions: `instruments.db` has `corporate_actions` table. `adjust_for_corporate_actions()` adjusts pre-action OHLCV for splits/bonuses/dividends. Applied in CLI before engine run.
- Reference symbols: `--reference-symbols` loads non-tradable index data (e.g., NIFTY 50) into snapshots. Engine ignores signals for reference symbols.
- Metrics: sample std dev (N-1) for Sharpe/Sortino, CAGR from actual equity curve, per-symbol breakdown, monthly returns, benchmark comparison (buy-and-hold alpha), trade duration stats.
- Reporting: `results show` displays per-symbol P&L table, monthly returns, benchmark/alpha, trade duration
- `ClosedTrade.pnl` is net of costs. `Fill.costs` tracks per-fill transaction costs passed to strategies via gRPC.
- `BarData` includes `timestamp_ms` for time-aware analysis in strategy history
- Lookback buffers are pre-populated from bars before `--from` date — strategies get full history from the first `on_bar` call, no warmup API calls
- Test data generator uses IST (UTC+5:30) timestamps and skips NSE holidays
- Zerodha charges zero brokerage on all equity trades (CNC and MIS). ₹20/order (per side) applies to F&O only.
- Cost model (`ZerodhaCostModel`): zero equity brokerage, STT (0.1% delivery both sides, 0.025% intraday sell-only), transaction charges, GST, SEBI fees, stamp duty
- CLI `run` command supports `--exchange` flag (default NSE) for BSE/MCX backtesting
- Instrument metadata is loaded from `instruments.db` and passed to the engine for correct cost model (instrument type) and strategy use
