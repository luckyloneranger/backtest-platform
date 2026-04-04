# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

### Rust Engine
```bash
cd engine
cargo build                          # build all crates
cargo build --release -p backtest-cli # release build of CLI binary
cargo test                           # run all tests (98 tests across workspace)
cargo test -p backtest-core           # test single crate
cargo test -p backtest-core -- matching  # test single module
```

### Python Strategy Server
```bash
cd strategies
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
./generate_proto.sh                  # regenerate gRPC stubs from proto
pytest tests/ -v                     # run strategy tests
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
$CLI run --strategy sma_crossover --symbols RELIANCE --from 2024-01-01 --to 2025-03-31 --capital 1000000 --interval day --params '{"fast_period": 10, "slow_period": 30}'
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
- `server/registry.py` — `@register("name")` decorator for strategy discovery
- `server/server.py` — gRPC server: handles `GetRequirements`, `Initialize`, `OnBar`, `OnComplete`
- New strategies go in `strategies/examples/`, decorated with `@register`

### Strategy Data Flow

**Strategy declares requirements → Engine provides data → Strategy decides → Engine executes.**

1. Engine calls `GetRequirements` RPC → strategy returns intervals + lookback per interval
2. Engine loads candle data for each (symbol, interval) from CandleStore
3. Engine ticks at the finest declared interval
4. Each tick: `MarketSnapshot.timeframes` is `{interval → {symbol → BarData}}`. Coarser bars only appear when their candle closes.
5. Strategy returns signals → engine processes orders

### Event Loop (`BacktestEngine::run`)

Per timestamp: process pending orders (with circuit limit + margin checks) → update portfolio prices → build `MarketSnapshot` with all timeframes, lookback, fills, rejections, trades, instruments → call strategy via gRPC → submit new orders from signals. Market orders fill at next bar's open. Costs applied per fill via `ZerodhaCostModel`.

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
- Supported intervals: `minute`, `3minute`, `5minute`, `10minute`, `15minute`, `30minute`, `60minute`, `day`
- `fetch_candles_chunked` auto-splits requests to stay under Kite's 2000-candle limit
- The `StrategyClient` trait in `engine.rs` is the abstraction boundary — `GrpcStrategyClient` is the production impl, tests use mock impls
- `CircuitLimits` in `OrderMatcher` are optional — set via `set_circuit_limits()`, unset means no checking
- `BacktestConfig.margin_available` is optional — when `Some`, orders exceeding the margin are skipped
- Order rejections (circuit limit, margin) are tracked and sent to strategies via `MarketSnapshot.rejections`
- Strategies declare data needs via `required_data()` — engine loads and serves accordingly
- Multi-timeframe: engine ticks at finest interval, coarser bars appear only when their candle closes
