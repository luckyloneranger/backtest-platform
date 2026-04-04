# Backtest Platform

An Indian market backtesting platform for evaluating trading strategies against historical data from Zerodha Kite Connect API.

**Rust engine** for high-performance event-driven backtesting + **Python strategy server** for flexible strategy development, connected via gRPC.

## Features

- **Event-driven backtesting** — bar-by-bar simulation with realistic order matching (market, limit, stop-loss)
- **Zerodha Kite Connect integration** — fetch instruments, historical candles (minute to daily), OI data, continuous futures
- **Indian market cost model** — Zerodha brokerage, STT, GST, SEBI fees, stamp duty
- **Circuit limit checking** — optional order rejection at upper/lower circuit bounds
- **Margin validation** — optional position sizing limits
- **Performance metrics** — Sharpe, Sortino, Calmar, max drawdown, CAGR, win rate, profit factor
- **All Kite intervals** — minute, 3min, 5min, 10min, 15min, 30min, 60min, daily
- **Auto candle chunking** — transparently handles Kite's 2000-candle API limit

## Architecture

```
CLI (Rust) ─┬─ Data Manager ──► SQLite (instruments) + Parquet (candles)
             ├─ Backtest Engine ──gRPC──► Python Strategy Server
             └─ Results Reporter ──► JSON + Parquet (results)
```

| Component | Language | Purpose |
|-----------|----------|---------|
| `engine/crates/core` | Rust | Backtest engine, order matching, portfolio, costs, metrics |
| `engine/crates/data` | Rust | Kite API client, SQLite instruments, Parquet candle storage |
| `engine/crates/proto` | Rust | gRPC proto definitions + codegen |
| `engine/crates/cli` | Rust | CLI binary (`backtest`) |
| `strategies/` | Python | gRPC strategy server + strategy implementations |

## Prerequisites

- **Rust** (1.75+)
- **Python** (3.11+)
- **protoc** (Protocol Buffers compiler) — `brew install protobuf`
- **Zerodha Kite Connect** API subscription (₹2000/mo) for live data — or use synthetic data for testing

## Quick Start

### 1. Build the engine

```bash
cd engine
cargo build --release -p backtest-cli
```

### 2. Set up the Python strategy server

```bash
cd strategies
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
./generate_proto.sh
```

### 3. Generate test data (no API key needed)

```bash
cd ..
./engine/target/release/backtest data generate-test-data \
  --symbol TESTSTOCK --from 2023-01-01 --to 2023-12-31 \
  --interval day --start-price 1000
```

### 4. Start the strategy server

```bash
cd strategies
source .venv/bin/activate
python -m server.server
```

### 5. Run a backtest (in another terminal)

```bash
./engine/target/release/backtest run \
  --strategy sma_crossover \
  --symbols TESTSTOCK \
  --from 2023-01-01 --to 2023-12-31 \
  --capital 1000000 --interval day \
  --params '{"fast_period": 10, "slow_period": 30}'
```

### 6. View results

```bash
./engine/target/release/backtest results list
./engine/target/release/backtest results show <backtest_id>
```

## Using Real Market Data

### Set up Kite Connect credentials

```bash
# .env (already gitignored)
KITE_API_KEY=your_api_key
KITE_API_SECRET=your_api_secret
KITE_ACCESS_TOKEN=your_daily_access_token
```

The access token must be refreshed daily via Kite's login flow:
1. Visit `https://kite.zerodha.com/connect/login?v=3&api_key=YOUR_API_KEY`
2. Log in → redirected with `request_token`
3. Exchange for access token via `POST /session/token`

### Fetch data

```bash
export KITE_API_KEY=... KITE_ACCESS_TOKEN=...

# Fetch all instrument metadata (run once)
backtest data fetch-instruments

# Fetch candles (token auto-resolved from instruments.db)
backtest data fetch --symbol RELIANCE --from 2024-01-01 --to 2025-03-31 --interval day
backtest data fetch --symbol RELIANCE --from 2025-03-01 --to 2025-03-31 --interval 5minute

# Continuous futures data
backtest data fetch --symbol NIFTY25APRFUT --from 2024-01-01 --to 2025-03-31 --continuous

# List cached data
backtest data list
```

## Writing a Strategy

Create a Python class in `strategies/strategies/examples/`:

```python
from collections import deque
from server.registry import register
from strategies.base import Strategy, Bar, Portfolio, Signal

@register("my_strategy")
class MyStrategy(Strategy):
    def initialize(self, config: dict) -> None:
        self.period = config.get("period", 20)
        self.prices = deque(maxlen=self.period)

    def on_bar(self, bar: Bar, portfolio: Portfolio) -> list[Signal]:
        self.prices.append(bar.close)
        if len(self.prices) < self.period:
            return []

        # Your logic here
        avg = sum(self.prices) / len(self.prices)
        if bar.close > avg * 1.02:
            return [Signal(action="BUY", symbol=bar.symbol, quantity=1)]
        elif bar.close < avg * 0.98:
            return [Signal(action="SELL", symbol=bar.symbol, quantity=1)]
        return []

    def on_complete(self) -> dict:
        return {"custom_metric": 42}
```

Import it in `strategies/server/server.py`:
```python
import strategies.examples.my_strategy  # noqa: F401
```

Run:
```bash
backtest run --strategy my_strategy --symbols RELIANCE --from 2024-01-01 --to 2025-03-31 \
  --capital 1000000 --interval day --params '{"period": 20}'
```

## Strategy Interface

| Method | Called | Returns |
|--------|--------|---------|
| `initialize(config)` | Once at backtest start | None |
| `on_bar(bar, portfolio)` | Every bar | `list[Signal]` |
| `on_complete()` | After last bar | `dict` (custom metrics) |

**Signal fields**: `action` (BUY/SELL/HOLD), `symbol`, `quantity`, `order_type` (MARKET/LIMIT/SL/SL_M), `limit_price`, `stop_price`

**Bar fields**: `timestamp_ms`, `symbol`, `open`, `high`, `low`, `close`, `volume`, `oi`

**Portfolio fields**: `cash`, `equity`, `positions` (list of `Position` with symbol, quantity, avg_price, unrealized_pnl)

## Running Tests

```bash
# Rust (98 tests)
cd engine && cargo test

# Python (3 tests)
cd strategies && source .venv/bin/activate && pytest tests/ -v

# End-to-end
./tests/e2e_test.sh
```

## Project Structure

```
backtest-platform/
├── engine/                     # Rust workspace
│   ├── Cargo.toml
│   └── crates/
│       ├── core/               # Engine, matching, portfolio, costs, metrics
│       ├── data/               # Kite API, SQLite, Parquet storage
│       ├── proto/              # gRPC proto + codegen
│       └── cli/                # CLI binary
├── strategies/                 # Python strategy server
│   ├── server/                 # gRPC server + registry
│   ├── strategies/             # Strategy base class + examples
│   └── tests/
├── data/                       # Local data cache (gitignored)
├── results/                    # Backtest results (gitignored)
├── tests/                      # E2E test scripts
└── docs/plans/                 # Design documents
```

## License

MIT
