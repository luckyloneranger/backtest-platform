# Backtest Platform

An Indian market backtesting platform for evaluating trading strategies against historical data from Zerodha Kite Connect API.

**Rust engine** for high-performance event-driven backtesting + **Python strategy server** for flexible strategy development, connected via gRPC.

## Features

- **Event-driven backtesting** — bar-by-bar simulation with realistic order matching (market, limit, stop-loss)
- **Multi-timeframe support** — strategies declare which intervals they need; engine loads and serves all of them
- **Rich strategy context** — strategies receive lookback bars, instrument metadata, fill feedback, order rejections, trade history, and session context via `MarketSnapshot`
- **Zerodha Kite Connect integration** — fetch instruments, historical candles (minute to daily), OI data, continuous futures
- **Indian market cost model** — zero equity brokerage (Zerodha current pricing), STT, GST, SEBI fees, stamp duty; ₹20/order for F&O only
- **Product type support** — strategies choose CNC (delivery), MIS (intraday), or NRML (F&O) per signal, with correct cost calculation for each
- **Circuit limit checking** — optional order rejection at upper/lower circuit bounds
- **Margin validation** — optional position sizing limits
- **Performance metrics** — Sharpe, Sortino, Calmar, max drawdown, CAGR, win rate, profit factor
- **All Kite intervals** — minute, 3min, 5min, 10min, 15min, 30min, 60min, daily
- **Auto candle chunking** — transparently handles Kite's 2000-candle API limit
- **Multi-symbol backtests** — all symbols grouped per timestamp in one `on_bar` call

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
cp .env.example .env
# Edit .env with your API key and secret
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

Strategies declare what data they need, receive a rich `MarketSnapshot`, and return trading signals.

Create a Python class in `strategies/strategies/examples/`:

```python
from server.registry import register
from strategies.base import Strategy, MarketSnapshot, InstrumentInfo, Signal

@register("my_strategy")
class MyStrategy(Strategy):
    def required_data(self) -> list[dict]:
        """Declare which timeframes and lookback this strategy needs."""
        return [
            {"interval": "5minute", "lookback": 50},
            {"interval": "day", "lookback": 200},
        ]

    def initialize(self, config: dict, instruments: dict[str, InstrumentInfo]) -> None:
        self.threshold = config.get("threshold", 0.02)
        self.risk_pct = config.get("risk_pct", 0.2)  # 20% of capital per trade

    def on_bar(self, snapshot: MarketSnapshot) -> list[Signal]:
        signals = []

        # 5-minute bars available every tick
        if "5minute" in snapshot.timeframes:
            for symbol, bar in snapshot.timeframes["5minute"].items():
                # Dynamic position sizing: allocate risk_pct of available cash
                qty = int(snapshot.portfolio.cash * self.risk_pct / bar.close)
                if qty <= 0:
                    continue

                # Check daily trend for confirmation
                daily_bars = snapshot.history.get((symbol, "day"), [])
                if daily_bars and bar.close > daily_bars[-1].close * (1 + self.threshold):
                    signals.append(Signal(action="BUY", symbol=symbol, quantity=qty, product_type="MIS"))

        # Check for rejected orders from last bar
        for rejection in snapshot.rejections:
            print(f"Order rejected: {rejection.symbol} {rejection.reason}")

        return signals

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
  --capital 1000000 --params '{"threshold": 0.02, "risk_pct": 0.2}'
```

### Included strategies

| Strategy | Timeframes | Description |
|----------|-----------|-------------|
| `sma_crossover` | day | Simple Moving Average crossover (golden/death cross) |
| `rsi_daily_trend` | 15min + day | RSI for entry timing, daily EMA for trend filter, dynamic position sizing |
| `donchian_breakout` | 15min + day | Donchian channel breakout with volume confirmation and ATR trailing stop |

## Strategy Interface

| Method | Called | Receives | Returns |
|--------|--------|----------|---------|
| `required_data()` | Before init | Nothing | `list[dict]` — intervals + lookback |
| `initialize(config, instruments)` | Once at start | Config dict + instrument metadata | None |
| `on_bar(snapshot)` | Every bar (finest interval) | `MarketSnapshot` | `list[Signal]` |
| `on_complete()` | After last bar | Nothing | `dict` (custom metrics) |

### MarketSnapshot fields

| Field | Type | Description |
|-------|------|-------------|
| `timestamp_ms` | int | Current timestamp |
| `timeframes` | `dict[str, dict[str, BarData]]` | `interval → symbol → bar` (only intervals with new candles) |
| `history` | `dict[tuple[str, str], list[BarData]]` | `(symbol, interval) → last N bars` |
| `portfolio` | `Portfolio` | Cash, equity, positions |
| `instruments` | `dict[str, InstrumentInfo]` | lot_size, tick_size, expiry, strike, circuit limits |
| `fills` | `list[FillInfo]` | Fills from previous bar |
| `rejections` | `list[OrderRejection]` | Rejected orders with reasons |
| `closed_trades` | `list[TradeInfo]` | All completed trades |
| `context` | `SessionContext` | initial_capital, bar_number, total_bars, dates, intervals |

### Signal fields

`action` (BUY/SELL/HOLD), `symbol`, `quantity`, `order_type` (MARKET/LIMIT/SL/SL_M), `limit_price`, `stop_price`, `product_type` (CNC/MIS/NRML)

## Running Tests

```bash
# Rust (98 tests)
cd engine && cargo test

# Python (21 tests)
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
