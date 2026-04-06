# Backtest Platform

An Indian market backtesting platform for evaluating trading strategies against historical data from Zerodha Kite Connect API.

**Rust engine** for high-performance event-driven backtesting + **Python strategy server** for flexible strategy development, connected via gRPC.

## Features

- **Event-driven backtesting** — bar-by-bar simulation with realistic order matching (market, limit, stop-loss, SL-M) and gap handling
- **Short selling** — strategies can go long and short; positions can be negative
- **Multi-timeframe support** — strategies declare which intervals they need; lookback buffers pre-populated from warmup data
- **Rich strategy context** — strategies receive lookback bars, instrument metadata, fill feedback (with costs), order rejections, pending orders, trade history, and session context via `MarketSnapshot`
- **Zerodha Kite Connect integration** — fetch instruments, historical candles (minute to daily), OI data, continuous futures
- **Indian market cost model** — zero equity brokerage (Zerodha current pricing), STT, GST, SEBI fees, stamp duty; ₹20/order (per side) for F&O only
- **Product type support** — strategies choose CNC (delivery), MIS (intraday), or NRML (F&O) per signal, with correct cost calculation derived from instrument metadata
- **MIS auto-squareoff** — intraday positions automatically closed at 3:20 PM IST
- **Order management** — cancel pending limit/SL orders; volume constraints limit fill qty to % of bar volume
- **Risk controls** — max drawdown kill switch, daily loss limit, per-symbol position limit, portfolio exposure limit
- **Circuit limit checking** — optional order rejection at upper/lower circuit bounds
- **Performance metrics** — Sharpe, Sortino (sample std dev), Calmar, max drawdown, CAGR, win rate, profit factor, per-symbol breakdown, monthly returns, benchmark comparison, trade duration
- **Trading calendar** — NSE holidays 2020-2027, test data generator skips holidays
- **Corporate actions** — split/bonus/dividend adjustments for historical prices
- **Reference symbols** — non-tradable index data (e.g., NIFTY 50) for regime detection
- **All Kite intervals** — minute, 3min, 5min, 10min, 15min, 30min, 60min, daily
- **Auto candle chunking** — transparently handles Kite's 2000-candle API limit
- **Multi-symbol backtests** — all symbols grouped per timestamp in one `on_bar` call
- **Multi-exchange support** — `--exchange` flag supports NSE, BSE, MCX
- **LLM-based strategies** — Azure OpenAI integration for AI-powered signal generation
- **Candle data merging** — fetching additional date ranges merges with existing data, never overwrites

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
- **Azure OpenAI** (optional) — for LLM-based strategies

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
  --capital 1000000 --interval day --exchange NSE \
  --params '{"fast_period": 10, "slow_period": 30}'
```

### 6. View results

```bash
./engine/target/release/backtest results list
./engine/target/release/backtest results show <backtest_id>
```

## Using Real Market Data

### Set up credentials

```bash
cp .env.example .env
# Edit .env with your Kite Connect and (optionally) Azure OpenAI credentials
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

### Set up Azure OpenAI (for LLM strategies)

```bash
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
export AZURE_OPENAI_API_KEY=your_api_key
export AZURE_OPENAI_DEPLOYMENT=gpt-4o
```

Or add these to your `.env` file. Required only when running LLM-based strategies like `llm_signal_generator`.

## Writing a Strategy

Strategies declare what data they need, receive a rich `MarketSnapshot`, and return trading signals.

Create a Python class in `strategies/strategies/deterministic/` (for rule-based strategies) or `strategies/strategies/llm/` (for LLM-powered strategies):

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
import strategies.deterministic.my_strategy  # noqa: F401
```

Run:
```bash
backtest run --strategy my_strategy --symbols RELIANCE --from 2024-01-01 --to 2025-03-31 \
  --capital 1000000 --params '{"threshold": 0.02, "risk_pct": 0.2}'
```

### Included strategies

| Strategy | Type | Timeframes | Description |
|----------|------|-----------|-------------|
| `sma_crossover` | Deterministic | day | SMA crossover (10/30) with ATR sizing, trailing stops, pyramiding, long+short |
| `rsi_daily_trend` | Deterministic | 15min + day | RSI mean reversion with pyramid entries (RSI 35/25/15), partial exits, ATR stops, cooldown, long+short |
| `donchian_breakout` | Deterministic | 15min + day | Donchian channel breakout with risk-based sizing (0.01), wider stops (2x ATR), long+short |
| `confluence` | Deterministic | day | Multi-indicator confluence (RSI+MACD+Bollinger+ADX+OBV) — trades when 2+ indicators agree |
| `pairs_trading` | Deterministic | day | Statistical pairs trading via cointegration — market-neutral, trades spread z-score |
| `regime_adaptive` | Deterministic | 15min + day | Regime detection (trending/ranging/volatile) with 3-bar smoothing — switches sub-strategies |
| `vwap_reversion` | Intraday | 5min | VWAP mean reversion — buy below VWAP-2.5σ, exit at VWAP. Pure MIS, daily reset. |
| `bollinger_squeeze` | Intraday | 5min | Bollinger squeeze breakout — enter on volatility expansion after compression. Pure MIS. |
| `orb_breakout` | Intraday | 5min | Opening Range Breakout — trade first 30-min range breakout with volume. Pure MIS. |
| `portfolio_combiner` | Adaptive | 15min + day | **Best strategy** — Donchian breakout when trending (ADX>25), RSI mean-reversion when ranging (ADX<20). +32%/+39% across 2024/2025. |
| `intraday_momentum` | Intraday | 5min | Pure momentum — enter on 3-bar burst > 1.5x ATR with volume. Trail 1x ATR. |
| `time_adaptive` | Intraday | 5min | Switches momentum/reversion by time-of-day (opening/midday/closing sessions). |
| `relative_strength` | Intraday | 15min | Long top 3 stocks, short bottom 3 by opening momentum. Market-neutral. |
| `multi_tf_confirm` | Multi-TF | 5+15min+day | Triple confirmation: daily EMA direction + 15min MACD + 5min RSI timing. |
| `llm_signal_generator` | LLM | day | Direct signal generation via Azure OpenAI — full order type control |

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
| `pending_orders` | `list[PendingOrder]` | Unfilled limit/SL orders from previous bars |
| `fills` | `list[FillInfo]` | Fills from previous bar (includes per-fill costs) |
| `rejections` | `list[OrderRejection]` | Rejected orders with reasons |
| `closed_trades` | `list[TradeInfo]` | All completed trades |
| `context` | `SessionContext` | initial_capital, bar_number, total_bars, dates, intervals |

### Signal fields

`action` (BUY/SELL/HOLD/CANCEL), `symbol`, `quantity`, `order_type` (MARKET/LIMIT/SL/SL_M), `limit_price`, `stop_price`, `trigger_price`, `product_type` (CNC/MIS/NRML), `validity` (DAY/IOC), `cancel_order_id`

## Running Tests

```bash
# Rust (176 tests)
cd engine && cargo test

# Python (255 tests)
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
│   ├── strategies/             # Strategy base classes + implementations
│   │   ├── deterministic/      # Rule-based strategies
│   │   └── llm/                # LLM-powered strategies
│   └── tests/
├── data/                       # Local data cache (gitignored)
├── results/                    # Backtest results (gitignored)
├── tests/                      # E2E test scripts
└── docs/plans/                 # Design documents
```

## License

MIT
